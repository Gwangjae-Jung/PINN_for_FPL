import argparse
import os
import shutil
import time
from   typing                  import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from   torch.optim             import Adam
from   torch.optim.lr_scheduler import ExponentialLR
import tqdm

from   .data                   import GaussianRandomField1D, generate_biot_test_data
from   .helpers                import mse, mse_single, relative_l2_error
from   .models                 import SeparableDeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_ics(
    model:     SeparableDeepONet,
    u_batch:   torch.Tensor,
    t_ics:     torch.Tensor,
    z_ics:     torch.Tensor,
    p0_batch:  torch.Tensor,
) -> torch.Tensor:
    """Computes initial condition loss for Biot's consolidation theory.

    ## Description
    Enforces initial displacement $u(z, 0) = 0$ and initial pore pressure $\\tilde{p}(z, 0) = f(0)$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet instance.
        `u_batch` (`torch.Tensor`): Input loading function $f(t)$ at $z = 0$.
        `t_ics` (`torch.Tensor`): Initial time coordinate ($t=0$).
        `z_ics` (`torch.Tensor`): Spatial coordinate grid on $[0, L]$.
        `p0_batch` (`torch.Tensor`): Target initial pore pressure values.

    Returns:
        `torch.Tensor`: Composite initial condition MSE loss.
    """
    # Prediction shape: (N, 1, N_z, 2)
    s_pred: torch.Tensor = model(u_batch, t_ics, z_ics, compute_derivatives=False)
    u_pred: torch.Tensor = s_pred[:, 0, :, 0]
    p_pred: torch.Tensor = s_pred[:, 0, :, 1]

    loss_ic_u: torch.Tensor = mse_single(u_pred)
    loss_ic_p: torch.Tensor = mse(p0_batch.reshape(-1), p_pred.reshape(-1))

    return loss_ic_u + loss_ic_p


def loss_bcs(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_bcs:   torch.Tensor,
    z_bc_0:  torch.Tensor,
    z_bc_1:  torch.Tensor,
) -> torch.Tensor:
    """Computes boundary condition losses for Biot's consolidation problem.

    ## Description
    Enforces:
    - At drained top surface $z = 0$: $u(0, t) = 0$ (or Dirichlet displacement) and $\\tilde{p}(0, t) = 0$.
    - At impermeable bottom $z = L$: $u(L, t) = 0$ and $\\frac{\\partial \\tilde{p}}{\\partial z}(L, t) = 0$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input load functions.
        `t_bcs` (`torch.Tensor`): Boundary time points.
        `z_bc_0` (`torch.Tensor`): Top boundary coordinate ($z=0$).
        `z_bc_1` (`torch.Tensor`): Bottom boundary coordinate ($z=L$).

    Returns:
        `torch.Tensor`: Scalar boundary condition loss.
    """
    # Surface z = 0: u(0, t) = f(t), p(0, t) = 0
    s_0: torch.Tensor = model(u_batch, t_bcs, z_bc_0, compute_derivatives=False)
    u_0: torch.Tensor = s_0[:, :, 0, 0]
    p_0: torch.Tensor = s_0[:, :, 0, 1]

    # Bottom z = 1: u(L, t) = 0, dp/dz(L, t) = 0
    s_1:   torch.Tensor = model(u_batch, t_bcs, z_bc_1, compute_derivatives=True)
    u_1:   torch.Tensor = s_1[:, :, 0, 0]
    dp_dz: torch.Tensor = model.compute_gradient([0, 1])[:, :, 0, 1]

    loss_bc: torch.Tensor = mse_single(p_0) + mse_single(u_1) + mse_single(dp_dz)
    return loss_bc


def loss_res(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_res:   torch.Tensor,
    z_res:   torch.Tensor,
) -> torch.Tensor:
    """Computes coupled PDE residual losses for Biot's consolidation theory:
    $(\\lambda + 2\\mu) u_{zz} - p_z = 0$ and $u_{tz} - \\frac{k}{\\rho g} p_{zz} = 0$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input functions.
        `t_res` (`torch.Tensor`): Collocation time points.
        `z_res` (`torch.Tensor`): Collocation space points.

    Returns:
        `torch.Tensor`: Coupled PDE residual MSE loss.
    """
    _s:    torch.Tensor = model(u_batch, t_res, z_res, compute_derivatives=True)
    s_z:   torch.Tensor = model.compute_gradient([0, 1])
    s_zz:  torch.Tensor = model.compute_gradient([0, 2])
    s_tz:  torch.Tensor = model.compute_gradient([1, 1])

    # Component 0: displacement u; Component 1: pore pressure p
    u_zz:  torch.Tensor = s_zz[:, :, :, 0]
    p_z:   torch.Tensor = s_z[:, :, :, 1]
    u_tz:  torch.Tensor = s_tz[:, :, :, 0]
    p_zz:  torch.Tensor = s_zz[:, :, :, 1]

    # Normalized constants (lambda + 2*mu = 3, k/(rho*g) = 1)
    res_1: torch.Tensor = 3.0 * u_zz - p_z
    res_2: torch.Tensor = u_tz - p_zz

    return mse_single(res_1) + mse_single(res_2)


def loss_fn(
    model:     SeparableDeepONet,
    u_batch:   torch.Tensor,
    t_ics:     torch.Tensor,
    z_ics:     torch.Tensor,
    p0_batch:  torch.Tensor,
    t_bcs:     torch.Tensor,
    z_bc_0:    torch.Tensor,
    z_bc_1:    torch.Tensor,
    t_res:     torch.Tensor,
    z_res:     torch.Tensor,
    lambda_ic: float = 10.0,
    lambda_bc: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates total loss for Biot's consolidation problem.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input functions.
        `t_ics` (`torch.Tensor`): Initial time coordinate.
        `z_ics` (`torch.Tensor`): Spatial coordinates for IC.
        `p0_batch` (`torch.Tensor`): Initial pressure targets.
        `t_bcs` (`torch.Tensor`): Boundary time points.
        `z_bc_0` (`torch.Tensor`): Top boundary coordinate.
        `z_bc_1` (`torch.Tensor`): Bottom boundary coordinate.
        `t_res` (`torch.Tensor`): Interior collocation time points.
        `z_res` (`torch.Tensor`): Interior collocation spatial points.
        `lambda_ic` (`float`, default: `10.0`): Weighting penalty for initial condition.
        `lambda_bc` (`float`, default: `10.0`): Weighting penalty for boundary condition.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metric dictionary.
    """
    l_ic:  torch.Tensor = loss_ics(model, u_batch, t_ics, z_ics, p0_batch)
    l_bc:  torch.Tensor = loss_bcs(model, u_batch, t_bcs, z_bc_0, z_bc_1)
    l_res: torch.Tensor = loss_res(model, u_batch, t_res, z_res)

    total: torch.Tensor = lambda_ic * l_ic + lambda_bc * l_bc + l_res
    logs: Dict[str, float] = {
        'loss_ic':    float(l_ic.item()),
        'loss_bc':    float(l_bc.item()),
        'loss_res':   float(l_res.item()),
        'total_loss': float(total.item()),
    }
    return total, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes training and evaluation of Separable PI-DeepONet on Biot's consolidation theory."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing Biot's Consolidation on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Dataset loading / synthesis
    npy_path: str = os.path.join(os.getcwd(), 'data/biot/Y.npy')
    if os.path.exists(npy_path):
        u_sol = np.load(npy_path)
    else:
        print(f"Dataset not found at '{npy_path}'. Generating synthetic Biot load distributions...")
        grf = GaussianRandomField1D(n_points=args.n_sensors, seed=args.seed)
        loads = grf.sample(args.n_train + args.n_test).numpy()
        u_sol = np.zeros((args.n_train + args.n_test, 101, 101, 2), dtype=np.float32)
        u_sol[:, 0, :, 0] = loads

    u_sol_tensor: torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)
    u0_train:     torch.Tensor = u_sol_tensor[:args.n_train, 0, :, 0].to(device)
    p0_train:     torch.Tensor = u_sol_tensor[:args.n_train, -1, 0, 1].to(device)

    # 2. Collocation Coordinates
    t_ics:  torch.Tensor = torch.zeros((1, 1), device=device)
    z_ics:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_ics_train, device=device).reshape(-1, 1)

    t_bcs:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)
    z_bc_0: torch.Tensor = torch.zeros((1, 1), device=device)
    z_bc_1: torch.Tensor = torch.ones((1, 1), device=device)

    t_res:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_t_train, device=device).reshape(-1, 1)
    z_res:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_z_train, device=device).reshape(-1, 1)

    # 3. Model setup
    args.separable            = True
    args.num_outputs          = 2  # Multi-output: [u, p]
    args.split_trunk          = True
    args.split_branch         = True
    args.trunk_input_features = 2
    args.deg_grad             = [1, 2]  # t degree 1, z degree 2

    config, model = setup_deeponet(args)
    model         = model.to(device)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'biot_sep_p{args.hidden_dim}_r{args.r}')
    os.makedirs(result_dir, exist_ok=True)

    print(f"Starting training on Biot's Consolidation for {args.epochs} epochs...")
    start_time: float = time.time()

    pbar = tqdm.tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if args.batch_size < args.n_train:
            idx = torch.randperm(args.n_train)[:args.batch_size]
            u_batch  = u0_train[idx]
            p0_batch = p0_train[idx]
        else:
            u_batch  = u0_train
            p0_batch = p0_train

        loss, log_dict = loss_fn(
            model     = model,
            u_batch   = u_batch,
            t_ics     = t_ics,
            z_ics     = z_ics,
            p0_batch  = p0_batch,
            t_bcs     = t_bcs,
            z_bc_0    = z_bc_0,
            z_bc_1    = z_bc_1,
            t_res     = t_res,
            z_res     = z_res,
            lambda_ic = args.lambda_ic,
            lambda_bc = args.lambda_bc,
        )

        loss.backward()
        optimizer.step()

        if epoch % args.decay_step == 0:
            scheduler.step()

        if epoch % args.log_step == 0 or epoch == 1:
            pbar.set_postfix({'loss': f"{log_dict['total_loss']:.3e}"})

    elapsed: float = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} s ({elapsed / args.epochs * 1000:.2f} ms/epoch).")

    return None


##################################################
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separable PI-DeepONet for Biot Consolidation Theory")
    parser.add_argument('--seed',               type=int,   default=42)
    parser.add_argument('--epochs',             type=int,   default=1000)
    parser.add_argument('--batch_size',         type=int,   default=100)
    parser.add_argument('--n_train',            type=int,   default=1000)
    parser.add_argument('--n_test',             type=int,   default=100)
    parser.add_argument('--n_sensors',          type=int,   default=101)
    parser.add_argument('--branch_input_features', type=int, default=1)
    parser.add_argument('--p_ics_train',        type=int,   default=101)
    parser.add_argument('--p_bcs_train',        type=int,   default=100)
    parser.add_argument('--p_res_t_train',      type=int,   default=50)
    parser.add_argument('--p_res_z_train',      type=int,   default=50)
    parser.add_argument('--hidden_dim',         type=int,   default=50)
    parser.add_argument('--r',                  type=int,   default=50)
    parser.add_argument('--branch_layers',      nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--trunk_layers',       nargs='+',  type=int, default=[50, 50, 50, 50, 50])
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--gamma',              type=float, default=0.95)
    parser.add_argument('--decay_step',         type=int,   default=1000)
    parser.add_argument('--lambda_ic',          type=float, default=10.0)
    parser.add_argument('--lambda_bc',          type=float, default=10.0)
    parser.add_argument('--log_step',           type=int,   default=100)
    parser.add_argument('--result_dir',         type=str,   default='./results/biot')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
