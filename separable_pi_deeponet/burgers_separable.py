import argparse
import os
import shutil
import time
from   typing                  import Any, Dict, List, Optional, Sequence, Tuple, Union
from   typing_extensions       import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from   torch.optim             import Adam
from   torch.optim.lr_scheduler import ExponentialLR
import tqdm

from   .autograd               import hvp_fwdfwd
from   .data                   import GaussianRandomField1D, generate_burgers_test_data
from   .helpers                import apply_net_sep, mse, mse_single, relative_l2_error
from   .models                 import SeparableDeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_ics(
    model:     SeparableDeepONet,
    u_batch:   torch.Tensor,
    t_ics:     torch.Tensor,
    x_ics:     torch.Tensor,
    s_targets: torch.Tensor,
) -> torch.Tensor:
    """Computes initial condition loss for the 1D Burgers' equation.

    Arguments:
        `model` (`SeparableDeepONet`): The Separable DeepONet model instance.
        `u_batch` (`torch.Tensor`): Batch of input functions evaluated at sensors, shape `(N, n_sensors)`.
        `t_ics` (`torch.Tensor`): Initial time coordinate tensor ($t = 0$), shape `(1, 1)`.
        `x_ics` (`torch.Tensor`): Spatial coordinate tensor on $[0, 1]$, shape `(N_x, 1)`.
        `s_targets` (`torch.Tensor`): Ground truth initial condition values, shape `(N, N_x)`.

    Returns:
        `torch.Tensor`: Scalar MSE loss for initial condition.
    """
    s_pred: torch.Tensor = model(u_batch, t_ics, x_ics, compute_derivatives=False)
    # Remove single time and channel dimension: (N, 1, N_x, 1) -> (N, N_x)
    s_pred_flat: torch.Tensor = s_pred.squeeze(-1).squeeze(1)
    return mse(s_targets.reshape(-1), s_pred_flat.reshape(-1))


def loss_bcs(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_bcs:   torch.Tensor,
    x_bc_0:  torch.Tensor,
    x_bc_1:  torch.Tensor,
) -> torch.Tensor:
    """Computes periodic boundary condition loss for 1D Burgers' equation.

    ## Description
    Enforces periodic conditions $u(t, 0) = u(t, 1)$ and $u_x(t, 0) = u_x(t, 1)$.

    Arguments:
        `model` (`SeparableDeepONet`): The Separable DeepONet model instance.
        `u_batch` (`torch.Tensor`): Batch of input functions, shape `(N, n_sensors)`.
        `t_bcs` (`torch.Tensor`): Collocation time points on $(0, 1]$, shape `(N_t, 1)`.
        `x_bc_0` (`torch.Tensor`): Left spatial boundary coordinate ($x = 0$), shape `(1, 1)`.
        `x_bc_1` (`torch.Tensor`): Right spatial boundary coordinate ($x = 1$), shape `(1, 1)`.

    Returns:
        `torch.Tensor`: Scalar boundary condition loss.
    """
    # Evaluate at left boundary x = 0
    s_0: torch.Tensor = model(u_batch, t_bcs, x_bc_0, compute_derivatives=True)
    s_x_0: torch.Tensor = model.compute_gradient([0, 1])

    # Evaluate at right boundary x = 1
    s_1: torch.Tensor = model(u_batch, t_bcs, x_bc_1, compute_derivatives=True)
    s_x_1: torch.Tensor = model.compute_gradient([0, 1])

    loss_s:   torch.Tensor = mse(s_0.reshape(-1), s_1.reshape(-1))
    loss_s_x: torch.Tensor = mse(s_x_0.reshape(-1), s_x_1.reshape(-1))
    return loss_s + loss_s_x


def loss_res(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_res:   torch.Tensor,
    x_res:   torch.Tensor,
    nu:      float = 0.01,
) -> torch.Tensor:
    """Computes PDE residual loss for the 1D viscous Burgers' equation: $u_t + u u_x - \\nu u_{xx} = 0$.

    Arguments:
        `model` (`SeparableDeepONet`): The Separable DeepONet model instance.
        `u_batch` (`torch.Tensor`): Batch of input functions, shape `(N, n_sensors)`.
        `t_res` (`torch.Tensor`): Collocation time points, shape `(N_t, 1)`.
        `x_res` (`torch.Tensor`): Collocation space points, shape `(N_x, 1)`.
        `nu` (`float`, default: `0.01`): Kinematic viscosity parameter.

    Returns:
        `torch.Tensor`: Scalar residual MSE loss.
    """
    # Forward pass caching derivatives: t degree 1, x degree 2
    s:    torch.Tensor = model(u_batch, t_res, x_res, compute_derivatives=True)
    s_t:  torch.Tensor = model.compute_gradient([1, 0])
    s_x:  torch.Tensor = model.compute_gradient([0, 1])
    s_xx: torch.Tensor = model.compute_gradient([0, 2])

    residual: torch.Tensor = s_t + s * s_x - nu * s_xx
    return mse_single(residual)


def loss_fn(
    model:     SeparableDeepONet,
    u_batch:   torch.Tensor,
    t_ics:     torch.Tensor,
    x_ics:     torch.Tensor,
    s_ics:     torch.Tensor,
    t_bcs:     torch.Tensor,
    x_bc_0:    torch.Tensor,
    x_bc_1:    torch.Tensor,
    t_res:     torch.Tensor,
    x_res:     torch.Tensor,
    lambda_ic: float = 20.0,
    nu:        float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates composite physics-informed loss for Burgers' equation.

    Arguments:
        `model` (`SeparableDeepONet`): The Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Batch of input functions.
        `t_ics` (`torch.Tensor`): Initial time coordinate.
        `x_ics` (`torch.Tensor`): Initial space coordinates.
        `s_ics` (`torch.Tensor`): Initial condition target values.
        `t_bcs` (`torch.Tensor`): Boundary time coordinates.
        `x_bc_0` (`torch.Tensor`): Left boundary location.
        `x_bc_1` (`torch.Tensor`): Right boundary location.
        `t_res` (`torch.Tensor`): Interior collocation time points.
        `x_res` (`torch.Tensor`): Interior collocation spatial points.
        `lambda_ic` (`float`, default: `20.0`): Weighting penalty for initial condition.
        `nu` (`float`, default: `0.01`): Kinematic viscosity.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and dictionary of individual loss components.
    """
    l_ic:  torch.Tensor = loss_ics(model, u_batch, t_ics, x_ics, s_ics)
    l_bc:  torch.Tensor = loss_bcs(model, u_batch, t_bcs, x_bc_0, x_bc_1)
    l_res: torch.Tensor = loss_res(model, u_batch, t_res, x_res, nu)

    total_loss: torch.Tensor = lambda_ic * l_ic + l_bc + l_res
    logs: Dict[str, float] = {
        'loss_ic':    float(l_ic.item()),
        'loss_bc':    float(l_bc.item()),
        'loss_res':   float(l_res.item()),
        'total_loss': float(total_loss.item()),
    }
    return total_loss, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes the training and evaluation routine for Separable PI-DeepONet on 1D Burgers' equation.

    Arguments:
        `args` (`argparse.Namespace`): Command line argument options.

    Returns:
        This function returns no values.
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load or synthesize dataset
    mat_path: str = os.path.join(os.getcwd(), 'data/burgers/Burger.mat')
    if os.path.exists(mat_path):
        data = scipy.io.loadmat(mat_path)
        u_sol = data['output']  # (N, n_t, n_x)
    else:
        print(f"Data file not detected at '{mat_path}'. Generating synthetic Gaussian random field dataset...")
        grf = GaussianRandomField1D(n_points=args.n_sensors, seed=args.seed)
        u_samples = grf.sample(args.n_train + args.n_test)
        u_sol = np.zeros((args.n_train + args.n_test, 101, 101), dtype=np.float32)
        u_sol[:, 0, :] = u_samples.numpy()

    u_sol_tensor: torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)
    u0_train:     torch.Tensor = u_sol_tensor[:args.n_train, 0, :].to(device)

    # 2. Build Collocation Points
    t_ics:  torch.Tensor = torch.zeros((1, 1), device=device)
    x_ics:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_ics_train, device=device).reshape(-1, 1)
    s_ics:  torch.Tensor = u0_train

    t_bcs:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)
    x_bc_0: torch.Tensor = torch.zeros((1, 1), device=device)
    x_bc_1: torch.Tensor = torch.ones((1, 1), device=device)

    t_res:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_t_train, device=device).reshape(-1, 1)
    x_res:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_x_train, device=device).reshape(-1, 1)

    # 3. Model setup
    args.separable             = True
    args.trunk_input_features  = 2
    args.deg_grad              = [1, 2]  # t degree 1, x degree 2

    config, model = setup_deeponet(args)
    model         = model.to(device)

    # 4. Optimizer and scheduler
    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'burgers_sep_p{args.hidden_dim}_r{args.r}')
    os.makedirs(result_dir, exist_ok=True)

    print(f"Starting training for {args.epochs} epochs...")
    start_time: float = time.time()

    pbar = tqdm.tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Mini-batch subsampling of initial conditions
        if args.batch_size < args.n_train:
            idx = torch.randperm(args.n_train)[:args.batch_size]
            u_batch = u0_train[idx]
            s_ics_batch = s_ics[idx]
        else:
            u_batch = u0_train
            s_ics_batch = s_ics

        loss, log_dict = loss_fn(
            model     = model,
            u_batch   = u_batch,
            t_ics     = t_ics,
            x_ics     = x_ics,
            s_ics     = s_ics_batch,
            t_bcs     = t_bcs,
            x_bc_0    = x_bc_0,
            x_bc_1    = x_bc_1,
            t_res     = t_res,
            x_res     = x_res,
            lambda_ic = args.lambda_ic,
            nu        = args.nu,
        )

        loss.backward()
        optimizer.step()

        if epoch % args.decay_step == 0:
            scheduler.step()

        if epoch % args.log_step == 0 or epoch == 1:
            pbar.set_postfix({
                'loss': f"{log_dict['total_loss']:.3e}",
                'ic':   f"{log_dict['loss_ic']:.3e}",
                'res':  f"{log_dict['loss_res']:.3e}",
            })

    elapsed: float = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds ({elapsed / args.epochs * 1000:.2f} ms/epoch).")

    # Save model checkpoint
    ckpt_path: str = os.path.join(result_dir, 'model_checkpoint.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to '{ckpt_path}'.")

    return None


##################################################
def get_args() -> argparse.Namespace:
    """Parses command line arguments for the Burgers' equation training."""
    parser = argparse.ArgumentParser(description="Separable PI-DeepONet for 1D Burgers' Equation")
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
    parser.add_argument('--p_res_x_train',      type=int,   default=50)
    parser.add_argument('--hidden_dim',         type=int,   default=50)
    parser.add_argument('--r',                  type=int,   default=50)
    parser.add_argument('--branch_layers',      nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--trunk_layers',       nargs='+',  type=int, default=[50, 50, 50, 50, 50])
    parser.add_argument('--num_outputs',        type=int,   default=1)
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--gamma',              type=float, default=0.95)
    parser.add_argument('--decay_step',         type=int,   default=1000)
    parser.add_argument('--lambda_ic',          type=float, default=20.0)
    parser.add_argument('--nu',                 type=float, default=0.01)
    parser.add_argument('--log_step',           type=int,   default=100)
    parser.add_argument('--result_dir',         type=str,   default='./results/burgers')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
