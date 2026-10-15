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

from   .data                   import GaussianRandomField1D
from   .helpers                import mse, mse_single
from   .models                 import SeparableDeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_res_conservative(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_res:   torch.Tensor,
    x_res:   torch.Tensor,
    nu:      float = 0.01,
) -> torch.Tensor:
    """Computes residual loss for Burgers' equation in conservative form: $u_t + (u^2/2)_x - \\nu u_{xx} = 0$.

    Arguments:
        `model` (`SeparableDeepONet`): The Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input functions evaluated at sensors.
        `t_res` (`torch.Tensor`): Collocation time points.
        `x_res` (`torch.Tensor`): Collocation space points.
        `nu` (`float`, default: `0.01`): Kinematic viscosity.

    Returns:
        `torch.Tensor`: Scalar residual loss.
    """
    s:    torch.Tensor = model(u_batch, t_res, x_res, compute_derivatives=True)
    s_t:  torch.Tensor = model.compute_gradient([1, 0])
    s_x:  torch.Tensor = model.compute_gradient([0, 1])
    s_xx: torch.Tensor = model.compute_gradient([0, 2])

    # Conservative flux derivative: (u^2 / 2)_x = u * u_x
    flux_x:   torch.Tensor = s * s_x
    residual: torch.Tensor = s_t + flux_x - nu * s_xx
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
    """Calculates composite physics-informed loss with conservative PDE residual.

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
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metrics.
    """
    # 1. IC loss
    s_ic_pred: torch.Tensor = model(u_batch, t_ics, x_ics, compute_derivatives=False).squeeze(-1).squeeze(1)
    l_ic: torch.Tensor = mse(s_ics.reshape(-1), s_ic_pred.reshape(-1))

    # 2. BC loss
    s_0:   torch.Tensor = model(u_batch, t_bcs, x_bc_0, compute_derivatives=True)
    s_x_0: torch.Tensor = model.compute_gradient([0, 1])
    s_1:   torch.Tensor = model(u_batch, t_bcs, x_bc_1, compute_derivatives=True)
    s_x_1: torch.Tensor = model.compute_gradient([0, 1])
    l_bc:  torch.Tensor = mse(s_0.reshape(-1), s_1.reshape(-1)) + mse(s_x_0.reshape(-1), s_x_1.reshape(-1))

    # 3. Residual loss
    l_res: torch.Tensor = loss_res_conservative(model, u_batch, t_res, x_res, nu)

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
    """Executes conservative Burgers' equation training routine."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing conservative Burgers on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mat_path: str = os.path.join(os.getcwd(), 'data/burgers/Burger.mat')
    if os.path.exists(mat_path):
        data = scipy.io.loadmat(mat_path)
        u_sol = data['output']
    else:
        grf = GaussianRandomField1D(n_points=args.n_sensors, seed=args.seed)
        u_samples = grf.sample(args.n_train + args.n_test)
        u_sol = np.zeros((args.n_train + args.n_test, 101, 101), dtype=np.float32)
        u_sol[:, 0, :] = u_samples.numpy()

    u_sol_tensor: torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)
    u0_train:     torch.Tensor = u_sol_tensor[:args.n_train, 0, :].to(device)

    t_ics:  torch.Tensor = torch.zeros((1, 1), device=device)
    x_ics:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_ics_train, device=device).reshape(-1, 1)
    s_ics:  torch.Tensor = u0_train

    t_bcs:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)
    x_bc_0: torch.Tensor = torch.zeros((1, 1), device=device)
    x_bc_1: torch.Tensor = torch.ones((1, 1), device=device)

    t_res:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_t_train, device=device).reshape(-1, 1)
    x_res:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_x_train, device=device).reshape(-1, 1)

    args.separable            = True
    args.trunk_input_features = 2
    args.deg_grad             = [1, 2]

    config, model = setup_deeponet(args)
    model         = model.to(device)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'burgers_sep_conservative_p{args.hidden_dim}_r{args.r}')
    os.makedirs(result_dir, exist_ok=True)

    pbar = tqdm.tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

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
            pbar.set_postfix({'loss': f"{log_dict['total_loss']:.3e}"})

    return None


##################################################
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separable PI-DeepONet (Conservative) for Burgers' Equation")
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
    parser.add_argument('--result_dir',         type=str,   default='./results/burgers_conservative')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
