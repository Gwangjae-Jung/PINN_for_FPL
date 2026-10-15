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

from   .data                   import generate_heat_test_data
from   .helpers                import mse, mse_single, relative_l2_error
from   .models                 import SeparableDeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_ic(
    model:     SeparableDeepONet,
    u_batch:   torch.Tensor,
    t_ics:     torch.Tensor,
    x_ics:     torch.Tensor,
    y_ics:     torch.Tensor,
    c_ics:     torch.Tensor,
    s_targets: torch.Tensor,
) -> torch.Tensor:
    """Computes initial condition loss for the 4D parameterized heat equation: $T(x, y, 0) = T_0$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet instance.
        `u_batch` (`torch.Tensor`): Initial temperature field input, shape `(N, n_sensors)`.
        `t_ics` (`torch.Tensor`): Initial time coordinate ($t=0$), shape `(1, 1)`.
        `x_ics` (`torch.Tensor`): Spatial $x$ coordinates, shape `(N_x, 1)`.
        `y_ics` (`torch.Tensor`): Spatial $y$ coordinates, shape `(N_y, 1)`.
        `c_ics` (`torch.Tensor`): Diffusivity parameter values $c = \\sqrt{\\alpha}$, shape `(N_c, 1)`.
        `s_targets` (`torch.Tensor`): Ground truth initial temperature targets, shape `(N, 1)`.

    Returns:
        `torch.Tensor`: Scalar IC loss.
    """
    # Evaluate at t = 0: shape (N, 1, N_x, N_y, N_c, 1)
    s_pred: torch.Tensor = model(u_batch, t_ics, x_ics, y_ics, c_ics, compute_derivatives=False)
    # Target shape: (N, 1, 1, 1, 1, 1)
    diff: torch.Tensor = s_pred - s_targets.reshape(-1, 1, 1, 1, 1, 1)
    return mse_single(diff)


def loss_bc_split(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_bcs:   torch.Tensor,
    x_bcs:   torch.Tensor,
    y_bcs:   torch.Tensor,
    c_bcs:   torch.Tensor,
) -> torch.Tensor:
    """Computes Dirichlet boundary loss for zero edge temperature $T = 0$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Branch input functions.
        `t_bcs` (`torch.Tensor`): Boundary time coordinates.
        `x_bcs` (`torch.Tensor`): Boundary $x$ coordinates.
        `y_bcs` (`torch.Tensor`): Boundary $y$ coordinates.
        `c_bcs` (`torch.Tensor`): Boundary diffusivity coordinates.

    Returns:
        `torch.Tensor`: Scalar boundary loss.
    """
    s_pred: torch.Tensor = model(u_batch, t_bcs, x_bcs, y_bcs, c_bcs, compute_derivatives=False)
    return mse_single(s_pred)


def loss_bc(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_bcs:   torch.Tensor,
    x_bc_01: torch.Tensor,
    x_line:  torch.Tensor,
    y_bc_01: torch.Tensor,
    y_line:  torch.Tensor,
    c_bcs:   torch.Tensor,
) -> torch.Tensor:
    """Computes average boundary condition loss along $x \\in \\{0, 1\\}$ and $y \\in \\{0, 1\\}$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input functions.
        `t_bcs` (`torch.Tensor`): Boundary time points.
        `x_bc_01` (`torch.Tensor`): Boundary spatial $x$ values ($0$ and $1$).
        `x_line` (`torch.Tensor`): Interior spatial $x$ line coordinates.
        `y_bc_01` (`torch.Tensor`): Boundary spatial $y$ values ($0$ and $1$).
        `y_line` (`torch.Tensor`): Interior spatial $y$ line coordinates.
        `c_bcs` (`torch.Tensor`): Parameter $c$ coordinates.

    Returns:
        `torch.Tensor`: Scalar boundary loss.
    """
    # Boundary at x = 0 and x = 1 across all y
    loss_x: torch.Tensor = loss_bc_split(model, u_batch, t_bcs, x_bc_01, y_line, c_bcs)
    # Boundary at y = 0 and y = 1 across all x
    loss_y: torch.Tensor = loss_bc_split(model, u_batch, t_bcs, x_line, y_bc_01, c_bcs)
    return 0.5 * (loss_x + loss_y)


def loss_res(
    model:   SeparableDeepONet,
    u_batch: torch.Tensor,
    t_res:   torch.Tensor,
    x_res:   torch.Tensor,
    y_res:   torch.Tensor,
    c_res:   torch.Tensor,
) -> torch.Tensor:
    """Computes PDE residual loss for 4D heat equation: $T_t - c^2 (T_{xx} + T_{yy}) = 0$.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet instance.
        `u_batch` (`torch.Tensor`): Branch input functions.
        `t_res` (`torch.Tensor`): Collocation time points.
        `x_res` (`torch.Tensor`): Collocation $x$ points.
        `y_res` (`torch.Tensor`): Collocation $y$ points.
        `c_res` (`torch.Tensor`): Collocation $c = \\sqrt{\\alpha}$ points.

    Returns:
        `torch.Tensor`: Scalar residual MSE loss.
    """
    # Forward pass caching derivatives: t order 1, x order 2, y order 2, c order 0
    _T:   torch.Tensor = model(u_batch, t_res, x_res, y_res, c_res, compute_derivatives=True)
    T_t:  torch.Tensor = model.compute_gradient([1, 0, 0, 0])
    T_xx: torch.Tensor = model.compute_gradient([0, 2, 0, 0])
    T_yy: torch.Tensor = model.compute_gradient([0, 0, 2, 0])

    # Broadcast c^2 = alpha across [N, N_t, N_x, N_y, N_c, 1]
    c_squared: torch.Tensor = torch.square(c_res).reshape(1, 1, 1, 1, -1, 1)

    residual: torch.Tensor = T_t - c_squared * (T_xx + T_yy)
    return mse_single(residual)


def loss_fn(
    model:      SeparableDeepONet,
    u_batch:    torch.Tensor,
    t_ics:      torch.Tensor,
    x_ics:      torch.Tensor,
    y_ics:      torch.Tensor,
    c_ics:      torch.Tensor,
    s_ics:      torch.Tensor,
    t_bcs:      torch.Tensor,
    x_bc_01:    torch.Tensor,
    x_line:     torch.Tensor,
    y_bc_01:    torch.Tensor,
    y_line:     torch.Tensor,
    c_bcs:      torch.Tensor,
    t_res:      torch.Tensor,
    x_res:      torch.Tensor,
    y_res:      torch.Tensor,
    c_res:      torch.Tensor,
    lambda_res: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates composite physics-informed loss for parameterized 4D heat equation.

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input functions.
        `t_ics` (`torch.Tensor`): Initial time coordinate.
        `x_ics` (`torch.Tensor`): Initial $x$ coordinates.
        `y_ics` (`torch.Tensor`): Initial $y$ coordinates.
        `c_ics` (`torch.Tensor`): Initial diffusivity coordinates.
        `s_ics` (`torch.Tensor`): Initial temperature target values.
        `t_bcs` (`torch.Tensor`): Boundary time coordinates.
        `x_bc_01` (`torch.Tensor`): Boundary $x$ points (0, 1).
        `x_line` (`torch.Tensor`): Interior $x$ line.
        `y_bc_01` (`torch.Tensor`): Boundary $y$ points (0, 1).
        `y_line` (`torch.Tensor`): Interior $y$ line.
        `c_bcs` (`torch.Tensor`): Boundary diffusivity points.
        `t_res` (`torch.Tensor`): Residual collocation time points.
        `x_res` (`torch.Tensor`): Residual collocation $x$ points.
        `y_res` (`torch.Tensor`): Residual collocation $y$ points.
        `c_res` (`torch.Tensor`): Residual collocation $c$ points.
        `lambda_res` (`float`, default: `4.0`): Weighting penalty for residual loss.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metric dictionary.
    """
    l_ic:  torch.Tensor = loss_ic(model, u_batch, t_ics, x_ics, y_ics, c_ics, s_ics)
    l_bc:  torch.Tensor = loss_bc(model, u_batch, t_bcs, x_bc_01, x_line, y_bc_01, y_line, c_bcs)
    l_res: torch.Tensor = loss_res(model, u_batch, t_res, x_res, y_res, c_res)

    total_loss: torch.Tensor = l_ic + l_bc + lambda_res * l_res
    logs: Dict[str, float] = {
        'loss_ic':    float(l_ic.item()),
        'loss_bc':    float(l_bc.item()),
        'loss_res':   float(l_res.item()),
        'total_loss': float(total_loss.item()),
    }
    return total_loss, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes training and evaluation of Separable PI-DeepONet on 4D heat equation."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing 4D Heat equation on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Dataset loading / synthesis
    mat_path: str = os.path.join(os.getcwd(), 'data/heat/heat_const.mat')
    if os.path.exists(mat_path):
        data = scipy.io.loadmat(mat_path)
        u_sol = data['u_sol']
        c_ref = np.sqrt(data['alpha_ref'].reshape(-1, 1))
        t_ic  = data['t_ic'].reshape(-1, 1)
    else:
        print(f"Dataset not found at '{mat_path}'. Generating synthetic parameter distributions...")
        t_ic  = np.random.uniform(0.0, 1.0, (args.n_train + args.n_test, 1)).astype(np.float32)
        c_ref = np.sqrt(np.random.uniform(0.01, 1.0, (args.n_train + args.n_test, 1))).astype(np.float32)
        u_sol = np.zeros((args.n_train + args.n_test, 51, 51, 51), dtype=np.float32)

    u0_train: torch.Tensor = torch.as_tensor(t_ic[:args.n_train], dtype=torch.float32, device=device)

    # 2. Build Collocation Points
    t_ics: torch.Tensor = torch.zeros((1, 1), device=device)
    x_ics: torch.Tensor = torch.linspace(0.0, 1.0, args.p_ics_train, device=device).reshape(-1, 1)
    y_ics: torch.Tensor = torch.linspace(0.0, 1.0, args.p_ics_train, device=device).reshape(-1, 1)
    c_ics: torch.Tensor = torch.linspace(0.1, 1.0, args.p_ics_train, device=device).reshape(-1, 1)
    s_ics: torch.Tensor = u0_train

    t_bcs:   torch.Tensor = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)
    x_bc_01: torch.Tensor = torch.tensor([[0.0], [1.0]], device=device)
    x_line:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)
    y_bc_01: torch.Tensor = torch.tensor([[0.0], [1.0]], device=device)
    y_line:  torch.Tensor = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)
    c_bcs:   torch.Tensor = torch.linspace(0.1, 1.0, args.p_bcs_train, device=device).reshape(-1, 1)

    t_res: torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_train, device=device).reshape(-1, 1)
    x_res: torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_train, device=device).reshape(-1, 1)
    y_res: torch.Tensor = torch.linspace(0.0, 1.0, args.p_res_train, device=device).reshape(-1, 1)
    c_res: torch.Tensor = torch.linspace(0.1, 1.0, args.p_res_train, device=device).reshape(-1, 1)

    # 3. Model setup
    args.separable             = True
    args.trunk_input_features  = 4
    args.deg_grad              = [1, 2, 2, 0]  # t degree 1, x degree 2, y degree 2, c degree 0

    config, model = setup_deeponet(args)
    model         = model.to(device)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'heat_sep_p{args.hidden_dim}_r{args.r}')
    os.makedirs(result_dir, exist_ok=True)

    print(f"Starting training on 4D Heat equation for {args.epochs} epochs...")
    start_time: float = time.time()

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
            model      = model,
            u_batch    = u_batch,
            t_ics      = t_ics,
            x_ics      = x_ics,
            y_ics      = y_ics,
            c_ics      = c_ics,
            s_ics      = s_ics_batch,
            t_bcs      = t_bcs,
            x_bc_01    = x_bc_01,
            x_line     = x_line,
            y_bc_01    = y_bc_01,
            y_line     = y_line,
            c_bcs      = c_bcs,
            t_res      = t_res,
            x_res      = x_res,
            y_res      = y_res,
            c_res      = c_res,
            lambda_res = args.lambda_res,
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
    parser = argparse.ArgumentParser(description="Separable PI-DeepONet for 4D Heat Equation")
    parser.add_argument('--seed',               type=int,   default=42)
    parser.add_argument('--epochs',             type=int,   default=1000)
    parser.add_argument('--batch_size',         type=int,   default=25)
    parser.add_argument('--n_train',            type=int,   default=25)
    parser.add_argument('--n_test',             type=int,   default=50)
    parser.add_argument('--n_sensors',          type=int,   default=1)
    parser.add_argument('--branch_input_features', type=int, default=1)
    parser.add_argument('--p_ics_train',        type=int,   default=51)
    parser.add_argument('--p_bcs_train',        type=int,   default=51)
    parser.add_argument('--p_res_train',        type=int,   default=31)
    parser.add_argument('--hidden_dim',         type=int,   default=50)
    parser.add_argument('--r',                  type=int,   default=50)
    parser.add_argument('--branch_layers',      nargs='+',  type=int, default=[50, 50, 50, 50, 50])
    parser.add_argument('--trunk_layers',       nargs='+',  type=int, default=[50, 50, 50, 50, 50])
    parser.add_argument('--num_outputs',        type=int,   default=1)
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--gamma',              type=float, default=0.9)
    parser.add_argument('--decay_step',         type=int,   default=1500)
    parser.add_argument('--lambda_res',         type=float, default=4.0)
    parser.add_argument('--log_step',           type=int,   default=100)
    parser.add_argument('--result_dir',         type=str,   default='./results/heat')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
