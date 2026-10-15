import argparse
import os
import time
from   typing                  import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from   torch.optim             import Adam
from   torch.optim.lr_scheduler import ExponentialLR
import tqdm

from   .autograd               import compute_grad
from   .data                   import GaussianRandomField1D
from   .helpers                import mse, mse_single
from   .models                 import DeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_res_vanilla(
    model:    DeepONet,
    u_batch:  torch.Tensor,
    tx_res:   torch.Tensor,
    nu:       float = 0.01,
) -> torch.Tensor:
    """Computes residual loss for vanilla PI-DeepONet using reverse-mode automatic differentiation.

    Arguments:
        `model` (`DeepONet`): Vanilla DeepONet instance.
        `u_batch` (`torch.Tensor`): Input functions, shape `(N, n_sensors)`.
        `tx_res` (`torch.Tensor`): Coordinate pairs `(t, x)`, shape `(N_pts, 2)` with `requires_grad=True`.
        `nu` (`float`, default: `0.01`): Kinematic viscosity.

    Returns:
        `torch.Tensor`: Scalar residual loss.
    """
    # Forward pass: (N, p, 1) and (N_pts, p, 1) -> (N, N_pts, 1)
    s: torch.Tensor = model(u_batch, tx_res).squeeze(-1)

    # First derivatives
    grads: torch.Tensor = compute_grad(s.sum(), tx_res, create_graph=True)
    s_t:   torch.Tensor = grads[:, 0].reshape(1, -1)
    s_x:   torch.Tensor = grads[:, 1].reshape(1, -1)

    # Second derivative s_xx
    s_xx:  torch.Tensor = compute_grad(s_x.sum(), tx_res, create_graph=True)[:, 1].reshape(1, -1)

    residual: torch.Tensor = s_t + s * s_x - nu * s_xx
    return mse_single(residual)


def loss_fn(
    model:     DeepONet,
    u_batch:   torch.Tensor,
    tx_ics:    torch.Tensor,
    s_ics:     torch.Tensor,
    tx_bc_0:   torch.Tensor,
    tx_bc_1:   torch.Tensor,
    tx_res:    torch.Tensor,
    lambda_ic: float = 20.0,
    nu:        float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluates composite physics-informed loss for conventional PI-DeepONet.

    Arguments:
        `model` (`DeepONet`): Vanilla DeepONet model.
        `u_batch` (`torch.Tensor`): Branch input functions.
        `tx_ics` (`torch.Tensor`): Initial condition coordinates $(0, x)$.
        `s_ics` (`torch.Tensor`): Initial condition ground truth.
        `tx_bc_0` (`torch.Tensor`): Left boundary coordinates $(t, 0)$.
        `tx_bc_1` (`torch.Tensor`): Right boundary coordinates $(t, 1)$.
        `tx_res` (`torch.Tensor`): Interior collocation coordinates $(t, x)$.
        `lambda_ic` (`float`, default: `20.0`): Initial condition penalty weight.
        `nu` (`float`, default: `0.01`): Kinematic viscosity.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metric dictionary.
    """
    # 1. IC loss
    s_ic: torch.Tensor = model(u_batch, tx_ics).squeeze(-1)
    l_ic: torch.Tensor = mse(s_ics.reshape(-1), s_ic.reshape(-1))

    # 2. BC loss
    s_0: torch.Tensor = model(u_batch, tx_bc_0).squeeze(-1)
    s_1: torch.Tensor = model(u_batch, tx_bc_1).squeeze(-1)
    l_bc: torch.Tensor = mse(s_0.reshape(-1), s_1.reshape(-1))

    # 3. Residual loss
    l_res: torch.Tensor = loss_res_vanilla(model, u_batch, tx_res, nu)

    total: torch.Tensor = lambda_ic * l_ic + l_bc + l_res
    logs: Dict[str, float] = {
        'loss_ic':    float(l_ic.item()),
        'loss_bc':    float(l_bc.item()),
        'loss_res':   float(l_res.item()),
        'total_loss': float(total.item()),
    }
    return total, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes training of conventional vanilla PI-DeepONet on 1D Burgers' equation."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing Vanilla DeepONet Burgers on device: {device}")

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

    # Initial condition coordinates: t = 0, x in [0, 1]
    x_lin = torch.linspace(0.0, 1.0, args.p_ics_train, device=device)
    tx_ics = torch.stack([torch.zeros_like(x_lin), x_lin], dim=-1)
    s_ics  = u0_train

    # Boundary condition coordinates: t in [0, 1], x = 0 and x = 1
    t_lin = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device)
    tx_bc_0 = torch.stack([t_lin, torch.zeros_like(t_lin)], dim=-1)
    tx_bc_1 = torch.stack([t_lin, torch.ones_like(t_lin)], dim=-1)

    # Collocation grid
    t_f = torch.linspace(0.0, 1.0, args.p_res_t_train, device=device)
    x_f = torch.linspace(0.0, 1.0, args.p_res_x_train, device=device)
    grid_t, grid_x = torch.meshgrid(t_f, x_f, indexing='ij')
    tx_res = torch.stack([grid_t.reshape(-1), grid_x.reshape(-1)], dim=-1).requires_grad_(True)

    args.separable            = False
    args.trunk_input_features = 2

    config, model = setup_deeponet(args)
    model         = model.to(device)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'burgers_vanilla_p{args.hidden_dim}')
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
            tx_ics    = tx_ics,
            s_ics     = s_ics_batch,
            tx_bc_0   = tx_bc_0,
            tx_bc_1   = tx_bc_1,
            tx_res    = tx_res,
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
    parser = argparse.ArgumentParser(description="Vanilla Conventional PI-DeepONet for Burgers' Equation")
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
    parser.add_argument('--hidden_dim',         type=int,   default=100)
    parser.add_argument('--branch_layers',      nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--trunk_layers',       nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--num_outputs',        type=int,   default=1)
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--gamma',              type=float, default=0.95)
    parser.add_argument('--decay_step',         type=int,   default=1000)
    parser.add_argument('--lambda_ic',          type=float, default=20.0)
    parser.add_argument('--nu',                 type=float, default=0.01)
    parser.add_argument('--log_step',           type=int,   default=100)
    parser.add_argument('--result_dir',         type=str,   default='./results/burgers_vanilla')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
