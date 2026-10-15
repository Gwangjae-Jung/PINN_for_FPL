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
    tz_res:   torch.Tensor,
) -> torch.Tensor:
    """Computes residual loss for Biot's consolidation theory using conventional DeepONet with autograd.

    Arguments:
        `model` (`DeepONet`): Conventional DeepONet model with 2 outputs.
        `u_batch` (`torch.Tensor`): Input loading functions.
        `tz_res` (`torch.Tensor`): Collocation points `(t, z)` with `requires_grad=True`.

    Returns:
        `torch.Tensor`: Scalar coupled residual loss.
    """
    s: torch.Tensor = model(u_batch, tz_res)
    # s has shape (N, N_pts, 2)
    u_field: torch.Tensor = s[:, :, 0]
    p_field: torch.Tensor = s[:, :, 1]

    # Derivatives for u
    grad_u: torch.Tensor = compute_grad(u_field.sum(), tz_res, create_graph=True)
    u_t:    torch.Tensor = grad_u[:, 0].reshape(1, -1)
    u_z:    torch.Tensor = grad_u[:, 1].reshape(1, -1)
    u_zz:   torch.Tensor = compute_grad(u_z.sum(), tz_res, create_graph=True)[:, 1].reshape(1, -1)
    u_tz:   torch.Tensor = compute_grad(u_t.sum(), tz_res, create_graph=True)[:, 1].reshape(1, -1)

    # Derivatives for p
    grad_p: torch.Tensor = compute_grad(p_field.sum(), tz_res, create_graph=True)
    p_z:    torch.Tensor = grad_p[:, 1].reshape(1, -1)
    p_zz:   torch.Tensor = compute_grad(p_z.sum(), tz_res, create_graph=True)[:, 1].reshape(1, -1)

    res_1:  torch.Tensor = 3.0 * u_zz - p_z
    res_2:  torch.Tensor = u_tz - p_zz

    return mse_single(res_1) + mse_single(res_2)


def loss_fn(
    model:     DeepONet,
    u_batch:   torch.Tensor,
    tz_ics:    torch.Tensor,
    p0_batch:  torch.Tensor,
    tz_bc_0:   torch.Tensor,
    tz_bc_1:   torch.Tensor,
    tz_res:    torch.Tensor,
    lambda_ic: float = 10.0,
    lambda_bc: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates composite physics-informed loss for Biot's consolidation theory in conventional DeepONet."""
    # 1. IC loss: u(z, 0) = 0, p(z, 0) = f(0)
    s_ic:   torch.Tensor = model(u_batch, tz_ics)
    u_ic:   torch.Tensor = s_ic[:, :, 0]
    p_ic:   torch.Tensor = s_ic[:, :, 1]
    l_ic:   torch.Tensor = mse_single(u_ic) + mse(p0_batch.reshape(-1), p_ic.reshape(-1))

    # 2. BC loss: p(0, t) = 0, u(L, t) = 0, dp/dz(L, t) = 0
    s_bc0:  torch.Tensor = model(u_batch, tz_bc_0)
    p_bc0:  torch.Tensor = s_bc0[:, :, 1]

    s_bc1:  torch.Tensor = model(u_batch, tz_bc_1)
    u_bc1:  torch.Tensor = s_bc1[:, :, 0]
    grad_p1 = compute_grad(s_bc1[:, :, 1].sum(), tz_bc_1, create_graph=True)
    p_z_bc1 = grad_p1[:, 1].reshape(1, -1)

    l_bc:   torch.Tensor = mse_single(p_bc0) + mse_single(u_bc1) + mse_single(p_z_bc1)

    # 3. Residual loss
    l_res:  torch.Tensor = loss_res_vanilla(model, u_batch, tz_res)

    total:  torch.Tensor = lambda_ic * l_ic + lambda_bc * l_bc + l_res
    logs: Dict[str, float] = {
        'loss_ic':    float(l_ic.item()),
        'loss_bc':    float(l_bc.item()),
        'loss_res':   float(l_res.item()),
        'total_loss': float(total.item()),
    }
    return total, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes training of conventional vanilla PI-DeepONet on Biot's consolidation theory."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing Vanilla DeepONet Biot on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npy_path: str = os.path.join(os.getcwd(), 'data/biot/Y.npy')
    if os.path.exists(npy_path):
        u_sol = np.load(npy_path)
    else:
        grf = GaussianRandomField1D(n_points=args.n_sensors, seed=args.seed)
        loads = grf.sample(args.n_train + args.n_test).numpy()
        u_sol = np.zeros((args.n_train + args.n_test, 101, 101, 2), dtype=np.float32)
        u_sol[:, 0, :, 0] = loads

    u_sol_tensor: torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)
    u0_train:     torch.Tensor = u_sol_tensor[:args.n_train, 0, :, 0].to(device)
    p0_train:     torch.Tensor = u_sol_tensor[:args.n_train, -1, 0, 1].to(device)

    # Coordinates for IC (t = 0)
    z_lin = torch.linspace(0.0, 1.0, args.p_ics_train, device=device)
    tz_ics = torch.stack([torch.zeros_like(z_lin), z_lin], dim=-1)

    # Coordinates for BC
    t_lin = torch.linspace(0.0, 1.0, args.p_bcs_train, device=device)
    tz_bc_0 = torch.stack([t_lin, torch.zeros_like(t_lin)], dim=-1)
    tz_bc_1 = torch.stack([t_lin, torch.ones_like(t_lin)], dim=-1).requires_grad_(True)

    # Collocation grid for Residual
    t_f = torch.linspace(0.0, 1.0, args.p_res_t_train, device=device)
    z_f = torch.linspace(0.0, 1.0, args.p_res_z_train, device=device)
    grid_t, grid_z = torch.meshgrid(t_f, z_f, indexing='ij')
    tz_res = torch.stack([grid_t.reshape(-1), grid_z.reshape(-1)], dim=-1).requires_grad_(True)

    args.separable            = False
    args.num_outputs          = 2
    args.split_trunk          = True
    args.split_branch         = True
    args.trunk_input_features = 2

    config, model = setup_deeponet(args)
    model         = model.to(device)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'biot_vanilla_p{args.hidden_dim}')
    os.makedirs(result_dir, exist_ok=True)

    pbar = tqdm.tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if args.batch_size < args.n_train:
            idx = torch.randperm(args.n_train)[:args.batch_size]
            u_batch = u0_train[idx]
            p0_batch = p0_train[idx]
        else:
            u_batch = u0_train
            p0_batch = p0_train

        loss, log_dict = loss_fn(
            model     = model,
            u_batch   = u_batch,
            tz_ics    = tz_ics,
            p0_batch  = p0_batch,
            tz_bc_0   = tz_bc_0,
            tz_bc_1   = tz_bc_1,
            tz_res    = tz_res,
            lambda_ic = args.lambda_ic,
            lambda_bc = args.lambda_bc,
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
    parser = argparse.ArgumentParser(description="Vanilla Conventional PI-DeepONet for Biot Consolidation")
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
    parser.add_argument('--hidden_dim',         type=int,   default=100)
    parser.add_argument('--branch_layers',      nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--trunk_layers',       nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--gamma',              type=float, default=0.95)
    parser.add_argument('--decay_step',         type=int,   default=1000)
    parser.add_argument('--lambda_ic',          type=float, default=10.0)
    parser.add_argument('--lambda_bc',          type=float, default=10.0)
    parser.add_argument('--log_step',           type=int,   default=100)
    parser.add_argument('--result_dir',         type=str,   default='./results/biot_vanilla')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
