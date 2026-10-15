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
from   .helpers                import mse, mse_single
from   .models                 import DeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_res_vanilla(
    model:     DeepONet,
    u_batch:   torch.Tensor,
    txyc_res:  torch.Tensor,
) -> torch.Tensor:
    """Computes residual loss for 4D heat equation using vanilla PI-DeepONet with reverse-mode autodiff.

    Arguments:
        `model` (`DeepONet`): Conventional DeepONet model.
        `u_batch` (`torch.Tensor`): Input temperature field.
        `txyc_res` (`torch.Tensor`): Collocation 4D points `(t, x, y, c)` with `requires_grad=True`.

    Returns:
        `torch.Tensor`: Scalar residual MSE loss.
    """
    T: torch.Tensor = model(u_batch, txyc_res).squeeze(-1)

    # First derivatives
    grads: torch.Tensor = compute_grad(T.sum(), txyc_res, create_graph=True)
    T_t:   torch.Tensor = grads[:, 0].reshape(1, -1)
    T_x:   torch.Tensor = grads[:, 1].reshape(1, -1)
    T_y:   torch.Tensor = grads[:, 2].reshape(1, -1)

    # Second derivatives
    T_xx:  torch.Tensor = compute_grad(T_x.sum(), txyc_res, create_graph=True)[:, 1].reshape(1, -1)
    T_yy:  torch.Tensor = compute_grad(T_y.sum(), txyc_res, create_graph=True)[:, 2].reshape(1, -1)

    c_vals:    torch.Tensor = txyc_res[:, 3].reshape(1, -1)
    c_squared: torch.Tensor = torch.square(c_vals)

    residual:  torch.Tensor = T_t - c_squared * (T_xx + T_yy)
    return mse_single(residual)


def loss_fn(
    model:      DeepONet,
    u_batch:    torch.Tensor,
    txyc_ics:   torch.Tensor,
    s_ics:      torch.Tensor,
    txyc_bcs:   torch.Tensor,
    txyc_res:   torch.Tensor,
    lambda_res: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates composite physics-informed loss for 4D heat equation in conventional DeepONet.

    Arguments:
        `model` (`DeepONet`): Vanilla DeepONet instance.
        `u_batch` (`torch.Tensor`): Branch input temperature.
        `txyc_ics` (`torch.Tensor`): Initial condition coordinates `(0, x, y, c)`.
        `s_ics` (`torch.Tensor`): Initial condition targets.
        `txyc_bcs` (`torch.Tensor`): Boundary condition coordinates on $[0, 1]^2$ edges.
        `txyc_res` (`torch.Tensor`): Interior collocation coordinates `(t, x, y, c)`.
        `lambda_res` (`float`, default: `4.0`): Weighting factor for residual loss.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metric dictionary.
    """
    # 1. IC loss
    T_ic: torch.Tensor = model(u_batch, txyc_ics).squeeze(-1)
    l_ic: torch.Tensor = mse(s_ics.reshape(-1), T_ic.reshape(-1))

    # 2. BC loss
    T_bc: torch.Tensor = model(u_batch, txyc_bcs).squeeze(-1)
    l_bc: torch.Tensor = mse_single(T_bc)

    # 3. Residual loss
    l_res: torch.Tensor = loss_res_vanilla(model, u_batch, txyc_res)

    total: torch.Tensor = l_ic + l_bc + lambda_res * l_res
    logs: Dict[str, float] = {
        'loss_ic':    float(l_ic.item()),
        'loss_bc':    float(l_bc.item()),
        'loss_res':   float(l_res.item()),
        'total_loss': float(total.item()),
    }
    return total, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes training of conventional vanilla PI-DeepONet on 4D heat equation."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing Vanilla DeepONet 4D Heat on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mat_path: str = os.path.join(os.getcwd(), 'data/heat/heat_const.mat')
    if os.path.exists(mat_path):
        data = scipy.io.loadmat(mat_path)
        t_ic = data['t_ic'].reshape(-1, 1)
    else:
        t_ic = np.random.uniform(0.0, 1.0, (args.n_train + args.n_test, 1)).astype(np.float32)

    u0_train: torch.Tensor = torch.as_tensor(t_ic[:args.n_train], dtype=torch.float32, device=device)

    # Coordinates for IC (t = 0)
    x_lin = torch.linspace(0.0, 1.0, args.p_ics_train, device=device)
    y_lin = torch.linspace(0.0, 1.0, args.p_ics_train, device=device)
    c_lin = torch.linspace(0.1, 1.0, args.p_ics_train, device=device)
    # Sample points for IC
    N_ic_pts = args.p_ics_train * args.p_ics_train
    x_samp = torch.rand(N_ic_pts, device=device)
    y_samp = torch.rand(N_ic_pts, device=device)
    c_samp = torch.rand(N_ic_pts, device=device) * 0.9 + 0.1
    txyc_ics = torch.stack([torch.zeros_like(x_samp), x_samp, y_samp, c_samp], dim=-1)
    s_ics = u0_train

    # Sample points for BC
    N_bc_pts = args.p_bcs_train * 4
    t_bc = torch.rand(N_bc_pts, device=device)
    c_bc = torch.rand(N_bc_pts, device=device) * 0.9 + 0.1
    x_bc = torch.randint(0, 2, (N_bc_pts,), device=device).float()
    y_bc = torch.rand(N_bc_pts, device=device)
    txyc_bcs = torch.stack([t_bc, x_bc, y_bc, c_bc], dim=-1)

    # Collocation points for Residual
    N_res_pts = args.p_res_train**4
    # Subsample to keep manageable on conventional DeepONet
    N_subsample = min(N_res_pts, 5000)
    t_f = torch.rand(N_subsample, device=device)
    x_f = torch.rand(N_subsample, device=device)
    y_f = torch.rand(N_subsample, device=device)
    c_f = torch.rand(N_subsample, device=device) * 0.9 + 0.1
    txyc_res = torch.stack([t_f, x_f, y_f, c_f], dim=-1).requires_grad_(True)

    args.separable            = False
    args.trunk_input_features = 4

    config, model = setup_deeponet(args)
    model         = model.to(device)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'heat_vanilla_p{args.hidden_dim}')
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
            model      = model,
            u_batch    = u_batch,
            txyc_ics   = txyc_ics,
            s_ics      = s_ics_batch,
            txyc_bcs   = txyc_bcs,
            txyc_res   = txyc_res,
            lambda_res = args.lambda_res,
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
    parser = argparse.ArgumentParser(description="Vanilla Conventional PI-DeepONet for 4D Heat Equation")
    parser.add_argument('--seed',               type=int,   default=42)
    parser.add_argument('--epochs',             type=int,   default=1000)
    parser.add_argument('--batch_size',         type=int,   default=25)
    parser.add_argument('--n_train',            type=int,   default=25)
    parser.add_argument('--n_test',             type=int,   default=50)
    parser.add_argument('--n_sensors',          type=int,   default=1)
    parser.add_argument('--branch_input_features', type=int, default=1)
    parser.add_argument('--p_ics_train',        type=int,   default=51)
    parser.add_argument('--p_bcs_train',        type=int,   default=51)
    parser.add_argument('--p_res_train',        type=int,   default=10)
    parser.add_argument('--hidden_dim',         type=int,   default=100)
    parser.add_argument('--branch_layers',      nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--trunk_layers',       nargs='+',  type=int, default=[100, 100, 100, 100, 100])
    parser.add_argument('--num_outputs',        type=int,   default=1)
    parser.add_argument('--learning_rate',      type=float, default=1e-3)
    parser.add_argument('--gamma',              type=float, default=0.9)
    parser.add_argument('--decay_step',         type=int,   default=1500)
    parser.add_argument('--lambda_res',         type=float, default=4.0)
    parser.add_argument('--log_step',           type=int,   default=100)
    parser.add_argument('--result_dir',         type=str,   default='./results/heat_vanilla')
    parser.add_argument('--no_gpu',             action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
