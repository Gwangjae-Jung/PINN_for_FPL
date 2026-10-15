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

from   .data                   import GaussianRandomField2D
from   .helpers                import mse_single, relative_l2_error
from   .models                 import SeparableBranchCNNDeepONet, setup_deeponet


__all__: List[str] = ['main_routine', 'loss_fn']


##################################################
def loss_fn(
    model:     SeparableBranchCNNDeepONet,
    u_batch:   torch.Tensor,
    x_res:     torch.Tensor,
    y_res:     torch.Tensor,
    k_res:     torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates composite physics-informed loss for 2D Poisson equation with random field source.

    ## Description
    Enforces Dirichlet zero boundary conditions $S = 0$ along grid edges and evaluates
    the PDE residual $k (S_{xx} + S_{yy}) + u(x, y) = 0$ on the lattice grid.

    Arguments:
        `model` (`SeparableBranchCNNDeepONet`): The CNN-branch separable DeepONet model.
        `u_batch` (`torch.Tensor`): Input 2D source fields, shape `(N, 1, H, W)`.
        `x_res` (`torch.Tensor`): Collocation $x$ coordinates, shape `(N_x, 1)`.
        `y_res` (`torch.Tensor`): Collocation $y$ coordinates, shape `(N_y, 1)`.
        `k_res` (`torch.Tensor`): Diffusivity coefficients, shape `(N_k, 1)`.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metric dictionary.
    """
    # 1. Forward pass caching derivatives: x order 2, y order 2, k order 0
    # Output shape: (N, N_x, N_y, N_k, 1)
    s:    torch.Tensor = model(u_batch, x_res, y_res, k_res, compute_derivatives=True)
    s_xx: torch.Tensor = model.compute_gradient([2, 0, 0])
    s_yy: torch.Tensor = model.compute_gradient([0, 2, 0])

    # 2. Boundary Condition Loss: S = 0 along x = 0, x = 1, y = 0, y = 1
    bc_x_0:    torch.Tensor = s[:, 0, :, :, 0]
    bc_x_1:    torch.Tensor = s[:, -1, :, :, 0]
    bc_y_0:    torch.Tensor = s[:, :, 0, :, 0]
    bc_y_1:    torch.Tensor = s[:, :, -1, :, 0]
    loss_bc_x: torch.Tensor = mse_single(bc_x_0) + mse_single(bc_x_1)
    loss_bc_y: torch.Tensor = mse_single(bc_y_0) + mse_single(bc_y_1)
    loss_bc:   torch.Tensor = 0.5 * (loss_bc_x + loss_bc_y)

    # 3. Residual Loss: k * (s_xx + s_yy) + u(x, y) = 0
    # Broadcast k: shape (1, 1, 1, N_k, 1)
    k_broad: torch.Tensor = k_res.reshape(1, 1, 1, -1, 1)

    # Reshape u_batch from (N, 1, H, W) to (N, H, W, 1, 1) for broadcasting across k
    u_broad: torch.Tensor = u_batch.permute(0, 2, 3, 1).unsqueeze(-1) if u_batch.ndim == 4 else u_batch.unsqueeze(-1).unsqueeze(-1)

    pde:      torch.Tensor = k_broad * (s_xx + s_yy) + u_broad
    loss_res: torch.Tensor = mse_single(pde)

    total: torch.Tensor = loss_bc + loss_res
    logs: Dict[str, float] = {
        'loss_bc':    float(loss_bc.item()),
        'loss_res':   float(loss_res.item()),
        'total_loss': float(total.item()),
    }
    return total, logs


##################################################
def main_routine(args: argparse.Namespace) -> None:
    """Executes training of PI-Sep-CNN-DeepONet on 2D Poisson's equation."""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Executing 2D Poisson CNN-DeepONet on device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Dataset loading / synthesis
    mat_path: str = os.path.join(os.getcwd(), 'data/poisson/poisson.mat')
    if os.path.exists(mat_path):
        data = scipy.io.loadmat(mat_path)
        u_in = data['u_in']   # (N, H, W)
    else:
        print(f"Dataset not found at '{mat_path}'. Generating 2D Gaussian random source fields...")
        grf = GaussianRandomField2D(size=(51, 51), alpha=4.0, seed=args.seed)
        u_in = grf.sample(args.n_train + args.n_test).numpy()

    u_tensor: torch.Tensor = torch.as_tensor(u_in, dtype=torch.float32).unsqueeze(1)  # (N, 1, 51, 51)
    u_train:  torch.Tensor = u_tensor[:args.n_train].to(device)

    # Coordinates
    x_res: torch.Tensor = torch.linspace(0.0, 1.0, 51, device=device).reshape(-1, 1)
    y_res: torch.Tensor = torch.linspace(0.0, 1.0, 51, device=device).reshape(-1, 1)
    # Log-spaced diffusivity values from 10^-3 to 10^1
    k_log = np.logspace(-3, 1, 21).astype(np.float32)
    k_res: torch.Tensor = torch.as_tensor(k_log, device=device).reshape(-1, 1)

    # Architecture definition for CNN Branch
    cnn_blocks: List[List[Any]] = [
        [16, 3, 3, 'tanh'],
        ['avg_pool', 2, 2, 2, 2],
        [32, 3, 3, 'tanh'],
        ['avg_pool', 2, 2, 2, 2],
        [64, 3, 3, 'tanh'],
        ['avg_pool', 2, 2, 2, 2],
        ['flatten'],
        [400, 'relu'],
        [400, 'relu'],
    ]

    args.separable                 = True
    args.branch_cnn                = True
    args.branch_cnn_blocks         = cnn_blocks
    args.branch_cnn_input_channels = 1
    args.branch_cnn_input_size     = [51, 51]
    args.trunk_input_features      = 3
    args.num_outputs               = 1
    args.deg_grad                  = [2, 2, 0]  # x degree 2, y degree 2, k degree 0

    config, model = setup_deeponet(args)
    model         = model.to(device)

    # Initialize lazy layers if any by a dummy forward pass
    dummy_u = torch.zeros((1, 1, 51, 51), device=device)
    dummy_x = torch.zeros((1, 1), device=device)
    dummy_y = torch.zeros((1, 1), device=device)
    dummy_k = torch.zeros((1, 1), device=device)
    _ = model(dummy_u, dummy_x, dummy_y, dummy_k, compute_derivatives=False)

    optimizer: Adam = Adam(model.parameters(), lr=args.learning_rate)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=args.gamma)

    result_dir: str = os.path.join(args.result_dir, f'poisson_cnn_p{args.hidden_dim}_r{args.r}')
    os.makedirs(result_dir, exist_ok=True)

    print(f"Starting training on 2D Poisson for {args.epochs} epochs...")
    start_time: float = time.time()

    pbar = tqdm.tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        if args.batch_size < args.n_train:
            idx = torch.randperm(args.n_train)[:args.batch_size]
            u_batch = u_train[idx]
        else:
            u_batch = u_train

        loss, log_dict = loss_fn(
            model   = model,
            u_batch = u_batch,
            x_res   = x_res,
            y_res   = y_res,
            k_res   = k_res,
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
    parser = argparse.ArgumentParser(description="PI-Sep-CNN-DeepONet for 2D Poisson Equation")
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--epochs',        type=int,   default=1000)
    parser.add_argument('--batch_size',    type=int,   default=20)
    parser.add_argument('--n_train',       type=int,   default=100)
    parser.add_argument('--n_test',        type=int,   default=20)
    parser.add_argument('--hidden_dim',    type=int,   default=400)
    parser.add_argument('--r',             type=int,   default=80)
    parser.add_argument('--trunk_layers',  nargs='+',  type=int, default=[50, 50, 50, 50, 50])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma',         type=float, default=0.8)
    parser.add_argument('--decay_step',    type=int,   default=5000)
    parser.add_argument('--log_step',      type=int,   default=100)
    parser.add_argument('--result_dir',    type=str,   default='./results/poisson')
    parser.add_argument('--no_gpu',        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main_routine(get_args())
