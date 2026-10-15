import argparse
import os
from   pathlib                 import Path
import sys
import time
from   typing                  import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from   torch.optim             import Adam
from   torch.optim.lr_scheduler import ExponentialLR
from   torch.utils.data        import DataLoader
import tqdm

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from   separable_pi_deeponet   import (
    SeparableDeepONet,
    mse,
    mse_single,
    relative_l2_error,
    setup_deeponet,
)
from   .collision              import compute_conservation_loss, compute_positivity_loss
from   .config                 import FPLDeepONetConfig, get_default_bkw_config
from   .dataset                import BKWClassDataset


__all__: List[str] = ['train_bkw_routine', 'compute_bkw_loss']


##################################################
def compute_bkw_loss(
    model:       SeparableDeepONet,
    dataset:     BKWClassDataset,
    batch_idx:   torch.Tensor,
    branch_in:   torch.Tensor,
    ic_targets:  torch.Tensor,
    t_coord:     torch.Tensor,
    v1_coord:    torch.Tensor,
    v2_coord:    torch.Tensor,
    dv:          float,
    lambda_ic:   float = 20.0,
    lambda_res:  float = 1.0,
    lambda_cons: float = 5.0,
    lambda_pos:  float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluates composite physics-informed loss for BKW solution class operator learning.

    ## Description
    Computes initial condition error at $t=0$, PDE residual error against the analytical BKW
    temporal evolution $\\partial_t f = Q(f)$, macroscopic moment conservation (mass, momentum, energy),
    and physical non-negativity penalty ($f \\ge 0$).

    Arguments:
        `model` (`SeparableDeepONet`): Separable DeepONet model.
        `dataset` (`BKWClassDataset`): Dataset containing BKW sample parameters.
        `batch_idx` (`torch.Tensor`): Indices of the current batch samples.
        `branch_in` (`torch.Tensor`): Branch sensor evaluations, shape `(batch, m)`.
        `ic_targets` (`torch.Tensor`): Target initial condition grid, shape `(batch, N_v, N_v)`.
        `t_coord` (`torch.Tensor`): 1D coordinate vector along time axis.
        `v1_coord` (`torch.Tensor`): 1D coordinate vector along $v_1$ axis.
        `v2_coord` (`torch.Tensor`): 1D coordinate vector along $v_2$ axis.
        `dv` (`float`): Velocity grid interval $\\Delta v$.
        `lambda_ic` (`float`, default: `20.0`): Penalty factor for initial condition error.
        `lambda_res` (`float`, default: `1.0`): Penalty factor for residual error.
        `lambda_cons` (`float`, default: `5.0`): Penalty factor for moment conservation.
        `lambda_pos` (`float`, default: `10.0`): Penalty factor for negative distribution values.

    Returns:
        `Tuple[torch.Tensor, Dict[str, float]]`: Total loss tensor and logging metric dictionary.
    """
    # 1. Forward pass caching derivatives: t order 1, v1 order 0, v2 order 0
    # Output shape: (batch, N_t, N_v, N_v, 1)
    f_pred: torch.Tensor = model(branch_in, t_coord, v1_coord, v2_coord, compute_derivatives=True).squeeze(-1)
    df_dt:  torch.Tensor = model.compute_gradient([1, 0, 0]).squeeze(-1)

    # 2. Initial Condition Loss at t = 0 (first index of time axis)
    f_t0:    torch.Tensor = f_pred[:, 0, :, :]
    loss_ic: torch.Tensor = mse(f_t0, ic_targets)

    # 3. Residual Loss: evaluate true analytical time derivative Q_bkw = df_true/dt
    # Compute ground truth df_dt on the collocation lattice
    q_bkw_list: List[torch.Tensor] = []
    grid_v1, grid_v2 = torch.meshgrid(v1_coord.squeeze(), v2_coord.squeeze(), indexing='ij')
    v_pts: torch.Tensor = torch.stack([grid_v1.reshape(-1), grid_v2.reshape(-1)], dim=-1)

    n_v:       int   = v1_coord.shape[0]
    coeff_int: float = 2.0 * (2 - 1) * dataset.density * dataset.vhs_coeff
    dim:       int   = 2

    for b_i in range(batch_idx.shape[0]):
        c_ext: float = float(dataset.c_ext_vals[batch_idx[b_i].item()])
        t_vals: torch.Tensor = t_coord  # (N_t, 1)

        # K(t) and dK/dt
        K_t:   torch.Tensor = 1.0 - c_ext * torch.exp(-coeff_int * t_vals)
        dK_dt: torch.Tensor = c_ext * coeff_int * torch.exp(-coeff_int * t_vals)

        speed_sq: torch.Tensor = torch.sum(v_pts**2, dim=-1, keepdim=True).unsqueeze(0)  # (1, N_v^2, 1)
        K_t_b:    torch.Tensor = K_t.unsqueeze(1)                                          # (N_t, 1, 1)
        dK_dt_b:  torch.Tensor = dK_dt.unsqueeze(1)                                        # (N_t, 1, 1)

        # Analytical derivative of BKW with respect to K
        part1: torch.Tensor = dataset.density * torch.pow(2.0 * torch.pi * K_t_b, -dim / 2.0)
        part2: torch.Tensor = torch.exp(-speed_sq / (2.0 * K_t_b))
        part3: torch.Tensor = ((dim + 2.0) / 2.0) - ((dim + speed_sq) / 2.0) / K_t_b + (speed_sq / 2.0) / (K_t_b**2)
        f_val: torch.Tensor = part1 * part2 * part3

        # d(part1)/dK
        d_p1: torch.Tensor = - (dim / (2.0 * K_t_b)) * part1
        # d(part2)/dK
        d_p2: torch.Tensor = (speed_sq / (2.0 * K_t_b**2)) * part2
        # d(part3)/dK
        d_p3: torch.Tensor = ((dim + speed_sq) / (2.0 * K_t_b**2)) - (speed_sq / (K_t_b**3))

        df_dK: torch.Tensor = (d_p1 * part2 * part3) + (part1 * d_p2 * part3) + (part1 * part2 * d_p3)
        df_dt_exact: torch.Tensor = (df_dK * dK_dt_b).reshape(-1, n_v, n_v)
        q_bkw_list.append(df_dt_exact)

    q_true:   torch.Tensor = torch.stack(q_bkw_list, dim=0)
    loss_res: torch.Tensor = mse(df_dt, q_true)

    # 4. Conservation Loss (Mass, Momentum, Energy)
    loss_cons: torch.Tensor = compute_conservation_loss(f_pred, v1_coord, v2_coord, dv)

    # 5. Positivity Loss
    loss_pos: torch.Tensor = compute_positivity_loss(f_pred)

    total_loss: torch.Tensor = (
        lambda_ic   * loss_ic   +
        lambda_res  * loss_res  +
        lambda_cons * loss_cons +
        lambda_pos  * loss_pos
    )

    logs: Dict[str, float] = {
        'loss_total': float(total_loss.item()),
        'loss_ic':    float(loss_ic.item()),
        'loss_res':   float(loss_res.item()),
        'loss_cons':  float(loss_cons.item()),
        'loss_pos':   float(loss_pos.item()),
    }
    return total_loss, logs


##################################################
def train_bkw_routine(config: Optional[FPLDeepONetConfig] = None) -> None:
    """Executes training and evaluation of Separable PI-DeepONet for the BKW solution class.

    Arguments:
        `config` (`Optional[FPLDeepONetConfig]`, default: `None`): Problem configuration dictionary.

    Returns:
        This function returns no values.
    """
    if config is None:
        config = get_default_bkw_config()

    device_str: str          = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    device:     torch.device = torch.device(device_str)
    print(f"[BKW Operator Learning] Using device: {device}")

    seed: int = int(config.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Datasets
    n_train: int = int(config.get('num_train_samples', 200))
    n_test:  int = int(config.get('num_test_samples', 50))
    train_dataset: BKWClassDataset = BKWClassDataset(n_train, config, seed=seed)
    test_dataset:  BKWClassDataset = BKWClassDataset(n_test, config, seed=seed + 1000)

    # Collocation coordinates
    t_coord:  torch.Tensor = train_dataset.t_coord.to(device)
    v1_coord: torch.Tensor = train_dataset.v1_coord.to(device)
    v2_coord: torch.Tensor = train_dataset.v2_coord.to(device)

    v_max:    float = float(config.get('max_v', float(2.0 * torch.pi)))
    sample_v: int   = int(config.get('sample_v', 32))
    dv:       float = (2.0 * v_max) / sample_v

    # 2. Build Separable DeepONet Model
    sensor_res: int = int(config.get('sensor_res', 16))
    num_sensors: int = sensor_res * sensor_res

    deeponet_cfg: Dict[str, Any] = {
        'num_outputs':           1,
        'separable':             True,
        'r':                     int(config.get('r', 50)),
        'hidden_dim':            int(config.get('hidden_dim', 100)),
        'n_sensors':             num_sensors,
        'branch_input_features': 1,
        'trunk_input_features':  3,  # (t, v1, v2)
        'branch_layers':         config.get('branch_layers', [128, 128, 128, 100]),
        'trunk_layers':          config.get('trunk_layers', [64, 64, 64, 100]),
        'deg_grad':              [1, 0, 0],  # t order 1, v1 order 0, v2 order 0
        'activation':            str(config.get('activation', 'tanh')),
        'bias':                  True,
    }

    _, model = setup_deeponet(deeponet_cfg)
    model    = model.to(device)

    # 3. Optimizer and Scheduler
    lr:        float         = float(config.get('learning_rate', 1e-3))
    gamma:     float         = float(config.get('gamma', 0.95))
    decay_step: int          = int(config.get('decay_step', 200))
    epochs:    int           = int(config.get('num_epochs', 2000))
    batch_size: int          = int(config.get('batch_size', 16))

    optimizer: Adam          = Adam(model.parameters(), lr=lr)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=gamma)

    result_dir: str = str(config.get('result_dir', './results/fpl_bkw'))
    os.makedirs(result_dir, exist_ok=True)

    print(f"Beginning training for {epochs} epochs (Batch size: {batch_size}, Train samples: {n_train})...")
    start_time: float = time.time()

    train_indices: torch.Tensor = torch.arange(n_train)
    log_step: int = int(config.get('log_step', 50))

    pbar = tqdm.tqdm(range(1, epochs + 1))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Batch sampling
        perm = torch.randperm(n_train)[:batch_size]
        batch_idx = train_indices[perm]

        branch_in  = train_dataset.branch_data[batch_idx].to(device)
        ic_targets = train_dataset.ic_data[batch_idx].to(device)

        total_loss, logs = compute_bkw_loss(
            model       = model,
            dataset     = train_dataset,
            batch_idx   = batch_idx,
            branch_in   = branch_in,
            ic_targets  = ic_targets,
            t_coord     = t_coord,
            v1_coord    = v1_coord,
            v2_coord    = v2_coord,
            dv          = dv,
            lambda_ic   = float(config.get('lambda_ic', 20.0)),
            lambda_res  = float(config.get('lambda_res', 1.0)),
            lambda_cons = float(config.get('lambda_cons', 5.0)),
            lambda_pos  = float(config.get('lambda_pos', 10.0)),
        )

        total_loss.backward()
        optimizer.step()

        if epoch % decay_step == 0:
            scheduler.step()

        if epoch % log_step == 0 or epoch == 1:
            # Quick evaluation on a test sample
            model.eval()
            with torch.no_grad():
                test_idx: int = 0
                test_branch: torch.Tensor = test_dataset.branch_data[[test_idx]].to(device)
                f_test_pred: torch.Tensor = model(test_branch, t_coord, v1_coord, v2_coord, compute_derivatives=False)
                f_test_true: torch.Tensor = test_dataset.get_ground_truth_lattice(test_idx).to(device)
                test_rel_err: float = float(relative_l2_error(f_test_true, f_test_pred).item())

            pbar.set_postfix({
                'loss':     f"{logs['loss_total']:.3e}",
                'ic':       f"{logs['loss_ic']:.3e}",
                'res':      f"{logs['loss_res']:.3e}",
                'test_err': f"{test_rel_err:.3e}",
            })

    elapsed: float = time.time() - start_time
    print(f"BKW training complete in {elapsed:.2f} s ({elapsed / epochs * 1000:.2f} ms/epoch).")

    # Save model checkpoint
    ckpt_path: str = os.path.join(result_dir, 'fpl_bkw_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to '{ckpt_path}'.")

    return None


##################################################
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separable PI-DeepONet for 2D FPL Equation (BKW Solution Class)")
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--epochs',        type=int,   default=2000)
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--n_train',       type=int,   default=200)
    parser.add_argument('--n_test',        type=int,   default=50)
    parser.add_argument('--sample_t',      type=int,   default=16)
    parser.add_argument('--sample_v',      type=int,   default=32)
    parser.add_argument('--sensor_res',    type=int,   default=16)
    parser.add_argument('--hidden_dim',    type=int,   default=100)
    parser.add_argument('--r',             type=int,   default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma',         type=float, default=0.95)
    parser.add_argument('--decay_step',    type=int,   default=200)
    parser.add_argument('--lambda_ic',     type=float, default=20.0)
    parser.add_argument('--lambda_res',    type=float, default=1.0)
    parser.add_argument('--lambda_cons',   type=float, default=5.0)
    parser.add_argument('--lambda_pos',    type=float, default=10.0)
    parser.add_argument('--log_step',      type=int,   default=50)
    parser.add_argument('--result_dir',    type=str,   default='./results/fpl_bkw')
    parser.add_argument('--no_gpu',        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args: argparse.Namespace = get_args()
    cfg: FPLDeepONetConfig   = get_default_bkw_config()
    cfg['seed']              = args.seed
    cfg['num_epochs']        = args.epochs
    cfg['batch_size']        = args.batch_size
    cfg['num_train_samples'] = args.n_train
    cfg['num_test_samples']  = args.n_test
    cfg['sample_t']          = args.sample_t
    cfg['sample_v']          = args.sample_v
    cfg['sensor_res']        = args.sensor_res
    cfg['hidden_dim']        = args.hidden_dim
    cfg['r']                 = args.r
    cfg['learning_rate']     = args.learning_rate
    cfg['gamma']             = args.gamma
    cfg['decay_step']        = args.decay_step
    cfg['lambda_ic']         = args.lambda_ic
    cfg['lambda_res']        = args.lambda_res
    cfg['lambda_cons']       = args.lambda_cons
    cfg['lambda_pos']        = args.lambda_pos
    cfg['log_step']          = args.log_step
    cfg['result_dir']        = args.result_dir
    if args.no_gpu:
        cfg['device'] = 'cpu'
    train_bkw_routine(cfg)
