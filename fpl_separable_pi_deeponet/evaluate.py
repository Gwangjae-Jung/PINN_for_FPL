import argparse
import os
from   pathlib                 import Path
import sys
from   typing                  import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from   separable_pi_deeponet   import (
    SeparableDeepONet,
    relative_l2_error,
    setup_deeponet,
)
from   .collision              import compute_moments
from   .config                 import (
    FPLDeepONetConfig,
    get_default_bimaxwellian_config,
    get_default_bkw_config,
)
from   .dataset                import BKWClassDataset, BiMaxwellianClassDataset


__all__: List[str] = ['evaluate_model', 'plot_comparison']


##################################################
def plot_comparison(
    t_vals:       torch.Tensor,
    v1_coord:     torch.Tensor,
    v2_coord:     torch.Tensor,
    f_pred:       torch.Tensor,
    f_true:       Optional[torch.Tensor] = None,
    save_path:    str                    = 'evaluation_plot.png',
    title_prefix: str                    = 'FPL Sep-PI-DeepONet',
) -> None:
    """Generates 2D contour comparison plots across multiple time snapshots.

    Arguments:
        `t_vals` (`torch.Tensor`): Array of evaluation time points.
        `v1_coord` (`torch.Tensor`): Coordinate vector along $v_1$.
        `v2_coord` (`torch.Tensor`): Coordinate vector along $v_2$.
        `f_pred` (`torch.Tensor`): Predicted distribution tensor, shape `(N_t, N_v, N_v)`.
        `f_true` (`Optional[torch.Tensor]`, default: `None`): Ground truth tensor if available.
        `save_path` (`str`, default: `'evaluation_plot.png'`): Output file path for figure.
        `title_prefix` (`str`, default: `'FPL Sep-PI-DeepONet'`): Header text for the plot.

    Returns:
        This function returns no values.
    """
    n_t: int = f_pred.shape[0]
    time_indices: List[int] = [0, n_t // 4, n_t // 2, n_t - 1]
    n_cols: int = len(time_indices)
    n_rows: int = 3 if f_true is not None else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    v1_np: np.ndarray = v1_coord.cpu().squeeze().numpy()
    v2_np: np.ndarray = v2_coord.cpu().squeeze().numpy()
    extent = [v1_np[0], v1_np[-1], v2_np[0], v2_np[-1]]

    for col_idx, t_idx in enumerate(time_indices):
        t_cur: float = float(t_vals[t_idx].item())
        pred_slice: np.ndarray = f_pred[t_idx].cpu().numpy()

        # Row 0: Prediction
        im0 = axes[0, col_idx].imshow(pred_slice.T, origin='lower', extent=extent, cmap='viridis')
        axes[0, col_idx].set_title(f"Pred: t = {t_cur:.2f}")
        axes[0, col_idx].set_xlabel("$v_1$")
        axes[0, col_idx].set_ylabel("$v_2$")
        fig.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

        if f_true is not None:
            true_slice: np.ndarray = f_true[t_idx].cpu().numpy()
            diff_slice: np.ndarray = np.abs(pred_slice - true_slice)

            # Row 1: True reference
            im1 = axes[1, col_idx].imshow(true_slice.T, origin='lower', extent=extent, cmap='viridis')
            axes[1, col_idx].set_title(f"True: t = {t_cur:.2f}")
            axes[1, col_idx].set_xlabel("$v_1$")
            axes[1, col_idx].set_ylabel("$v_2$")
            fig.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

            # Row 2: Absolute Error
            im2 = axes[2, col_idx].imshow(diff_slice.T, origin='lower', extent=extent, cmap='magma')
            axes[2, col_idx].set_title(f"|Error|: max {diff_slice.max():.2e}")
            axes[2, col_idx].set_xlabel("$v_1$")
            axes[2, col_idx].set_ylabel("$v_2$")
            fig.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

    plt.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved evaluation comparison figure to '{save_path}'.")
    return None


def evaluate_model(
    checkpoint_path: str,
    init_type:       str                   = 'bkw',
    config:          Optional[FPLDeepONetConfig] = None,
    output_dir:      str                   = './evaluation_results',
) -> Dict[str, float]:
    """Loads a trained checkpoint and evaluates operator prediction metrics on test cases.

    Arguments:
        `checkpoint_path` (`str`): File path to saved model state dictionary.
        `init_type` (`str`, default: `'bkw'`): Target initial condition class ('bkw' or 'bimaxwellian').
        `config` (`Optional[FPLDeepONetConfig]`, default: `None`): Configuration options.
        `output_dir` (`str`, default: `'./evaluation_results'`): Destination folder for output figures.

    Returns:
        `Dict[str, float]`: Evaluated summary metrics dictionary.
    """
    if config is None:
        config = get_default_bkw_config() if init_type == 'bkw' else get_default_bimaxwellian_config()

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') != 'cpu' else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset
    num_sensors: int = int(config.get('sensor_res', 16))**2
    if init_type == 'bkw':
        test_dataset = BKWClassDataset(20, config, seed=999)
    else:
        test_dataset = BiMaxwellianClassDataset(20, config, seed=999)

    t_coord:  torch.Tensor = test_dataset.t_coord.to(device)
    v1_coord: torch.Tensor = test_dataset.v1_coord.to(device)
    v2_coord: torch.Tensor = test_dataset.v2_coord.to(device)

    # 2. Build Model & Load Weights
    deeponet_cfg: Dict[str, Any] = {
        'num_outputs':           1,
        'separable':             True,
        'r':                     int(config.get('r', 50)),
        'hidden_dim':            int(config.get('hidden_dim', 100)),
        'n_sensors':             num_sensors,
        'branch_input_features': 1,
        'trunk_input_features':  3,
        'branch_layers':         config.get('branch_layers', [128, 128, 128, 100]),
        'trunk_layers':          config.get('trunk_layers', [64, 64, 64, 100]),
        'deg_grad':              [1, 0, 0],
        'activation':            str(config.get('activation', 'tanh')),
        'bias':                  True,
    }

    _, model = setup_deeponet(deeponet_cfg)
    model    = model.to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from '{checkpoint_path}'.")
    else:
        print(f"Warning: Checkpoint not found at '{checkpoint_path}'. Evaluating untrained initialized model.")

    model.eval()

    rel_l2_errors: List[float] = []
    mass_drifts:   List[float] = []
    energy_drifts: List[float] = []

    v_max:    float = float(config.get('max_v', 5.0))
    sample_v: int   = int(config.get('sample_v', 32))
    dv:       float = (2.0 * v_max) / sample_v

    with torch.no_grad():
        for idx in range(min(len(test_dataset), 5)):
            branch_in: torch.Tensor = test_dataset.branch_data[[idx]].to(device)
            f_pred:    torch.Tensor = model(branch_in, t_coord, v1_coord, v2_coord, compute_derivatives=False).squeeze(-1).squeeze(0)

            f_true: Optional[torch.Tensor] = None
            if init_type == 'bkw' and isinstance(test_dataset, BKWClassDataset):
                f_true = test_dataset.get_ground_truth_lattice(idx).to(device).squeeze(-1)
                err_l2: float = float(relative_l2_error(f_true, f_pred).item())
                rel_l2_errors.append(err_l2)

            # Conservation metrics
            mass, momentum, energy = compute_moments(f_pred.unsqueeze(0), v1_coord, v2_coord, dv)
            d_mass:   float = float(torch.max(torch.abs(mass - mass[:, [0]])).item())
            d_energy: float = float(torch.max(torch.abs(energy - energy[:, [0]])).item())
            mass_drifts.append(d_mass)
            energy_drifts.append(d_energy)

            # Plot first 2 test cases
            if idx < 2:
                plot_file: str = os.path.join(output_dir, f'{init_type}_test_sample_{idx}.png')
                plot_comparison(
                    t_vals       = t_coord,
                    v1_coord     = v1_coord,
                    v2_coord     = v2_coord,
                    f_pred       = f_pred,
                    f_true       = f_true,
                    save_path    = plot_file,
                    title_prefix = f"{init_type.capitalize()} Test Case #{idx}",
                )

    metrics: Dict[str, float] = {
        'mean_mass_drift':   float(np.mean(mass_drifts)),
        'mean_energy_drift': float(np.mean(energy_drifts)),
    }
    if rel_l2_errors:
        metrics['mean_relative_l2_error'] = float(np.mean(rel_l2_errors))

    print("=" * 45)
    print("  Evaluation Summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4e}")
    print("=" * 45)

    return metrics


##################################################
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trained Separable PI-DeepONet on FPL Equation")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to .pt checkpoint file.")
    parser.add_argument('--init_type',  type=str, default='bkw', choices=['bkw', 'bimaxwellian'])
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    return parser.parse_args()


if __name__ == '__main__':
    args: argparse.Namespace = get_args()
    evaluate_model(
        checkpoint_path = args.checkpoint,
        init_type       = args.init_type,
        output_dir      = args.output_dir,
    )
