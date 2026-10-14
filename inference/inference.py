# Import necessary libraries and modules.
import argparse
from   itertools import accumulate, product
from   pathlib   import Path
import sys
import warnings

import matplotlib.pyplot as plt
import torch

try:
    from deprecated import deprecated
except ImportError:
    import functools

    def deprecated(reason: str = ""):
        def _decorator(func):
            @functools.wraps(func)
            def _wrapper(*args, **kwargs):
                warnings.warn(f"{func.__name__} is deprecated: {reason}", category=DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)
            return _wrapper
        return _decorator

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from   config                 import FurtherConfig
from   config.base_config__2d import (
    BKW_COEFF_EXT,
    DENSITY__DICT,
    DEPTH,
    DIMENSION,
    INIT_COND__DEV,
    INIT_COND__STD,
    MAX_T__DICT,
    MAX_V__DICT,
    SOFTPLUS,
    WIDTH,
)
from   models                 import FPL_spectral, PINN_FPL
from   utils                  import (
    absolute_error,
    bimaxwellian,
    bkw,
    compute_bulk_velocity,
    compute_energy_density,
    compute_entropy_density,
    compute_mass_density,
    maxwellian,
    relative_error,
    space_grid,
)

warnings.filterwarnings("ignore")


##################################################
_NUM_EPOCHS:     int         = 5000
_LIST_INDEX:     list[int]   = [1000 * k for k in (3, 5)]
_LIST_SEEDS:     list[int]   = list(range(5))
_LIST_COLORS:    list[str]   = ['red', 'orange', 'green', 'blue', 'purple']
_SIZE_SUPTITLE:  int         = 20
_SIZE_TITLE:     int         = 16
_SIZE_LABEL:     int         = 14
_LINEWIDTH:      int         = 3
_STYLE_TARGET:   str         = 'k--'
_STYLE_OPPINN:   str         = 'r-'
_STYLE_PINN:     str         = 'g-'
_DPI:            int         = 1000
_BBOX_TO_ANCHOR: tuple[float, float] = (0.5, -0.01)
_LABEL_OPPINN:   str         = 'opPINN'
_LABEL_SPECPINN: str         = 'specPINN'

_DICT_COEFF_STR: dict[float, str] = {
    5.0:  "5",
    3.0:  "3",
    1.0:  "1",
    5/16: "5/16",
}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference evaluation.

    ## Description
    Parses GPU index, gamma, time sampling mode, resolutions, and initial condition type.

    Arguments:
        This function does not take arguments.

    Returns:
        `argparse.Namespace`: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_index', type=int,   default=3, help='The index of the GPU to use.')
    parser.add_argument('--gamma',      type=float, help='The value of gamma in the Fokker-Planck-Landau equation.')
    parser.add_argument('--sample_t',   type=str,   help='The mod of sampling the time variable.')
    parser.add_argument('--res_t',      type=int,   help='The resolution in the time variable.')
    parser.add_argument('--res_v',      type=int,   help='The resolution in the velocity variable.')
    parser.add_argument('--init_type',  type=str,   help='The initial condition.')
    return parser.parse_args()


def _generate_pinn(
    dimension: int,
    depth:     int,
    width:     int,
    softplus:  float,
    device:    torch.device,
) -> PINN_FPL:
    """Instantiate a PINN_FPL model.

    ## Description
    Initializes a PINN model for the Fokker-Planck-Landau equation with specified
    architecture parameters and moves it to the target device.

    Arguments:
        `dimension` (`int`): Dimension of the velocity domain.
        `depth` (`int`): Depth (number of hidden layers) of the network.
        `width` (`int`): Width (neurons per layer) of the network.
        `softplus` (`float`): Softplus parameter for activation function.
        `device` (`torch.device`): Device on which the model is allocated.

    Returns:
        `PINN_FPL`: The initialized PINN model.
    """
    return PINN_FPL(
        dimension = dimension,
        depth     = depth,
        width     = width,
        softplus  = softplus,
        device    = device,
    )


def _get_prefix(
    index:        int,
    dimension:    int,
    vhs_coeff:    float,
    vhs_exponent: float,
    init_type:    str,
) -> str:
    """Generate a filename prefix string for experimental results.

    ## Description
    Constructs a formatted prefix containing dimension, VHS parameters,
    initial condition type, and checkpoint index.

    Arguments:
        `index` (`int`): Epoch index of the checkpoint.
        `dimension` (`int`): Dimension of the velocity domain.
        `vhs_coeff` (`float`): VHS model coefficient.
        `vhs_exponent` (`float`): VHS model exponent (gamma).
        `init_type` (`str`): Type of initial condition.

    Returns:
        `str`: Formatted prefix string.
    """
    return f"{dimension}D__vhs__coeff{vhs_coeff:.2f}_exponent_{vhs_exponent:.2f}__init_type_{init_type}__index_{index}"


def _validate_models(
    indices:                     list[int],
    seeds:                       list[int],
    path_base:                   Path,
    device:                      torch.device,
    base_shape:                  tuple[int, ...],
    grid_t:                      torch.Tensor,
    grid_v:                      torch.Tensor,
    v:                           torch.Tensor,
    target:                      torch.Tensor,
    dimension:                   int,
    depth:                       int,
    width:                       int,
    softplus:                    float,
    vhs_coeff:                   float,
    vhs_exponent:                float,
    init_type:                   str,
    sample_t:                    str,
    res_t:                       int,
    res_v:                       int,
    dict__pred__pinn:            dict[tuple[int, int], torch.Tensor],
    dict__pred__oppinn:          dict[tuple[int, int], torch.Tensor],
    dict__abs_error__pinn:       dict[tuple[int, int], torch.Tensor],
    dict__rel_error__pinn:       dict[tuple[int, int], torch.Tensor],
    dict__abs_error__oppinn:     dict[tuple[int, int], torch.Tensor],
    dict__rel_error__oppinn:     dict[tuple[int, int], torch.Tensor],
    dict__mass__pinn:            dict[tuple[int, int], torch.Tensor],
    dict__mass__oppinn:          dict[tuple[int, int], torch.Tensor],
    dict__mass__target:          dict[tuple[int, int], torch.Tensor],
    dict__momentum__pinn:        dict[tuple[int, int], torch.Tensor],
    dict__momentum__oppinn:      dict[tuple[int, int], torch.Tensor],
    dict__momentum__target:      dict[tuple[int, int], torch.Tensor],
    dict__energy__pinn:          dict[tuple[int, int], torch.Tensor],
    dict__energy__oppinn:        dict[tuple[int, int], torch.Tensor],
    dict__energy__target:        dict[tuple[int, int], torch.Tensor],
    dict__entropy__pinn:         dict[tuple[int, int], torch.Tensor],
    dict__entropy__oppinn:       dict[tuple[int, int], torch.Tensor],
    dict__entropy__target:       dict[tuple[int, int], torch.Tensor],
    dict__train_time__pinn:      dict[int, list[float]],
    dict__train_time__oppinn:    dict[int, list[float]],
    dict__residual_loss__pinn:   dict[int, list[float]],
    dict__residual_loss__oppinn: dict[int, list[float]],
    dict__initial_loss__pinn:    dict[int, list[float]],
    dict__initial_loss__oppinn:  dict[int, list[float]],
    save:                        bool = True,
) -> None:
    """Evaluate trained PINN and opPINN checkpoints and compute physical observables.

    ## Description
    Loads model checkpoints, evaluates predictions on space-time grids, computes
    L2 errors against the ground truth, evaluates conservation quantities (mass,
    momentum, energy, entropy), and archives inference results.

    Arguments:
        `indices` (`list[int]`): List of epoch indices to evaluate.
        `seeds` (`list[int]`): List of random seeds.
        `path_base` (`Path`): Directory where model training results are located.
        `device` (`torch.device`): Computation device.
        `base_shape` (`tuple[int, ...]`): Target output grid shape.
        `grid_t` (`torch.Tensor`): 1D temporal coordinate tensor.
        `grid_v` (`torch.Tensor`): 1D spatial/velocity coordinate tensor.
        `v` (`torch.Tensor`): Velocity mesh tensor.
        `target` (`torch.Tensor`): Reference solution tensor.
        `dimension` (`int`): Dimension of the velocity domain.
        `depth` (`int`): Model depth.
        `width` (`int`): Model width.
        `softplus` (`float`): Activation parameter.
        `vhs_coeff` (`float`): VHS coefficient.
        `vhs_exponent` (`float`): VHS exponent.
        `init_type` (`str`): Initial condition identifier.
        `sample_t` (`str`): Time sampling strategy.
        `res_t` (`int`): Temporal resolution.
        `res_v` (`int`): Velocity resolution.
        `dict__*`: Various metric dictionary containers.
        `save` (`bool`, default: `True`): Whether to save evaluation output to disk.

    Returns:
        This function does not return values.
    """
    assert min(indices) >= 1, "The minimum index must be greater than or equal to 1."
    for index, seed in product(indices, seeds):
        path__pinn   = path_base / f"pinn{dimension}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}"
        path__oppinn = path_base / f"oppinn{dimension}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}"

        path_checkpoint__pinn   = path__pinn / "checkpoints/"
        path_checkpoint__oppinn = path__oppinn / "checkpoints/"

        pinn   = _generate_pinn(dimension, depth, width, softplus, device)
        oppinn = _generate_pinn(dimension, depth, width, softplus, device)
        loaded_pinn   = torch.load(path_checkpoint__pinn / f"params__epoch{index:05d}.pth",   weights_only=True, map_location=device)
        loaded_oppinn = torch.load(path_checkpoint__oppinn / f"params__epoch{index:05d}.pth", weights_only=True, map_location=device)
        pinn.load_state_dict(loaded_pinn['state_dict'])
        oppinn.load_state_dict(loaded_oppinn['state_dict'])

        points = torch.cartesian_prod(grid_t, *(grid_v for _ in range(dimension)))
        pinn.eval()
        oppinn.eval()
        with torch.inference_mode():
            pred_pinn   = pinn.forward(points).cpu().reshape(base_shape)
            pred_oppinn = oppinn.forward(points).cpu().reshape(base_shape)

        dict__pred__pinn[(index, seed)]   = pred_pinn
        dict__pred__oppinn[(index, seed)] = pred_oppinn

        dict__abs_error__pinn[(index, seed)]   = absolute_error(pred_pinn, target)
        dict__abs_error__oppinn[(index, seed)] = absolute_error(pred_oppinn, target)
        dict__rel_error__pinn[(index, seed)]   = relative_error(pred_pinn, target)
        dict__rel_error__oppinn[(index, seed)] = relative_error(pred_oppinn, target)

        dict__mass__target[(index, seed)]     = compute_mass_density(target, v)
        dict__momentum__target[(index, seed)] = compute_bulk_velocity(target, v)
        dict__energy__target[(index, seed)]   = compute_energy_density(target, v)
        dict__entropy__target[(index, seed)]  = compute_entropy_density(target, v)
        dict__mass__pinn[(index, seed)]       = compute_mass_density(pred_pinn, v)
        dict__momentum__pinn[(index, seed)]   = compute_bulk_velocity(pred_pinn, v)
        dict__energy__pinn[(index, seed)]     = compute_energy_density(pred_pinn, v)
        dict__entropy__pinn[(index, seed)]    = compute_entropy_density(pred_pinn, v)
        dict__mass__oppinn[(index, seed)]     = compute_mass_density(pred_oppinn, v)
        dict__momentum__oppinn[(index, seed)] = compute_bulk_velocity(pred_oppinn, v)
        dict__energy__oppinn[(index, seed)]   = compute_energy_density(pred_oppinn, v)
        dict__entropy__oppinn[(index, seed)]  = compute_entropy_density(pred_oppinn, v)

        history_pinn   = torch.load(path__pinn / "train_history.pth", weights_only=False)
        history_oppinn = torch.load(path__oppinn / "train_history.pth", weights_only=False)
        train_time__pinn   = list(accumulate(history_pinn['train_time']))
        train_time__oppinn = list(accumulate(history_oppinn['train_time']))
        dict__train_time__pinn[seed]      = train_time__pinn
        dict__train_time__oppinn[seed]    = train_time__oppinn
        dict__residual_loss__pinn[seed]   = history_pinn['loss_residual']
        dict__residual_loss__oppinn[seed] = history_oppinn['loss_residual']
        dict__initial_loss__pinn[seed]    = history_pinn['loss_initial']
        dict__initial_loss__oppinn[seed]  = history_oppinn['loss_initial']

    if save:
        path_inference = path_base.parent / f"inference_data__{sample_t}"
        path_inference.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'abs_error__pinn':          dict__abs_error__pinn,
                'rel_error__pinn':          dict__rel_error__pinn,
                'abs_error__oppinn':        dict__abs_error__oppinn,
                'rel_error__oppinn':        dict__rel_error__oppinn,
                'mass__pinn':               dict__mass__pinn,
                'mass__oppinn':             dict__mass__oppinn,
                'mass__target':             dict__mass__target,
                'momentum__pinn':           dict__momentum__pinn,
                'momentum__oppinn':         dict__momentum__oppinn,
                'momentum__target':         dict__momentum__target,
                'energy__pinn':             dict__energy__pinn,
                'energy__oppinn':           dict__energy__oppinn,
                'energy__target':           dict__energy__target,
                'entropy__pinn':            dict__entropy__pinn,
                'entropy__oppinn':          dict__entropy__oppinn,
                'entropy__target':          dict__entropy__target,

                'train_time__pinn':         dict__train_time__pinn,
                'train_time__oppinn':       dict__train_time__oppinn,
                'residual_loss__pinn':      dict__residual_loss__pinn,
                'residual_loss__oppinn':    dict__residual_loss__oppinn,
                'initial_loss__pinn':       dict__initial_loss__pinn,
                'initial_loss__oppinn':     dict__initial_loss__oppinn,
            },
            path_inference / f"inference__vhs__coeff{vhs_coeff:.2f}_exponent{vhs_exponent:.2f}__init_type_{init_type}__res_t{res_t:04d}_v{res_v:03d}.pth",
        )
    return


def _draw_snapshots(
    index:              int,
    seed:               int,
    max_v:              float,
    vhs_exponent:       float,
    vhs_coeff:          float,
    init_type:          str,
    target:             torch.Tensor,
    dict__pred__oppinn: dict[tuple[int, int], torch.Tensor],
    dict__pred__pinn:   dict[tuple[int, int], torch.Tensor],
    grid_t:             torch.Tensor,
) -> tuple[plt.Figure, plt.Axes]:
    """Render 2D velocity distribution snapshot figures at selected time steps.

    ## Description
    Plots snapshots comparing ground truth, opPINN predictions, and specPINN predictions
    across multiple time instances.

    Arguments:
        `index` (`int`): Checkpoint epoch index.
        `seed` (`int`): Model seed.
        `max_v` (`float`): Maximum velocity boundary.
        `vhs_exponent` (`float`): VHS model exponent.
        `vhs_coeff` (`float`): VHS model coefficient.
        `init_type` (`str`): Initial condition type.
        `target` (`torch.Tensor`): Ground truth tensor.
        `dict__pred__oppinn` (`dict`): opPINN predictions.
        `dict__pred__pinn` (`dict`): specPINN predictions.
        `grid_t` (`torch.Tensor`): Temporal grid tensor.

    Returns:
        `tuple[plt.Figure, plt.Axes]`: Created matplotlib figure and axes array.
    """
    time_indices: list[int] = [0, 200, 400, 600]
    xy_ticks:     list[float] = [-max_v, 0, max_v]
    xy_tick_labels: list[str]
    if max_v == 5.0:
        xy_tick_labels = ["-5", "0", "5"]
    elif max_v == 2 * torch.pi:
        xy_tick_labels = ["$-2\\pi$", "0", "$2\\pi$"]
    else:
        xy_tick_labels = [f"{-max_v:.1f}", "0", f"{max_v:.1f}"]

    cfg_imshow = {'origin': 'lower', 'extent': [-max_v, max_v, -max_v, max_v]}
    fig, axes = plt.subplots(3, len(time_indices), figsize=(10, 8), dpi=_DPI, sharex=True, sharey=True)
    suptitle: str
    if init_type == 'bkw':
        suptitle = "Initial condition (5.3)"
    elif init_type == 'maxwellian':
        suptitle = "Initial condition (5.2)"
    elif init_type == 'bimaxwellian':
        suptitle = "Initial condition (5.1)"
    else:
        suptitle = "Initial condition"
    suptitle += f" ($\\gamma={int(vhs_exponent)}$, $\\Lambda={_DICT_COEFF_STR.get(vhs_coeff, str(vhs_coeff))}$)"
    if index < _NUM_EPOCHS:
        suptitle += f"\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=_SIZE_SUPTITLE)

    axes[0, 0].set_ylabel("Ground truth",        fontsize=_SIZE_TITLE, rotation=90)
    axes[1, 0].set_ylabel(_LABEL_OPPINN,         fontsize=_SIZE_TITLE, rotation=90)
    axes[2, 0].set_ylabel(f"{_LABEL_SPECPINN} (ours)", fontsize=_SIZE_TITLE, rotation=90)
    for c, idx_t in enumerate(time_indices):
        axes[0, c].set_title(f"$t={grid_t[idx_t]:.1f}$", fontsize=_SIZE_TITLE)
        axes[0, c].imshow(target[idx_t], **cfg_imshow)
        axes[1, c].imshow(dict__pred__oppinn[(index, seed)][idx_t], **cfg_imshow)
        axes[2, c].imshow(dict__pred__pinn[(index, seed)][idx_t],   **cfg_imshow)

    for ax in axes.ravel():
        ax.set_xticks(xy_ticks, xy_tick_labels)
        ax.set_yticks(xy_ticks, xy_tick_labels)
        ax.tick_params(axis='both', which='major', labelsize=_SIZE_LABEL)
    fig.tight_layout()
    return fig, axes


@deprecated(reason="The figure for relative error should be aggregated, which can be done using 'aggregate_relative_errors.ipynb'.")
def _plot_error(
    index:                   int,
    init_type:               str,
    vhs_coeff:               float,
    vhs_exponent:            float,
    max_t:                   float,
    t:                       torch.Tensor,
    dict__abs_error__pinn:   dict[tuple[int, int], torch.Tensor],
    dict__abs_error__oppinn: dict[tuple[int, int], torch.Tensor],
    dict__rel_error__pinn:   dict[tuple[int, int], torch.Tensor],
    dict__rel_error__oppinn: dict[tuple[int, int], torch.Tensor],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot temporal evolution of absolute and relative L2 errors.

    ## Description
    Generates comparative error plots for specPINN and opPINN across all seeds.

    Arguments:
        `index` (`int`): Checkpoint epoch index.
        `init_type` (`str`): Initial condition type.
        `vhs_coeff` (`float`): VHS coefficient.
        `vhs_exponent` (`float`): VHS exponent.
        `max_t` (`float`): Maximum evaluation time.
        `t` (`torch.Tensor`): Time tensor.
        `dict__*`: Error dictionary containers.

    Returns:
        `tuple[plt.Figure, plt.Axes]`: Created matplotlib figure and axes array.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), dpi=_DPI, sharex=True)
    suptitle: str
    if init_type == 'bkw':
        suptitle = f"BKW solution ($\\Lambda={vhs_coeff:.2f}$)"
    elif init_type == 'maxwellian':
        suptitle = f"Maxwellian distribution ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    elif init_type == 'bimaxwellian':
        suptitle = f"Sum of two Maxwellian distributions ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    else:
        suptitle = f"Solution ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    if index < _NUM_EPOCHS:
        suptitle += f"\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=_SIZE_SUPTITLE)
    axes[0].set_ylabel("Absolute\n$L^2$ error", fontsize=_SIZE_TITLE)
    axes[1].set_ylabel("Relative\n$L^2$ error", fontsize=_SIZE_TITLE)
    axes[-1].set_xlabel("$t$", fontsize=_SIZE_TITLE)

    list_abs_errors__pinn   = [dict__abs_error__pinn[(index, _seed)]   for _seed in _LIST_SEEDS]
    list_abs_errors__oppinn = [dict__abs_error__oppinn[(index, _seed)] for _seed in _LIST_SEEDS]
    list_rel_errors__pinn   = [dict__rel_error__pinn[(index, _seed)]   for _seed in _LIST_SEEDS]
    list_rel_errors__oppinn = [dict__rel_error__oppinn[(index, _seed)] for _seed in _LIST_SEEDS]

    for seed, color, abs_err__pinn, abs_err__oppinn in zip(_LIST_SEEDS, _LIST_COLORS, list_abs_errors__pinn, list_abs_errors__oppinn):
        axes[0].plot(t, abs_err__pinn,   linewidth=_LINEWIDTH, color=color, label=f"{_LABEL_SPECPINN} (seed: {seed})")
        axes[0].plot(t, abs_err__oppinn, linewidth=_LINEWIDTH, color=color, linestyle=':', label=f"{_LABEL_OPPINN} (seed: {seed})")

    for seed, color, rel_err__pinn, rel_err__oppinn in zip(_LIST_SEEDS, _LIST_COLORS, list_rel_errors__pinn, list_rel_errors__oppinn):
        axes[1].plot(t, rel_err__pinn,   linewidth=_LINEWIDTH, color=color, label=f"{_LABEL_SPECPINN} (seed: {seed})")
        axes[1].plot(t, rel_err__oppinn, linewidth=_LINEWIDTH, color=color, linestyle=':', label=f"{_LABEL_OPPINN} (seed: {seed})")

    for ax in axes.ravel():
        ax.set_xlim((0, max_t))
        ax.set_yscale('log')
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=_BBOX_TO_ANCHOR, ncols=len(_LIST_SEEDS), fontsize=_SIZE_LABEL)
    fig.tight_layout()
    return fig, axes


def _plot_quantities(
    index:                  int,
    seed:                   int,
    init_type:              str,
    vhs_exponent:           float,
    vhs_coeff:              float,
    max_t:                  float,
    t:                      torch.Tensor,
    dict__mass__target:     dict[tuple[int, int], torch.Tensor],
    dict__mass__oppinn:     dict[tuple[int, int], torch.Tensor],
    dict__mass__pinn:       dict[tuple[int, int], torch.Tensor],
    dict__momentum__target: dict[tuple[int, int], torch.Tensor],
    dict__momentum__oppinn: dict[tuple[int, int], torch.Tensor],
    dict__momentum__pinn:   dict[tuple[int, int], torch.Tensor],
    dict__energy__target:   dict[tuple[int, int], torch.Tensor],
    dict__energy__oppinn:   dict[tuple[int, int], torch.Tensor],
    dict__energy__pinn:     dict[tuple[int, int], torch.Tensor],
    dict__entropy__target:  dict[tuple[int, int], torch.Tensor],
    dict__entropy__oppinn:  dict[tuple[int, int], torch.Tensor],
    dict__entropy__pinn:    dict[tuple[int, int], torch.Tensor],
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot physical observables and conservation laws over time.

    ## Description
    Plots density, bulk velocity components, energy, and entropy density compared
    against the reference solution.

    Arguments:
        `index` (`int`): Checkpoint epoch index.
        `seed` (`int`): Model seed.
        `init_type` (`str`): Initial condition identifier.
        `vhs_exponent` (`float`): VHS model exponent.
        `vhs_coeff` (`float`): VHS model coefficient.
        `max_t` (`float`): Maximum evaluation time.
        `t` (`torch.Tensor`): Time tensor.
        `dict__*`: Observables dictionary containers.

    Returns:
        `tuple[plt.Figure, dict[str, plt.Axes]]`: Matplotlib figure and dictionary of axes.
    """
    layout = [
        ['density', 'vx'],
        ['density', 'vy'],
        ['energy',  'entropy'],
        ['energy',  'entropy'],
    ]
    fig, axd = plt.subplot_mosaic(layout, figsize=(10, 6), layout="constrained", sharex=True)
    suptitle: str
    if init_type == 'bkw':
        suptitle = "BKW solution"
    elif init_type == 'maxwellian':
        suptitle = "Maxwellian distribution"
    elif init_type == 'bimaxwellian':
        suptitle = "Sum of two Maxwellian distributions"
    else:
        suptitle = "Solution"
    suptitle += f" ($\\gamma={vhs_exponent:.2f}$, $\\Lambda={vhs_coeff:.2f}$)"
    if index < _NUM_EPOCHS:
        suptitle += f"\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=_SIZE_SUPTITLE)
    xlim   = [0, max_t]
    xticks = tuple(range(int(max_t) + 1))
    limit_bulk_speed = 1e-2

    zeros_t             = torch.zeros_like(t)
    true_density        = dict__mass__target[(index, seed)][0]
    true_bulk_velocity  = dict__momentum__target[(index, seed)][0]
    true_energy_density = dict__energy__target[(index, seed)][0]

    axd['density'].set_title(r'Mass density ($\rho$)', fontsize=_SIZE_TITLE)
    axd['density'].plot(t, true_density + zeros_t, _STYLE_TARGET, linewidth=_LINEWIDTH)
    axd['density'].plot(t, dict__mass__oppinn[(index, seed)], _STYLE_OPPINN, linewidth=_LINEWIDTH, label=_LABEL_OPPINN)
    axd['density'].plot(t, dict__mass__pinn[(index, seed)],   _STYLE_PINN,   linewidth=_LINEWIDTH, label=f'{_LABEL_SPECPINN} (ours)')
    axd['density'].set_xticks(xticks, xticks)
    axd['density'].set_xlim(xlim)
    axd['density'].set_ylim(0.0, 2 * dict__mass__target[(index, seed)].max().item())
    axd['density'].grid(True)

    axd['vx'].set_title(r'Bulk velocity ($u_x$)', fontsize=_SIZE_TITLE)
    axd['vx'].plot(t, true_bulk_velocity[0] + zeros_t, _STYLE_TARGET, linewidth=_LINEWIDTH)
    axd['vx'].plot(t, dict__momentum__oppinn[(index, seed)][:, 0], _STYLE_OPPINN, linewidth=_LINEWIDTH, label=_LABEL_OPPINN)
    axd['vx'].plot(t, dict__momentum__pinn[(index, seed)][:, 0],   _STYLE_PINN,   linewidth=_LINEWIDTH, label=f'{_LABEL_SPECPINN} (ours)')
    axd['vx'].set_xticks(xticks, xticks)
    axd['vx'].set_xlim(xlim)
    axd['vx'].set_ylim(true_bulk_velocity[0] - limit_bulk_speed, true_bulk_velocity[0] + limit_bulk_speed)
    axd['vx'].grid(True)

    axd['vy'].set_title(r'Bulk velocity ($u_y$)', fontsize=_SIZE_TITLE)
    axd['vy'].plot(t, true_bulk_velocity[1] + zeros_t, _STYLE_TARGET, linewidth=_LINEWIDTH)
    axd['vy'].plot(t, dict__momentum__oppinn[(index, seed)][:, 1], _STYLE_OPPINN, linewidth=_LINEWIDTH, label=_LABEL_OPPINN)
    axd['vy'].plot(t, dict__momentum__pinn[(index, seed)][:, 1],   _STYLE_PINN,   linewidth=_LINEWIDTH, label=f'{_LABEL_SPECPINN} (ours)')
    axd['vy'].set_xticks(xticks, xticks)
    axd['vy'].set_xlim(xlim)
    axd['vy'].set_ylim(true_bulk_velocity[1] - limit_bulk_speed, true_bulk_velocity[1] + limit_bulk_speed)
    axd['vy'].grid(True)

    axd['energy'].set_title(r'Energy density ($E$)', fontsize=_SIZE_TITLE)
    axd['energy'].plot(t, true_energy_density + zeros_t, _STYLE_TARGET, linewidth=_LINEWIDTH)
    axd['energy'].plot(t, dict__energy__oppinn[(index, seed)], _STYLE_OPPINN, linewidth=_LINEWIDTH, label=_LABEL_OPPINN)
    axd['energy'].plot(t, dict__energy__pinn[(index, seed)],   _STYLE_PINN,   linewidth=_LINEWIDTH, label=f'{_LABEL_SPECPINN} (ours)')
    axd['energy'].set_xlabel(r'$t$', fontsize=_SIZE_TITLE)
    axd['energy'].set_xticks(xticks, xticks)
    axd['energy'].set_xlim(xlim)
    axd['energy'].set_ylim(0.0, 2 * dict__energy__target[(index, seed)].max().item())
    axd['energy'].grid(True)

    axd['entropy'].set_title(r'Entropy density ($H$)', fontsize=_SIZE_TITLE)
    axd['entropy'].plot(t, dict__entropy__oppinn[(index, seed)], _STYLE_OPPINN, linewidth=_LINEWIDTH, label=_LABEL_OPPINN)
    axd['entropy'].plot(t, dict__entropy__pinn[(index, seed)],   _STYLE_PINN,   linewidth=_LINEWIDTH, label=f'{_LABEL_SPECPINN} (ours)')
    axd['entropy'].set_xlabel(r'$t$', fontsize=_SIZE_TITLE)
    axd['entropy'].set_xticks(xticks, xticks)
    axd['entropy'].set_xlim(xlim)
    limit_max_entropy = dict__entropy__target[(index, seed)].max().item()
    limit_min_entropy = dict__entropy__target[(index, seed)].min().item()
    width_entropy     = limit_max_entropy - limit_min_entropy
    limit_max_entropy += width_entropy
    limit_min_entropy -= width_entropy
    axd['entropy'].set_ylim(limit_min_entropy, limit_max_entropy)
    axd['entropy'].grid(True)

    handles, labels = axd['density'].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=_BBOX_TO_ANCHOR, ncols=3, fontsize=_SIZE_LABEL)
    fig.tight_layout()
    return fig, axd


def _main() -> None:
    """Execute the inference and verification pipeline for PINN and opPINN models.

    ## Description
    Loads configurations and solutions, runs validation across all checkpoints and seeds,
    and produces quantitative snapshots and conservation metric plots.

    Arguments:
        This function does not take arguments.

    Returns:
        This function does not return values.
    """
    args = _parse_args()
    cuda_index: int   = args.cuda_index
    gamma:      float = args.gamma
    sample_t:   str   = args.sample_t
    res_t:      int   = args.res_t
    res_v:      int   = args.res_v
    init_type:  str   = args.init_type

    device = torch.device(f'cuda:{cuda_index}')
    torch.set_default_device(device)

    center_1 = torch.tensor([*(-INIT_COND__DEV for _ in range(DIMENSION - 1)), 0.0])
    center_2 = torch.tensor([*(+INIT_COND__DEV for _ in range(DIMENSION - 1)), 0.0])
    std_1    = torch.tensor([INIT_COND__STD])
    std_2    = torch.tensor([INIT_COND__STD])

    base_shape: tuple[int, ...] = tuple((res_t, *(res_v for _ in range(DIMENSION))))

    # Resolve config path relative to script location or project root
    path_config_candidate = Path(f"../config/config__{DIMENSION}d__gamma{gamma:.1f}__{init_type}.yaml")
    if not path_config_candidate.exists():
        path_config_candidate = _PROJECT_ROOT / f"config/config__{DIMENSION}d__gamma{gamma:.1f}__{init_type}.yaml"
    config: FurtherConfig = FurtherConfig(path_config_candidate)

    # Resolve result path relative to script location or project root
    path_base_candidate = Path(f"../result__{sample_t}")
    if not path_base_candidate.exists():
        path_base_candidate = _PROJECT_ROOT / f"result__{sample_t}"
    path_base: Path = path_base_candidate.resolve()
    assert path_base.exists(), f"The path [{str(path_base)}] does not exist."

    vhs_coeff:    float = config.vhs_coeff
    vhs_exponent: float = config.vhs_exponent

    max_t:   float = MAX_T__DICT[init_type]
    max_v:   float = MAX_V__DICT[init_type]
    density: float = DENSITY__DICT[init_type]
    delta_t: float = max_t / (res_t - 1)

    grid_t:  torch.Tensor = torch.linspace(0, max_t, res_t)
    grid_v:  torch.Tensor = space_grid(1, res_v, max_v).flatten()
    points = torch.cartesian_prod(grid_t, *(grid_v for _ in range(DIMENSION)))
    v      = torch.cartesian_prod(*(grid_v for _ in range(DIMENSION)))

    dict__pred__pinn:        dict[tuple[int, int], torch.Tensor] = {}
    dict__pred__oppinn:      dict[tuple[int, int], torch.Tensor] = {}
    dict__abs_error__pinn:   dict[tuple[int, int], torch.Tensor] = {}
    dict__rel_error__pinn:   dict[tuple[int, int], torch.Tensor] = {}
    dict__abs_error__oppinn: dict[tuple[int, int], torch.Tensor] = {}
    dict__rel_error__oppinn: dict[tuple[int, int], torch.Tensor] = {}

    dict__mass__pinn:        dict[tuple[int, int], torch.Tensor] = {}
    dict__mass__oppinn:      dict[tuple[int, int], torch.Tensor] = {}
    dict__mass__target:      dict[tuple[int, int], torch.Tensor] = {}
    dict__momentum__pinn:    dict[tuple[int, int], torch.Tensor] = {}
    dict__momentum__oppinn:  dict[tuple[int, int], torch.Tensor] = {}
    dict__momentum__target:  dict[tuple[int, int], torch.Tensor] = {}
    dict__energy__pinn:      dict[tuple[int, int], torch.Tensor] = {}
    dict__energy__oppinn:    dict[tuple[int, int], torch.Tensor] = {}
    dict__energy__target:    dict[tuple[int, int], torch.Tensor] = {}
    dict__entropy__pinn:     dict[tuple[int, int], torch.Tensor] = {}
    dict__entropy__oppinn:   dict[tuple[int, int], torch.Tensor] = {}
    dict__entropy__target:   dict[tuple[int, int], torch.Tensor] = {}

    dict__train_time__pinn:      dict[int, list[float]] = {}
    dict__train_time__oppinn:    dict[int, list[float]] = {}
    dict__residual_loss__pinn:   dict[int, list[float]] = {}
    dict__residual_loss__oppinn: dict[int, list[float]] = {}
    dict__initial_loss__pinn:    dict[int, list[float]] = {}
    dict__initial_loss__oppinn:  dict[int, list[float]] = {}

    path_solution = path_base.parent / "solutions" / \
        f"vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__max_t{max_t:.2f}__res_t{res_t:04d}__max_v{max_v:.2f}__res_v{res_v:03d}.pth"
    path_solution.parent.mkdir(parents=True, exist_ok=True)
    col = FPL_spectral(DIMENSION, res_v, max_v, vhs_coeff, vhs_exponent, device=device)

    target: torch.Tensor
    if path_solution.exists():
        print(f"Loading the solution from [{str(path_solution)}]...")
        target = torch.load(path_solution, weights_only=False)
    else:
        points = torch.cartesian_prod(grid_t, *(grid_v for _ in range(DIMENSION)))
        v      = torch.cartesian_prod(*(grid_v for _ in range(DIMENSION)))
        if init_type == 'bkw':
            print("Generating the analytic BKW solution...")
            target = bkw(points, vhs_coeff, BKW_COEFF_EXT, density).reshape(base_shape).cpu()
        elif init_type == 'maxwellian':
            print("Generating the analytic Maxwellian solution...")
            target = maxwellian(DIMENSION, v, center_1, std_1, density).reshape(base_shape[1:]).cpu()
            target = target.unsqueeze(0).repeat(base_shape[0], *(1 for _ in range(DIMENSION)))
        elif init_type == 'bimaxwellian':
            print("Generating the numerical biMaxwellian solution...")
            f_init = bimaxwellian(DIMENSION, v, center_1, center_2, std_1, std_2, density)
            target = col.solve(0.0, max_t, delta_t, f_init).cpu().reshape(base_shape)
        else:
            warnings.warn(f"The initial condition [{init_type}] is not supported.", RuntimeWarning)
        torch.save(target, path_solution)
        print(f"\tSaved the solution at [{str(path_solution)}].")

    v = v.cpu()
    t = grid_t.cpu()
    torch.cuda.empty_cache()

    _validate_models(
        indices                     = _LIST_INDEX,
        seeds                       = _LIST_SEEDS,
        path_base                   = path_base,
        device                      = device,
        base_shape                  = base_shape,
        grid_t                      = grid_t,
        grid_v                      = grid_v,
        v                           = v,
        target                      = target,
        dimension                   = DIMENSION,
        depth                       = DEPTH,
        width                       = WIDTH,
        softplus                    = SOFTPLUS,
        vhs_coeff                   = vhs_coeff,
        vhs_exponent                = vhs_exponent,
        init_type                   = init_type,
        sample_t                    = sample_t,
        res_t                       = res_t,
        res_v                       = res_v,
        dict__pred__pinn            = dict__pred__pinn,
        dict__pred__oppinn          = dict__pred__oppinn,
        dict__abs_error__pinn       = dict__abs_error__pinn,
        dict__rel_error__pinn       = dict__rel_error__pinn,
        dict__abs_error__oppinn     = dict__abs_error__oppinn,
        dict__rel_error__oppinn     = dict__rel_error__oppinn,
        dict__mass__pinn            = dict__mass__pinn,
        dict__mass__oppinn          = dict__mass__oppinn,
        dict__mass__target          = dict__mass__target,
        dict__momentum__pinn        = dict__momentum__pinn,
        dict__momentum__oppinn      = dict__momentum__oppinn,
        dict__momentum__target      = dict__momentum__target,
        dict__energy__pinn          = dict__energy__pinn,
        dict__energy__oppinn        = dict__energy__oppinn,
        dict__energy__target        = dict__energy__target,
        dict__entropy__pinn         = dict__entropy__pinn,
        dict__entropy__oppinn       = dict__entropy__oppinn,
        dict__entropy__target       = dict__entropy__target,
        dict__train_time__pinn      = dict__train_time__pinn,
        dict__train_time__oppinn    = dict__train_time__oppinn,
        dict__residual_loss__pinn   = dict__residual_loss__pinn,
        dict__residual_loss__oppinn = dict__residual_loss__oppinn,
        dict__initial_loss__pinn    = dict__initial_loss__pinn,
        dict__initial_loss__oppinn  = dict__initial_loss__oppinn,
        save                        = True,
    )

    cfg_savefig = {'dpi': _DPI, 'bbox_inches': 'tight'}
    for index in _LIST_INDEX:
        prefix_str  = _get_prefix(index, DIMENSION, vhs_coeff, vhs_exponent, init_type)
        path_images = _PROJECT_ROOT / "inference_figures" / sample_t / prefix_str
        if not path_images.exists():
            path_images.mkdir(parents=True, exist_ok=True)

        fig_snapshots, _  = _draw_snapshots(
            index              = index,
            seed               = 0,
            max_v              = max_v,
            vhs_exponent       = vhs_exponent,
            vhs_coeff          = vhs_coeff,
            init_type          = init_type,
            target             = target,
            dict__pred__oppinn = dict__pred__oppinn,
            dict__pred__pinn   = dict__pred__pinn,
            grid_t             = grid_t,
        )
        fig_quantities, _ = _plot_quantities(
            index                  = index,
            seed                   = 0,
            init_type              = init_type,
            vhs_exponent           = vhs_exponent,
            vhs_coeff              = vhs_coeff,
            max_t                  = max_t,
            t                      = t,
            dict__mass__target     = dict__mass__target,
            dict__mass__oppinn     = dict__mass__oppinn,
            dict__mass__pinn       = dict__mass__pinn,
            dict__momentum__target = dict__momentum__target,
            dict__momentum__oppinn = dict__momentum__oppinn,
            dict__momentum__pinn   = dict__momentum__pinn,
            dict__energy__target   = dict__energy__target,
            dict__energy__oppinn   = dict__energy__oppinn,
            dict__energy__pinn     = dict__energy__pinn,
            dict__entropy__target  = dict__entropy__target,
            dict__entropy__oppinn  = dict__entropy__oppinn,
            dict__entropy__pinn    = dict__entropy__pinn,
        )
        fig_snapshots.savefig(path_images / "snapshots.pdf",   **cfg_savefig)
        fig_quantities.savefig(path_images / "quantities.pdf", **cfg_savefig)

    return


##################################################
if __name__ == '__main__':
    _main()