##################################################
# Import libraries
##################################################
import  argparse
import  warnings
from    pathlib                 import  Path
from    itertools               import  accumulate, product
import  matplotlib.pyplot       as      plt

import  torch
from    deep_numerical.utils    import  space_grid, relative_error

from    models                  import  *
from    utils                   import  absolute_error, relative_error, maxwellian, bimaxwellian, bkw
from    utils                   import  compute_mass_density, compute_bulk_velocity, compute_energy_density, compute_entropy_density


from    config.base_config__2d      import  *
from    config                      import  FurtherConfig


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=3, help='The index of the GPU to use.')
parser.add_argument('--gamma', type=float, help='The value of gamma in the Fokker-Planck-Landau equation.')
parser.add_argument('--sample_t', type=str, help='The mod of sampling the time variable.')
parser.add_argument('--res_t', type=int, help='The resolution in the time variable.')
parser.add_argument('--res_v', type=int, help='The resolution in the velocity variable.')
parser.add_argument('--init_type', type=str, help='The initial condition.')
args = parser.parse_args()
cuda_index: int     = args.cuda_index
gamma:      float   = args.gamma
sample_t:   str     = args.sample_t
res_t:      int     = args.res_t
res_v:      int     = args.res_v
init_type:  str     = args.init_type

DEVICE = torch.device(f'cuda:{cuda_index}')
torch.set_default_device(DEVICE)


NUM_EPOCHS: int = 5000
CENTER_1 = torch.tensor([*(-INIT_COND__DEV for _ in range(DIMENSION-1)), 0.0])
CENTER_2 = torch.tensor([*(+INIT_COND__DEV for _ in range(DIMENSION-1)), 0.0])
STD_1    = torch.tensor([INIT_COND__STD])
STD_2    = torch.tensor([INIT_COND__STD])


LIST_INDEX      = [1000*k for k in range(1, 6)]
LIST_SEEDS      = list(range(5))
LIST_COLORS     = ['red', 'orange', 'green', 'blue', 'purple']
SIZE_SUPTITLE   = 20
SIZE_TITLE      = 16
SIZE_LABEL      = 14
LINEWIDTH       = 3
STYLE_TARGET    = 'k--'
STYLE_OPPINN    = 'r-'
STYLE_PINN      = 'g-'
DPI             = 1000
BBOX_TO_ANCHOR  = tuple((0.5, -0.01))


##################################################
base_shape:     tuple[int, ...] = tuple((res_t, *(res_v for _ in range(DIMENSION))))
config:         FurtherConfig   = FurtherConfig(f"./config/config__{DIMENSION}d__gamma{gamma:.1f}__{init_type}.yaml")

path_base:      Path    = Path().cwd() / f"result__{sample_t}"
assert path_base.exists(), f"The path [{str(path_base)}] does not exist."

vhs_coeff:      float   = config.vhs_coeff
vhs_exponent:   float   = config.vhs_exponent

max_t:          float   = MAX_T__DICT[init_type]
max_v:          float   = MAX_V__DICT[init_type]
density:        float   = DENSITY__DICT[init_type]
delta_t:        float   = max_t/(res_t-1)
delta_v:        float   = (2*max_v)/(res_v-1)

grid_t:         torch.Tensor    = torch.linspace(0, max_t, res_t)
grid_v:         torch.Tensor    = space_grid(1, res_v, max_v).flatten()
points  = torch.cartesian_prod(grid_t, *(grid_v for _ in range(DIMENSION)))
v       = torch.cartesian_prod(*(grid_v for _ in range(DIMENSION)))

# Key: index, seed
dict__pred__pinn:           dict[tuple[int, int], torch.Tensor] = {}
dict__pred__oppinn:         dict[tuple[int, int], torch.Tensor] = {}

dict__abs_error__pinn:      dict[tuple[int, int], torch.Tensor] = {}
dict__rel_error__pinn:      dict[tuple[int, int], torch.Tensor] = {}
dict__abs_error__oppinn:    dict[tuple[int, int], torch.Tensor] = {}
dict__rel_error__oppinn:    dict[tuple[int, int], torch.Tensor] = {}

dict__mass__pinn:           dict[tuple[int, int], torch.Tensor] = {}
dict__mass__oppinn:         dict[tuple[int, int], torch.Tensor] = {}
dict__mass__target:         dict[tuple[int, int], torch.Tensor] = {}
dict__momentum__pinn:       dict[tuple[int, int], torch.Tensor] = {}
dict__momentum__oppinn:     dict[tuple[int, int], torch.Tensor] = {}
dict__momentum__target:     dict[tuple[int, int], torch.Tensor] = {}
dict__energy__pinn:         dict[tuple[int, int], torch.Tensor] = {}
dict__energy__oppinn:       dict[tuple[int, int], torch.Tensor] = {}
dict__energy__target:       dict[tuple[int, int], torch.Tensor] = {}
dict__entropy__pinn:        dict[tuple[int, int], torch.Tensor] = {}
dict__entropy__oppinn:      dict[tuple[int, int], torch.Tensor] = {}
dict__entropy__target:      dict[tuple[int, int], torch.Tensor] = {}

# Key: seed
dict__train_time__pinn:         dict[int, torch.Tensor] = {}
dict__train_time__oppinn:       dict[int, torch.Tensor] = {}
dict__residual_loss__pinn:      dict[int, torch.Tensor] = {}
dict__residual_loss__oppinn:    dict[int, torch.Tensor] = {}
dict__initial_loss__pinn:       dict[int, torch.Tensor] = {}
dict__initial_loss__oppinn:     dict[int, torch.Tensor] = {}


##################################################
path_solution = path_base.parent / "solutions" / \
    f"vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__max_t{max_t:.2f}__res_t{res_t:04d}__max_v{max_v:.2f}__res_v{res_v:03d}.pth"
path_solution.parent.mkdir(parents=True, exist_ok=True)
col = FPL_spectral(DIMENSION, res_v, max_v, vhs_coeff, vhs_exponent, device=DEVICE)
if path_solution.exists():
    print(f"Loading the solution from [{str(path_solution)}]...")
    target = torch.load(path_solution, weights_only=False)
else:
    points  = torch.cartesian_prod(grid_t, *(grid_v for _ in range(DIMENSION)))
    v       = torch.cartesian_prod(*(grid_v for _ in range(DIMENSION)))
    if init_type=='bkw':
        print("Generating the analytic BKW solution...")
        target  = bkw(points, vhs_coeff, BKW_COEFF_EXT, density).reshape(base_shape).cpu()
    elif init_type=='maxwellian':
        print("Generating the analytic Maxwellian solution...")
        target  = maxwellian(DIMENSION, v, CENTER_1, STD_1, density).reshape(base_shape[1:]).cpu()
        target  = target.unsqueeze(0).repeat(base_shape[0], *((1,) for _ in range(DIMENSION)))
    elif init_type=='bimaxwellian':
        print("Generating the numerical biMaxwellian solution...")
        f_init  = bimaxwellian(DIMENSION, v, CENTER_1, CENTER_2, STD_1, STD_2, density)
        target  = col.solve(0.0, max_t, delta_t, f_init).cpu().reshape(base_shape)
    else:
        warnings.warn(f"The initial condition [{init_type}] is not supported.", RuntimeWarning)
    torch.save(target, path_solution)
    print(f"\tSaved the solution at [{str(path_solution)}].")
v = v.cpu()
t = grid_t.cpu()
torch.cuda.empty_cache()


##################################################
def generate_pinn() -> PINN_FPL:
    return PINN_FPL(
        dimension   = DIMENSION,
        depth       = DEPTH,
        width       = WIDTH,
        softplus    = SOFTPLUS,
        device      = DEVICE,
    )


def get_prefix(index: int) -> str:
    return f"{DIMENSION}D__vhs__coeff{vhs_coeff:.2f}_exponent_{vhs_exponent:.2f}__init_type_{init_type}__index_{index}"


##################################################
def validate_models(indices: list[int] = LIST_INDEX, save: bool=True) -> None:
    assert min(indices)>=1, "The minimum index must be greater than or equal to 1."
    for index, seed in product(indices, LIST_SEEDS):
        path__pinn   = path_base / f"pinn{DIMENSION}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}"
        path__oppinn = path_base / f"oppinn{DIMENSION}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}"
        # The above two paths will be explicitly used later in this loop
        path_checkpoint__pinn   = path__pinn / "checkpoints/"
        path_checkpoint__oppinn = path__oppinn / "checkpoints/"
            
        ##################################################
        # Conduct infernence
        ##################################################
        pinn    = generate_pinn()
        oppinn  = generate_pinn()
        loaded_pinn = torch.load(
            path_checkpoint__pinn/f"params__epoch{index:05d}.pth",
            weights_only=True, map_location=DEVICE,
        )
        loaded_oppinn = torch.load(
            path_checkpoint__oppinn/f"params__epoch{index:05d}.pth",
            weights_only=True, map_location=DEVICE,
        )
        pinn.load_state_dict(   loaded_pinn['state_dict']   )
        oppinn.load_state_dict( loaded_oppinn['state_dict'] )
        
        points = torch.cartesian_prod(grid_t, *(grid_v for _ in range(DIMENSION)))
        # Evaluation
        pinn.eval()
        oppinn.eval()
        with torch.inference_mode():
            pred_pinn = pinn.forward(points).cpu().reshape(base_shape)
            pred_oppinn = oppinn.forward(points).cpu().reshape(base_shape)
        # Save to the dictionaries
        dict__pred__pinn[   (index, seed)] = pred_pinn
        dict__pred__oppinn[ (index, seed)] = pred_oppinn
        
        ##################################################
        # Compute errors and moments
        ##################################################
        # Compute errors
        dict__abs_error__pinn[  (index, seed)] = absolute_error(pred_pinn,   target)
        dict__abs_error__oppinn[(index, seed)] = absolute_error(pred_oppinn, target)
        dict__rel_error__pinn[  (index, seed)] = relative_error(pred_pinn,   target)
        dict__rel_error__oppinn[(index, seed)] = relative_error(pred_oppinn, target)
        
        # Compute moments and entropy
        dict__mass__target[    (index, seed)] = compute_mass_density(target, v)
        dict__momentum__target[(index, seed)] = compute_bulk_velocity(target, v)
        dict__energy__target[  (index, seed)] = compute_energy_density(target, v)
        dict__entropy__target[ (index, seed)] = compute_entropy_density(target, v)
        dict__mass__pinn[      (index, seed)] = compute_mass_density(pred_pinn, v)
        dict__momentum__pinn[  (index, seed)] = compute_bulk_velocity(pred_pinn, v)
        dict__energy__pinn[    (index, seed)] = compute_energy_density(pred_pinn, v)
        dict__entropy__pinn[   (index, seed)] = compute_entropy_density(pred_pinn, v)
        dict__mass__oppinn[    (index, seed)] = compute_mass_density(pred_oppinn, v)
        dict__momentum__oppinn[(index, seed)] = compute_bulk_velocity(pred_oppinn, v)
        dict__energy__oppinn[  (index, seed)] = compute_energy_density(pred_oppinn, v)
        dict__entropy__oppinn[ (index, seed)] = compute_entropy_density(pred_oppinn, v)
        
        
        ##################################################
        # Load training history
        ##################################################
        history_pinn    = torch.load(path__pinn / "train_history.pth", weights_only=False)
        history_oppinn  = torch.load(path__oppinn / "train_history.pth", weights_only=False)
        # Save training time (PINNs only, not the surrogate models)
        train_time__pinn    = list(accumulate(history_pinn['train_time']))
        train_time__oppinn  = list(accumulate(history_oppinn['train_time']))
        dict__train_time__pinn[  seed]      = train_time__pinn
        dict__train_time__oppinn[seed]      = train_time__oppinn
        dict__residual_loss__pinn[seed]     = history_pinn['loss_residual']
        dict__residual_loss__oppinn[seed]   = history_oppinn['loss_residual']
        dict__initial_loss__pinn[seed]      = history_pinn['loss_initial']
        dict__initial_loss__oppinn[seed]    = history_oppinn['loss_initial']
    
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
            path_inference / f"inference__vhs__coeff{vhs_coeff:.2f}_exponent{vhs_exponent:.2f}__init_type_{init_type}__res_t{res_t:04d}_v{res_v:03d}.pth"
        )
    return None


##################################################
def draw_snapshots(index: int, seed: int=0) -> tuple[plt.Figure, plt.Axes]:
    time_indices = [0, 200, 400, 600]
    xy_ticks = [-max_v, 0, max_v]
    xy_tick_labels = [f"{-max_v:.2f}", "0", f"{max_v:.2f}"]
    _cfg_imshow = {'origin': 'lower', 'extent': [-max_v, max_v, -max_v, max_v]}
    fig: plt.Figure
    axes: plt.Axes
    fig, axes = plt.subplots(3, len(time_indices), figsize=(10, 8), dpi=DPI, sharex=True, sharey=True)
    suptitle: str
    if init_type=='bkw':
        suptitle = f"BKW solution ($\\Lambda={vhs_coeff:.2f}$)"
    elif init_type=='maxwellian':
        suptitle = f"Maxwellian distribution ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    elif init_type=='bimaxwellian':
        suptitle = f"Sum of two Maxwellian distributions ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    if index<NUM_EPOCHS:
        suptitle += f"\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=SIZE_SUPTITLE)
    
    axes[0, 0].set_ylabel("Ground truth",   fontsize=SIZE_TITLE, rotation=90)
    axes[1, 0].set_ylabel("opPINN",         fontsize=SIZE_TITLE, rotation=90)
    axes[2, 0].set_ylabel("PINN (ours)",    fontsize=SIZE_TITLE, rotation=90)
    for c, idx_t in enumerate(time_indices):
        axes[0, c].set_title(f"$t={grid_t[idx_t]:.1f}$", fontsize=SIZE_TITLE)
        axes[0, c].imshow(target[idx_t], **_cfg_imshow)
        axes[1, c].imshow(dict__pred__oppinn[(index, seed)][idx_t], **_cfg_imshow)
        axes[2, c].imshow(dict__pred__pinn[(index, seed)][idx_t], **_cfg_imshow)
        axes
    ax: plt.Axes
    for ax in axes.ravel():
        ax.set_xticks(xy_ticks, xy_tick_labels)
        ax.set_yticks(xy_ticks, xy_tick_labels)
    fig.tight_layout()
    return fig, axes


def plot_error(index: int) -> tuple[plt.Figure, plt.Axes]:
    fig: plt.Figure
    axes: plt.Axes
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), dpi=DPI, sharex=True)
    suptitle: str
    if init_type=='bkw':
        suptitle = f"BKW solution ($\\Lambda={vhs_coeff:.2f}$)"
    elif init_type=='maxwellian':
        suptitle = f"Maxwellian distribution ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    elif init_type=='bimaxwellian':
        suptitle = f"Sum of two Maxwellian distributions ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    if index<NUM_EPOCHS:
        suptitle += f"\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=SIZE_SUPTITLE)
    axes[0].set_ylabel("Absolute\n$L^2$ error", fontsize=SIZE_TITLE)
    axes[1].set_ylabel("Relative\n$L^2$ error", fontsize=SIZE_TITLE)
    axes[-1].set_xlabel("$t$", fontsize=SIZE_TITLE)

    list_abs_errors__pinn   = [dict__abs_error__pinn[  (index, _seed)] for _seed in LIST_SEEDS]
    list_abs_errors__oppinn = [dict__abs_error__oppinn[(index, _seed)] for _seed in LIST_SEEDS]
    list_rel_errors__pinn   = [dict__rel_error__pinn[  (index, _seed)] for _seed in LIST_SEEDS]
    list_rel_errors__oppinn = [dict__rel_error__oppinn[(index, _seed)] for _seed in LIST_SEEDS]

    # Absolute errors
    for seed, _c, abs_err__pinn, abs_err__oppinn in zip(LIST_SEEDS, LIST_COLORS, list_abs_errors__pinn, list_abs_errors__oppinn):
        axes[0].plot(
            t, abs_err__pinn,
            linewidth=LINEWIDTH, color=_c,
            label=f"PINN (seed: {seed})",
        )
        axes[0].plot(
            t, abs_err__oppinn,
            linewidth=LINEWIDTH, color=_c, linestyle=':',
            label=f"opPINN (seed: {seed})",
        )
    # Relative errors
    for seed, _c, rel_err__pinn, rel_err__oppinn in zip(LIST_SEEDS, LIST_COLORS, list_rel_errors__pinn, list_rel_errors__oppinn):
        axes[1].plot(
            t, rel_err__pinn,
            linewidth=LINEWIDTH, color=_c,
            label=f"PINN (seed: {seed})",
        )
        axes[1].plot(
            t, rel_err__oppinn,
            linewidth=LINEWIDTH, color=_c, linestyle=':',
            label=f"opPINN (seed: {seed})",
        )

    for ax in axes.ravel():
        ax.set_xlim((0, max_t))
        ax.set_yscale('log')
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=BBOX_TO_ANCHOR, ncols=len(LIST_SEEDS), fontsize=SIZE_LABEL)
    fig.tight_layout()
    return fig, axes


def plot_quantities(index: int, seed: int=0) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    layout = [
        ['density', 'vx'],
        ['density', 'vy'],
        ['energy',  'entropy'],
        ['energy',  'entropy'],
    ]
    fig, axd = plt.subplot_mosaic(layout, figsize=(10, 6), layout="constrained", sharex=True)
    suptitle: str
    if init_type=='bkw':
        suptitle = f"BKW solution ($\\Lambda={vhs_coeff:.2f}$)"
    elif init_type=='maxwellian':
        suptitle = f"Maxwellian distribution ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    elif init_type=='bimaxwellian':
        suptitle = f"Sum of two Maxwellian distributions ($\\Lambda={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)"
    if index<NUM_EPOCHS:
        suptitle += f"\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=SIZE_SUPTITLE)
    xlim = [0, max_t]
    xticks = tuple(range(int(max_t)+1))
    _limit_bulk_speed = 1e-2
    
    zeros_t = torch.zeros_like(t)
    true_density = dict__mass__target[(index, seed)][0]
    true_bulk_velocity = dict__momentum__target[(index, seed)][0]
    true_energy_density = dict__energy__target[(index, seed)][0]
    
    axd['density'].set_title(r'Mass density ($\rho$)', fontsize=SIZE_TITLE)
    axd['density'].plot(t, true_density+zeros_t, STYLE_TARGET, linewidth=LINEWIDTH)
    axd['density'].plot(t, dict__mass__oppinn[(index, seed)], STYLE_OPPINN,  linewidth=LINEWIDTH, label='opPINN')
    axd['density'].plot(t, dict__mass__pinn[  (index, seed)], STYLE_PINN,  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['density'].set_xticks(xticks, xticks)
    axd['density'].set_xlim(xlim)
    axd['density'].set_ylim(0.0, 2*dict__mass__target[(index, seed)].max().item())
    axd['density'].grid(True)
    
    axd['vx'].set_title(r'Bulk velocity ($v_x$)', fontsize=SIZE_TITLE)
    axd['vx'].plot(t, true_bulk_velocity[0]+zeros_t, STYLE_TARGET, linewidth=LINEWIDTH)
    axd['vx'].plot(t, dict__momentum__oppinn[(index, seed)][:, 0], STYLE_OPPINN,  linewidth=LINEWIDTH, label='opPINN')
    axd['vx'].plot(t, dict__momentum__pinn[  (index, seed)][:, 0], STYLE_PINN,  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['vx'].set_xticks(xticks, xticks)
    axd['vx'].set_xlim(xlim)
    axd['vx'].set_ylim(-_limit_bulk_speed, _limit_bulk_speed)
    axd['vx'].grid(True)
    
    axd['vy'].set_title(r'Bulk velocity ($v_y$)', fontsize=SIZE_TITLE)
    axd['vy'].plot(t, true_bulk_velocity[1]+zeros_t, STYLE_TARGET, linewidth=LINEWIDTH)
    axd['vy'].plot(t, dict__momentum__oppinn[(index, seed)][:, 1], STYLE_OPPINN,  linewidth=LINEWIDTH, label='opPINN')
    axd['vy'].plot(t, dict__momentum__pinn[  (index, seed)][:, 1], STYLE_PINN,  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['vy'].set_xticks(xticks, xticks)
    axd['vy'].set_xlim(xlim)
    axd['vy'].set_ylim(-_limit_bulk_speed, _limit_bulk_speed)
    axd['vy'].grid(True)
    
    axd['energy'].set_title(r'Energy density ($E$)', fontsize=SIZE_TITLE)
    axd['energy'].plot(t, true_energy_density+zeros_t, STYLE_TARGET, linewidth=LINEWIDTH)
    axd['energy'].plot(t, dict__energy__oppinn[(index, seed)], STYLE_OPPINN,  linewidth=LINEWIDTH, label='opPINN')
    axd['energy'].plot(t, dict__energy__pinn[  (index, seed)], STYLE_PINN,  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['energy'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['energy'].set_xticks(xticks, xticks)
    axd['energy'].set_xlim(xlim)
    axd['energy'].set_ylim(0.0, 2*dict__energy__target[(index, seed)].max().item())
    axd['energy'].grid(True)
    
    axd['entropy'].set_title(r'Entropy density ($H$)', fontsize=SIZE_TITLE)
    axd['entropy'].plot(t, dict__entropy__oppinn[(index, seed)], STYLE_OPPINN,  linewidth=LINEWIDTH, label='opPINN')
    axd['entropy'].plot(t, dict__entropy__pinn[  (index, seed)], STYLE_PINN,  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['entropy'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['entropy'].set_xticks(xticks, xticks)
    axd['entropy'].set_xlim(xlim)
    _limit_max_entropy = dict__entropy__target[(index, seed)].max().item()
    _limit_min_entropy = dict__entropy__target[(index, seed)].min().item()
    _width_entropy = _limit_max_entropy-_limit_min_entropy
    _limit_max_entropy += _width_entropy
    _limit_min_entropy -= _width_entropy
    axd['entropy'].set_ylim(_limit_min_entropy, _limit_max_entropy)
    axd['entropy'].grid(True)
    
    handles, labels = axd['density'].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=BBOX_TO_ANCHOR, ncols=3, fontsize=SIZE_LABEL)

    fig.tight_layout()
    return fig, axd


##################################################
validate_models()

_cfg_savefig = {'dpi': DPI, 'bbox_inches': 'tight'}
for index in LIST_INDEX:
    path_images = Path().cwd() / "images" / sample_t / get_prefix(index)
    if path_images.exists() is False:   path_images.mkdir(parents=True, exist_ok=True)
    fig_snapshots, axes_snapshots = draw_snapshots(index)
    fig_errors, axes_errors = plot_error(index)
    fig_quantities, axes_quantities = plot_quantities(index)
    fig_snapshots.savefig(path_images/"snapshots.pdf", **_cfg_savefig)
    fig_errors.savefig(path_images/"errors.pdf", **_cfg_savefig)
    fig_quantities.savefig(path_images/"quantities.pdf", **_cfg_savefig)


##################################################
# End of file