##################################################
# Import libraries
##################################################
import  argparse
import  warnings
from    pathlib                 import  Path
from    itertools               import  product
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


CENTER_1 = torch.tensor([*(-INIT_COND__DEV for _ in range(DIMENSION-1)), 0.0])
CENTER_2 = torch.tensor([*(+INIT_COND__DEV for _ in range(DIMENSION-1)), 0.0])
STD_1    = torch.tensor([INIT_COND__STD])
STD_2    = torch.tensor([INIT_COND__STD])


LIST_INDEX      = [1000*k for k in range(1, 6)]
LIST_SEEDS      = list(range(5))
LIST_COLORS     = ['red', 'orange', 'green', 'blue', 'purple']
SIZE_SUPTITLE   = 20
SIZE_TITLE      = 16
LINEWIDTH       = 3
DPI             = 1000


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


##################################################
path_solution = path_base.parent / "solutions" / \
    f"vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__max_t{max_t:.2f}__res_t{res_t:04d}__max_v{max_v:.2f}__res_v{res_v:03d}.pth"
path_solution.parent.mkdir(parents=True, exist_ok=True)
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
        col     = FPL_spectral(DIMENSION, res_v, max_v, vhs_coeff, vhs_exponent, device=DEVICE)
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
def validate_models(indices: list[int] = LIST_INDEX, save: bool=False) -> None:
    for index, seed in product(indices, LIST_SEEDS):
        path_checkpoint__pinn = path_base / f"pinn{DIMENSION}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}" / "checkpoints/"
        path_checkpoint__oppinn = path_base / f"oppinn{DIMENSION}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}" / "checkpoints/"
            
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
        pinn.eval()
        oppinn.eval()
        with torch.inference_mode():
            pred_pinn   = pinn.forward(points).cpu().reshape(base_shape)
            pred_oppinn = oppinn.forward(points).cpu().reshape(base_shape)
            
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
        
        # Save to the dictionaries
        dict__pred__pinn[   (index, seed)] = pred_pinn
        dict__pred__oppinn[ (index, seed)] = pred_oppinn
    
    if save:
        torch.save(
            {
                'abs_error__pinn':        dict__abs_error__pinn,
                'rel_error__pinn':        dict__rel_error__pinn,
                'abs_error__oppinn':      dict__abs_error__oppinn,
                'rel_error__oppinn':      dict__rel_error__oppinn,
                'mass__pinn':             dict__mass__pinn,
                'mass__oppinn':           dict__mass__oppinn,
                'mass__target':           dict__mass__target,
                'momentum__pinn':         dict__momentum__pinn,
                'momentum__oppinn':       dict__momentum__oppinn,
                'momentum__target':       dict__momentum__target,
                'energy__pinn':           dict__energy__pinn,
                'energy__oppinn':         dict__energy__oppinn,
                'energy__target':         dict__energy__target,
                'entropy__pinn':          dict__entropy__pinn,
                'entropy__oppinn':        dict__entropy__oppinn,
                'entropy__target':        dict__entropy__target,
            },
            path_base / f"inference__vhs__coeff{vhs_coeff:.2f}_exponent_{vhs_exponent:.2f}__init_type_{init_type}__res_t{res_t:04d}_v{res_v:03d}.pth"
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
        suptitle = f"BKW solution ($C={vhs_coeff:.2f}$)\nTrained for {index} epochs"
    elif init_type=='maxwellian':
        suptitle = f"Maxwellian distribution ($C={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)\nTrained for {index} epochs"
    elif init_type=='bimaxwellian':
        suptitle = f"Sum of two Maxwellian distributions ($C={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)\nTrained for {index} epochs"
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
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=DPI, sharex=True)
    suptitle: str
    if init_type=='bkw':
        suptitle = f"BKW solution ($C={vhs_coeff:.2f}$)\nTrained for {index} epochs"
    elif init_type=='maxwellian':
        suptitle = f"Maxwellian distribution ($C={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)\nTrained for {index} epochs"
    elif init_type=='bimaxwellian':
        suptitle = f"Sum of two Maxwellian distributions ($C={vhs_coeff:.2f}$, $\\gamma={vhs_exponent:.2f}$)\nTrained for {index} epochs"
    fig.suptitle(suptitle, fontsize=SIZE_SUPTITLE)
    axes[0].set_ylabel("Absolute $L^2$ error", fontsize=SIZE_TITLE)
    axes[1].set_ylabel("Relative $L^2$ error", fontsize=SIZE_TITLE)
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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=len(LIST_SEEDS))
    fig.tight_layout()
    return fig, axes


def plot_quantities(index: int, seed: int=0) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    layout = [
        ['density', 'vx'],
        ['density', 'vy'],
        ['energy',  'entropy'],
        ['energy',  'entropy'],
    ]
    fig, axd = plt.subplot_mosaic(layout, figsize=(10, 8), layout="constrained", sharex=True)
    fig.suptitle(f"Physical Quantities\nTrained for {index} epochs", fontsize=SIZE_SUPTITLE)
    xticks = tuple(range(int(max_t)+1))
    
    axd['density'].set_title(r'Mass Density ($\rho$)')
    axd['density'].plot(t, dict__mass__target[(index, seed)], 'k--', linewidth=3*LINEWIDTH, label='Target')
    axd['density'].plot(t, dict__mass__oppinn[(index, seed)], 'r-',  linewidth=LINEWIDTH, label='opPINN')
    axd['density'].plot(t, dict__mass__pinn[  (index, seed)], 'g-',  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['density'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['density'].set_xticks(xticks, xticks)
    
    axd['vx'].set_title(r'Bulk Velocity ($v_x$)')
    axd['vx'].plot(t, dict__momentum__target[(index, seed)][:, 0], 'k--', linewidth=3*LINEWIDTH, label='Target')
    axd['vx'].plot(t, dict__momentum__oppinn[(index, seed)][:, 0], 'r-',  linewidth=LINEWIDTH, label='opPINN')
    axd['vx'].plot(t, dict__momentum__pinn[  (index, seed)][:, 0], 'g-',  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['vx'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['vx'].set_xticks(xticks, xticks)
    
    axd['vy'].set_title(r'Bulk Velocity ($v_y$)')
    axd['vy'].plot(t, dict__momentum__target[(index, seed)][:, 1], 'k--', linewidth=3*LINEWIDTH, label='Target')
    axd['vy'].plot(t, dict__momentum__oppinn[(index, seed)][:, 1], 'r-',  linewidth=LINEWIDTH, label='opPINN')
    axd['vy'].plot(t, dict__momentum__pinn[  (index, seed)][:, 1], 'g-',  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['vy'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['vy'].set_xticks(xticks, xticks)
    
    axd['energy'].set_title(r'Energy Density ($E$)')
    axd['energy'].plot(t, dict__energy__target[(index, seed)], 'k--', linewidth=3*LINEWIDTH, label='Target')
    axd['energy'].plot(t, dict__energy__oppinn[(index, seed)], 'r-',  linewidth=LINEWIDTH, label='opPINN')
    axd['energy'].plot(t, dict__energy__pinn[  (index, seed)], 'g-',  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['energy'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['energy'].set_xticks(xticks, xticks)
    
    axd['entropy'].set_title(r'Entropy Density ($S$)')
    axd['entropy'].plot(t, dict__entropy__target[(index, seed)], 'k--', linewidth=3*LINEWIDTH, label='Target')
    axd['entropy'].plot(t, dict__entropy__oppinn[(index, seed)], 'r-',  linewidth=LINEWIDTH, label='opPINN')
    axd['entropy'].plot(t, dict__entropy__pinn[  (index, seed)], 'g-',  linewidth=LINEWIDTH, label='PINN (ours)')
    axd['entropy'].set_xlabel(r'$t$', fontsize=SIZE_TITLE)
    axd['entropy'].set_xticks(xticks, xticks)
    
    handles, labels = axd['density'].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=3)

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