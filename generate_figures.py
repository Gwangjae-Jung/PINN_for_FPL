##################################################
# Import libraries
##################################################
import  argparse
import  warnings
from    pathlib                 import  Path
from    itertools               import  product
import  torch
from    deep_numerical.utils    import  space_grid, relative_error


import  matplotlib.pyplot       as      plt

from    config.base_config__2d      import  *
from    config                      import  FurtherConfig


warnings.filterwarnings("ignore", category=SyntaxWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, help='The value of gamma in the Fokker-Planck-Landau equation.')
parser.add_argument('--sample_t', type=str, help='The mod of sampling the time variable.')
parser.add_argument('--res_t', type=int, help='The resolution in the time variable.')
parser.add_argument('--res_v', type=int, help='The resolution in the velocity variable.')
parser.add_argument('--init_type', type=str, help='The initial condition.')
args = parser.parse_args()
gamma:      float   = args.gamma
sample_t:   str     = args.sample_t
res_t:      int     = args.res_t
res_v:      int     = args.res_v
init_type:  str     = args.init_type
DEVICE = torch.device('cpu')
torch.set_default_device(DEVICE)


LIST_GAMMA      = [-3.0+k for k in range(5)]
LIST_INDEX      = [1000*k for k in range(1, 6)]
LIST_SAMPLE_T   = ['fixed_t', 'random_t']
LIST_RES_T      = [601]
LIST_RES_V      = [64]
LIST_INIT_TYPE  = ['bimaxwellian', 'bkw']


CENTER_1    = torch.tensor([*(-INIT_COND__DEV for _ in range(DIMENSION-1)), 0.0])
CENTER_2    = torch.tensor([*(+INIT_COND__DEV for _ in range(DIMENSION-1)), 0.0])
STD_1       = torch.tensor([INIT_COND__STD])
STD_2       = torch.tensor([INIT_COND__STD])


LIST_SEEDS      = [0, 1, 2, 3, 4]
LIST_COLORS     = ['red', 'orange', 'green', 'blue', 'purple']
SIZE_SUPTITLE   = 20
SIZE_TITLE      = 16
LINEWIDTH       = 3
DPI             = 1000


##################################################
base_shape:     tuple[int, ...] = tuple((res_t, *(res_v for _ in range(DIMENSION))))
config:         FurtherConfig   = FurtherConfig(f"./config/config__{DIMENSION}d__gamma{gamma:.1f}__{init_type}.yaml")

path_base:      Path    = Path().cwd() / f"result__{sample_t}"
path_images:    Path    = Path().cwd() / "images" / sample_t
assert path_base.exists(), f"The path [{str(path_base)}] does not exist."
if path_images.exists() is False:   path_images.mkdir(parents=True, exist_ok=True)

vhs_coeff:      float   = config.vhs_coeff
vhs_exponent:   float   = config.vhs_exponent

max_t:          float   = MAX_T__DICT[init_type]
max_v:          float   = MAX_V__DICT[init_type]

grid_t:         torch.Tensor    = torch.linspace(0, max_t, res_t)


##################################################
inference_data:     dict[str, dict[tuple[int, int], torch.Tensor]]  = \
    torch.load(
        path_base / \
        f"inference__vhs__coeff{vhs_coeff:.2f}_exponent_{vhs_exponent:.2f}__init_type_{init_type}__res_t{res_t:04d}_v{res_v:03d}.pth",
        weights_only = False,
    )

dict__abs_error__pinn         = inference_data['abs_error__pinn']
dict__rel_error__pinn         = inference_data['rel_error__pinn']
dict__abs_error__oppinn       = inference_data['abs_error__oppinn']
dict__rel_error__oppinn       = inference_data['rel_error__oppinn']
dict__mass__pinn              = inference_data['mass__pinn']
dict__mass__oppinn            = inference_data['mass__oppinn']
dict__mass__target            = inference_data['mass__target']
dict__momentum__pinn          = inference_data['momentum__pinn']
dict__momentum__oppinn        = inference_data['momentum__oppinn']
dict__momentum__target        = inference_data['momentum__target']
dict__energy__pinn            = inference_data['energy__pinn']
dict__energy__oppinn          = inference_data['energy__oppinn']
dict__energy__target          = inference_data['energy__target']
dict__entropy__pinn           = inference_data['entropy__pinn']
dict__entropy__oppinn         = inference_data['entropy__oppinn']
dict__entropy__target         = inference_data['entropy__target']


path_solution = path_base.parent / "solutions" / \
    f"vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__max_t{max_t:.2f}__res_t{res_t:04d}__max_v{max_v:.2f}__res_v{res_v:03d}.pth"
print(f"Loading the solution from [{str(path_solution)}]...")
try:
    target = torch.load(path_solution, weights_only=False)
except FileNotFoundError:
    warnings.warn("File is not found.", UserWarning)


##################################################
def show_snapshots(index: int) -> None:
    ...


def plot_error(index: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=DPI, sharex=True)
    title: str
    if init_type=='bkw':
        title = f"BKW solution ($C={vhs_coeff:.2f}$)\nTrained for {index} epochs"
    elif init_type=='maxwellian':
        title = f"Maxwellian distribution ($C={vhs_coeff:.2f}$, $\gamma={vhs_exponent:.2f}$)\nTrained for {index} epochs"
    elif init_type=='bimaxwellian':
        title = f"Sum of two Maxwellian distributions ($C={vhs_coeff:.2f}$, $\gamma={vhs_exponent:.2f}$)\nTrained for {index} epochs"
    fig.suptitle(title, fontsize=SIZE_SUPTITLE)
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
            grid_t.cpu(), abs_err__pinn,
            linewidth=LINEWIDTH, color=_c,
            label=f"PINN (seed: {seed})",
        )
        axes[0].plot(
            grid_t.cpu(), abs_err__oppinn,
            linewidth=LINEWIDTH, color=_c, linestyle=':',
            label=f"opPINN (seed: {seed})",
        )
    # Relative errors
    for seed, _c, rel_err__pinn, rel_err__oppinn in zip(LIST_SEEDS, LIST_COLORS, list_rel_errors__pinn, list_rel_errors__oppinn):
        axes[1].plot(
            grid_t.cpu(), rel_err__pinn,
            linewidth=LINEWIDTH, color=_c,
            label=f"PINN (seed: {seed})",
        )
        axes[1].plot(
            grid_t.cpu(), rel_err__oppinn,
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
    fig.savefig(
        path_images / \
        f"{DIMENSION}D__vhs__coeff{vhs_coeff:.2f}_exponent_{vhs_exponent:.2f}__init_type_{init_type}__index_{index}.pdf", bbox_inches="tight",
        dpi=DPI,
    )
    
    return None


def plot_quantities(index: int) -> None:
    ...


##################################################
for index in LIST_INDEX:
    show_snapshots(index)
    plot_error(index)
    plot_quantities(index)


##################################################
# End of file