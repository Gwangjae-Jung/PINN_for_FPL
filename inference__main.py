##################################################
# Import libraries
##################################################
import  argparse
import  warnings
from    pathlib                 import  Path
from    itertools               import  product

import  torch
from    deep_numerical.utils    import  space_grid, relative_error

from    models                  import  *
from    utils                   import  absolute_error, relative_error, maxwellian, bimaxwellian, bkw
from    utils                   import  compute_mass_density, compute_bulk_velocity, compute_energy_density, compute_entropy_density


from    config.base_config__2d      import  *
from    config                      import  FurtherConfig


warnings.filterwarnings("ignore", category=SyntaxWarning)


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
path_images:    Path    = Path().cwd() / "images" / sample_t
assert path_base.exists(), f"The path [{str(path_base)}] does not exist."
if path_images.exists() is False:   path_images.mkdir(parents=True, exist_ok=True)

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


##################################################
def validate_models(indices: list[int] = LIST_INDEX) -> None:
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
validate_models()


##################################################
# End of file