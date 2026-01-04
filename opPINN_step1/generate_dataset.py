import  argparse
import  gc
from    typing      import  Callable
from    pathlib     import  Path
from    time        import  time

import  torch

import  sys
sys.path.append('..')

from    models      import  FPL_finite_difference   as  FPL_FD
from    utils       import  maxwellian, bimaxwellian, perturbed_maxwellian


##################################################
parser = argparse.ArgumentParser()
# Arguments with default values
parser.add_argument('--cuda_index', type=int, default=3, help='CUDA device index.')
parser.add_argument('--dimension', type=int, default=2, help='Dimension of the velocity space.')
parser.add_argument('--resolution', type=int, default=64, help='Number of grid points per dimension.')
parser.add_argument('--v_max', type=float, default=5.0, help='Maximum velocity in each dimension.')
# Arguments without default values
parser.add_argument('--gamma', type=float, required=True, help='VHS exponent (gamma) for the FPL equation.')
args = parser.parse_args()


##################################################
torch.set_default_device(f'cuda:{args.cuda_index}')


DIMENSION:  int     = args.dimension
RESOLUTION: int     = args.resolution
V_MAX:      float   = args.v_max
DELTA_V:    float   = 2.0*V_MAX / RESOLUTION
DENSITY:    float   = 1.0
GAMMA:      float   = args.gamma
for APPENDIX in ["train", "validation"]:
    BASE_SIZE:      int = 100 if APPENDIX=="train" else 10
    SIZE__TYPE_1:   int = BASE_SIZE
    SIZE__TYPE_2:   int = BASE_SIZE
    SIZE__TYPE_3:   int = BASE_SIZE


    path_data = Path(f"./{DIMENSION}D__gamma{GAMMA:.1f}")
    path_data.mkdir(exist_ok=True)

    __sup_v = V_MAX - DELTA_V/2
    v_1d = torch.linspace(-__sup_v, __sup_v, RESOLUTION)
    v = torch.cartesian_prod(*[v_1d for _ in range(DIMENSION)])
    fpl_col = FPL_FD(DIMENSION, GAMMA, V_MAX, RESOLUTION)

    
    ##################################################
    # Type 1 Maxwellian distribution
    data_f__type_1:   list[torch.Tensor] = []
    data_Df__type_1:  list[torch.Tensor] = []
    data_Ff__type_1:  list[torch.Tensor] = []
    elapsed_time__type_1: float = time()
    for _ in range(SIZE__TYPE_1):
        random_center = torch.rand(DIMENSION)*2.0 - 1.0 # Uniform in `[-1, 1]^DIMENSION`
        random_sigma  = torch.rand(1)*0.4 + 0.8 # Uniform in `[0.8, 1.2]`
        func_1: Callable[[torch.Tensor], torch.Tensor] = \
            lambda _v: maxwellian(DIMENSION, _v, random_center, random_sigma, DENSITY)
        f = func_1(v)
        Df, Ff = fpl_col.compute_suboperators(func_1)
        data_f__type_1.append(f.cpu())
        data_Df__type_1.append(Df.cpu())
        data_Ff__type_1.append(Ff.cpu())
        del(random_center, random_sigma, func_1, f, Df, Ff)
        gc.collect()
        torch.cuda.empty_cache()
    elapsed_time__type_1 = time() - elapsed_time__type_1
    print(f"Elapsed time (type 1): {int(elapsed_time__type_1)} seconds", flush=True)
    data_f__type_1  = torch.stack(data_f__type_1, dim=0)
    data_Df__type_1 = torch.stack(data_Df__type_1, dim=0)
    data_Ff__type_1 = torch.stack(data_Ff__type_1, dim=0)

    
    # Type 2: Sum of two Maxwellian distributions
    data_f__type_2:   list[torch.Tensor] = []
    data_Df__type_2:  list[torch.Tensor] = []
    data_Ff__type_2:  list[torch.Tensor] = []
    elapsed_time__type_2: float = time()
    for _ in range(SIZE__TYPE_2):
        random_center_1 = torch.rand(DIMENSION)*2.0 - 1.0 # Uniform in `[-1, 1]^DIMENSION`
        random_center_2 = torch.rand(DIMENSION)*2.0 - 1.0 # Uniform in `[-1, 1]^DIMENSION`
        random_sigma_1  = torch.rand(1)*0.4 + 0.8 # Uniform in `[0.8, 1.2]`
        random_sigma_2  = torch.rand(1)*0.4 + 0.8 # Uniform in `[0.8, 1.2]`
        func_2: Callable[[torch.Tensor], torch.Tensor] = \
            lambda _v: bimaxwellian(DIMENSION, _v, random_center_1, random_center_2, random_sigma_1, random_sigma_2, DENSITY)
        f = func_2(v)
        Df, Ff = fpl_col.compute_suboperators(func_2)
        data_f__type_2.append(f.cpu())
        data_Df__type_2.append(Df.cpu())
        data_Ff__type_2.append(Ff.cpu())
        del(f, Df, Ff, func_2)
        gc.collect()
        torch.cuda.empty_cache()
    elapsed_time__type_2 = time() - elapsed_time__type_2
    print(f"Elapsed time (type 2): {int(elapsed_time__type_2)} seconds", flush=True)
    data_f__type_2  = torch.stack(data_f__type_2, dim=0)
    data_Df__type_2 = torch.stack(data_Df__type_2, dim=0)
    data_Ff__type_2 = torch.stack(data_Ff__type_2, dim=0)

    
    # Type 3: Perturbed Maxwellian distribution
    data_f__type_3:   list[torch.Tensor] = []
    data_Df__type_3:  list[torch.Tensor] = []
    data_Ff__type_3:  list[torch.Tensor] = []
    elapsed_time__type_3: float = time()
    for _ in range(SIZE__TYPE_3):
        random_center   = torch.rand(DIMENSION)*2.0 - 1.0   # Uniform in `[-1, 1]^DIMENSION`
        random_sigma    = torch.rand(1)*0.4 + 0.8           # Uniform in `[0.8, 1.2]`
        perturb_0 = torch.rand(1)
        perturb_1 = torch.rand(DIMENSION)
        perturb_2 = torch.rand(DIMENSION, DIMENSION)
        func_3: Callable[[torch.Tensor], torch.Tensor] = \
            lambda _v: perturbed_maxwellian(DIMENSION, _v, random_center, random_sigma, DENSITY, (perturb_0, perturb_1, perturb_2))
        f = func_3(v)
        Df, Ff = fpl_col.compute_suboperators(func_3)
        data_f__type_3.append(f.cpu())
        data_Df__type_3.append(Df.cpu())
        data_Ff__type_3.append(Ff.cpu())
        del(f, Df, Ff, func_3)
        gc.collect()
        torch.cuda.empty_cache()
    elapsed_time__type_3 = time() - elapsed_time__type_3
    print(f"Elapsed time (type 3): {int(elapsed_time__type_3)} seconds", flush=True)
    data_f__type_3  = torch.stack(data_f__type_3, dim=0)
    data_Df__type_3 = torch.stack(data_Df__type_3, dim=0)
    data_Ff__type_3 = torch.stack(data_Ff__type_3, dim=0)

    
    ##################################################
    data_f:     torch.Tensor    = torch.cat([data_f__type_1, data_f__type_2, data_f__type_3], dim=0).cpu()
    data_Df:    torch.Tensor    = torch.cat([data_Df__type_1, data_Df__type_2, data_Df__type_3], dim=0).cpu()
    data_Ff:    torch.Tensor    = torch.cat([data_Ff__type_1, data_Ff__type_2, data_Ff__type_3], dim=0).cpu()
    elapsed_time = elapsed_time__type_1 + elapsed_time__type_2 + elapsed_time__type_3
    
    batched_density = data_f.sum(dim=tuple(range(1, data_f.ndim))) * (DELTA_V**DIMENSION)
    data_f  = data_f  / batched_density.reshape(-1, 1, 1)
    data_Df = data_Df / batched_density.reshape(-1, 1, 1, 1)
    data_Ff = data_Ff / batched_density.reshape(-1, 1, 1)
    # Note that the opearatos `D` and `F` are linear
    # As the following dataset contains data for unit density,
    # for an arbitrarily given density `rho`, one just has to multiply each entry (`f`, `Df`, and `Ff`) by `rho`

    torch.save(
        {'f': data_f, 'Df': data_Df, 'Ff': data_Ff, 'elapsed_time': elapsed_time},
        path_data/f"data__FPL{DIMENSION}D__gamma{GAMMA:.1f}__res{RESOLUTION:03d}__{APPENDIX}.pth"
    )


##################################################
##################################################
# End of file