import  argparse
import  subprocess
from    config.further_config   import  FurtherConfig


DEFAULT_ARG__CUDA_INDEX = -1
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, help='CUDA device index.', default=DEFAULT_ARG__CUDA_INDEX)
parser.add_argument('--dim', type=int, help='Dimension of the problem (2 or 3).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
parser.add_argument('--path_config', type=str, help='Path to configuration file.')
parser.add_argument('--surrogate', action='store_true', help='Use surrogate operators (pretrained operators).')
parser.add_argument('--random_time', action='store_true', help='Sample the time variable randomly.')
args = parser.parse_args()

DIMENSION       = args.dim
if DIMENSION==2:
    from    config.base_config__2d  import  *
elif DIMENSION==3:
    from    config.base_config__3d  import  *
else:
    raise ValueError(f"Unsupported dimension: {DIMENSION}")
SEED            = args.seed
config = FurtherConfig(args.path_config)

CUDA_INDEX      = config.cuda_index if args.cuda_index==DEFAULT_ARG__CUDA_INDEX else args.cuda_index
VHS_COEFF       = config.vhs_coeff
VHS_EXPONENT    = config.vhs_exponent
INIT_TYPE       = config.init_type
PATH_D          = config.path_D
PATH_F          = config.path_F


# Load the configuration up to the initial condition
MAX_T   = MAX_T__DICT[INIT_TYPE]
MAX_V   = MAX_V__DICT[INIT_TYPE]
DENSITY = DENSITY__DICT[INIT_TYPE]


print(f"Start training with seed {SEED}...")
python_args = [
    "python", "train.py",
    
    "--seed",           str(SEED),
    "--cuda_index",     str(CUDA_INDEX),
    
    "--dimension",      str(DIMENSION),
    "--max_t",          str(MAX_T),
    "--max_v",          str(MAX_V),
    "--sample_t",       str(SAMPLE_T),
    "--sample_v",       str(SAMPLE_V),
    "--sample_v_init",  str(SAMPLE_V_INIT),
    "--vhs_coeff",      str(VHS_COEFF),
    "--vhs_exponent",   str(VHS_EXPONENT),
    "--density",        str(DENSITY),
    "--init_type",      INIT_TYPE,
    
    "--depth",          str(DEPTH),
    "--width",          str(WIDTH),
    "--softplus",       str(SOFTPLUS),
    "--path_D",         str(PATH_D),
    "--path_F",         str(PATH_F),
    
    "--learning_rate",  str(LEARNING_RATE),
    "--num_epochs",     str(NUM_EPOCHS),
    "--num_iterations", str(NUM_ITERATIONS),
    
    "--init_cond__dev", str(INIT_COND__DEV),
    "--init_cond__std", str(INIT_COND__STD),
    "--bkw_coeff_ext",  str(BKW_COEFF_EXT),
]
if args.surrogate:      python_args.append("--surrogate")
if args.random_time:    python_args.append("--random_time")
subprocess.run(python_args)
print("Training completed.\n")