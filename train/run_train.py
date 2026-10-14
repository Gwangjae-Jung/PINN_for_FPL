# Import necessary libraries and modules.
import argparse
from   pathlib import Path
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from   config.further_config import FurtherConfig


##################################################
_DEFAULT_ARG__CUDA_INDEX: int = -1


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running the training workflow.

    ## Description
    Parses configuration path, problem dimension, seed, CUDA device index,
    and optional training flags.

    Arguments:
        This function does not take arguments.

    Returns:
        `argparse.Namespace`: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_index',  type=int, default=_DEFAULT_ARG__CUDA_INDEX, help='CUDA device index.')
    parser.add_argument('--dim',         type=int, help='Dimension of the problem (2 or 3).')
    parser.add_argument('--seed',        type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--path_config', type=str, help='Path to configuration file.')
    parser.add_argument('--surrogate',   action='store_true', help='Use surrogate operators (pretrained operators).')
    parser.add_argument('--random_time', action='store_true', help='Sample the time variable randomly.')
    return parser.parse_args()


def _main() -> None:
    """Execute the training script with parsed configurations.

    ## Description
    Loads dimension-specific and experiment-specific configurations,
    assembles command-line arguments, and invokes `train.py` as a subprocess.

    Arguments:
        This function does not take arguments.

    Returns:
        This function does not return values.
    """
    args = _parse_args()

    dimension: int = args.dim
    if dimension == 2:
        from config.base_config__2d import (
            BKW_COEFF_EXT,
            DENSITY__DICT,
            DEPTH,
            INIT_COND__DEV,
            INIT_COND__STD,
            LEARNING_RATE,
            MAX_T__DICT,
            MAX_V__DICT,
            NUM_EPOCHS,
            NUM_ITERATIONS,
            SAMPLE_T,
            SAMPLE_V,
            SAMPLE_V_INIT,
            SOFTPLUS,
            WIDTH,
        )
    elif dimension == 3:
        from config.base_config__3d import (
            BKW_COEFF_EXT,
            DENSITY__DICT,
            DEPTH,
            INIT_COND__DEV,
            INIT_COND__STD,
            LEARNING_RATE,
            MAX_T__DICT,
            MAX_V__DICT,
            NUM_EPOCHS,
            NUM_ITERATIONS,
            SAMPLE_T,
            SAMPLE_V,
            SAMPLE_V_INIT,
            SOFTPLUS,
            WIDTH,
        )
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    seed:   int           = args.seed
    config: FurtherConfig = FurtherConfig(args.path_config)

    cuda_index:   int   = config.cuda_index if args.cuda_index == _DEFAULT_ARG__CUDA_INDEX else args.cuda_index
    vhs_coeff:    float = config.vhs_coeff
    vhs_exponent: float = config.vhs_exponent
    init_type:    str   = config.init_type
    path_d:       str   = config.path_D
    path_f:       str   = config.path_F

    # Load the configuration up to the initial condition
    max_t:   float = MAX_T__DICT[init_type]
    max_v:   float = MAX_V__DICT[init_type]
    density: float = DENSITY__DICT[init_type]

    print(f"Start training with seed {seed}...")
    train_script_path = str(Path(__file__).resolve().parent / "train.py")
    python_args: list[str] = [
        sys.executable,     train_script_path,

        "--seed",           str(seed),
        "--cuda_index",     str(cuda_index),

        "--dimension",      str(dimension),
        "--max_t",          str(max_t),
        "--max_v",          str(max_v),
        "--sample_t",       str(SAMPLE_T),
        "--sample_v",       str(SAMPLE_V),
        "--sample_v_init",  str(SAMPLE_V_INIT),
        "--vhs_coeff",      str(vhs_coeff),
        "--vhs_exponent",   str(vhs_exponent),
        "--density",        str(density),
        "--init_type",      init_type,

        "--depth",          str(DEPTH),
        "--width",          str(WIDTH),
        "--softplus",       str(SOFTPLUS),
        "--path_D",         str(path_d),
        "--path_F",         str(path_f),

        "--learning_rate",  str(LEARNING_RATE),
        "--num_epochs",     str(NUM_EPOCHS),
        "--num_iterations", str(NUM_ITERATIONS),

        "--init_cond__dev", str(INIT_COND__DEV),
        "--init_cond__std", str(INIT_COND__STD),
        "--bkw_coeff_ext",  str(BKW_COEFF_EXT),
    ]
    if args.surrogate:
        python_args.append("--surrogate")
    if args.random_time:
        python_args.append("--random_time")

    subprocess.run(python_args)
    print("Training completed.\n")
    return


##################################################
if __name__ == '__main__':
    _main()