# Import necessary libraries and modules.
import argparse
from   itertools import product
from   pathlib   import Path
import subprocess
import sys
import warnings

from   tqdm import tqdm

warnings.filterwarnings("ignore")


##################################################
_LIST_GAMMA:     list[float] = [-3.0 + k for k in range(5)]
_LIST_SAMPLE_T:  list[str]   = ['fixed_t']
_LIST_RES_T:     list[int]   = [601]
_LIST_RES_V:     list[int]   = [64]
_LIST_INIT_TYPE: list[str]   = ['bimaxwellian', 'maxwellian', 'bkw']


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running inference batch jobs.

    ## Description
    Parses the CUDA device index used to run inference scripts.

    Arguments:
        This function does not take arguments.

    Returns:
        `argparse.Namespace`: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_index', type=int, default=3, help='The index of the GPU to use.')
    return parser.parse_args()


def _main() -> None:
    """Run batch inference jobs for different initial conditions and gammas.

    ## Description
    Iterates through all combinations of gammas, sampling modes, resolutions,
    and initial conditions, executing `inference.py` for each valid configuration.

    Arguments:
        This function does not take arguments.

    Returns:
        This function does not return values.
    """
    args = _parse_args()
    cuda_index: int = args.cuda_index

    inference_script_path = str(Path(__file__).resolve().parent / "inference.py")

    for gamma, sample_t, res_t, res_v, init_type in tqdm(
        product(
            _LIST_GAMMA,
            _LIST_SAMPLE_T,
            _LIST_RES_T,
            _LIST_RES_V,
            _LIST_INIT_TYPE,
        )
    ):
        if init_type in ['bkw', 'maxwellian'] and gamma != 0.0:
            continue
        subprocess.run([
            sys.executable, inference_script_path,
            "--cuda_index", str(cuda_index),
            "--gamma",      str(gamma),
            "--sample_t",   str(sample_t),
            "--res_t",      str(res_t),
            "--res_v",      str(res_v),
            "--init_type",  str(init_type),
        ])

    print("Inference jobs completed.")
    return


##################################################
if __name__ == '__main__':
    _main()