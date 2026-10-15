from   typing            import Any, Dict, List, Optional, Sequence, Tuple, Union
from   typing_extensions import TypedDict
import torch


__all__: List[str] = ['FPLDeepONetConfig', 'get_default_bimaxwellian_config', 'get_default_bkw_config']


##################################################
class FPLDeepONetConfig(TypedDict, total=False):
    """Configuration dictionary for Separable PI-DeepONet applied to the FPL equation.

    ## Description
    Specifies physical parameters, discretization resolutions, initial condition class bounds,
    neural network architecture, and training hyperparameters for operator learning of the
    2D homogeneous Fokker-Planck-Landau equation.
    """
    # Equation configuration
    dimension:                  int
    """Velocity domain dimension ($d=2$)."""
    init_type:                  str
    """Class of initial conditions ('bimaxwellian' or 'bkw')."""
    max_t:                      float
    """Maximum time horizon $T_{\\text{max}}$."""
    max_v:                      float
    """Velocity truncation bound $V_{\\text{max}}$ defining $[-V_{\\text{max}}, V_{\\text{max}}]^d$."""
    density:                    float
    """Total mass density $\\rho$."""
    vhs_coeff:                  float
    """Variable Hard Sphere (VHS) collision coefficient $\\kappa$."""
    vhs_exponent:               float
    """Variable Hard Sphere (VHS) collision exponent $\\gamma$ (0.0 for Maxwellian interactions)."""

    # Discretization
    sample_t:                   int
    """Collocation resolution along time axis."""
    sample_v:                   int
    """Collocation resolution along each velocity axis."""
    sensor_res:                 int
    """Grid resolution for branch sensor measurements ($m = \\text{sensor\\_res}^2$)."""

    # Initial Condition Class Distributions
    # For Bi-Maxwellian:
    center_dev_min:             float
    """Minimum displacement for Bi-Maxwellian mode centers."""
    center_dev_max:             float
    """Maximum displacement for Bi-Maxwellian mode centers."""
    std_min:                    float
    """Minimum standard deviation (temperature) for Bi-Maxwellian modes."""
    std_max:                    float
    """Maximum standard deviation (temperature) for Bi-Maxwellian modes."""
    weight_min:                 float
    """Minimum mixing weight for the first mode."""
    weight_max:                 float
    """Maximum mixing weight for the first mode."""

    # For BKW:
    bkw_coeff_ext_min:          float
    """Minimum external relaxation coefficient $c_{\\text{ext}}$ for BKW class."""
    bkw_coeff_ext_max:          float
    """Maximum external relaxation coefficient $c_{\\text{ext}}$ for BKW class."""

    # Architecture parameters
    branch_layers:              Sequence[int]
    """Hidden layer width sequence for branch MLP."""
    trunk_layers:               Sequence[int]
    """Hidden layer width sequence for each 1D trunk network."""
    r:                          int
    """Tensor rank approximation parameter."""
    hidden_dim:                 int
    """Latent embedding dimension $p$."""
    activation:                 str
    """Hidden activation function ('tanh', 'relu', etc.)."""
    output_dim:                 int
    """Output solution field dimensionality (1 for scalar distribution $f$)."""

    # Optimization parameters
    learning_rate:              float
    """Initial learning rate for Adam optimizer."""
    gamma:                      float
    """Exponential learning rate decay multiplier."""
    decay_step:                 int
    """Epoch period for learning rate decay."""
    num_epochs:                 int
    """Total training epochs."""
    batch_size:                 int
    """Batch size of initial condition function samples."""
    num_train_samples:          int
    """Number of distinct initial condition realizations in training set."""
    num_test_samples:           int
    """Number of distinct initial condition realizations in test set."""

    # Loss weights
    lambda_ic:                  float
    """Penalty weight for initial condition error."""
    lambda_res:                 float
    """Penalty weight for PDE collision residual error."""
    lambda_cons:                float
    """Penalty weight for macroscopic mass, momentum, and energy conservation."""
    lambda_pos:                 float
    """Penalty weight for non-negativity violation ($f < 0$)."""

    # Runtime
    seed:                       int
    """Random seed for reproducibility."""
    log_step:                   int
    """Logging interval in epochs."""
    device:                     str
    """Target computation device ('cuda' or 'cpu')."""
    result_dir:                 str
    """Output directory path for checkpoints and evaluation results."""


def get_default_bimaxwellian_config() -> FPLDeepONetConfig:
    """Creates default configuration for Bi-Maxwellian initial condition class operator learning.

    ## Description
    Initializes default parameters based on `train/` PINN configurations for 2D Bi-Maxwellian
    relaxation: $V_{\\text{max}} = 5.0$, $T_{\\text{max}} = 3.0$, $\\rho = 0.2$, $\\gamma = 0.0$.

    Returns:
        `FPLDeepONetConfig`: Populated configuration dictionary.
    """
    config: FPLDeepONetConfig = {
        'dimension':          2,
        'init_type':          'bimaxwellian',
        'max_t':              3.0,
        'max_v':              5.0,
        'density':            0.2,
        'vhs_coeff':          0.3125,
        'vhs_exponent':       0.0,

        'sample_t':           16,
        'sample_v':           32,
        'sensor_res':         16,

        'center_dev_min':     0.6,
        'center_dev_max':     1.4,
        'std_min':            0.6,
        'std_max':            1.0,
        'weight_min':         0.4,
        'weight_max':         0.6,

        'branch_layers':      [128, 128, 128, 100],
        'trunk_layers':       [64, 64, 64, 100],
        'r':                  50,
        'hidden_dim':         100,
        'activation':         'tanh',
        'output_dim':         1,

        'learning_rate':      1e-3,
        'gamma':              0.95,
        'decay_step':         200,
        'num_epochs':         2000,
        'batch_size':         16,
        'num_train_samples':  200,
        'num_test_samples':   50,

        'lambda_ic':          20.0,
        'lambda_res':         1.0,
        'lambda_cons':        5.0,
        'lambda_pos':         10.0,

        'seed':               42,
        'log_step':           50,
        'device':             'cuda' if torch.cuda.is_available() else 'cpu',
        'result_dir':         './results/fpl_bimaxwellian',
    }
    return config


def get_default_bkw_config() -> FPLDeepONetConfig:
    """Creates default configuration for BKW initial condition class operator learning.

    ## Description
    Initializes default parameters based on `train/` PINN configurations for 2D BKW relaxation:
    $V_{\\text{max}} = 2\\pi \\approx 6.28318$, $T_{\\text{max}} = 3.0$, $\\rho = 0.2$, $\\gamma = 0.0$,
    with $c_{\\text{ext}} \\in [0.1, 0.5]$.

    Returns:
        `FPLDeepONetConfig`: Populated configuration dictionary.
    """
    config: FPLDeepONetConfig = {
        'dimension':          2,
        'init_type':          'bkw',
        'max_t':              3.0,
        'max_v':              float(2.0 * torch.pi),
        'density':            0.2,
        'vhs_coeff':          0.3125,
        'vhs_exponent':       0.0,

        'sample_t':           16,
        'sample_v':           32,
        'sensor_res':         16,

        'bkw_coeff_ext_min':  0.1,
        'bkw_coeff_ext_max':  0.5,

        'branch_layers':      [128, 128, 128, 100],
        'trunk_layers':       [64, 64, 64, 100],
        'r':                  50,
        'hidden_dim':         100,
        'activation':         'tanh',
        'output_dim':         1,

        'learning_rate':      1e-3,
        'gamma':              0.95,
        'decay_step':         200,
        'num_epochs':         2000,
        'batch_size':         16,
        'num_train_samples':  200,
        'num_test_samples':   50,

        'lambda_ic':          20.0,
        'lambda_res':         1.0,
        'lambda_cons':        5.0,
        'lambda_pos':         10.0,

        'seed':               42,
        'log_step':           50,
        'device':             'cuda' if torch.cuda.is_available() else 'cpu',
        'result_dir':         './results/fpl_bkw',
    }
    return config


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
