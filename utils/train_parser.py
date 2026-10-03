from    typing          import  Self
import  torch


__all__: list[str] = ['TrainParser']


##################################################
##################################################
class TrainParser():
    """Command-line argument parser and configuration container for PINN training.

    ## Description
    Parses all command-line arguments required to run a PINN training experiment for
    the Fokker-Planck-Landau equation. Arguments are grouped into five categories:
    general, equation, PINN model, training procedure, and initial condition.
    After parsing, computed attributes such as the computing device and initial
    condition center tensors are pre-built for convenience.

    ## Arguments
    None. All configuration is read from `sys.argv` via `argparse`.

    ## Returns
    `None`: None.
    """
    def __init__(self) -> Self:
        from    argparse    import  ArgumentParser

        parser = ArgumentParser()
        group_general  = parser.add_argument_group("General configuration")
        group_equation = parser.add_argument_group("Configuration for the Fokker-Planck-Landau equation")
        group_pinn     = parser.add_argument_group("Configuration for the PINN model")
        group_train    = parser.add_argument_group("Configuration for the training procedure")
        group_ic       = parser.add_argument_group("Configuration for the initial condition")

        group_general.add_argument('--seed',        type=int, default=0,
            help='Random seed for reproducibility.')
        group_general.add_argument('--cuda_index',  type=int,
            help='The index of the CUDA device to be used for computations.')

        group_equation.add_argument('--dimension',      type=int,   choices=[2, 3],
            help='The dimension of the velocity space (2D or 3D).')
        group_equation.add_argument('--max_t',          type=float,
            help='The maximum time for the simulation.')
        group_equation.add_argument('--max_v',          type=float, default=5.0,
            help='The maximum velocity in each dimension. (Default: 5.0)')
        group_equation.add_argument('--sample_t',       type=int,   default=10,
            help='The number of sample points in the time dimension. (Default: 10)')
        group_equation.add_argument('--sample_v',       type=int,   default=64,
            help='The number of sample points in each velocity dimension. (Default: 64)')
        group_equation.add_argument('--sample_v_init',  type=int,   default=64,
            help='The number of velocity sample points for the initial condition. (Default: 64)')
        group_equation.add_argument('--vhs_coeff',      type=float,
            help='The VHS coefficient for the collision operator.')
        group_equation.add_argument('--vhs_exponent',   type=float,
            help='The VHS exponent for the collision operator.')
        group_equation.add_argument('--density',        type=float, default=0.2,
            help='The density of the initial distribution. (Default: 0.2)')
        group_equation.add_argument('--init_type',      type=str,
            choices=['bkw', 'maxwellian', 'bimaxwellian'],
            help="The type of the initial condition. ('bkw', 'maxwellian', or 'bimaxwellian')")

        group_pinn.add_argument('--depth',      type=int,   default=4,
            help='The depth of the PINN model. (Default: 4)')
        group_pinn.add_argument('--width',      type=int,   default=100,
            help='The width of the PINN model. (Default: 100)')
        group_pinn.add_argument('--softplus',   type=float, default=1.0,
            help='The softplus beta parameter for the output activation. (Default: 1.0)')
        group_pinn.add_argument('--path_D',     type=str,   default='',
            help='Path to the pre-trained neural network for the diffusion operator D(v).')
        group_pinn.add_argument('--path_F',     type=str,   default='',
            help='Path to the pre-trained neural network for the friction operator F(v).')

        group_train.add_argument('--surrogate',       action='store_true',
            help='Use surrogate operators instead of the exact spectral method.')
        group_train.add_argument('--learning_rate',   type=float, default=1e-3,
            help='The learning rate for the optimizer. (Default: 1e-3)')
        group_train.add_argument('--num_epochs',      type=int,   default=int(1e4),
            help='The number of training epochs. (Default: 10000)')
        group_train.add_argument('--num_iterations',  type=int,   default=20,
            help='The number of training iterations per epoch. (Default: 20)')
        group_train.add_argument('--period_save',     type=int,   default=1000,
            help='The epoch period to save a model checkpoint. (Default: 1000)')
        group_train.add_argument('--random_time',     action='store_true',
            help='Sample the time variable randomly at each iteration.')

        group_ic.add_argument('--init_cond__dev',   type=float, default=1.0,
            help='Deviation of each biMaxwellian mode centre from the origin. (Default: 1.0)')
        group_ic.add_argument('--init_cond__std',   type=float, default=0.8,
            help='Standard deviation of the initial condition distribution. (Default: 0.8)')
        group_ic.add_argument('--bkw_coeff_ext',    type=float, default=0.5,
            help='External relaxation coefficient for the BKW analytic solution.')

        self.__args = parser.parse_args()

        # Pre-build the device handle
        self.__device: torch.device = torch.device(f'cuda:{self.cuda_index}')

        # Pre-build initial condition centre tensors (used as targets during training)
        self.__init_cond__centers: torch.Tensor = torch.tensor(
            [
                [*(-self.__args.init_cond__dev for _ in range(self.dimension-1)), 0.0],
                [*(+self.__args.init_cond__dev for _ in range(self.dimension-1)), 0.0],
                [0.0, *(-self.__args.init_cond__dev for _ in range(self.dimension-1))],
                [0.0, *(+self.__args.init_cond__dev for _ in range(self.dimension-1))],
            ],
            device=self.device,
        )
        self.__init_cond__std: torch.Tensor = torch.tensor(
            [self.__args.init_cond__std],
            device=self.device,
        )
        return


    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    @property
    def args(self) -> object:               return self.__args

    @property
    def seed(self) -> int:                  return self.__args.seed
    @property
    def cuda_index(self) -> int:            return self.__args.cuda_index
    @property
    def device(self) -> torch.device:       return self.__device

    # ------------------------------------------------------------------
    # Equation
    # ------------------------------------------------------------------
    @property
    def dimension(self) -> int:             return self.__args.dimension
    @property
    def max_t(self) -> float:               return self.__args.max_t
    @property
    def max_v(self) -> float:               return self.__args.max_v
    @property
    def sample_t(self) -> int:              return self.__args.sample_t
    @property
    def sample_v(self) -> int:              return self.__args.sample_v
    @property
    def sample_v_init(self) -> int:         return self.__args.sample_v_init
    @property
    def vhs_coeff(self) -> float:           return self.__args.vhs_coeff
    @property
    def vhs_exponent(self) -> float:        return self.__args.vhs_exponent
    @property
    def density(self) -> float:             return self.__args.density
    @property
    def init_type(self) -> str:             return self.__args.init_type

    # ------------------------------------------------------------------
    # PINN model
    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:                 return self.__args.depth
    @property
    def width(self) -> int:                 return self.__args.width
    @property
    def softplus(self) -> float:            return self.__args.softplus
    @property
    def path_D(self) -> str:                return self.__args.path_D
    @property
    def path_F(self) -> str:                return self.__args.path_F

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @property
    def surrogate(self) -> bool:            return self.__args.surrogate
    @property
    def learning_rate(self) -> float:       return self.__args.learning_rate
    @property
    def num_epochs(self) -> int:            return self.__args.num_epochs
    @property
    def num_iterations(self) -> int:        return self.__args.num_iterations
    @property
    def period_save(self) -> int:           return self.__args.period_save
    @property
    def random_time(self) -> bool:          return self.__args.random_time

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    @property
    def init_cond__centers(self) -> torch.Tensor:   return self.__init_cond__centers
    @property
    def init_cond__std(self) -> torch.Tensor:       return self.__init_cond__std
    @property
    def bkw_coeff_ext(self) -> float:               return self.__args.bkw_coeff_ext


    def summary(self) -> None:
        """Print a human-readable summary of the current training configuration.

        ## Description
        Outputs all key hyperparameters to stdout, including device name, velocity and
        time domain bounds, VHS model parameters, PINN architecture, and training
        schedule.

        ## Arguments
        None.

        ## Returns
        `None`: None.
        """
        from    torch.cuda      import  get_device_name
        desc__time_domain: str = "Random (uniform) sampling" if self.random_time else "Fixed sampling"
        print("="*50, flush=True)
        print(f"Training {'op' if self.surrogate else ''}PINN for the FPL equation:", flush=True)
        print(f"* Random seed:          {self.seed}", flush=True)
        print(f"* Device:               {get_device_name(self.cuda_index)} ({self.cuda_index})", flush=True)
        print(f"* Dimension:            {self.dimension}", flush=True)
        print(f"* Time domain:          [0.0, {self.max_t:.1f}] with {self.sample_t} samples ({desc__time_domain})", flush=True)
        print(f"* Velocity domain:      [-{self.max_v:.1f}, {self.max_v:.1f}] with {self.sample_v} samples/dim "
              f"({self.sample_v_init} for the initial condition)", flush=True)
        print(f"* VHS coefficient:      {self.vhs_coeff:.2f}", flush=True)
        print(f"* VHS exponent:         {self.vhs_exponent:.2f}", flush=True)
        print(f"* Initial condition:    {self.init_type}", flush=True)
        print(f"* Model:                depth {self.depth}, width {self.width}, softplus {self.softplus:.1f}", flush=True)
        if self.surrogate:
            print(f"* Pre-trained operators", flush=True)
            print(f"  - D: {self.path_D}", flush=True)
            print(f"  - F: {self.path_F}", flush=True)
        print(f"* Training:             {self.num_epochs} epochs, {self.num_iterations} iters/epoch, "
              f"lr {self.learning_rate:.2e}", flush=True)
        print("="*50, flush=True)
        return


##################################################
##################################################
# End of file