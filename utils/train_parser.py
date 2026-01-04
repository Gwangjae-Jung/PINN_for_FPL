from    typing          import  Self
import  torch


__all__: list[str] = ['TrainParser']


class TrainParser():
    def __init__(self) -> Self:
        from    argparse    import  ArgumentParser
        
        parser = ArgumentParser()
        group_general = parser.add_argument_group("General configuration")
        group_equation = parser.add_argument_group("Configuration for the Fokker-Planck-Landau equation")
        group_pinn = parser.add_argument_group("Configuration for the PINN model")
        group_train = parser.add_argument_group("Configuration for the training procedure")
        group_ic = parser.add_argument_group("Configuration for the initial condition")
        
        group_general.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
        group_general.add_argument('--cuda_index', type=int, help='The index of the CUDA device to be used for computations.')
        
        group_equation.add_argument('--dimension', type=int, choices=[2,3], help='The dimension of the velocity space (2D or 3D).')
        group_equation.add_argument('--max_t', type=float, help='The maximum time for the simulation.')
        group_equation.add_argument('--max_v', type=float, default=5.0, help='The maximum velocity in each dimension. (Default: 5.0)')
        group_equation.add_argument('--sample_t', type=int, default=10, help='The number of sample points in the time dimension. (Default: 10)')
        group_equation.add_argument('--sample_v', type=int, default=64, help='The number of sample points in each velocity dimension. (Default: 64)')
        group_equation.add_argument('--sample_v_init', type=int, default=64, help='The number of sample points in each velocity dimension for the initial condition. (Default: 100)')
        group_equation.add_argument('--vhs_coeff', type=float, help='The VHS coefficient for the collision operator.)')
        group_equation.add_argument('--vhs_exponent', type=float, help='The VHS exponent for the collision operator.')
        group_equation.add_argument('--density', type=float, default=0.2, help='The density of the initial bimaxwellian distribution. (Default: 0.2)')
        group_equation.add_argument('--init_type', type=str, choices=['bkw', 'maxwellian', 'bimaxwellian'], help="The type of the initial condition. ('bkw', 'maxwellian', or 'bimaxwellian')")
        
        group_pinn.add_argument('--depth', type=int, default=4, help='The depth of the PINN model. (Default: 4)')
        group_pinn.add_argument('--width', type=int, default=100, help='The width of the PINN model. (Default: 100)')
        group_pinn.add_argument('--softplus', type=float, default=1.0, help='The softplus parameter for the activation function. (Default: 1.0)')
        group_pinn.add_argument('--path_D', type=str, default='', help='The file path to the pre-trained neural network for the diffusion coefficient D(v). (Default: empty string)')
        group_pinn.add_argument('--path_F', type=str, default='', help='The file path to the pre-trained neural network for the friction coefficient F(v). (Default: empty string)')
        
        group_train.add_argument('--surrogate', action='store_true', help='Flag indicating whether to use the surrogate operators for training the models.')
        group_train.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate for the optimizer. (Default: 1e-3)')
        group_train.add_argument('--num_epochs', type=int, default=int(1e4), help='The number of epochs for training. (Default: 10000)')
        group_train.add_argument('--num_iterations', type=int, default=20, help='The number of training iterations. (Default: 20)')
        group_train.add_argument('--period_save', type=int, default=1000, help='The period (in epochs) to save the model checkpoint. (Default: 1000)')
        group_train.add_argument('--random_time', action='store_true', help='Sample the time variable randomly.')

        group_ic.add_argument('--init_cond__dev', type=float, default=1.0, help='The deviation of the center of each mode in the initial condition distribution from the origin. (Default: 1.0)')
        group_ic.add_argument('--init_cond__std', type=float, default=0.8, help='The standard deviation of the initial condition distribution. (Default: 0.8)')
        group_ic.add_argument('--bkw_coeff_ext', type=float, default=0.5, help='A parameter of the BKW solution, which is a known analytic solution to the homogeneous Fokker-Planck-Landau equation.')
        
        self.__args = parser.parse_args()
        
        self.__device = torch.device(f'cuda:{self.cuda_index}')
        
        self.__init_cond__centers = torch.tensor(
            [
                [*(-self.__args.init_cond__dev for _ in range(self.dimension-1)), 0.0],
                [*(+self.__args.init_cond__dev for _ in range(self.dimension-1)), 0.0],
                [0.0, *(-self.__args.init_cond__dev for _ in range(self.dimension-1))],
                [0.0, *(+self.__args.init_cond__dev for _ in range(self.dimension-1))],
            ],
            device=self.device,
        )
        self.__init_cond__std   = torch.tensor([self.__args.init_cond__std], device=self.device)
        return
    

    @property
    def args(self) -> object:           return self.__args
    
    @property
    def seed(self) -> int:              return self.__args.seed
    @property
    def cuda_index(self) -> int:        return self.__args.cuda_index
    @property
    def device(self) -> torch.device:   return self.__device
    
    @property
    def dimension(self) -> int:         return self.__args.dimension
    @property
    def max_t(self) -> float:           return self.__args.max_t
    @property
    def max_v(self) -> float:           return self.__args.max_v
    @property
    def sample_t(self) -> int:          return self.__args.sample_t
    @property
    def sample_v(self) -> int:          return self.__args.sample_v
    @property
    def sample_v_init(self) -> int:     return self.__args.sample_v_init
    @property
    def vhs_coeff(self) -> float:       return self.__args.vhs_coeff
    @property
    def vhs_exponent(self) -> float:    return self.__args.vhs_exponent
    @property
    def density(self) -> float:         return self.__args.density
    @property
    def init_type(self) -> str:         return self.__args.init_type
    
    @property
    def depth(self) -> int:             return self.__args.depth
    @property
    def width(self) -> int:             return self.__args.width
    @property
    def softplus(self) -> float:        return self.__args.softplus
    @property
    def path_D(self) -> str:            return self.__args.path_D
    @property
    def path_F(self) -> str:            return self.__args.path_F
    
    @property
    def surrogate(self) -> bool:        return self.__args.surrogate
    @property
    def learning_rate(self) -> float:   return self.__args.learning_rate
    @property
    def num_epochs(self) -> int:        return self.__args.num_epochs
    @property
    def num_iterations(self) -> int:    return self.__args.num_iterations
    @property
    def period_save(self) -> int:       return self.__args.period_save
    @property
    def random_time(self) -> bool:      return self.__args.random_time
    
    @property
    def init_cond__centers(self) -> torch.Tensor:   return self.__init_cond__centers
    @property
    def init_cond__std(self) -> torch.Tensor:       return self.__init_cond__std
    @property
    def bkw_coeff_ext(self) -> float:               return self.__args.bkw_coeff_ext
    
    
    def summary(self) -> None:
        from    torch.cuda      import  get_device_name
        desc__time_domain = "Random (uniform) sampling" if self.random_time else "Fixed sampling"
        print("="*50, flush=True)
        print(f"Training {'op' if self.surrogate else ''}PINN for the FPL equation with the following configuration:", flush=True)
        print(f"* Random seed:          {self.seed}", flush=True)
        print(f"* Device:               {get_device_name(self.cuda_index)} ({self.cuda_index})", flush=True)
        print(f"* Dimension:            {self.dimension}", flush=True)
        print(f"* Time domain:          [0.0, {self.max_t:.1f}] with {self.sample_t} samples ({desc__time_domain})", flush=True)
        print(f"* Velocity domain:      [-{self.max_v:.1f}, {self.max_v:.1f}] with {self.sample_v} samples per dimension ({self.sample_v_init} for the initial condition)", flush=True)
        print(f"* VHS coefficient:      {self.vhs_coeff:.2f}", flush=True)
        print(f"* VHS exponent:         {self.vhs_exponent:.2f}", flush=True)
        print(f"* Initial condition:    {self.init_type}", flush=True)
        print(f"* Model:                depth {self.depth}, width {self.width}, softplus {self.softplus:.1f}", flush=True)
        if self.surrogate:
            print(f"* Pre-trained operators", flush=True)
            print(f"  - D: {self.path_D}", flush=True)
            print(f"  - F: {self.path_F}", flush=True)
        print(f"* Training:             {self.num_epochs} epochs, {self.num_iterations} iterations per epoch, learning rate {self.learning_rate:.2e}", flush=True)
        print("="*50, flush=True)
        return


##################################################
##################################################
# End of file