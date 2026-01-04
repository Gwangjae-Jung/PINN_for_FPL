from    typing      import  Self, Optional
import  torch


__all__: list[str] = ['GridGenerator',]


def space_grid_1d(res_x: int, max_x: float, device: torch.device) -> torch.Tensor:
    dx = (2*max_x)/res_x
    a = max_x - dx/2
    return torch.linspace(-a, a, res_x, device=device)

    
class GridGenerator():
    def __init__(
            self,
            dim:        int,
            num_t:      int,
            max_t:      float,
            num_v:      int,
            max_v:      float,
            num_v_init: Optional[int] = None,
            device:     Optional[torch.device] = None,
        ) -> Self:
        if device is None:  device = torch.get_default_device()
        self.__dim          = dim
        self.__num_t        = num_t
        self.__max_t        = max_t
        self.__num_v        = num_v
        self.__max_v        = max_v
        self.__num_v_init   = num_v_init if num_v_init is not None else num_v
        self.__device       = device
        
        self.__t        = torch.linspace(0, self.__max_t, self.__num_t, device=device)
        self.__v_1d     = space_grid_1d(num_v, max_v, device=device)
        self.__v0_1d    = space_grid_1d(self.__num_v_init, max_v, device=device)
        self.__tv       = torch.cartesian_prod(self.__t, *(self.__v_1d for _ in range(self.__dim)))
        self.__t0v      = torch.cartesian_prod(torch.tensor([0.0], device=device), *(self.__v0_1d for _ in range(self.__dim)))
        
        return
    
    
    @property
    def dim(self) -> int:       return self.__dim
    @property
    def delta_t(self) -> float: return self.__max_t/(self.__num_t-1)
    @property
    def delta_v(self) -> float: return (2*self.__max_v)/self.__num_v
    @property
    def dv(self) -> float:      return self.delta_v**self.__dim
    @property
    def num_t(self) -> int:     return self.__num_t
    @property
    def num_v(self) -> int:     return self.__num_v
    @property
    def tv(self) -> torch.Tensor:   return self.__tv
    @property
    def t0v(self) -> torch.Tensor:  return self.__t0v
    
    
    def sample_tv(self, is_time_fixed: bool) -> torch.Tensor:
        grid_t: torch.Tensor
        if is_time_fixed:
            grid_t = self.__t
        else:
            grid_t = self.__max_t * torch.sort(torch.rand(self.__num_t, device=self.__device))[0]
        return torch.cartesian_prod(grid_t, *(self.__v_1d for _ in range(self.__dim)))


##################################################
##################################################
# End of file