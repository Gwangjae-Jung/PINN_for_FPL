from    typing      import  Self, Optional
import  torch
from    torch       import  nn


__all__: list[str] = ['PINN_FPL']


class PINN_FPL(nn.Module):
    def __init__(
            self,
            dimension:  int,
            depth:      int,
            width:      int,            
            softplus:   float = 1.0,
            
            dtype:      Optional[torch.dtype]   = None,
            device:     Optional[torch.device]  = None,
        ) -> Self:
        super().__init__()
        
        if dtype is not None:   dtype   = torch.get_default_dtype()
        if device is not None:  device  = torch.get_default_device()
        
        self.network = nn.Sequential()
        self.network.append(nn.Linear(1+dimension, width, dtype=dtype))
        self.network.append(nn.Tanh())
        for _ in range(depth-1):
            self.network.append(nn.Linear(width, width))
            self.network.append(nn.Tanh())
        self.network.append(nn.Linear(width, 1, dtype=dtype))
        self.network.append(nn.Softplus(beta=softplus))
        self.network.to(device)
        
        return

    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `points` (`torch.Tensor`): A tensor of shape `(num_points, in_channels)`. If `points` saves points on a grid, then the points should be aligned with respect to the `ij`-indexing style.
        """
        return self.network(points)


##################################################
##################################################
# End of file