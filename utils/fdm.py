from    typing      import  Self, Optional
import  torch
from    torch.nn    import  functional  as  F


__all__: list[str] = ['FiniteDifferenceMethod']


##################################################
##################################################
class FiniteDifferenceMethod():
    def __init__(
            self,
            dim:    int,
            dx:     float,
            order:  int                     = 2,
            device: Optional[torch.device]  = None,
        ) -> Self:
        if not (isinstance(order, int) and order in [2, 4]):
            raise ValueError(f"'order' should be in [2, 4], but got order={order}.")
        device = device if device is not None else torch.get_default_device()
        self.__dim:             int                         = dim
        self.__dx:              float                       = dx
        self.__device:          torch.device                = device
        self.__order:           int                         = order
        self.__filters:         tuple[torch.Tensor, ...]    = tuple([])
        self.__padding_size:    int                         = 0
        self.__reset_filters()
        return
            
    
    def __reset_filters(self) -> None:
        if      self.__order==2:  self.__padding_size = 1
        elif    self.__order==4:  self.__padding_size = 2
        else:   raise ValueError(f"'order' should be in [2, 4], but got order={self.__order}.")
        filters = []
        for idx in range(self.__dim):
            filters.append(getattr(self, f'get_filter__order_{self.__order}')(idx))
        self.__filters = tuple(filters)
        return
    
    
    def get_filter__order_2(self, index: int) -> torch.Tensor:
        f = torch.zeros([3 for _ in range(self.__dim)], device=self.__device)
        _prefix, _appendix = [1]*index, [1]*(self.__dim-1-index)
        f[*_prefix, 0, *_appendix] = -0.5/self.__dx
        f[*_prefix, 2, *_appendix] = +0.5/self.__dx
        return f
    
    
    def get_filter__order_4(self, index: int) -> torch.Tensor:
        f = torch.zeros([5 for _ in range(self.__dim)], device=self.__device)
        _prefix, _appendix = [2]*index, [2]*(self.__dim-1-index)
        f[*_prefix, 0, *_appendix] = +(1.0/12.0)/self.__dx
        f[*_prefix, 1, *_appendix] = -(2.0/ 3.0)/self.__dx
        f[*_prefix, 3, *_appendix] = +(2.0/ 3.0)/self.__dx
        f[*_prefix, 4, *_appendix] = -(1.0/12.0)/self.__dx
        return f
    
    
    def compute_derivative(self, u: torch.Tensor, index: int) -> torch.Tensor:
        """
        Arguments:
            `u`     (`torch.Tensor`): A tensor of shape `(num_t, *domain)`.
            `index` (`int`): The index of the variable to be differentiated. `index` should be in `[0, dim-1]`.
        Returns:
            (`torch.Tensor`): A tensor of shape `(num_t, *domain)`, which is the shape of `u`. Note that the boundary values are padded with zeros.
        """
        if index<0 or index>=self.__dim:
            raise ValueError(f"'index' should be in [0, {self.__dim-1}], but got index={index}.")
        weight = self.__filters[index][None, None]    # Add the batch-channel dimensions
        conv = getattr(F, f'conv{self.__dim}d')
        du = conv(u.unsqueeze(1), weight, padding=0)
        du = F.pad(du, pad=[self.__padding_size for _ in range(2*self.__dim)], mode='constant', value=0)
        return du.squeeze(1)


##################################################
##################################################
# End of file