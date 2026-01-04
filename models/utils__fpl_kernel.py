from    typing      import  Callable, Self, Optional
import  torch
import  torch.nn.functional     as      F
from    deep_numerical.utils    import  compute_grad


FFT_NORM: str = 'forward'
__all__: list[str] = ['projection_matrix', 'FPL_kernel', 'LandauCollisionOperator_FiniteDifference']


##################################################
##################################################
class FiniteDifferenceMethod():
    def __init__(
            self,
            dim:    int,
            dx:     float,
            order:  int                     = 1,
            device: Optional[torch.device]  = None,
        ) -> Self:
        if not isinstance(order, int) or order<1:
            raise ValueError(f"'order' should be a positive integer, but got order={order}.")
        device = device if device is not None else torch.get_default_device()
        self.__dim:     int             = dim
        self.__dx:      float           = dx
        self.__device:  torch.device    = device
        self.__order:   int             = order
        self.__filters:  tuple[torch.Tensor, ...] = tuple([])
        self.__reset_filters()
        return
    
    
    def __reset_filters(self) -> None:
        dx = self.__dx
        filters = []
        __prefix, __appendix = [], [1 for _ in range(self.__dim-1)]
        for _ in range(self.__dim):
            f = torch.zeros([3 for _ in range(self.__dim)], device=self.__device)
            f[*__prefix, 0, *__appendix] = -0.5/dx
            f[*__prefix, 2, *__appendix] = +0.5/dx
            filters.append(f)
            if __appendix:  __prefix.append(__appendix.pop(-1))
        self.__filters = tuple(filters)
        return
    
    
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
        du = F.pad(du, pad=[1 for _ in range(2*self.__dim)], mode='constant', value=0)
        return du.squeeze(1)


##################################################
##################################################
def projection_matrix(points: torch.Tensor) -> torch.Tensor:
    """Compute the projection matrix onto the direction of the given points.
    
    Arguments:
        `points` (`torch.Tensor`):
            A tensor of shape `(*alignment_of_points, dim)`, where `dim` is the spatial dimension.
    Returns:
        `torch.Tensor`:
            The projection matrix tensor of shape `(*alignment_of_points, dim, dim)`.
    """
    unit = torch.where(points!=0.0, points/points.norm(dim=-1, keepdim=True), torch.zeros_like(points, device=points.device))
    return unit[..., :, None]*unit[..., None, :]


##################################################
##################################################
def FPL_kernel(v: torch.Tensor, gamma: float, scale: float=1.0) -> torch.Tensor:
    """Compute the collision kernel function for the Fokker-Planck-Landau equation.
    
    Arguments:
        `gamma` (`float`):
            The exponent in the kernel function.
        `v` (`torch.Tensor`):
            The relative velocity tensor of shape `(*alignment_of_points, dim)`, where `dim` is the spatial dimension.
        `scale` (`float`): A scaling factor for the kernel. Default is `1.0`.

    Returns:
        `torch.Tensor`:
            The computed kernel tensor of shape `(*alignment_of_points, dim, dim)`. Note that the collision kernel of the Fokker-Planck-Landau equation is a matrix-valued function.
    """
    _power: torch.Tensor
    if gamma>=0:    _power = v.norm(dim=-1).pow(2+gamma)
    else:           _power = v.norm(dim=-1).pow(2) / (v.norm(dim=-1).pow(-gamma)+1e-16)
    _power = _power[..., None, None]
    _proj = torch.eye(v.size(-1))-projection_matrix(v)
    return scale*_power*_proj


##################################################
##################################################
class LandauCollisionOperator_FiniteDifference():
    def __init__(
            self,
            dim:        int,
            gamma:      float,
            max_v:      float,
            res_v:      int,
            quad_order: int = 100,
            scale:      float = 1.0,
            order:      int = 1,
            device:     Optional[torch.device] = None,
        ) -> Self:
        if device is None:  device = torch.get_default_device()
        self.__dim          = dim
        self.__gamma        = gamma
        self.__res_v        = res_v
        self.__scale        = scale
        
        from    scipy.special   import  roots_legendre
        _qv1, _qw1 = roots_legendre(quad_order)
        self.__quad_v1  = torch.tensor(max_v*_qv1, dtype=torch.float, device=device)
        self.__quad_w1  = torch.tensor(max_v*_qw1, dtype=torch.float, device=device)
        self.__quad_v   = torch.cartesian_prod(*(self.__quad_v1 for _ in range(dim)))
        self.__quad_w   = torch.cartesian_prod(*(self.__quad_w1 for _ in range(dim))).prod(dim=1, keepdim=True)
        
        self.__delta_v  = (2*max_v)/res_v
        _a              = max_v-self.__delta_v/2
        self.__v1       = torch.linspace(-_a, _a, res_v)
        self.__v        = torch.cartesian_prod(*(self.__v1 for _ in range(dim)))
        
        self.fdm = FiniteDifferenceMethod(dim=dim, dx=self.__delta_v, order=order, device=device)
        return

        
    def integration(self, func: Callable[[torch.Tensor, object], torch.Tensor]) -> torch.Tensor:
        """Conducts numerical integration of `func`.
        
        Arguments:
            `func` (`Callable[[torch.Tensor, object], torch.Tensor]`):
                A function that takes a tensor of shape `(num_points, dim)` and returns a tensor of shape `(num_points, 1)`.
        
        Returns:
            `torch.Tensor`:
                The result of the numerical integration as a tensor of shape `(1,)`.
        """
        return torch.sum(func(self.__quad_v) * self.__quad_w).squeeze()
    
    
    def FPL_kernel(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the collision kernel function for the Fokker-Planck-Landau equation.
        
        Arguments:
            `points` (`torch.Tensor`):
                A tensor of shape `(*alignment_of_points, dim)`, where `dim` is the spatial dimension.
        
        Returns:
            `torch.Tensor`:
                The computed kernel tensor of shape `(*alignment_of_points, dim, dim)`. Note that the collision kernel of the Fokker-Planck-Landau equation is a matrix-valued function.
        """
        gamma = self.__gamma
        _power: torch.Tensor
        if gamma>=0.0:  _power = points.norm(dim=-1).pow(2+gamma)
        else:           _power = points.norm(dim=-1).pow(2) / (points.norm(dim=-1).pow(-gamma)+1e-2)
        _power = _power[..., None, None]
        _proj = torch.eye(points.size(-1))-projection_matrix(points)
        return self.__scale*_power*_proj
    
    
    def compute_suboperators(
            self,
            func:   Callable[[torch.Tensor], torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sub-operators comprising the Fokker-Planck-Landau collision operator.
        Arguments:
            `func` (`Callable[[torch.Tensor], torch.Tensor]`):
                A function that takes a tensor of shape `(num_points, dim)` and returns a tensor of shape `(num_points, 1)`.
        Returns:
            `tuple[torch.Tensor, torch.Tensor]`:
                A tuple containing two tensors:
                - The first tensor represents the term `D(f)(v)` and has shape `(num_points, dim, dim)`.
                - The second tensor represents the term `F(f)(v)` and has shape `(num_points, dim)`.
        """
        points_diff = self.__v[:, None, :] - self.__quad_v[None, :, :]  # shape: (num_v, num_qv, dim)
        kernel_diff = self.FPL_kernel(points_diff)  # shape: (num_v, num_qv, dim, dim)
        quad_v = self.__quad_v.clone()
        quad_v.requires_grad_(True)
        f_quad      = func(quad_v).flatten()        # shape: (num_qv,)
        w_quad      = self.__quad_w.flatten()       # shape: (num_qv,)
        df_quad = compute_grad(f_quad, quad_v, False)    # shape: (num_qv, dim)
        
        Df = torch.einsum("vqij, q, q -> vij", kernel_diff, f_quad, w_quad)
        Ff = torch.einsum("vqij, qj, q -> vi", kernel_diff, df_quad, w_quad)
        return Df, Ff

    
    def forward(
            self,
            func:   Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
        """Compute the Fokker-Planck-Landau collision operator using the finite difference method.
        
        ### Note
        This method computes only for a single time instant.
        
        Arguments:
            `func` (`Callable[[torch.Tensor], torch.Tensor]`):
                A function that takes a tensor of shape `(num_points, dim)` and returns a tensor of shape `(num_points, 1)`.
        Returns:
            `torch.Tensor`:
                The computed Fokker-Planck-Landau collision operator tensor of shape `(num_points,)`.
        """
        dim = self.__dim
        domain = tuple([self.__res_v for _ in range(dim)])
        v = self.__v.clone()
        v.requires_grad_(True)
        f = func(v)
        grad_f = compute_grad(f, v, False)
        Df, Ff = self.compute_suboperators(func)
        
        f = f.reshape(*domain)
        grad_f = grad_f.reshape(*(f.shape), dim)
        Df = Df.reshape(*domain, dim, dim)
        Ff = Ff.reshape(*domain, dim)
        # So far, the tensors to be used are reshaped as follows:
        # * f:        (*domain)
        # * grad_f:   (*domain, dim)
        # * Df:       (*domain, dim, dim)
        # * Ff:       (*domain, dim)
        print("Shapes:", f.shape, grad_f.shape, Df.shape, Ff.shape)
        
        operands: list[torch.Tensor] = [
            torch.einsum("...j, ...j -> ...", Df[..., d, :], grad_f) - Ff[..., d]*f
            for d in range(dim)
        ]
        # `operands` saves tensors of shape `(*domain)`,
        # and the `i`-th tensor will be differentiated with respect to `v_i`
        operands = [
            self.fdm.compute_derivative(_op.unsqueeze(0), _idx)
            for _idx, _op in enumerate(operands)
        ]
        q = torch.stack(operands, dim=-1).sum(dim=-1)
        return q.flatten()


##################################################
##################################################
# End of file