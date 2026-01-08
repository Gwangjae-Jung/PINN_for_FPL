from    typing      import  Callable, Self, Optional
import  torch
from    deep_numerical.numerical.solvers    import  FastSM_Landau_VHS
from    utils       import  compute_grad, FiniteDifferenceMethod as FDM


__all__: list[str] = ['FPL_spectral', 'FPL_finite_difference']


##################################################
##################################################
class FPL_spectral():
    def __init__(
            self,
            dimension:  int,
            v_num_grid: int,
            v_max:      float,
            vhs_coeff:  float,
            vhs_alpha:  float,
            dtype:      Optional[torch.dtype]   = None,
            device:     Optional[torch.device]  = None,
        ) -> Self:
        self.fsm = FastSM_Landau_VHS(
            dimension   = dimension,
            v_num_grid  = v_num_grid,
            v_max       = v_max,
            vhs_coeff   = vhs_coeff,
            vhs_alpha   = vhs_alpha,
            dtype       = dtype,
            device      = device,
        )
        self.__shape = tuple([-1, *(1 for _ in range(dimension)), *(v_num_grid for _ in range(dimension)), 1])
        self.__fft_dim = tuple(range(1+dimension, 1+2*dimension))
        pass
    
    
    @property
    def shape(self) -> tuple[int, ...]: return self.__shape
    @property
    def fft_dim(self) -> tuple[int, ...]: return self.__fft_dim
    
    
    def precompute(self) -> None:
        """Precomputes the internal variables required for the fast spectral method.
        This method should be called before using the `forward` or `solve` methods.
        """
        self.fsm.precompute()
        return
    
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `f` (`torch.Tensor`): A tensor of shape `(num_points, in_channels)` aligned in the `ij`-indexing style. Here, `num_points` should be equal to `v_num_grid ** dimension`.
        """
        if f.ndim!=2:
            raise ValueError(f"'f' should be a 2-tensor of shape '(num_points, in_channels)`, but the shape of 'f' is {list(f.shape)}.")
        in_channels = f.size(-1)
        assert in_channels == 1
        f = f.reshape(self.shape)
        f_fft: torch.Tensor = torch.fft.fftn(f, dim=self.fft_dim, norm=self.fsm.internal_fft_norm)
        q_fft = self.fsm.compute_fft(None, f_fft)
        q = torch.real(torch.fft.ifftn(q_fft, dim=self.fft_dim, norm=self.fsm.internal_fft_norm))
        return q.reshape((-1, in_channels))
    
    
    def solve(self, t_init: float, t_final: float, delta_t: float, f_init: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `points` (`torch.Tensor`): A tensor of shape `(num_points, in_channels)` aligned in the `ij`-indexing style. Here, `num_points` should be equal to `v_num_grid ** dimension`.
        """
        from    deep_numerical.utils    import  one_step_RK4_classic
        if f_init.ndim!=2:
            raise ValueError(f"'f' should be a 2-tensor of shape '(num_points, in_channels)`, but the shape of 'f' is {list(f_init.shape)}.")
        in_channels = f_init.size(-1)
        assert in_channels == 1
        f_init = f_init.reshape(self.shape)
        sol = self.fsm.solve(t_init, t_final, delta_t, f_init, one_step_RK4_classic)
        return sol.reshape((-1, in_channels))


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


class FPL_finite_difference():
    def __init__(
            self,
            dim:        int,
            gamma:      float,
            max_v:      float,
            res_v:      int,
            quad_order: int = 100,
            scale:      float = 1.0,
            order:      int = 2,
            device:     Optional[torch.device] = None,
        ) -> Self:
        from    scipy.special   import  roots_legendre
        
        if device is None:  device = torch.get_default_device()
        self.__dim          = dim
        self.__gamma        = gamma
        self.__res_v        = res_v
        self.__scale        = scale
        
        _qv1, _qw1 = roots_legendre(quad_order)
        self.__quad_v1  = torch.tensor(max_v*_qv1, dtype=torch.float, device=device)
        self.__quad_w1  = torch.tensor(max_v*_qw1, dtype=torch.float, device=device)
        self.__quad_v   = torch.cartesian_prod(*(self.__quad_v1 for _ in range(dim)))
        self.__quad_w   = torch.cartesian_prod(*(self.__quad_w1 for _ in range(dim))).prod(dim=1, keepdim=True)
        
        self.__delta_v  = (2*max_v)/res_v
        _a              = max_v-self.__delta_v/2
        self.__v1       = torch.linspace(-_a, _a, res_v)
        self.__v        = torch.cartesian_prod(*(self.__v1 for _ in range(dim)))
        
        self.fdm = FDM(dim=dim, dx=self.__delta_v, order=order, device=device)
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
        df_quad = compute_grad(f_quad, quad_v, create_graph=False)    # shape: (num_qv, dim)
        f_quad = f_quad.detach()    # <-- THIS LINE PREVENTS THE ACCUMULATION OF COMPUTATIONAL GRAPH
        
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
        f = f.detach()  # <-- THIS LINE PREVENTS THE ACCUMULATION OF COMPUTATIONAL GRAPH
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