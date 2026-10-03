from   typing import Callable, Optional
import torch
from   solver import FastSM_Landau_VHS, one_step_RK4_classic
from   utils  import FiniteDifferenceMethod as FDM, compute_grad


__all__: list[str] = ['FPL_spectral', 'FPL_finite_difference']


##################################################
##################################################
class FPL_spectral():
    """Fast spectral method wrapper for Fokker-Planck-Landau collision evaluation.

    ## Description
    Wraps `FastSM_Landau_VHS` to expose a flat `(num_points, 1)`-compatible interface
    for computing the FPL collision operator $Q(f)$ and for integrating the spatially
    homogeneous FPL equation forward in time. Internally uses `ij`-indexing convention
    for the velocity grid.

    ## Arguments
    `dimension` (`int`): Dimension of the velocity domain.
    `v_num_grid` (`int`): Number of grid intervals along each velocity axis.
    `v_max` (`float`): Velocity cutoff bound defining `[-v_max, v_max]^d`.
    `vhs_coeff` (`float`): Interaction coefficient in the VHS model.
    `vhs_alpha` (`float`): Interaction exponent in the VHS model.
    `dtype` (`Optional[torch.dtype]`, default: `None`): PyTorch tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `None`: None.
    """
    def __init__(
            self,
            dimension:  int,
            v_num_grid: int,
            v_max:      float,
            vhs_coeff:  float,
            vhs_alpha:  float,
            dtype:      Optional[torch.dtype]  = None,
            device:     Optional[torch.device] = None,
        ) -> None:
        self.fsm: FastSM_Landau_VHS = FastSM_Landau_VHS(
            dimension  = dimension,
            v_num_grid = v_num_grid,
            v_max      = v_max,
            vhs_coeff  = vhs_coeff,
            vhs_alpha  = vhs_alpha,
            dtype      = dtype,
            device     = device,
        )
        # Shape used to map the flat (N, 1) tensor to the grid layout expected by the FSM
        self.__shape:   tuple[int, ...] = tuple(
            [-1, *(1 for _ in range(dimension)), *(v_num_grid for _ in range(dimension)), 1]
        )
        self.__fft_dim: tuple[int, ...] = tuple(range(1 + dimension, 1 + 2 * dimension))
        return

    @property
    def shape(self) -> tuple[int, ...]:
        """Internal grid reshape tuple used when converting flat tensors to grid layout."""
        return self.__shape

    @property
    def fft_dim(self) -> tuple[int, ...]:
        """Axis indices over which the FFT is computed (velocity axes only)."""
        return self.__fft_dim

    def precompute(self) -> None:
        """Precompute internal spectral variables.

        ## Description
        Delegates to `FastSM_Landau_VHS.precompute()` to execute the precomputation
        of characteristic functions, gain tensors, and loss tensors required by the
        fast spectral method.

        ## Arguments
        None.

        ## Returns
        `None`: None.
        """
        self.fsm.precompute()
        return

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Evaluate the spectral collision operator $Q(f)$.

        ## Description
        Computes the fast spectral evaluation of the Fokker-Planck-Landau collision
        operator for a flat distribution tensor `f`. The evaluation is performed in
        Fourier space and the result is inverse-transformed back to physical space.

        ## Arguments
        `f` (`torch.Tensor`): Distribution tensor of shape `(num_points, 1)`.

        ## Returns
        `torch.Tensor`: Evaluated collision operator tensor of shape `(num_points, 1)`.
        """
        if f.ndim != 2:
            raise ValueError(
                f"'f' should be a 2-tensor of shape '(num_points, in_channels)', "
                f"but got shape {list(f.shape)}."
            )
        in_channels: int = f.size(-1)
        if in_channels != 1:
            raise ValueError(f"'in_channels' must be 1, but got {in_channels}.")

        f_reshaped: torch.Tensor = f.reshape(self.shape)
        f_fft: torch.Tensor      = torch.fft.fftn(f_reshaped, dim=self.fft_dim, norm=self.fsm.internal_fft_norm)
        q_fft: torch.Tensor      = self.fsm.compute_fft(None, f_fft)
        q:     torch.Tensor      = torch.real(torch.fft.ifftn(q_fft, dim=self.fft_dim, norm=self.fsm.internal_fft_norm))
        return q.reshape((-1, in_channels))

    def solve(
            self,
            t_init:  float,
            t_final: float,
            delta_t: float,
            f_init:  torch.Tensor,
        ) -> torch.Tensor:
        """Integrate the FPL equation from `t_init` to `t_final`.

        ## Description
        Advances the initial distribution `f_init` forward in time using the fast
        spectral method combined with the classical RK4 integrator.

        ## Arguments
        `t_init` (`float`): Initial time instant.
        `t_final` (`float`): Final time instant.
        `delta_t` (`float`): Time step size.
        `f_init` (`torch.Tensor`): Initial distribution tensor of shape
        `(num_points, 1)`.

        ## Returns
        `torch.Tensor`: Trajectory tensor of shape `(num_points, 1)` at `t_final`.
        """
        if f_init.ndim != 2:
            raise ValueError(
                f"'f_init' should be a 2-tensor of shape '(num_points, in_channels)', "
                f"but got shape {list(f_init.shape)}."
            )
        in_channels: int = f_init.size(-1)
        if in_channels != 1:
            raise ValueError(f"'in_channels' must be 1, but got {in_channels}.")

        f_init_reshaped: torch.Tensor = f_init.reshape(self.shape)
        sol: torch.Tensor = self.fsm.solve(t_init, t_final, delta_t, f_init_reshaped, one_step_RK4_classic)
        return sol.reshape((-1, in_channels))


##################################################
##################################################
class FPL_finite_difference():
    """Finite difference solver for the Fokker-Planck-Landau collision operator.

    ## Description
    Computes the Fokker-Planck-Landau collision operator using Gauss-Legendre
    numerical quadrature for the sub-operators $D(f)$ and $F(f)$, and central finite
    difference differentiation for the divergence. Suitable for reference solutions
    or as a baseline for the neural surrogate operator.

    ## Arguments
    `dim` (`int`): Dimension of velocity domain.
    `gamma` (`float`): Exponent parameter of the collision kernel.
    `max_v` (`float`): Maximum velocity boundary.
    `res_v` (`int`): Number of grid points along each velocity axis.
    `quad_order` (`int`, default: `100`): Quadrature order for Gauss-Legendre
    integration.
    `scale` (`float`, default: `1.0`): Overall scaling factor applied to the kernel.
    `order` (`int`, default: `2`): Finite difference stencil order.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `None`: None.
    """
    def __init__(
            self,
            dim:        int,
            gamma:      float,
            max_v:      float,
            res_v:      int,
            quad_order: int                    = 100,
            scale:      float                  = 1.0,
            order:      int                    = 2,
            device:     Optional[torch.device] = None,
        ) -> None:
        from scipy.special import roots_legendre

        if device is None:
            device = torch.get_default_device()

        self.__dim:   int   = dim
        self.__gamma: float = gamma
        self.__res_v: int   = res_v
        self.__scale: float = scale

        # Build Gauss-Legendre quadrature grid over the velocity domain
        _qv1, _qw1 = roots_legendre(quad_order)
        self.__quad_v1: torch.Tensor = torch.tensor(max_v * _qv1, dtype=torch.float, device=device)
        self.__quad_w1: torch.Tensor = torch.tensor(max_v * _qw1, dtype=torch.float, device=device)
        self.__quad_v:  torch.Tensor = torch.cartesian_prod(*(self.__quad_v1 for _ in range(dim)))
        self.__quad_w:  torch.Tensor = torch.cartesian_prod(
            *(self.__quad_w1 for _ in range(dim))
        ).prod(dim=1, keepdim=True)

        # Build cell-centered velocity grid
        self.__delta_v: float        = (2.0 * max_v) / res_v
        _a:             float        = max_v - self.__delta_v / 2.0
        self.__v1:      torch.Tensor = torch.linspace(-_a, _a, res_v, device=device)
        self.__v:       torch.Tensor = torch.cartesian_prod(*(self.__v1 for _ in range(dim)))

        self.fdm: FDM = FDM(dim=dim, dx=self.__delta_v, order=order, device=device)
        return

    def integration(self, func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Numerically integrate `func` over the velocity domain.

        ## Description
        Integrates `func` over the velocity domain using the precomputed Gauss-Legendre
        quadrature grid and weights.

        ## Arguments
        `func` (`Callable[[torch.Tensor], torch.Tensor]`): Function mapping a coordinate
        tensor of shape `(num_points, dim)` to `(num_points, 1)`.

        ## Returns
        `torch.Tensor`: Integrated scalar value.
        """
        return torch.sum(func(self.__quad_v) * self.__quad_w).squeeze()

    def FPL_kernel(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the matrix-valued Landau collision kernel.

        ## Description
        Computes the matrix-valued collision kernel $A(v) = |v|^{2+\gamma} \Pi(v)$,
        where $\Pi(v)$ is the orthogonal projection onto the hyperplane perpendicular
        to $v$. Near-singular behaviour for negative `gamma` is regularized.

        ## Arguments
        `points` (`torch.Tensor`): Relative velocity tensor of shape
        `(*alignment_of_points, dim)`.

        ## Returns
        `torch.Tensor`: Kernel tensor of shape `(*alignment_of_points, dim, dim)`.
        """
        gamma:   float        = self.__gamma
        norm_v:  torch.Tensor = points.norm(dim=-1)
        if gamma >= 0.0:
            power_v: torch.Tensor = norm_v.pow(2.0 + gamma)
        else:
            power_v = norm_v.pow(2.0) / (norm_v.pow(-gamma) + 1e-2)
        power_v = power_v[..., None, None]
        proj: torch.Tensor = torch.eye(points.size(-1), device=points.device) - projection_matrix(points)
        return self.__scale * power_v * proj

    def compute_suboperators(
            self,
            func: Callable[[torch.Tensor], torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the $D(f)$ and $F(f)$ sub-operators.

        ## Description
        Evaluates the matrix diffusion term $D(f)(v)$ and vector drift term $F(f)(v)$
        that compose the Landau collision operator, using the precomputed quadrature
        grid and the collision kernel.

        ## Arguments
        `func` (`Callable[[torch.Tensor], torch.Tensor]`): Distribution function
        evaluated at a coordinate tensor of shape `(num_points, dim)`.

        ## Returns
        `tuple[torch.Tensor, torch.Tensor]`: Tuple `(Df, Ff)` where `Df` has shape
        `(num_points, dim, dim)` and `Ff` has shape `(num_points, dim)`.
        """
        points_diff: torch.Tensor = self.__v[:, None, :] - self.__quad_v[None, :, :]
        kernel_diff: torch.Tensor = self.FPL_kernel(points_diff)

        # Compute gradient of f at quadrature points
        quad_v:  torch.Tensor = self.__quad_v.clone().requires_grad_(True)
        f_quad:  torch.Tensor = func(quad_v).flatten()
        w_quad:  torch.Tensor = self.__quad_w.flatten()
        df_quad: torch.Tensor = compute_grad(f_quad, quad_v, create_graph=False)
        f_quad = f_quad.detach()

        Df: torch.Tensor = torch.einsum("vqij, q, q -> vij", kernel_diff, f_quad,  w_quad)
        Ff: torch.Tensor = torch.einsum("vqij, qj, q -> vi",  kernel_diff, df_quad, w_quad)
        return Df, Ff

    def forward(
            self,
            func: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
        """Compute the Landau collision operator via finite difference divergence.

        ## Description
        Evaluates the divergence form of the Landau collision operator
        $Q(f) = \nabla_v \cdot (D(f) \nabla_v f - F(f) f)$ on the precomputed
        velocity grid using central finite differences.

        ## Arguments
        `func` (`Callable[[torch.Tensor], torch.Tensor]`): Distribution function
        evaluated at a coordinate tensor of shape `(num_points, dim)`.

        ## Returns
        `torch.Tensor`: Flattened collision operator tensor of shape `(num_points,)`.
        """
        dim:    int            = self.__dim
        domain: tuple[int, ...] = tuple(self.__res_v for _ in range(dim))

        v:      torch.Tensor = self.__v.clone().requires_grad_(True)
        f:      torch.Tensor = func(v)
        grad_f: torch.Tensor = compute_grad(f, v, False)
        f = f.detach()
        Df, Ff = self.compute_suboperators(func)

        # Reshape tensors to the velocity grid layout for FDM
        f      = f.reshape(*domain)
        grad_f = grad_f.reshape(*(f.shape), dim)
        Df     = Df.reshape(*domain, dim, dim)
        Ff     = Ff.reshape(*domain, dim)

        # Per-axis flux and divergence
        operands: list[torch.Tensor] = [
            torch.einsum("...j, ...j -> ...", Df[..., d, :], grad_f) - Ff[..., d] * f
            for d in range(dim)
        ]
        diff_operands: list[torch.Tensor] = [
            self.fdm.compute_derivative(op.unsqueeze(0), idx)
            for idx, op in enumerate(operands)
        ]
        q: torch.Tensor = torch.stack(diff_operands, dim=-1).sum(dim=-1)
        return q.flatten()


##################################################
def projection_matrix(points: torch.Tensor) -> torch.Tensor:
    """Compute the projection matrix onto the direction of each point.

    ## Description
    For each point $v$ in `points`, computes the rank-1 projection matrix
    $\hat{v} \hat{v}^T$ where $\hat{v} = v / |v|$. The zero vector maps to the zero
    matrix.

    ## Arguments
    `points` (`torch.Tensor`): Tensor of shape `(*alignment_of_points, dim)`.

    ## Returns
    `torch.Tensor`: Projection matrix tensor of shape
    `(*alignment_of_points, dim, dim)`.
    """
    norm: torch.Tensor = points.norm(dim=-1, keepdim=True)
    unit: torch.Tensor = torch.where(norm != 0.0, points / norm, torch.zeros_like(points))
    return unit[..., :, None] * unit[..., None, :]


##################################################
##################################################
# End of file