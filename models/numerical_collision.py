from   typing import Callable, Optional
import torch
from   solver import FastSM_Landau_VHS, one_step_RK4_classic
from   utils  import FiniteDifferenceMethod as FDM, compute_grad


__all__: list[str] = ['FPL_spectral', 'FPL_finite_difference']


class FPL_spectral():
    """## Fast spectral method wrapper for Fokker-Planck-Landau collision evaluation

## Description
Wraps `FastSM_Landau_VHS` to compute the Fokker-Planck-Landau collision operator and solve initial value problems
for 2D distribution functions flattened with `ij`-indexing convention.

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
        self.__shape:   tuple[int, ...] = tuple([-1, *(1 for _ in range(dimension)), *(v_num_grid for _ in range(dimension)), 1])
        self.__fft_dim: tuple[int, ...] = tuple(range(1 + dimension, 1 + 2 * dimension))
        return

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__shape

    @property
    def fft_dim(self) -> tuple[int, ...]:
        return self.__fft_dim

    def precompute(self) -> None:
        """## Precompute internal spectral variables

## Description
Executes required precomputations of characteristic functions, gain tensors, and loss tensors.

## Arguments
None.

## Returns
`None`: None.
"""
        self.fsm.precompute()
        return

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """## Forward evaluation of the spectral collision operator

## Description
Computes the fast spectral evaluation of the Fokker-Planck-Landau collision operator for distribution `f`.

## Arguments
`f` (`torch.Tensor`): A 2-tensor of shape `(num_points, in_channels)` with `in_channels == 1`.

## Returns
`torch.Tensor`: Evaluated collision operator tensor of shape `(num_points, in_channels)`.
"""
        if f.ndim != 2:
            raise ValueError(f"'f' should be a 2-tensor of shape '(num_points, in_channels)', but got shape {list(f.shape)}.")
        in_channels: int = f.size(-1)
        if in_channels != 1:
            raise ValueError(f"'in_channels' must be 1, but got {in_channels}.")

        f_reshaped: torch.Tensor = f.reshape(self.shape)
        f_fft: torch.Tensor = torch.fft.fftn(f_reshaped, dim=self.fft_dim, norm=self.fsm.internal_fft_norm)
        q_fft: torch.Tensor = self.fsm.compute_fft(None, f_fft)
        q: torch.Tensor = torch.real(torch.fft.ifftn(q_fft, dim=self.fft_dim, norm=self.fsm.internal_fft_norm))
        return q.reshape((-1, in_channels))

    def solve(
            self,
            t_init:  float,
            t_final: float,
            delta_t: float,
            f_init:  torch.Tensor,
        ) -> torch.Tensor:
        """## Solve the Fokker-Planck-Landau equation across time

## Description
Integrates the FPL distribution forward from `t_init` to `t_final` with step size `delta_t` using the fast spectral method and RK4.

## Arguments
`t_init` (`float`): Initial time instant.
`t_final` (`float`): Final time instant.
`delta_t` (`float`): Time step size.
`f_init` (`torch.Tensor`): Initial distribution tensor of shape `(num_points, in_channels)`.

## Returns
`torch.Tensor`: Trajectory tensor of shape `(num_points, in_channels)`.
"""
        if f_init.ndim != 2:
            raise ValueError(f"'f_init' should be a 2-tensor of shape '(num_points, in_channels)', but got shape {list(f_init.shape)}.")
        in_channels: int = f_init.size(-1)
        if in_channels != 1:
            raise ValueError(f"'in_channels' must be 1, but got {in_channels}.")

        f_init_reshaped: torch.Tensor = f_init.reshape(self.shape)
        sol: torch.Tensor = self.fsm.solve(t_init, t_final, delta_t, f_init_reshaped, one_step_RK4_classic)
        return sol.reshape((-1, in_channels))


class FPL_finite_difference():
    """## Finite difference solver for the Fokker-Planck-Landau collision operator

## Description
Computes the Fokker-Planck-Landau collision operator using numerical quadrature for the sub-operators $D(f)$ and $F(f)$
and standard finite difference differentiation for the divergence.

## Arguments
`dim` (`int`): Dimension of velocity domain.
`gamma` (`float`): Exponent parameter of collision kernel.
`max_v` (`float`): Maximum velocity boundary.
`res_v` (`int`): Number of grid points along each velocity axis.
`quad_order` (`int`, default: `100`): Quadrature order for Gauss-Legendre integration.
`scale` (`float`, default: `1.0`): Overall scaling factor.
`order` (`int`, default: `2`): Finite difference order.
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

        self.__dim:     int          = dim
        self.__gamma:   float        = gamma
        self.__res_v:   int          = res_v
        self.__scale:   float        = scale

        _qv1, _qw1 = roots_legendre(quad_order)
        self.__quad_v1: torch.Tensor = torch.tensor(max_v * _qv1, dtype=torch.float, device=device)
        self.__quad_w1: torch.Tensor = torch.tensor(max_v * _qw1, dtype=torch.float, device=device)
        self.__quad_v:  torch.Tensor = torch.cartesian_prod(*(self.__quad_v1 for _ in range(dim)))
        self.__quad_w:  torch.Tensor = torch.cartesian_prod(*(self.__quad_w1 for _ in range(dim))).prod(dim=1, keepdim=True)

        self.__delta_v: float        = (2.0 * max_v) / res_v
        _a:             float        = max_v - self.__delta_v / 2.0
        self.__v1:      torch.Tensor = torch.linspace(-_a, _a, res_v, device=device)
        self.__v:       torch.Tensor = torch.cartesian_prod(*(self.__v1 for _ in range(dim)))

        self.fdm: FDM = FDM(dim=dim, dx=self.__delta_v, order=order, device=device)
        return

    def integration(self, func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """## Numerical quadrature integration of function over velocity domain

## Description
Integrates `func` over the velocity domain using the precomputed Gauss-Legendre quadrature grid and weights.

## Arguments
`func` (`Callable[[torch.Tensor], torch.Tensor]`): Function mapping coordinate tensor `(num_points, dim)` to `(num_points, 1)`.

## Returns
`torch.Tensor`: Integrated scalar value tensor.
"""
        return torch.sum(func(self.__quad_v) * self.__quad_w).squeeze()

    def FPL_kernel(self, points: torch.Tensor) -> torch.Tensor:
        """## Collision kernel matrix for the Fokker-Planck-Landau equation

## Description
Evaluates the matrix-valued Landau collision kernel at relative velocity points.

## Arguments
`points` (`torch.Tensor`): Coordinate tensor of shape `(*alignment_of_points, dim)`.

## Returns
`torch.Tensor`: Matrix-valued kernel tensor of shape `(*alignment_of_points, dim, dim)`.
"""
        gamma: float = self.__gamma
        norm_v: torch.Tensor = points.norm(dim=-1)
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
        """## Compute D(f) and F(f) sub-operators

## Description
Computes the matrix diffusion term $D(f)(v)$ and vector drift term $F(f)(v)$ composing the Landau collision operator.

## Arguments
`func` (`Callable[[torch.Tensor], torch.Tensor]`): Distribution function evaluated at coordinate tensor `(num_points, dim)`.

## Returns
`tuple[torch.Tensor, torch.Tensor]`: Tuple `(Df, Ff)` where `Df` has shape `(num_points, dim, dim)` and `Ff` has shape `(num_points, dim)`.
"""
        points_diff: torch.Tensor = self.__v[:, None, :] - self.__quad_v[None, :, :]
        kernel_diff: torch.Tensor = self.FPL_kernel(points_diff)

        quad_v: torch.Tensor = self.__quad_v.clone().requires_grad_(True)
        f_quad: torch.Tensor = func(quad_v).flatten()
        w_quad: torch.Tensor = self.__quad_w.flatten()
        df_quad: torch.Tensor = compute_grad(f_quad, quad_v, create_graph=False)
        f_quad = f_quad.detach()

        Df: torch.Tensor = torch.einsum("vqij, q, q -> vij", kernel_diff, f_quad, w_quad)
        Ff: torch.Tensor = torch.einsum("vqij, qj, q -> vi", kernel_diff, df_quad, w_quad)
        return Df, Ff

    def forward(
            self,
            func: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
        """## Compute Landau collision operator via finite difference divergence

## Description
Evaluates the divergence of the Landau collision fluxes using the finite difference method.

## Arguments
`func` (`Callable[[torch.Tensor], torch.Tensor]`): Distribution function evaluated at coordinate tensor `(num_points, dim)`.

## Returns
`torch.Tensor`: Flattened collision operator tensor of shape `(num_points,)`.
"""
        dim: int = self.__dim
        domain: tuple[int, ...] = tuple(self.__res_v for _ in range(dim))
        v: torch.Tensor = self.__v.clone().requires_grad_(True)
        f: torch.Tensor = func(v)
        grad_f: torch.Tensor = compute_grad(f, v, False)
        f = f.detach()
        Df, Ff = self.compute_suboperators(func)

        f = f.reshape(*domain)
        grad_f = grad_f.reshape(*(f.shape), dim)
        Df = Df.reshape(*domain, dim, dim)
        Ff = Ff.reshape(*domain, dim)

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
    """## Compute projection matrix onto point directions

## Description
Computes the projection matrix tensor onto the direction of each point vector.

## Arguments
`points` (`torch.Tensor`): Tensor of shape `(*alignment_of_points, dim)` where `dim` is spatial dimension.

## Returns
`torch.Tensor`: Projection matrix tensor of shape `(*alignment_of_points, dim, dim)`.
"""
    norm: torch.Tensor = points.norm(dim=-1, keepdim=True)
    unit: torch.Tensor = torch.where(norm != 0.0, points / norm, torch.zeros_like(points))
    return unit[..., :, None] * unit[..., None, :]