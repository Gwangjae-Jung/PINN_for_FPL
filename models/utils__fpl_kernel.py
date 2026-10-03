from   typing import Callable, Optional
import torch
import torch.nn.functional as F
from   utils  import compute_grad


FFT_NORM: str = 'forward'

__all__: list[str] = ['projection_matrix', 'FPL_kernel', 'LandauCollisionOperator_FiniteDifference']


##################################################
##################################################
class FiniteDifferenceMethod():
    """Multidimensional central finite difference differentiator (local copy).

    ## Description
    Applies second-order central difference convolutional filters to compute partial
    derivatives of grid-based tensors. This is a local, simplified version of
    `utils.FiniteDifferenceMethod` that supports only first-order stencils and is
    used internally within `LandauCollisionOperator_FiniteDifference`.

    ## Arguments
    `dim` (`int`): Spatial dimension of the domain.
    `dx` (`float`): Uniform grid spacing.
    `order` (`int`, default: `1`): Finite difference order placeholder (always uses
    second-order central differences regardless of this value).
    `device` (`Optional[torch.device]`, default: `None`): Computing device for filter
    tensors. Defaults to `torch.get_default_device()`.

    ## Returns
    `None`: None.
    """
    def __init__(
            self,
            dim:    int,
            dx:     float,
            order:  int                    = 1,
            device: Optional[torch.device] = None,
        ) -> None:
        if not isinstance(order, int) or order < 1:
            raise ValueError(f"'order' should be a positive integer, but got order={order}.")
        if device is None:
            device = torch.get_default_device()

        self.__dim:     int                        = dim
        self.__dx:      float                      = dx
        self.__device:  torch.device               = device
        self.__order:   int                        = order
        self.__filters: tuple[torch.Tensor, ...]   = tuple([])
        self.__reset_filters()
        return

    def __reset_filters(self) -> None:
        """Build second-order central difference filters for all spatial axes."""
        dx:       float               = self.__dx
        filters:  list[torch.Tensor]  = []
        prefix:   list[int]           = []
        appendix: list[int]           = [1 for _ in range(self.__dim - 1)]
        for _ in range(self.__dim):
            f: torch.Tensor = torch.zeros([3 for _ in range(self.__dim)], device=self.__device)
            f[*prefix, 0, *appendix] = -0.5 / dx
            f[*prefix, 2, *appendix] = +0.5 / dx
            filters.append(f)
            if appendix:
                prefix.append(appendix.pop(-1))
        self.__filters = tuple(filters)
        return

    def compute_derivative(self, u: torch.Tensor, index: int) -> torch.Tensor:
        """Compute the partial derivative of `u` along axis `index`.

        ## Description
        Applies the central difference convolutional filter along the specified spatial
        axis, zero-padding the boundary to preserve the original tensor shape.

        ## Arguments
        `u` (`torch.Tensor`): Input tensor of shape `(num_t, *domain)`.
        `index` (`int`): Spatial axis index in `[0, dim-1]`.

        ## Returns
        `torch.Tensor`: Differentiated tensor of shape `(num_t, *domain)`.
        """
        if index < 0 or index >= self.__dim:
            raise ValueError(f"'index' should be in [0, {self.__dim - 1}], but got index={index}.")
        weight: torch.Tensor = self.__filters[index][None, None]
        conv: Callable[..., torch.Tensor] = getattr(F, f'conv{self.__dim}d')
        du: torch.Tensor = conv(u.unsqueeze(1), weight, padding=0)
        du = F.pad(du, pad=[1 for _ in range(2 * self.__dim)], mode='constant', value=0.0)
        return du.squeeze(1)


##################################################
##################################################
class LandauCollisionOperator_FiniteDifference():
    """Finite difference evaluator for the Landau collision operator.

    ## Description
    Computes the Fokker-Planck-Landau collision operator using Gauss-Legendre
    numerical quadrature for the sub-operators $D(f)$ and $F(f)$, and central
    finite difference differentiation for the divergence. Analogous to
    `models.FPL_finite_difference` but uses the local `FiniteDifferenceMethod`
    with a first-order stencil interface.

    ## Arguments
    `dim` (`int`): Dimension of velocity domain.
    `gamma` (`float`): Exponent parameter of the collision kernel.
    `max_v` (`float`): Maximum velocity boundary.
    `res_v` (`int`): Number of grid points along each velocity axis.
    `quad_order` (`int`, default: `100`): Quadrature order for Gauss-Legendre
    integration.
    `scale` (`float`, default: `1.0`): Overall scaling factor applied to the kernel.
    `order` (`int`, default: `1`): Finite difference order (passed to
    `FiniteDifferenceMethod`).
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
            order:      int                    = 1,
            device:     Optional[torch.device] = None,
        ) -> None:
        if device is None:
            device = torch.get_default_device()

        self.__dim:   int   = dim
        self.__gamma: float = gamma
        self.__res_v: int   = res_v
        self.__scale: float = scale

        # Build Gauss-Legendre quadrature grid over the velocity domain
        from scipy.special import roots_legendre
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

        self.fdm: FiniteDifferenceMethod = FiniteDifferenceMethod(
            dim=dim, dx=self.__delta_v, order=order, device=device,
        )
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
        Computes $A(v) = |v|^{2+\gamma} \Pi(v)$ where $\Pi(v)$ is the projection onto
        the hyperplane perpendicular to $v$. Regularizes the near-singular case for
        negative `gamma`.

        ## Arguments
        `points` (`torch.Tensor`): Relative velocity tensor of shape
        `(*alignment_of_points, dim)`.

        ## Returns
        `torch.Tensor`: Kernel tensor of shape `(*alignment_of_points, dim, dim)`.
        """
        gamma:  float        = self.__gamma
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
        """Compute the $D(f)$ and $F(f)$ sub-operators.

        ## Description
        Evaluates the matrix diffusion term $D(f)(v)$ and vector drift term $F(f)(v)$
        composing the Landau collision operator via numerical quadrature.

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
        df_quad: torch.Tensor = compute_grad(f_quad, quad_v, False)

        Df: torch.Tensor = torch.einsum("vqij, q, q -> vij", kernel_diff, f_quad,  w_quad)
        Ff: torch.Tensor = torch.einsum("vqij, qj, q -> vi",  kernel_diff, df_quad, w_quad)
        return Df, Ff

    def forward(
            self,
            func: Callable[[torch.Tensor], torch.Tensor],
        ) -> torch.Tensor:
        """Compute the Landau collision operator via finite difference divergence.

        ## Description
        Evaluates $Q(f) = \nabla_v \cdot (D(f) \nabla_v f - F(f) f)$ on the
        precomputed velocity grid using central finite differences.

        ## Arguments
        `func` (`Callable[[torch.Tensor], torch.Tensor]`): Distribution function
        evaluated at a coordinate tensor of shape `(num_points, dim)`.

        ## Returns
        `torch.Tensor`: Flattened collision operator tensor of shape `(num_points,)`.
        """
        dim:    int             = self.__dim
        domain: tuple[int, ...] = tuple(self.__res_v for _ in range(dim))

        v:      torch.Tensor = self.__v.clone().requires_grad_(True)
        f:      torch.Tensor = func(v)
        grad_f: torch.Tensor = compute_grad(f, v, False)
        Df, Ff = self.compute_suboperators(func)

        # Reshape tensors to the velocity grid layout for FDM
        f      = f.reshape(*domain)
        grad_f = grad_f.reshape(*(f.shape), dim)
        Df     = Df.reshape(*domain, dim, dim)
        Ff     = Ff.reshape(*domain, dim)

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
def FPL_kernel(v: torch.Tensor, gamma: float, scale: float = 1.0) -> torch.Tensor:
    """Evaluate the collision kernel matrix for the Fokker-Planck-Landau equation.

    ## Description
    Computes $A(v) = \text{scale} \cdot |v|^{2+\gamma} \Pi(v)$ at the given relative
    velocity points, where $\Pi(v)$ is the orthogonal projection onto the hyperplane
    perpendicular to $v$. Handles the near-singular case for negative `gamma` via
    a small regularizer.

    ## Arguments
    `v` (`torch.Tensor`): Relative velocity tensor of shape
    `(*alignment_of_points, dim)`.
    `gamma` (`float`): Exponent in the kernel function.
    `scale` (`float`, default: `1.0`): Scaling factor for the kernel.

    ## Returns
    `torch.Tensor`: Matrix-valued kernel tensor of shape
    `(*alignment_of_points, dim, dim)`.
    """
    norm_v: torch.Tensor = v.norm(dim=-1)
    if gamma >= 0.0:
        power_v: torch.Tensor = norm_v.pow(2.0 + gamma)
    else:
        power_v = norm_v.pow(2.0) / (norm_v.pow(-gamma) + 1e-16)
    power_v = power_v[..., None, None]
    proj: torch.Tensor = torch.eye(v.size(-1), device=v.device) - projection_matrix(v)
    return scale * power_v * proj


##################################################
##################################################
# End of file