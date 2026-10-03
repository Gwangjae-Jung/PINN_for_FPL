import  torch


__all__: list[str] = ['bkw', 'maxwellian', 'bimaxwellian', 'perturbed_maxwellian']


##################################################
##################################################
def bkw(
        points:     torch.Tensor,
        kernel:     float,
        coeff_ext:  float,
        density:    float = 0.2,
    ) -> torch.Tensor:
    """Evaluate the analytic BKW solution of the homogeneous Fokker-Planck-Landau equation.

    ## Description
    Computes the BKW (Bobylev-Krook-Wu) self-similar solution $f(t, v)$ of the
    spatially homogeneous Fokker-Planck-Landau equation. The solution is parameterized
    by an internal collision coefficient derived from `kernel` and `density`, and an
    external relaxation coefficient `coeff_ext`.

    ## Arguments
    `points` (`torch.Tensor`): Coordinate tensor of shape `(N, 1+dim)`, where the
    first column is time `t` and the remaining columns are velocity `v`.
    `kernel` (`float`): Interaction (VHS) coefficient that determines `coeff_int` via
    `coeff_int = 2*(dim-1)*density*kernel`.
    `coeff_ext` (`float`): External relaxation coefficient controlling the exponential
    decay rate of the shape function `K(t)`.
    `density` (`float`, default: `0.2`): Mass density of the distribution.

    ## Returns
    `torch.Tensor`: Evaluated BKW distribution tensor of shape `(N, 1)`.
    """
    t:   torch.Tensor = points[..., [0]]
    v:   torch.Tensor = points[..., 1:]
    dim: int          = v.shape[-1]

    coeff_int: float = 2 * (dim - 1) * density * kernel

    speed_sq: torch.Tensor = torch.sum(v**2, dim=-1, keepdim=True)
    K_t:      torch.Tensor = 1 - coeff_ext * torch.exp(-coeff_int * t)

    _part1: torch.Tensor = density * torch.pow(2 * torch.pi * K_t, -dim / 2)
    _part2: torch.Tensor = torch.exp(-speed_sq / (2 * K_t))
    _part3: torch.Tensor = ((dim + 2) / 2) - ((dim + speed_sq) / 2) / K_t + (speed_sq / 2) / (K_t**2)

    return _part1 * _part2 * _part3


##################################################
def maxwellian(
        dim:        int,
        points:     torch.Tensor,
        center:     torch.Tensor,
        sigma:      float,
        density:    float,
    ) -> torch.Tensor:
    """Evaluate an isotropic Maxwellian distribution.

    ## Description
    Computes the `dim`-dimensional isotropic Gaussian (Maxwellian) distribution
    centred at `center` with standard deviation `sigma` and total mass `density`.
    If `points` includes a time column (shape `(N, 1+dim)`), it is discarded.

    ## Arguments
    `dim` (`int`): Dimension of the velocity space.
    `points` (`torch.Tensor`): Coordinate tensor of shape `(N, dim)` or `(N, 1+dim)`.
    `center` (`torch.Tensor`): Mean velocity vector of shape `(dim,)` or broadcastable.
    `sigma` (`float`): Standard deviation (isotropic temperature).
    `density` (`float`): Total mass density.

    ## Returns
    `torch.Tensor`: Evaluated Maxwellian distribution of shape `(N, 1)`.
    """
    if not (points.size(-1) in (dim, 1+dim)):
        raise ValueError(f"'points' should have size(-1) in {{{dim}, {1+dim}}}, but got {list(points.shape)}.")
    if points.size(-1) == 1+dim:
        points = points[:, 1:]              # Discard the time column if present
    if isinstance(sigma, float):
        sigma = torch.tensor([sigma], device=points.device)

    _cfg:  dict = {'dim': -1, 'keepdim': True}
    _dim:  int  = points.size(-1)
    mode: torch.Tensor = torch.exp(
        -(points - center).pow(2).sum(**_cfg) /
        (2 * (sigma**2))
    ) / (2 * torch.pi * (sigma**2))**(0.5 * _dim)

    return density * mode


##################################################
def bimaxwellian(
        dim:        int,
        points:     torch.Tensor,
        center_1:   torch.Tensor,
        center_2:   torch.Tensor,
        sigma_1:    float,
        sigma_2:    float,
        density:    float,
    ) -> torch.Tensor:
    """Evaluate a symmetric biMaxwellian distribution (mixture of two Maxwellians).

    ## Description
    Computes the equal-weight mixture of two isotropic Maxwellian distributions,
    each centred at `center_1` and `center_2` with standard deviations `sigma_1`
    and `sigma_2`, respectively, and total mass `density`.

    ## Arguments
    `dim` (`int`): Dimension of the velocity space.
    `points` (`torch.Tensor`): Coordinate tensor of shape `(N, dim)` or `(N, 1+dim)`.
    `center_1` (`torch.Tensor`): Mean velocity of the first mode.
    `center_2` (`torch.Tensor`): Mean velocity of the second mode.
    `sigma_1` (`float`): Standard deviation of the first mode.
    `sigma_2` (`float`): Standard deviation of the second mode.
    `density` (`float`): Total mass density of the mixture.

    ## Returns
    `torch.Tensor`: Evaluated biMaxwellian distribution of shape `(N, 1)`.
    """
    if not (points.size(-1) in (dim, 1+dim)):
        raise ValueError(f"'points' should have size(-1) in {{{dim}, {1+dim}}}, but got {list(points.shape)}.")
    if points.size(-1) == 1+dim:
        points = points[:, 1:]              # Discard the time column if present
    if isinstance(sigma_1, float):  sigma_1 = torch.tensor([sigma_1], device=points.device)
    if isinstance(sigma_2, float):  sigma_2 = torch.tensor([sigma_2], device=points.device)

    _cfg:  dict = {'dim': -1, 'keepdim': True}
    _dim:  int  = points.size(-1)
    mode_1: torch.Tensor = torch.exp(
        -(points - center_1).pow(2).sum(**_cfg) /
        (2 * (sigma_1**2))
    ) / (2 * torch.pi * sigma_1.pow(2)).pow(0.5 * _dim)
    mode_2: torch.Tensor = torch.exp(
        -(points - center_2).pow(2).sum(**_cfg) /
        (2 * (sigma_2**2))
    ) / (2 * torch.pi * sigma_2.pow(2)).pow(0.5 * _dim)

    return density * ((mode_1 + mode_2) / 2)


##################################################
def perturbed_maxwellian(
        dim:            int,
        points:         torch.Tensor,
        center:         torch.Tensor,
        sigma:          float,
        density:        float,
        perturbation:   tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
    """Evaluate a polynomially perturbed Maxwellian distribution.

    ## Description
    Computes a Maxwellian distribution multiplied by a quadratic perturbation
    polynomial in velocity. The perturbation is parameterized by three tensors
    `(p0, p1, p2)` representing the constant, linear, and quadratic coefficients,
    respectively.

    ## Arguments
    `dim` (`int`): Dimension of the velocity space.
    `points` (`torch.Tensor`): Coordinate tensor of shape `(N, dim)` or `(N, 1+dim)`.
    `center` (`torch.Tensor`): Mean velocity of the base Maxwellian.
    `sigma` (`float`): Standard deviation of the base Maxwellian.
    `density` (`float`): Total mass density.
    `perturbation` (`tuple[torch.Tensor, torch.Tensor, torch.Tensor]`): Tuple
    `(p0, p1, p2)` where `p0` is a scalar tensor (1 element), `p1` is a vector
    tensor of `dim` elements, and `p2` is a matrix tensor of `dim^2` elements,
    encoding the quadratic perturbation `1 + p0 + p1·v + v^T p2 v`.

    ## Returns
    `torch.Tensor`: Evaluated perturbed Maxwellian distribution of shape `(N, 1)`.
    """
    p0, p1, p2 = perturbation
    if any(not isinstance(p, torch.Tensor) for p in perturbation):
        raise ValueError(
            f"All elements of 'perturbation' should be 'torch.Tensor', "
            f"but got {[type(p) for p in perturbation]}."
        )
    if p0.numel() != 1:
        raise ValueError(
            f"The first element of 'perturbation' should be a single scalar tensor, "
            f"but got {p0.numel()} entries."
        )
    if p1.numel() != dim:
        raise ValueError(
            f"The second element of 'perturbation' should be a tensor with {dim} elements, "
            f"but got {p1.numel()} entries."
        )
    if p2.numel() != dim**2:
        raise ValueError(
            f"The third element of 'perturbation' should be a tensor with {dim**2} elements, "
            f"but got {p2.numel()} entries."
        )
    p0 = p0.reshape((1,))
    p1 = p1.reshape((dim,))
    p2 = p2.reshape((dim, dim))

    base:    torch.Tensor = maxwellian(dim, points, center, sigma, 1.0)
    perturb: torch.Tensor = (
        p0 +
        (points * p1).sum(dim=-1, keepdim=True) +
        torch.einsum('...i,ij,...j->...', points, p2, points).unsqueeze(-1)
    )
    return density * base * (1 + perturb)


##################################################
##################################################
# End of file