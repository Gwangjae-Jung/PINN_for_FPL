from    typing      import  Self
import  torch


__all__: list[str] = [
    'abs_error', 'rel_error', 'rmse_error',
    'absolute_error', 'relative_error',
    'compute_mass_density', 'compute_bulk_velocity', 'compute_energy_density', 'compute_entropy_density',
    'AverageMeter',
]


EPS: float = 1e-16  # Small constant to avoid log(0) or division by zero


##################################################
def abs_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute error between `pred` and `target`.

    ## Description
    Returns the mean of the element-wise absolute difference between prediction and
    target tensors.

    ## Arguments
    `pred` (`torch.Tensor`): Predicted tensor.
    `target` (`torch.Tensor`): Ground truth tensor of the same shape as `pred`.

    ## Returns
    `torch.Tensor`: Scalar mean absolute error.
    """
    return torch.mean(torch.abs(pred - target))


def rel_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean relative error between `pred` and `target`.

    ## Description
    Returns the mean absolute difference normalized by the mean absolute value of
    the target tensor.

    ## Arguments
    `pred` (`torch.Tensor`): Predicted tensor.
    `target` (`torch.Tensor`): Ground truth tensor of the same shape as `pred`.

    ## Returns
    `torch.Tensor`: Scalar mean relative error.
    """
    return torch.mean(torch.abs(pred - target)) / torch.mean(torch.abs(target))


def rmse_error(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute root-mean-square error between `pred` and `target`.

    ## Description
    Returns the RMSE with a small additive epsilon inside the square root to avoid
    numerical issues near zero.

    ## Arguments
    `pred` (`torch.Tensor`): Predicted tensor.
    `target` (`torch.Tensor`): Ground truth tensor of the same shape as `pred`.
    `eps` (`float`, default: `1e-16`): Small constant added inside the square root
    for numerical stability.

    ## Returns
    `torch.Tensor`: Scalar RMSE value.
    """
    diff: torch.Tensor = torch.abs(pred - target)
    return torch.sqrt(diff.pow(2).mean() + eps)


##################################################
def absolute_error(pred: torch.Tensor, target: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    r"""Compute the batched absolute $L^p$ error between `pred` and `target`.

    ## Description
    For each sample in the batch (first axis), computes the $L^p$ norm of the
    element-wise absolute difference, averaged over all non-batch dimensions.

    ## Arguments
    `pred` (`torch.Tensor`): Predicted tensor of shape `(B, ...)`.
    `target` (`torch.Tensor`): Ground truth tensor of the same shape as `pred`.
    `p` (`float`, default: `2.0`): Order of the $L^p$ norm.

    ## Returns
    `torch.Tensor`: Per-sample absolute error tensor of shape `(B,)`.
    """
    assert pred.shape == target.shape
    ndim: int         = pred.ndim
    dims: tuple       = tuple(range(1, ndim))
    return ((pred - target).abs().pow(p).mean(dim=dims)).pow(1 / p)


def relative_error(pred: torch.Tensor, target: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    r"""Compute the batched relative $L^p$ error between `pred` and `target`.

    ## Description
    For each sample in the batch (first axis), computes the ratio of the $L^p$ norm
    of the absolute difference to the $L^p$ norm of the target, both averaged over
    all non-batch dimensions.

    ## Arguments
    `pred` (`torch.Tensor`): Predicted tensor of shape `(B, ...)`.
    `target` (`torch.Tensor`): Ground truth tensor of the same shape as `pred`.
    `p` (`float`, default: `2.0`): Order of the $L^p$ norm.

    ## Returns
    `torch.Tensor`: Per-sample relative error tensor of shape `(B,)`.
    """
    assert pred.shape == target.shape
    ndim: int         = pred.ndim
    dims: tuple       = tuple(range(1, ndim))
    numerator:   torch.Tensor = ((pred - target).abs().pow(p).mean(dim=dims)).pow(1 / p)
    denominator: torch.Tensor = target.abs().pow(p).mean(dim=dims).pow(1 / p)
    return numerator / denominator


##################################################
def compute_mass_density(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the mass density from a distribution function.

    ## Description
    Approximates $\\rho(t) = \\int f(t, v)\\,dv$ by summing over all velocity axes and
    multiplying by the volume element `dv`.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(N_t, *repeat(N_v, d))`.
    `v` (`torch.Tensor`): Velocity grid tensor of shape `(N_v^d, d)` or
    `(N_t, N_v^d, d+1)`. In the latter case the zeroth column (time) is ignored.

    ## Returns
    `torch.Tensor`: Mass density tensor of shape `(N_t,)`.
    """
    ndim:      int   = f.ndim
    dimension: int   = ndim - 1
    assert v.size(-1) in [dimension, ndim]
    if v.size(-1) == ndim:
        v = v[..., 1:]
    dv: float = (v[1] - v[0]).norm().pow(dimension).item()
    mass_density: torch.Tensor = f.sum(dim=tuple(range(1, ndim))) * dv
    return mass_density


def compute_bulk_velocity(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the bulk velocity from a distribution function.

    ## Description
    Approximates $u(t) = \\int v\\,f(t, v)\\,dv$ by a weighted sum over velocity axes.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(N_t, *repeat(N_v, d))`.
    `v` (`torch.Tensor`): Velocity grid tensor of shape `(N_v^d, d)` or
    `(N_t, N_v^d, d+1)`. In the latter case the zeroth column (time) is ignored.

    ## Returns
    `torch.Tensor`: Bulk velocity tensor of shape `(N_t, d)`.
    """
    ndim:      int   = f.ndim
    dimension: int   = ndim - 1
    assert v.size(-1) in [dimension, ndim]
    if v.size(-1) == ndim:
        v = v[..., 1:]
    dv: float = (v[1] - v[0]).norm().pow(dimension).item()
    v = v.reshape(*f.shape[1:], dimension)
    f = f.unsqueeze(-1)
    mass_velocity: torch.Tensor = (f * v).sum(dim=tuple(range(1, ndim))) * dv
    return mass_velocity


def compute_energy_density(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the kinetic energy density from a distribution function.

    ## Description
    Approximates $E(t) = \\frac{1}{2}\\int |v|^2\\,f(t, v)\\,dv$ by a weighted sum
    over all velocity axes.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(N_t, *repeat(N_v, d))`.
    `v` (`torch.Tensor`): Velocity grid tensor of shape `(N_v^d, d)` or
    `(N_t, N_v^d, d+1)`. In the latter case the zeroth column (time) is ignored.

    ## Returns
    `torch.Tensor`: Energy density tensor of shape `(N_t,)`.
    """
    ndim:      int   = f.ndim
    dimension: int   = ndim - 1
    assert v.size(-1) in [dimension, ndim], f"{ndim=}"
    if v.size(-1) == ndim:
        v = v[..., 1:]
    dv: float = (v[1] - v[0]).norm().pow(dimension).item()
    v = v.reshape(*f.shape[1:], dimension)
    speed_squared:   torch.Tensor = v.pow(2).sum(dim=-1)
    energy_density:  torch.Tensor = (f * speed_squared).sum(dim=tuple(range(1, ndim))) * dv / 2
    return energy_density


def compute_entropy_density(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the entropy density from a distribution function.

    ## Description
    Approximates $H(t) = \\int f(t, v)\\log f(t, v)\\,dv$ by a weighted sum over
    velocity axes. If `f` has negative values, a warning is emitted and the negative
    entries are clipped before the logarithm is applied.

    ## Arguments
    `f` (`torch.Tensor`): Distribution tensor of shape `(N_t, *repeat(N_v, d))`.
    `v` (`torch.Tensor`): Velocity grid tensor of shape `(N_v^d, d)` or
    `(N_t, N_v^d, d+1)`. In the latter case the zeroth column (time) is ignored.

    ## Returns
    `torch.Tensor`: Entropy density tensor of shape `(N_t,)`.
    """
    import  warnings
    ndim:      int   = f.ndim
    dimension: int   = ndim - 1
    assert v.size(-1) in [dimension, ndim]
    if v.size(-1) == ndim:
        v = v[..., 1:]
    dv:              float         = (v[1] - v[0]).norm().pow(dimension).item()
    entropy_density: torch.Tensor
    _min: float = f.min().item()
    if _min > 0.0:
        entropy_density = (f * f.log()).sum(dim=tuple(range(1, ndim))) * dv
    else:
        warnings.warn(
            f"Encountered a distribution function with negative values (min={_min}). "
            f"Clipping to compute entropy density."
        )
        f_safe:         torch.Tensor = torch.where(f > -_min, f, -_min + 1e-16)
        entropy_density              = (f_safe * f_safe.log()).sum(dim=tuple(range(1, ndim))) * dv
    return entropy_density


##################################################
##################################################
class AverageMeter():
    """Online running-average tracker.

    ## Description
    Maintains a running sum and count to compute the cumulative mean of a sequence
    of scalar values. Useful for aggregating per-iteration loss values during training.
    Adapted from the PyTorch ImageNet example:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262

    ## Arguments
    None.

    ## Returns
    `None`: None.
    """
    def __init__(self) -> Self:
        self.reset()
        return

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.last_value: float = 0.0
        self.mean:       float = 0.0
        self.sum:        float = 0.0
        self.count:      int   = 0
        return

    def update(self, value: object, n: int = 1) -> None:
        """Update the running mean with `value` counted `n` times.

        ## Description
        Adds `value * n` to the running sum and increments the count by `n`,
        then recomputes the mean.

        ## Arguments
        `value` (`object`): Scalar value to accumulate.
        `n` (`int`, default: `1`): Weight (multiplicity) of this update.

        ## Returns
        `None`: None.
        """
        self.last_value  = value
        self.sum        += value * n
        self.count      += n
        self.mean        = self.sum / self.count
        return


##################################################
##################################################
# End of file