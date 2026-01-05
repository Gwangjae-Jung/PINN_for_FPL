from    typing      import  Self
import  torch


__all__: list[str] = [
    'abs_error', 'rel_error', 'rmse_error',
    'absolute_error', 'relative_error',
    'compute_mass_density', 'compute_bulk_velocity', 'compute_energy_density', 'compute_entropy_density',
    'AverageMeter',
]


EPS = 1e-16
def abs_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred-target))
def rel_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred-target)) / torch.mean(torch.abs(target))
def rmse_error(pred: torch.Tensor, target: torch.Tensor, eps: float=EPS) -> torch.Tensor:
    diff = torch.abs(pred-target)
    return torch.sqrt(diff.pow(2).mean()+eps)


def absolute_error(pred: torch.Tensor, target: torch.Tensor, p: float=2.0) -> torch.Tensor:
    r"""Returns the absolute L^p error for batched predictions and targets."""
    assert pred.shape==target.shape
    ndim = pred.ndim
    dims = tuple(range(1, ndim))
    return ((pred-target).abs().pow(p).mean(dim=dims)).pow(1/p)
def relative_error(pred: torch.Tensor, target: torch.Tensor, p: float=2.0) -> torch.Tensor:
    r"""Returns the relative L^p error for batched predictions and targets."""
    assert pred.shape==target.shape
    ndim = pred.ndim
    dims = tuple(range(1, ndim))
    numerator   = ((pred-target).abs().pow(p).mean(dim=dims)).pow(1/p)
    denominator = target.abs().pow(p).mean(dim=dims).pow(1/p)
    return numerator/denominator


def compute_mass_density(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Given a spatio-temporal tensor of distribution functions, this function computes the mass density.
    
    Arguments:
        `f`: A tensor of shape `(N_t, *repeat(N_v, d))`, where `N_t` is the number of time steps, `N_v` is the number of velocity discretization points along each axis, and `d` is the dimension.
        `v`: A tensor of shape `(N_v**d, d)` or `(N_t, N_v**d, d+1)`, representing the velocity grid. As for the latter, the zeroth column is ignored.
    
    Returns:
        A tensor of shape `(N_t,)`, representing the mass density at each time step.
    """
    ndim = f.ndim
    dimension = ndim-1
    assert v.size(-1) in [dimension, ndim]
    if v.size(-1)==ndim:    v = v[..., 1:]
    dv = (v[1]-v[0]).norm().pow(dimension).item()
    mass_density = f.sum(dim=tuple(range(1, ndim))) * dv
    return mass_density


def compute_bulk_velocity(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Given a spatio-temporal tensor of distribution functions, this function computes the bulk velocity.
    
    Arguments:
        `f`: A tensor of shape `(N_t, *repeat(N_v, d))`, where `N_t` is the number of time steps, `N_v` is the number of velocity discretization points along each axis, and `d` is the dimension.
        `v`: A tensor of shape `(N_v**d, d)` or `(N_t, N_v**d, d+1)`, representing the velocity grid. As for the latter, the zeroth column is ignored.
    
    Returns:
        A tensor of shape `(N_t, d)`, representing the bulk velocity at each time step.
    """
    ndim = f.ndim
    dimension = ndim-1
    assert v.size(-1) in [dimension, ndim]
    if v.size(-1)==ndim:    v = v[..., 1:]
    dv = (v[1]-v[0]).norm().pow(dimension).item()
    v = v.reshape(*f.shape[1:], dimension)
    f = f.unsqueeze(-1)
    mass_velocity = (f*v).sum(dim=tuple(range(1, ndim))) * dv
    return mass_velocity


def compute_energy_density(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Given a spatio-temporal tensor of distribution functions, this function computes the energy density.
    
    Arguments:
        `f`: A tensor of shape `(N_t, *repeat(N_v, d))`, where `N_t` is the number of time steps, `N_v` is the number of velocity discretization points along each axis, and `d` is the dimension.
        `v`: A tensor of shape `(N_v**d, d)` or `(N_t, N_v**d, d+1)`, representing the velocity grid. As for the latter, the zeroth column is ignored.
    
    Returns:
        A tensor of shape `(N_t,)`, representing the energy density at each time step.
    """
    ndim = f.ndim
    dimension = ndim-1
    assert v.size(-1) in [dimension, ndim], f"{ndim=}"
    if v.size(-1)==ndim:    v = v[..., 1:]
    dv = (v[1]-v[0]).norm().pow(dimension).item()
    v = v.reshape(*f.shape[1:], dimension)
    speed_squared = v.pow(2).sum(dim=-1)
    energy_density = (f*speed_squared).sum(dim=tuple(range(1, ndim))) * dv / 2
    return energy_density


def compute_entropy_density(f: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Given a spatio-temporal tensor of distribution functions, this function computes the entropy density.
    
    Arguments:
        `f`: A tensor of shape `(N_t, *repeat(N_v, d))`, where `N_t` is the number of time steps, `N_v` is the number of velocity discretization points along each axis, and `d` is the dimension.
        `v`: A tensor of shape `(N_v**d, d)` or `(N_t, N_v**d, d+1)`, representing the velocity grid. As for the latter, the zeroth column is ignored.
    
    Returns:
        A tensor of shape `(N_t,)`, representing the entropy density at each time step.
    """
    import  warnings
    ndim = f.ndim
    dimension = ndim-1
    assert v.size(-1) in [dimension, ndim]
    if v.size(-1)==ndim:    v = v[..., 1:]
    dv = (v[1]-v[0]).norm().pow(dimension).item()
    entropy_density: torch.Tensor
    _min = f.min().item()
    if _min > 0.0:
        entropy_density = (f*f.log()).sum(dim=tuple(range(1, ndim))) * dv
    else:
        warnings.warn(f"Encountered a distribution function with negative values (min={_min}). Clipping to compute entropy density.")
        f_safe = torch.where(f > -_min, f, -_min + 1e-16)
        entropy_density = (f_safe*f_safe.log()).sum(dim=tuple(range(1, ndim))) * dv
    return entropy_density


class AverageMeter():
    """Computes and stores the average and current value. Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self) -> Self:
        self.reset()
        return

    def reset(self) -> None:
        self.last_value = 0.0
        self.mean       = 0.0
        self.sum        = 0.0
        self.count      = 0
        return

    def update(self, value: object, n: int=1) -> None:
        self.last_value = value
        self.sum        += value * n
        self.count      += n
        self.mean       = self.sum/self.count
        return


##################################################
##################################################
# End of file