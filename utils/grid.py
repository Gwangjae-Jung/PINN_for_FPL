from   typing import Literal, Optional, Sequence, Union
import torch


__all__: list[str] = ['space_grid_1d', 'space_grid', 'GridGenerator']


##################################################
##################################################
class GridGenerator():
    """Spatiotemporal coordinate grid generator for PINN training.

    ## Description
    Generates temporal and velocity Cartesian product coordinate grids for training
    physics-informed neural networks on the Fokker-Planck-Landau equation. The time
    grid spans `[0, max_t]` with `num_t` uniform points, and the velocity grid covers
    `[-max_v, max_v]^dim` with `num_v` cell-centered points per axis.

    ## Arguments
    `dim` (`int`): Velocity dimension.
    `num_t` (`int`): Number of time discretization points.
    `max_t` (`float`): Maximum simulation time.
    `num_v` (`int`): Number of velocity grid points along each velocity axis.
    `max_v` (`float`): Velocity cutoff bound defining `[-max_v, max_v]^dim`.
    `num_v_init` (`Optional[int]`, default: `None`): Velocity resolution for the
    initial condition grid at `t=0`. Defaults to `num_v` if not given.
    `device` (`Optional[torch.device]`, default: `None`): Computing device for
    generated tensors. Defaults to `torch.get_default_device()`.

    ## Returns
    `None`: None.
    """
    def __init__(
            self,
            dim:        int,
            num_t:      int,
            max_t:      float,
            num_v:      int,
            max_v:      float,
            num_v_init: Optional[int]          = None,
            device:     Optional[torch.device] = None,
        ) -> None:
        if device is None:
            device = torch.get_default_device()

        self.__dim:        int          = dim
        self.__num_t:      int          = num_t
        self.__max_t:      float        = max_t
        self.__num_v:      int          = num_v
        self.__max_v:      float        = max_v
        self.__num_v_init: int          = num_v_init if num_v_init is not None else num_v
        self.__device:     torch.device = device

        # Precompute the fixed 1-D grids for time and velocity
        self.__t:    torch.Tensor = torch.linspace(0.0, self.__max_t, self.__num_t, device=device)
        self.__v_1d: torch.Tensor = space_grid_1d(num_v, max_v, device=device)
        self.__v0_1d: torch.Tensor = space_grid_1d(self.__num_v_init, max_v, device=device)

        # Precompute the full spatiotemporal Cartesian products
        self.__tv:  torch.Tensor = torch.cartesian_prod(self.__t, *(self.__v_1d for _ in range(self.__dim)))
        self.__t0v: torch.Tensor = torch.cartesian_prod(
            torch.tensor([0.0], device=device),
            *(self.__v0_1d for _ in range(self.__dim)),
        )
        return

    @property
    def dim(self) -> int:
        """Velocity-space dimension."""
        return self.__dim

    @property
    def delta_t(self) -> float:
        """Uniform time step size."""
        return self.__max_t / (self.__num_t - 1)

    @property
    def delta_v(self) -> float:
        """Uniform velocity grid spacing along each axis."""
        return (2.0 * self.__max_v) / self.__num_v

    @property
    def dv(self) -> float:
        """Volume element of the velocity grid cell."""
        return self.delta_v ** self.__dim

    @property
    def num_t(self) -> int:
        """Number of time grid points."""
        return self.__num_t

    @property
    def num_v(self) -> int:
        """Number of velocity grid points along each axis."""
        return self.__num_v

    @property
    def tv(self) -> torch.Tensor:
        """Fixed Cartesian product grid of shape `(num_t * num_v^dim, 1+dim)`."""
        return self.__tv

    @property
    def t0v(self) -> torch.Tensor:
        """Cartesian product grid at `t=0` of shape `(num_v_init^dim, 1+dim)`."""
        return self.__t0v

    def sample_tv(self, is_time_fixed: bool) -> torch.Tensor:
        """Sample collocation points in time-velocity space.

        ## Description
        Draws collocation points from the Cartesian product of the time domain and
        velocity grid, either with a fixed equispaced time grid or randomly sampled
        (sorted) time coordinates.

        ## Arguments
        `is_time_fixed` (`bool`): If `True`, uses the precomputed uniform time grid;
        if `False`, draws `num_t` random sorted time values from `[0, max_t]`.

        ## Returns
        `torch.Tensor`: Collocation tensor of shape `(num_t * num_v^dim, 1+dim)`.
        """
        grid_t: torch.Tensor
        if is_time_fixed:
            grid_t = self.__t
        else:
            grid_t = self.__max_t * torch.sort(torch.rand(self.__num_t, device=self.__device))[0]
        return torch.cartesian_prod(grid_t, *(self.__v_1d for _ in range(self.__dim)))


##################################################
def space_grid_1d(
        res_x:  int,
        max_x:  float,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Generate a 1D cell-centered uniform velocity grid.

    ## Description
    Constructs a 1-dimensional cell-centered grid of `res_x` points within
    `[-max_x, max_x]`, excluding the boundary endpoints. The grid spacing is
    `dx = 2 * max_x / res_x`, so the leftmost point is `-max_x + dx/2` and the
    rightmost is `max_x - dx/2`.

    ## Arguments
    `res_x` (`int`): Number of grid points.
    `max_x` (`float`): Cutoff boundary coordinate.
    `device` (`Optional[torch.device]`, default: `None`): PyTorch device on which
    the tensor is created. Defaults to `torch.get_default_device()`.

    ## Returns
    `torch.Tensor`: 1D tensor of cell-centered grid coordinates of shape `(res_x,)`.
    """
    if device is None:
        device = torch.get_default_device()
    dx:      float = (2.0 * max_x) / res_x
    half_dx: float = max_x - dx / 2.0
    return torch.linspace(-half_dx, half_dx, res_x, device=device)


##################################################
def space_grid(
        dimension:    int,
        num_grids:    Union[int, Sequence[int]],
        max_values:   Union[float, Sequence[float]],
        min_values:   Optional[Union[float, Sequence[float]]]            = None,
        where_closed: Optional[Literal['both', 'left', 'right', 'none']] = None,
        dtype:        Optional[torch.dtype]                              = None,
        device:       Optional[torch.device]                             = None,
    ) -> torch.Tensor:
    """Generate a multidimensional Cartesian spatial grid.

    ## Description
    Constructs a multidimensional Cartesian coordinate grid with user-specified
    resolution, domain bounds, and boundary closure conventions. The last dimension
    of the returned tensor indexes the spatial coordinate.

    ## Arguments
    `dimension` (`int`): Dimension of the spatial/velocity domain.
    `num_grids` (`Union[int, Sequence[int]]`): Number of discretization points along
    each axis. A single `int` applies the same count to all axes.
    `max_values` (`Union[float, Sequence[float]]`): Maximum coordinate value along
    each axis.
    `min_values` (`Optional[Union[float, Sequence[float]]]`, default: `None`): Minimum
    coordinate value along each axis. Defaults to `-max_values` per axis.
    `where_closed` (`Optional[Literal['both', 'left', 'right', 'none']]`, default:
    `None`): Boundary closure rule. `'both'` includes both endpoints, `'left'` /
    `'right'` includes only one, and `'none'` (default) uses cell-centered points.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Data type of the generated
    tensor.
    `device` (`Optional[torch.device]`, default: `None`): Computing device.

    ## Returns
    `torch.Tensor`: Coordinate grid tensor of shape `(*num_grids, dimension)`.
    """
    if isinstance(num_grids, int):
        num_list: tuple[int, ...] = tuple(num_grids for _ in range(dimension))
    else:
        num_list = tuple(num_grids)

    if isinstance(max_values, (int, float)):
        max_list: tuple[float, ...] = tuple(float(max_values) for _ in range(dimension))
    else:
        max_list = tuple(float(v) for v in max_values)

    if min_values is None:
        min_list: tuple[float, ...] = tuple(-m for m in max_list)
    elif isinstance(min_values, (int, float)):
        min_list = tuple(float(min_values) for _ in range(dimension))
    else:
        min_list = tuple(float(v) for v in min_values)

    closure_mode: str = where_closed if where_closed is not None else 'none'

    axis_grids: list[torch.Tensor] = []
    for d in range(dimension):
        n_d:   int   = num_list[d]
        max_d: float = max_list[d]
        min_d: float = min_list[d]
        dx_d:  float = (max_d - min_d) / n_d

        if closure_mode == 'both':
            left:  float = min_d
            right: float = max_d
        elif closure_mode == 'left':
            left  = min_d
            right = max_d - dx_d
        elif closure_mode == 'right':
            left  = min_d + dx_d
            right = max_d
        elif closure_mode == 'none':
            left  = min_d + dx_d / 2.0
            right = max_d - dx_d / 2.0
        else:
            raise ValueError(f"Unknown closure mode: {closure_mode}")

        axis_grids.append(torch.linspace(left, right, n_d, dtype=dtype, device=device))

    return torch.stack(torch.meshgrid(*axis_grids, indexing='ij'), dim=-1)


##################################################
##################################################
# End of file