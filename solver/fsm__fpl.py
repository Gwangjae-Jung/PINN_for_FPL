from   typing        import Callable, Optional, Sequence
import torch
from   torch.special import bessel_j0 as j0
from   scipy.special import roots_legendre
from   .runge_kutta  import one_step_RK4_classic


FFT_NORM:   str   = 'forward'
LAMBDA_FPL: float = 0.5

__all__: list[str] = ['FastSM_Landau', 'FastSM_Landau_VHS']


class FastSM_Landau():
    """Fast spectral method base solver for the Fokker-Planck-Landau equation.

    ## Description
    Provides fundamental algorithms, data structures, and frequency-space convolution
    routines for solving the Fokker-Planck-Landau (FPL) collision operator using the
    fast Fourier spectral method.
    Reference: L. Pareschi, G. Russo, G. Toscani, J. Comput. Phys., 165(1):216-236, 2000.

    ## Arguments
    `dimension` (`int`): Dimension of the velocity domain (`2` or `3`).
    `v_num_grid` (`int`): Number of grid points along each velocity dimension.
    `v_max` (`float`): Maximum velocity value defining truncation domain `[-v_max, v_max]^d`.
    `x_num_grid` (`Optional[int]`, default: `None`): Spatial grid resolution (unused for
    homogeneous FPL).
    `x_max` (`Optional[float]`, default: `None`): Spatial domain maximum (unused for
    homogeneous FPL).
    `quad_order_uniform` (`Optional[int]`, default: `None`): Quadrature order for angular
    integration.
    `quad_order_legendre` (`Optional[int]`, default: `None`): Quadrature order for radial
    Legendre integration.
    `dtype` (`Optional[torch.dtype]`, default: `None`): PyTorch tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `None`: None.
    """
    def __init__(
            self,
            dimension:           int,
            v_num_grid:          int,
            v_max:               float,
            x_num_grid:          Optional[int]          = None,
            x_max:               Optional[float]        = None,
            quad_order_uniform:  Optional[int]          = None,
            quad_order_legendre: Optional[int]          = None,
            dtype:               Optional[torch.dtype]  = None,
            device:              Optional[torch.device] = None,
        ) -> None:
        if dimension not in (2, 3):
            raise NotImplementedError(f"Fast spectral method is implemented only for dimensions 2 and 3, got dimension={dimension}.")

        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        self._dimension:           int                    = dimension
        self._v_num_grid:          int                    = v_num_grid
        self._v_max:               float                  = v_max
        self._delta_v:             float                  = (2.0 * v_max) / v_num_grid
        self._x_num_grid:          Optional[int]          = x_num_grid
        self._x_max:               Optional[float]        = x_max
        self._dtype:               torch.dtype            = dtype
        self._device:              torch.device           = device

        self._quad_order_uniform:  int                    = quad_order_uniform if isinstance(quad_order_uniform, int) else v_num_grid
        self._quad_order_legendre: int                    = quad_order_legendre if isinstance(quad_order_legendre, int) else v_num_grid

        self._fpl_character_1:     Optional[torch.Tensor] = None
        self._fpl_character_2:     Optional[torch.Tensor] = None
        self._fpl_gain_tensor_1:   Optional[torch.Tensor] = None
        self._fpl_gain_tensor_2:   Optional[torch.Tensor] = None
        self._fpl_loss_tensor:     Optional[torch.Tensor] = None

        raw_freqs: torch.Tensor = freq_tensor(dimension, v_num_grid, keepdim=True, dtype=torch.long, device=device)
        self._freqs: torch.Tensor = raw_freqs.reshape(
            1,
            *(1 for _ in range(dimension)),
            *(v_num_grid for _ in range(dimension)),
            dimension,
        )
        self._freq_norms: torch.Tensor = torch.norm(self._freqs.type(dtype), p=2, dim=-1, keepdim=True)
        return

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def v_num_grid(self) -> int:
        return self._v_num_grid

    @property
    def v_max(self) -> float:
        return self._v_max

    @property
    def v_ratio(self) -> float:
        return float(torch.pi / self._v_max)

    @property
    def v_radius(self) -> float:
        return self._v_max * LAMBDA_FPL

    @property
    def v_axes(self) -> tuple[int, ...]:
        return tuple(range(1 + self._dimension, 1 + 2 * self._dimension))

    @property
    def internal_fft_norm(self) -> str:
        return FFT_NORM

    @property
    def dtype_and_device(self) -> dict[str, object]:
        return {'dtype': self._dtype, 'device': self._device}

    @property
    def freqs(self) -> torch.Tensor:
        return self._freqs

    @property
    def freq_norms(self) -> torch.Tensor:
        return self._freq_norms

    @property
    def fpl_base_shape(self) -> tuple[int, ...]:
        return tuple([1, *(1 for _ in range(self._dimension)), *(self._v_num_grid for _ in range(self._dimension)), 1])

    @property
    def fpl_gain_tensor_1(self) -> Optional[torch.Tensor]:
        return self._fpl_gain_tensor_1

    @property
    def fpl_gain_tensor_2(self) -> Optional[torch.Tensor]:
        return self._fpl_gain_tensor_2

    @property
    def fpl_loss_tensor(self) -> Optional[torch.Tensor]:
        return self._fpl_loss_tensor

    def precompute(self) -> None:
        """Precompute characteristic functions and convolution weights.

        ## Description
        Executes dimensionality-specific precomputations of characteristic weights, gain
        tensors, and loss tensors required by the fast spectral method.

        ## Arguments
        None.

        ## Returns
        `None`: None.
        """
        if self._dimension in (2, 3):
            getattr(self, f"_precompute_fpl_character_1_{self._dimension}D")()
            getattr(self, f"_precompute_fpl_character_2_{self._dimension}D")()
            getattr(self, f"_precompute_fpl_gain_tensors_{self._dimension}D")()
            self._precompute_fpl_loss_tensor()
        return

    def _precompute_fpl_character_1_2D(self) -> None:
        pass

    def _precompute_fpl_character_2_2D(self) -> None:
        pass

    def _precompute_fpl_gain_tensors_2D(self) -> None:
        pass

    def _precompute_fpl_character_1_3D(self) -> None:
        pass

    def _precompute_fpl_character_2_3D(self) -> None:
        pass

    def _precompute_fpl_gain_tensors_3D(self) -> None:
        pass

    def _precompute_fpl_loss_tensor(self) -> None:
        freq_norms_sq: torch.Tensor = torch.sum(self._freqs**2, dim=-1, keepdim=True).type(self._dtype)
        positive: torch.Tensor = freq_norms_sq * self._fpl_gain_tensor_1
        freqs_typed: torch.Tensor = self._freqs.type(self._dtype)
        negative: torch.Tensor = torch.einsum("...i, ...vij, ...j -> ...v", freqs_typed, self._fpl_gain_tensor_2, freqs_typed)
        self._fpl_loss_tensor = positive - negative
        return

    def compute_gain_fft(
            self,
            _placeholder_t: float,
            fft_curr:       torch.Tensor,
        ) -> torch.Tensor:
        """Compute the gain term of the FPL collision operator in Fourier space.

        ## Description
        Evaluates the gain term of the Fokker-Planck-Landau operator in the Fourier
        frequency domain via spectral convolution.

        ## Arguments
        `_placeholder_t` (`float`): Current temporal coordinate (unused placeholder for
        ODE solver compatibility).
        `fft_curr` (`torch.Tensor`): Current Fourier transform of the distribution
        function.

        ## Returns
        `torch.Tensor`: Evaluated gain term in Fourier space.
        """
        if self._dimension in (2, 3):
            gain_fft_positive: torch.Tensor = convolve_freqs(
                (self._freq_norms**2) * fft_curr,
                self._fpl_gain_tensor_1 * fft_curr,
                dim = self.v_axes,
            )
            gain_fft_negative: torch.Tensor = torch.sum(
                torch.stack(
                    [
                        convolve_freqs(
                            self._freqs[..., [i]] * self._freqs[..., [j]] * fft_curr,
                            self._fpl_gain_tensor_2[..., i, j] * fft_curr,
                            dim = self.v_axes,
                        )
                        for i in range(self._dimension) for j in range(self._dimension)
                    ],
                    dim = -1,
                ),
                dim = -1,
            )
            return -(gain_fft_positive - gain_fft_negative)
        return torch.zeros_like(fft_curr)

    def compute_loss_fft(
            self,
            _placeholder_t: float,
            fft_curr:       torch.Tensor,
        ) -> torch.Tensor:
        """Compute the loss term of the FPL collision operator in Fourier space.

        ## Description
        Evaluates the loss term of the Fokker-Planck-Landau operator in the Fourier
        frequency domain via spectral convolution.

        ## Arguments
        `_placeholder_t` (`float`): Current temporal coordinate (unused placeholder for
        ODE solver compatibility).
        `fft_curr` (`torch.Tensor`): Current Fourier transform of the distribution
        function.

        ## Returns
        `torch.Tensor`: Evaluated loss term in Fourier space.
        """
        if self._dimension in (2, 3):
            loss_fft: torch.Tensor = convolve_freqs(
                fft_curr,
                fft_curr * self._fpl_loss_tensor,
                dim = self.v_axes,
            )
            return -loss_fft
        return torch.zeros_like(fft_curr)

    def compute_fft(
            self,
            _placeholder_t: Optional[float],
            fft_curr:       torch.Tensor,
        ) -> torch.Tensor:
        """Compute the net collision operator in Fourier space.

        ## Description
        Returns `gain_fft - loss_fft` in the Fourier frequency domain.

        ## Arguments
        `_placeholder_t` (`Optional[float]`): Current temporal coordinate.
        `fft_curr` (`torch.Tensor`): Current Fourier transform of the distribution
        function.

        ## Returns
        `torch.Tensor`: Total collision operator evaluated in Fourier space.
        """
        return self.compute_gain_fft(0.0, fft_curr) - self.compute_loss_fft(0.0, fft_curr)

    def solve(
            self,
            t_init:  float,
            t_final: float,
            delta_t: float,
            f_init:  torch.Tensor,
            RK_fcn:  Callable[
                [float, torch.Tensor, float, Callable[[float, torch.Tensor], torch.Tensor]],
                torch.Tensor,
            ] = one_step_RK4_classic,
        ) -> torch.Tensor:
        """Numerically integrate the Fokker-Planck-Landau equation forward in time.

        ## Description
        Advances the initial distribution `f_init` from `t_init` to `t_final` with step
        size `delta_t` using the fast spectral method and the specified Runge-Kutta
        integrator.

        ## Arguments
        `t_init` (`float`): Initial time instant.
        `t_final` (`float`): Final time instant.
        `delta_t` (`float`): Step size for temporal discretization.
        `f_init` (`torch.Tensor`): Initial distribution function in physical space.
        `RK_fcn` (`Callable[...]`, default: `one_step_RK4_classic`): Single-step
        Runge-Kutta integration function conforming to the `one_step_*` interface.

        ## Returns
        `torch.Tensor`: Trajectory tensor of shape
        `(batch, num_time_steps+1, *space, *velocity, channel)`.
        """
        try:
            from tqdm import trange
        except ImportError:
            trange = range
        num_updates: int = int((t_final - t_init) / delta_t + 0.1)
        ret: list[torch.Tensor] = [f_init]
        f_fft_curr: torch.Tensor = torch.fft.fftn(f_init, dim=self.v_axes, norm=self.internal_fft_norm)
        for _ in trange(num_updates):
            f_fft_curr = RK_fcn(0.0, f_fft_curr, delta_t, self.compute_fft)
            ret.append(torch.real(torch.fft.ifftn(f_fft_curr, dim=self.v_axes, norm=self.internal_fft_norm)))
        return torch.stack(ret, dim=1)

    def to(self, device: torch.device) -> 'FastSM_Landau':
        """Move all solver internal tensors to the target device.

        ## Description
        Transfers all precomputed tensors and coordinate grids to the specified
        PyTorch device.

        ## Arguments
        `device` (`torch.device`): Target computing device.

        ## Returns
        `FastSM_Landau`: Self reference with tensors moved to `device`.
        """
        self._device = device
        self._freqs = self._freqs.to(device)
        self._freq_norms = self._freq_norms.to(device)
        if self._fpl_character_1 is not None:
            self._fpl_character_1 = self._fpl_character_1.to(device)
        if self._fpl_character_2 is not None:
            self._fpl_character_2 = self._fpl_character_2.to(device)
        if self._fpl_gain_tensor_1 is not None:
            self._fpl_gain_tensor_1 = self._fpl_gain_tensor_1.to(device)
        if self._fpl_gain_tensor_2 is not None:
            self._fpl_gain_tensor_2 = self._fpl_gain_tensor_2.to(device)
        if self._fpl_loss_tensor is not None:
            self._fpl_loss_tensor = self._fpl_loss_tensor.to(device)
        return self


class FastSM_Landau_VHS(FastSM_Landau):
    """Fast spectral method solver for the FPL equation with the VHS interaction kernel.

    ## Description
    Extends `FastSM_Landau` to implement precomputation and evaluation of the Variable
    Hard Sphere (VHS) collision kernel, computing collision integrals in
    $O(N^d \\log N)$ complexity.
    Reference: L. Pareschi, G. Russo, G. Toscani, J. Comput. Phys., 165(1):216-236, 2000.

    ## Arguments
    `dimension` (`int`): Dimension of the velocity domain (`2` or `3`).
    `v_num_grid` (`int`): Number of grid points along each velocity dimension.
    `v_max` (`float`): Maximum velocity value defining domain `[-v_max, v_max]^d`.
    `x_num_grid` (`Optional[int]`, default: `None`): Spatial grid resolution.
    `x_max` (`Optional[float]`, default: `None`): Spatial domain maximum coordinate.
    `vhs_coeff` (`Optional[float]`, default: `None`): Collision kernel coefficient.
    `vhs_alpha` (`Optional[float]`, default: `None`): Exponent of relative velocity in
    the collision kernel.
    `quad_order_uniform` (`Optional[int]`, default: `None`): Quadrature order for
    angular integration.
    `quad_order_legendre` (`Optional[int]`, default: `None`): Quadrature order for
    radial Legendre integration.
    `dtype` (`Optional[torch.dtype]`, default: `None`): PyTorch tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `None`: None.
    """
    def __init__(
            self,
            dimension:           int,
            v_num_grid:          int,
            v_max:               float,
            x_num_grid:          Optional[int]          = None,
            x_max:               Optional[float]        = None,
            vhs_coeff:           Optional[float]        = None,
            vhs_alpha:           Optional[float]        = None,
            quad_order_uniform:  Optional[int]          = None,
            quad_order_legendre: Optional[int]          = None,
            dtype:               Optional[torch.dtype]  = None,
            device:              Optional[torch.device] = None,
        ) -> None:
        super().__init__(
            dimension           = dimension,
            v_num_grid          = v_num_grid,
            v_max               = v_max,
            x_num_grid          = x_num_grid,
            x_max               = x_max,
            quad_order_uniform  = quad_order_uniform,
            quad_order_legendre = quad_order_legendre,
            dtype               = dtype,
            device              = device,
        )
        self._vhs_coeff: float = float(vhs_coeff) if vhs_coeff is not None else 1.0
        self._vhs_alpha: float = float(vhs_alpha) if vhs_alpha is not None else 0.0
        self.precompute()
        return

    @property
    def vhs_coeff(self) -> float:
        return self._vhs_coeff

    @property
    def vhs_alpha(self) -> float:
        return self._vhs_alpha

    def _precompute_fpl_character_1_2D(self) -> None:
        len_norms: int = int(self._dimension * (self._v_num_grid // 2)**2 + 1)
        norms: torch.Tensor = torch.sqrt(torch.arange(len_norms, **self.dtype_and_device))[:, None]
        r, w = roots_legendre_shifted(self._quad_order_legendre, 0.0, torch.pi, **self.dtype_and_device)
        r = r.reshape(1, -1)
        w = w.reshape(1, -1)

        scale: float = float((2.0 * torch.pi * self._vhs_coeff) / (self.v_ratio**(2.0 + self._vhs_alpha)))
        power_r: torch.Tensor = torch.pow(r, 3.0 + self._vhs_alpha)
        fcn_j0: torch.Tensor = j0(norms * r)
        integrand: torch.Tensor = scale * power_r * fcn_j0
        self._fpl_character_1 = torch.sum(w * integrand, dim=-1)
        return

    def _precompute_fpl_character_2_2D(self) -> None:
        len_norms: int = int(self._dimension * (self._v_num_grid // 2)**2 + 1)
        norms: torch.Tensor = torch.sqrt(torch.arange(len_norms, **self.dtype_and_device))[:, None]
        r, w = roots_legendre_shifted(self._quad_order_legendre, 0.0, torch.pi, **self.dtype_and_device)
        r = r.reshape(1, -1)
        w = w.reshape(1, -1)

        scale: float = float(self._vhs_coeff / (self.v_ratio**(2.0 + self._vhs_alpha)))
        power_r: torch.Tensor = torch.pow(r, 3.0 + self._vhs_alpha)
        func_c: torch.Tensor = _fpl_character_2__weight_C(r * norms, self._quad_order_uniform, **self.dtype_and_device)
        integrand: torch.Tensor = power_r * func_c
        self._fpl_character_2 = scale * torch.sum(integrand * w, dim=-1)
        return

    def _precompute_fpl_gain_tensors_2D(self) -> None:
        freq_norms_sq: torch.Tensor = torch.sum(self._freqs**2, dim=-1, keepdim=True)
        F: torch.Tensor = self._fpl_character_1[freq_norms_sq]
        G: torch.Tensor = self._fpl_character_2[freq_norms_sq]
        freq_norms_sq = freq_norms_sq.type(self._dtype)

        Q: torch.Tensor = torch.zeros((*self.fpl_base_shape, self._dimension, self._dimension), **self.dtype_and_device)
        idx_i: torch.Tensor = self._freqs[..., [0]].type(self._dtype)
        idx_j: torch.Tensor = self._freqs[..., [1]].type(self._dtype)
        double_cos: torch.Tensor = (idx_i**2 - idx_j**2) / freq_norms_sq
        double_sin: torch.Tensor = (2.0 * idx_i * idx_j) / freq_norms_sq

        zero_origin: tuple[int, ...] = tuple(0 for _ in range(1 + 2 * self._dimension))
        double_cos[zero_origin] = 0.0
        double_sin[zero_origin] = 0.0

        Q[..., 0, 0] = 0.5 * (F + double_cos * G)
        Q[..., 1, 1] = 0.5 * (F - double_cos * G)
        Q[..., 0, 1] = 0.5 * double_sin * G
        Q[..., 1, 0] = Q[..., 0, 1]

        self._fpl_gain_tensor_1 = F
        self._fpl_gain_tensor_2 = Q
        return

    def _precompute_fpl_character_1_3D(self) -> None:
        return

    def _precompute_fpl_character_2_3D(self) -> None:
        return

    def _precompute_fpl_gain_tensors_3D(self) -> None:
        self.__precompute_fpl_gain_tensors_3D__positive()
        self.__precompute_fpl_gain_tensors_3D__negative()
        return

    def __precompute_fpl_gain_tensors_3D__positive(self) -> None:
        len_norms: int = int(1 + self._dimension * ((self._v_num_grid // 2)**2))
        norms: torch.Tensor = torch.sqrt(torch.arange(len_norms, **self.dtype_and_device))[:, None]
        r, w = roots_legendre_shifted(self._quad_order_legendre, 0.0, torch.pi, **self.dtype_and_device)
        r = r.reshape(1, -1)
        w = w.reshape(1, -1)

        coeff: float = float((4.0 * torch.pi * self._vhs_coeff) / (self.v_ratio**(3.0 + self._vhs_alpha)))
        power_r: torch.Tensor = torch.pow(r, 4.0 + self._vhs_alpha)
        func_sinc: torch.Tensor = sinc(norms * r)
        integrand: torch.Tensor = coeff * power_r * func_sinc

        gain_tensor_1: torch.Tensor = torch.sum(integrand * w, dim=-1)
        freq_norms_sq: torch.Tensor = torch.sum(self._freqs**2, dim=-1, keepdim=True)
        self._fpl_gain_tensor_1 = gain_tensor_1[freq_norms_sq]
        return

    def __precompute_fpl_gain_tensors_3D__negative(self) -> None:
        self._fpl_gain_tensor_2 = torch.zeros(
            (1, *(1 for _ in range(self._dimension)), *(self._v_num_grid for _ in range(self._dimension)), 1, 3, 3),
            **self.dtype_and_device,
        )

        freqs_xy: torch.Tensor = self._freqs[..., :2].type(self._dtype)
        freqs_z: torch.Tensor = self._freqs[..., [2]].type(self._dtype)
        freq_norms_xy: torch.Tensor = torch.norm(freqs_xy, p=2, dim=-1, keepdim=True)
        r, w = roots_legendre_shifted(self._quad_order_legendre, 0.0, torch.pi, **self.dtype_and_device)

        freqs_xy = freqs_xy[..., None]
        freqs_z = freqs_z[..., None]
        freq_norms_xy = freq_norms_xy[..., None]
        r = r.reshape(*(1 for _ in range(freqs_xy.ndim - 1)), -1)
        w = w.reshape(*(1 for _ in range(freqs_xy.ndim - 1)), -1)

        coeff_common: float = float(self._vhs_coeff / (self.v_ratio**(3.0 + self._vhs_alpha)))
        power_r: torch.Tensor = torch.pow(r, 4.0 + self._vhs_alpha)

        entry_01_scale: torch.Tensor = (
            coeff_common * freqs_xy.prod(dim=-1, keepdim=True) / torch.pow(freq_norms_xy, 2)
        )
        entry_01_scale[tuple(0 for _ in range(2 * self._dimension))] = 0.0
        entry_01_weight: torch.Tensor = _fpl_character_2_3D__entry01_weight(
            a                  = freqs_z,
            b                  = freq_norms_xy,
            scale_factor       = r,
            quad_order_uniform = self._quad_order_uniform,
            **self.dtype_and_device,
        )
        entry_01: torch.Tensor = torch.sum(entry_01_scale * power_r * entry_01_weight * w, dim=-1)

        entry_22_scale: float = coeff_common * (2.0 * torch.pi)
        entry_22_weight: torch.Tensor = _fpl_character_2_3D__entry22_weight(
            a                  = freqs_z,
            b                  = freq_norms_xy,
            scale_factor       = r,
            quad_order_uniform = self._quad_order_uniform,
            **self.dtype_and_device,
        )
        entry_22: torch.Tensor = torch.sum(entry_22_scale * power_r * entry_22_weight * w, dim=-1)

        reduce_idx: tuple[object, ...] = tuple([0, *(0 for _ in range(self._dimension)), ..., 0])
        entry_01 = entry_01[reduce_idx]
        entry_22 = entry_22[reduce_idx]

        self._fpl_gain_tensor_2[*reduce_idx, 0, 1] = entry_01
        self._fpl_gain_tensor_2[*reduce_idx, 2, 2] = entry_22
        self._fpl_gain_tensor_2[*reduce_idx, 0, 0] = entry_22.permute((1, 2, 0))
        self._fpl_gain_tensor_2[*reduce_idx, 1, 1] = entry_22.permute((2, 0, 1))
        self._fpl_gain_tensor_2[*reduce_idx, 0, 2] = entry_01.permute((2, 0, 1))
        self._fpl_gain_tensor_2[*reduce_idx, 1, 2] = entry_01.permute((1, 2, 0))
        self._fpl_gain_tensor_2[*reduce_idx, 1, 0] = self._fpl_gain_tensor_2[*reduce_idx, 0, 1]
        self._fpl_gain_tensor_2[*reduce_idx, 2, 0] = self._fpl_gain_tensor_2[*reduce_idx, 0, 2]
        self._fpl_gain_tensor_2[*reduce_idx, 2, 1] = self._fpl_gain_tensor_2[*reduce_idx, 1, 2]
        return


##################################################
def fft_index(
        n:      int,
        dtype:  torch.dtype            = torch.long,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Generate 1D discrete Fourier transform frequency indices.

    ## Description
    Generates a 1-dimensional array of standard DFT frequency indices ordered as
    `[0, 1, ..., (n+1)//2 - 1, -(n//2), ..., -1]`.

    ## Arguments
    `n` (`int`): Grid resolution.
    `dtype` (`torch.dtype`, default: `torch.long`): Tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `torch.Tensor`: 1D tensor of frequency indices with shape `(n,)`.
    """
    return torch.cat(
        (
            torch.arange((n + 1) // 2, dtype=dtype, device=device),
            torch.arange(-(n // 2), 0, dtype=dtype, device=device),
        )
    )


def freq_tensor(
        dimension: int,
        num_grid:  int,
        keepdim:   bool                   = False,
        dtype:     torch.dtype             = torch.long,
        device:    Optional[torch.device]  = None,
    ) -> torch.Tensor:
    """Construct the multidimensional DFT frequency tensor.

    ## Description
    Constructs the full multidimensional grid of DFT frequency modes across all spatial
    velocity axes, with optional grid-shape retention.

    ## Arguments
    `dimension` (`int`): Dimension of the frequency domain.
    `num_grid` (`int`): Number of grid points along each dimension.
    `keepdim` (`bool`, default: `False`): Whether to retain the grid shape
    `(*repeat(num_grid, dimension), dimension)`.
    `dtype` (`torch.dtype`, default: `torch.long`): Tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `torch.Tensor`: Multidimensional frequency tensor.
    """
    idx_1d: torch.Tensor = fft_index(num_grid, dtype=dtype, device=device)
    freqs: torch.Tensor = torch.stack(
        torch.meshgrid(*(idx_1d for _ in range(dimension)), indexing='ij'),
        dim = -1,
    )
    if keepdim:
        return freqs
    return freqs.reshape(-1, dimension)


def convolve_freqs(
        x1_fft: torch.Tensor,
        x2_fft: torch.Tensor,
        dim:    Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
    """Compute the convolution of two frequency-domain signals via inverse FFT.

    ## Description
    Computes the convolution of two DFT representations by taking the inverse FFT,
    multiplying in physical space, and transforming back to frequency space with
    forward normalization.

    ## Arguments
    `x1_fft` (`torch.Tensor`): First frequency-domain tensor.
    `x2_fft` (`torch.Tensor`): Second frequency-domain tensor of the same shape as
    `x1_fft`.
    `dim` (`Optional[Sequence[int]]`, default: `None`): Axes along which convolution is
    computed. Defaults to all axes.

    ## Returns
    `torch.Tensor`: Convolved frequency-domain tensor matching input shape.
    """
    if x1_fft.shape != x2_fft.shape:
        raise ValueError(f"Tensors must have matching shape, got x1={list(x1_fft.shape)} and x2={list(x2_fft.shape)}.")
    if dim is None:
        dim = tuple(range(x1_fft.ndim))
    else:
        dim = tuple(dim)
    x1: torch.Tensor = torch.fft.ifftn(x1_fft, dim=dim, norm=FFT_NORM)
    x2: torch.Tensor = torch.fft.ifftn(x2_fft, dim=dim, norm=FFT_NORM)
    return torch.fft.fftn(x1 * x2, dim=dim, norm=FFT_NORM)


def roots_legendre_shifted(
        n:      int,
        a:      float,
        b:      float,
        dtype:  Optional[torch.dtype]  = None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shifted Gauss-Legendre quadrature nodes and weights on `[a, b]`.

    ## Description
    Computes nodes and weights of the Gauss-Legendre quadrature shifted from the
    standard interval `[-1, 1]` to an arbitrary real interval `[a, b]`.

    ## Arguments
    `n` (`int`): Quadrature order (number of integration nodes).
    `a` (`float`): Lower bound of integration interval.
    `b` (`float`): Upper bound of integration interval.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `tuple[torch.Tensor, torch.Tensor]`: Tuple `(nodes, weights)` of 1D tensors of
    length `n`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    roots, weights = roots_legendre(n)
    nodes_t: torch.Tensor = torch.tensor((a + b) / 2.0 + (b - a) * roots / 2.0, dtype=dtype, device=device)
    weights_t: torch.Tensor = torch.tensor((b - a) * weights / 2.0, dtype=dtype, device=device)
    return nodes_t, weights_t


def sinc(x: torch.Tensor) -> torch.Tensor:
    """Compute the unnormalized cardinal sine `sin(x) / x`.

    ## Description
    Evaluates the unnormalized sinc function with the removable singularity at `x = 0`
    handled continuously so that `sinc(0) = 1`.

    ## Arguments
    `x` (`torch.Tensor`): Input coordinate tensor.

    ## Returns
    `torch.Tensor`: Elementwise `sin(x) / x` tensor.
    """
    return torch.where(
        x != 0.0,
        torch.sin(x) / x,
        torch.ones_like(x),
    )


def _fpl_character_2__weight_C(
        x:                  torch.Tensor,
        quad_order_uniform: int,
        dtype:              Optional[torch.dtype]  = None,
        device:             Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Compute the angular weight $C(x)$ for the 2D FPL characteristic function.

    ## Description
    Evaluates $C(x) = \\int_{0}^{2\\pi} \\cos(2t) \\cos(x \\cos(t))\\,dt$ using
    Gauss-Legendre quadrature on $[0, 2\\pi]$.

    ## Arguments
    `x` (`torch.Tensor`): Input radial-frequency coordinate tensor.
    `quad_order_uniform` (`int`): Number of integration quadrature nodes.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `torch.Tensor`: Evaluated angular weight integral tensor matching shape of `x`.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    r, w = roots_legendre_shifted(quad_order_uniform, 0.0, 2.0 * torch.pi, dtype=dtype, device=device)
    r = r.reshape(*(1 for _ in range(x.ndim)), -1)
    w = w.reshape(*(1 for _ in range(x.ndim)), -1)
    integrand: torch.Tensor = torch.cos(2.0 * r) * torch.cos(x[..., None] * torch.cos(r))
    return torch.sum(integrand * w, dim=-1)


def _fpl_character_2_3D__entry01_weight(
        a:                  torch.Tensor,
        b:                  torch.Tensor,
        scale_factor:       torch.Tensor,
        quad_order_uniform: int,
        dtype:              Optional[torch.dtype]  = None,
        device:             Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Compute the off-diagonal angular weight for the 3D FPL gain tensor.

    ## Description
    Evaluates $\\int_0^\\pi \\cos(a\\cos t)\\sin^3(t)\\,C(b\\sin t)\\,dt$ using
    Gauss-Legendre quadrature, corresponding to the off-diagonal entries of the 3D
    FPL gain tensor $\\hat{G}_2$.

    ## Arguments
    `a` (`torch.Tensor`): Coordinate tensor along the z-axis frequency direction.
    `b` (`torch.Tensor`): Coordinate tensor along the xy-plane frequency direction.
    `scale_factor` (`torch.Tensor`): Radial scaling factor tensor.
    `quad_order_uniform` (`int`): Number of quadrature nodes.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `torch.Tensor`: Integrated weight tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    t, w = roots_legendre_shifted(quad_order_uniform, 0.0, torch.pi, dtype=dtype, device=device)
    ndim: int = scale_factor.ndim
    t = t.reshape(*(1 for _ in range(ndim)), -1)
    w = w.reshape(*(1 for _ in range(ndim)), -1)

    scaled_a: torch.Tensor = scale_factor * a
    scaled_b: torch.Tensor = scale_factor * b
    prod_1: torch.Tensor = torch.cos(scaled_a[..., None] * torch.cos(t))
    prod_2: torch.Tensor = torch.sin(t)**3
    prod_3: torch.Tensor = _fpl_character_2__weight_C(scaled_b[..., None] * torch.sin(t), quad_order_uniform, dtype=dtype, device=device)
    integrand: torch.Tensor = prod_1 * prod_2 * prod_3
    return torch.sum(integrand * w, dim=-1)


def _fpl_character_2_3D__entry22_weight(
        a:                  torch.Tensor,
        b:                  torch.Tensor,
        scale_factor:       torch.Tensor,
        quad_order_uniform: int,
        dtype:              Optional[torch.dtype]  = None,
        device:             Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Compute the diagonal angular weight for the 3D FPL gain tensor.

    ## Description
    Evaluates $\\int_0^\\pi \\cos(a\\cos t)\\cos^2(t)\\sin(t)\\,J_0(b\\sin t)\\,dt$ using
    Gauss-Legendre quadrature, corresponding to the diagonal entries of the 3D FPL
    gain tensor $\\hat{G}_2$.

    ## Arguments
    `a` (`torch.Tensor`): Coordinate tensor along the z-axis frequency direction.
    `b` (`torch.Tensor`): Coordinate tensor along the xy-plane frequency direction.
    `scale_factor` (`torch.Tensor`): Radial scaling factor tensor.
    `quad_order_uniform` (`int`): Number of quadrature nodes.
    `dtype` (`Optional[torch.dtype]`, default: `None`): Tensor data type.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `torch.Tensor`: Integrated weight tensor.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    t, w = roots_legendre_shifted(quad_order_uniform, 0.0, torch.pi, dtype=dtype, device=device)
    ndim: int = scale_factor.ndim
    t = t.reshape(*(1 for _ in range(ndim)), -1)
    w = w.reshape(*(1 for _ in range(ndim)), -1)

    scaled_a: torch.Tensor = scale_factor * a
    scaled_b: torch.Tensor = scale_factor * b
    prod_1: torch.Tensor = torch.cos(scaled_a[..., None] * torch.cos(t))
    prod_2: torch.Tensor = (torch.cos(t)**2) * torch.sin(t)
    prod_3: torch.Tensor = j0(scaled_b[..., None] * torch.sin(t))
    integrand: torch.Tensor = prod_1 * prod_2 * prod_3
    return torch.sum(integrand * w, dim=-1)


##################################################
##################################################
# End of file
