from   pathlib                 import Path
import sys
from   typing                  import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from   models.numerical_collision import FPL_spectral
from   solver                     import FastSM_Landau_VHS


__all__: List[str] = [
    'FPLSpectralCollision',
    'compute_moments',
    'compute_conservation_loss',
    'compute_positivity_loss',
]


##################################################
class FPLSpectralCollision:
    """Wrapper around Fast Spectral Method (FSM) for evaluating the FPL collision operator $Q(f)$.

    ## Description
    Wraps `FastSM_Landau_VHS` to evaluate the 2D Fokker-Planck-Landau collision operator
    on batches of velocity grids of shape `(batch, sample_v, sample_v)`.
    Precomputes spectral weights and characteristic functions for high-speed forward passes.

    -----
    ### References
    [1] L. Pareschi, G. Russo, and G. Toscani.
        Fast spectral methods for the Fokker-Planck-Landau collision operator.
        Journal of Computational Physics, 165(1):216–236, 2000.
    """

    def __init__(
            self,
            v_num_grid: int,
            v_max:      float,
            vhs_coeff:  float                  = 0.3125,
            vhs_alpha:  float                  = 0.0,
            device:     Optional[torch.device] = None,
        ) -> None:
        """Initializes the FPL spectral collision operator.

        Arguments:
            `v_num_grid` (`int`): Grid intervals along each velocity axis.
            `v_max` (`float`): Truncation velocity bound defining `[-v_max, v_max]^2`.
            `vhs_coeff` (`float`, default: `0.3125`): VHS interaction coefficient.
            `vhs_alpha` (`float`, default: `0.0`): VHS interaction exponent (0 for Maxwellian molecules).
            `device` (`Optional[torch.device]`, default: `None`): Computation device.

        Returns:
            This method returns no values.
        """
        self.dimension:  int          = 2
        self.v_num_grid: int          = v_num_grid
        self.v_max:      float        = v_max
        self.device:     torch.device = device if device is not None else torch.get_default_device()

        self.fsm: FastSM_Landau_VHS = FastSM_Landau_VHS(
            dimension  = self.dimension,
            v_num_grid = v_num_grid,
            v_max      = v_max,
            vhs_coeff  = vhs_coeff,
            vhs_alpha  = vhs_alpha,
            device     = self.device,
        )
        self.fsm.precompute()
        return None

    def __call__(self, f_grid: torch.Tensor) -> torch.Tensor:
        """Evaluates $Q(f)$ on a batch of 2D velocity distribution functions.

        Arguments:
            `f_grid` (`torch.Tensor`): Distribution tensor on velocity grid, shape `(batch, N_v, N_v)`.

        Returns:
            `torch.Tensor`: Evaluated collision operator tensor of shape `(batch, N_v, N_v)`.
        """
        batch_size: int = f_grid.shape[0]
        # Reshape to expected FSM grid layout: (batch, 1, 1, N_v, N_v, 1)
        f_in:   torch.Tensor = f_grid.reshape(batch_size, 1, 1, self.v_num_grid, self.v_num_grid, 1)
        fft_dim: Tuple[int, int] = (3, 4)

        f_fft: torch.Tensor = torch.fft.fftn(f_in, dim=fft_dim, norm=self.fsm.internal_fft_norm)
        q_fft: torch.Tensor = self.fsm.compute_fft(None, f_fft)
        q_out: torch.Tensor = torch.real(torch.fft.ifftn(q_fft, dim=fft_dim, norm=self.fsm.internal_fft_norm))

        return q_out.reshape(batch_size, self.v_num_grid, self.v_num_grid)


def compute_moments(
    f:    torch.Tensor,
    v1:   torch.Tensor,
    v2:   torch.Tensor,
    dv:   float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes macroscopic conservation quantities (mass, momentum vector, and kinetic energy).

    ## Description
    Calculates numerical integrals on the uniform 2D velocity lattice:
    - Mass: $\\rho(t) = \\iint f(t, v) \\, dv_1 dv_2$
    - Momentum: $P(t) = \\iint v f(t, v) \\, dv_1 dv_2$
    - Energy: $E(t) = \\iint (v_1^2 + v_2^2) f(t, v) \\, dv_1 dv_2$

    Arguments:
        `f` (`torch.Tensor`): Distribution tensor of shape `(batch, N_t, N_v, N_v)`.
        `v1` (`torch.Tensor`): 1D coordinate vector along $v_1$, shape `(N_v, 1)`.
        `v2` (`torch.Tensor`): 1D coordinate vector along $v_2$, shape `(N_v, 1)`.
        `dv` (`float`): Grid cell spacing $\\Delta v$.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Tuple of `(mass, momentum, energy)` tensors.
    """
    dv_sq: float = dv * dv
    # Mass: sum over (N_v, N_v)
    mass: torch.Tensor = torch.sum(f, dim=(-2, -1)) * dv_sq  # (batch, N_t)

    # Momentum: v1 and v2
    v1_grid, v2_grid = torch.meshgrid(v1.squeeze(), v2.squeeze(), indexing='ij')
    v1_grid = v1_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, N_v, N_v)
    v2_grid = v2_grid.unsqueeze(0).unsqueeze(0)

    p1: torch.Tensor = torch.sum(f * v1_grid, dim=(-2, -1)) * dv_sq
    p2: torch.Tensor = torch.sum(f * v2_grid, dim=(-2, -1)) * dv_sq
    momentum: torch.Tensor = torch.stack([p1, p2], dim=-1)  # (batch, N_t, 2)

    # Energy: (v1^2 + v2^2)
    speed_sq: torch.Tensor = (v1_grid**2 + v2_grid**2)
    energy:   torch.Tensor = torch.sum(f * speed_sq, dim=(-2, -1)) * dv_sq  # (batch, N_t)

    return mass, momentum, energy


def compute_conservation_loss(
    f:    torch.Tensor,
    v1:   torch.Tensor,
    v2:   torch.Tensor,
    dv:   float,
) -> torch.Tensor:
    """Computes macroscopic conservation loss enforcing constant mass, momentum, and energy over time.

    Arguments:
        `f` (`torch.Tensor`): Distribution tensor, shape `(batch, N_t, N_v, N_v)`.
        `v1` (`torch.Tensor`): Velocity coordinate vector $v_1$.
        `v2` (`torch.Tensor`): Velocity coordinate vector $v_2$.
        `dv` (`float`): Grid spacing $\\Delta v$.

    Returns:
        `torch.Tensor`: Scalar conservation MSE loss.
    """
    mass, momentum, energy = compute_moments(f, v1, v2, dv)

    # Reference values at initial time t = 0 (first index along time axis)
    mass_0:     torch.Tensor = mass[:, [0]]
    momentum_0: torch.Tensor = momentum[:, [0], :]
    energy_0:   torch.Tensor = energy[:, [0]]

    loss_mass:     torch.Tensor = torch.mean((mass - mass_0)**2)
    loss_momentum: torch.Tensor = torch.mean((momentum - momentum_0)**2)
    loss_energy:   torch.Tensor = torch.mean((energy - energy_0)**2)

    return loss_mass + loss_momentum + loss_energy


def compute_positivity_loss(f: torch.Tensor) -> torch.Tensor:
    """Penalizes negative distribution function values to enforce physical non-negativity $f \\ge 0$.

    Arguments:
        `f` (`torch.Tensor`): Distribution tensor.

    Returns:
        `torch.Tensor`: Mean squared magnitude of negative values.
    """
    neg_vals: torch.Tensor = torch.relu(-f)
    return torch.mean(torch.square(neg_vals))


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
