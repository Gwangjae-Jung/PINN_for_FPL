from   typing                  import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from   torch.utils.data        import Dataset

from   .config                 import FPLDeepONetConfig


__all__: List[str] = [
    'create_sensor_grid',
    'create_collocation_coords',
    'BiMaxwellianClassDataset',
    'BKWClassDataset',
]


##################################################
def create_sensor_grid(v_max: float, sensor_res: int) -> torch.Tensor:
    """Generates uniform 2D sensor points in velocity space $[-V_{\\text{max}}, V_{\\text{max}}]^2$.

    Arguments:
        `v_max` (`float`): Maximum velocity truncation bound.
        `sensor_res` (`int`): Number of sensor intervals along each velocity axis.

    Returns:
        `torch.Tensor`: Tensor of sensor coordinates of shape `(sensor_res * sensor_res, 2)`.
    """
    v_lin:   torch.Tensor = torch.linspace(-v_max, v_max, sensor_res)
    grid_v1, grid_v2      = torch.meshgrid(v_lin, v_lin, indexing='ij')
    sensors: torch.Tensor = torch.stack([grid_v1.reshape(-1), grid_v2.reshape(-1)], dim=-1)
    return sensors


def create_collocation_coords(
    max_t:    float,
    max_v:    float,
    sample_t: int,
    sample_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates 1D factorized coordinate vectors for $(t, v_1, v_2)$.

    Arguments:
        `max_t` (`float`): Maximum time bound $T_{\\text{max}}$.
        `max_v` (`float`): Velocity cutoff bound $V_{\\text{max}}$.
        `sample_t` (`int`): Grid resolution in time axis.
        `sample_v` (`int`): Grid resolution in velocity axes.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Coordinate vectors $(t, v_1, v_2)$, each of shape `(N, 1)`.
    """
    t_coord:  torch.Tensor = torch.linspace(0.0, max_t, sample_t).reshape(-1, 1)
    v1_coord: torch.Tensor = torch.linspace(-max_v, max_v, sample_v).reshape(-1, 1)
    v2_coord: torch.Tensor = torch.linspace(-max_v, max_v, sample_v).reshape(-1, 1)
    return t_coord, v1_coord, v2_coord


class BiMaxwellianClassDataset(Dataset):
    """Dataset generator representing the parametric class of Bi-Maxwellian initial conditions.

    ## Description
    Generates diverse realizations of non-equilibrium mixtures of two Gaussian distributions in 2D velocity space:
    $$f_0(v) = \\rho \\left[ w_1 \\mathcal{M}(v; u_1, \\sigma_1) + w_2 \\mathcal{M}(v; u_2, \\sigma_2) \\right]$$
    with opposite bulk velocities ($u_1 = -u_2$) ensuring zero mean net momentum,
    and randomized thermal spreads $\\sigma_1, \\sigma_2$ and mixing ratios $w_1, w_2$.
    """

    def __init__(
            self,
            num_samples: int,
            config:      FPLDeepONetConfig,
            seed:        int = 42,
        ) -> None:
        """Initializes the Bi-Maxwellian class dataset.

        Arguments:
            `num_samples` (`int`): Total number of realization functions to generate.
            `config` (`FPLDeepONetConfig`): Problem configuration dictionary.
            `seed` (`int`, default: `42`): Random number generator seed.

        Returns:
            This method returns no values.
        """
        super().__init__()
        self.num_samples: int               = num_samples
        self.config:      FPLDeepONetConfig = config
        self.rng:         np.random.Generator = np.random.default_rng(seed)

        v_max:      float = float(config.get('max_v', 5.0))
        sensor_res: int   = int(config.get('sensor_res', 16))
        sample_v:   int   = int(config.get('sample_v', 32))
        sample_t:   int   = int(config.get('sample_t', 16))
        max_t:      float = float(config.get('max_t', 3.0))

        self.sensors: torch.Tensor = create_sensor_grid(v_max, sensor_res)
        self.t_coord, self.v1_coord, self.v2_coord = create_collocation_coords(max_t, v_max, sample_t, sample_v)

        # Generate sample parameters
        d_min:  float = float(config.get('center_dev_min', 0.6))
        d_max:  float = float(config.get('center_dev_max', 1.4))
        s_min:  float = float(config.get('std_min', 0.6))
        s_max:  float = float(config.get('std_max', 1.0))
        w_min:  float = float(config.get('weight_min', 0.4))
        w_max:  float = float(config.get('weight_max', 0.6))
        rho:    float = float(config.get('density', 0.2))

        # Sample parameter arrays
        d_vals:      np.ndarray = self.rng.uniform(d_min, d_max, num_samples)
        angles:      np.ndarray = self.rng.uniform(0.0, 2.0 * np.pi, num_samples)
        s1_vals:     np.ndarray = self.rng.uniform(s_min, s_max, num_samples)
        s2_vals:     np.ndarray = self.rng.uniform(s_min, s_max, num_samples)
        w1_vals:     np.ndarray = self.rng.uniform(w_min, w_max, num_samples)

        # Precompute sensor evaluations and target initial distributions
        branch_inputs: List[torch.Tensor] = []
        ic_targets:    List[torch.Tensor] = []

        grid_v1, grid_v2 = torch.meshgrid(self.v1_coord.squeeze(), self.v2_coord.squeeze(), indexing='ij')
        target_pts: torch.Tensor = torch.stack([grid_v1.reshape(-1), grid_v2.reshape(-1)], dim=-1)

        for i in range(num_samples):
            u1_np: np.ndarray   = np.array([d_vals[i] * np.cos(angles[i]), d_vals[i] * np.sin(angles[i])])
            u1:    torch.Tensor = torch.tensor(u1_np, dtype=torch.float32)
            u2:    torch.Tensor = -u1
            sig1:  float        = float(s1_vals[i])
            sig2:  float        = float(s2_vals[i])
            w1:    float        = float(w1_vals[i])
            w2:    float        = 1.0 - w1

            # Evaluate at sensor points for branch input
            f_sensors: torch.Tensor = self._evaluate_mixture(self.sensors, u1, u2, sig1, sig2, w1, w2, rho)
            branch_inputs.append(f_sensors)

            # Evaluate at collocation grid for t=0 initial condition loss
            f_grid: torch.Tensor = self._evaluate_mixture(target_pts, u1, u2, sig1, sig2, w1, w2, rho)
            ic_targets.append(f_grid.reshape(sample_v, sample_v))

        self.branch_data: torch.Tensor = torch.stack(branch_inputs, dim=0)
        self.ic_data:     torch.Tensor = torch.stack(ic_targets, dim=0)
        return None

    @staticmethod
    def _evaluate_mixture(
        pts:      torch.Tensor,
        center_1: torch.Tensor,
        center_2: torch.Tensor,
        sigma_1:  float,
        sigma_2:  float,
        weight_1: float,
        weight_2: float,
        density:  float,
    ) -> torch.Tensor:
        """Evaluates weighted mixture of two 2D Maxwellian distributions.

        Arguments:
            `pts` (`torch.Tensor`): Velocity coordinates, shape `(N, 2)`.
            `center_1` (`torch.Tensor`): Center of mode 1.
            `center_2` (`torch.Tensor`): Center of mode 2.
            `sigma_1` (`float`): Standard deviation of mode 1.
            `sigma_2` (`float`): Standard deviation of mode 2.
            `weight_1` (`float`): Mixing weight of mode 1.
            `weight_2` (`float`): Mixing weight of mode 2.
            `density` (`float`): Overall mass density.

        Returns:
            `torch.Tensor`: Evaluated distribution values, shape `(N,)`.
        """
        diff1_sq: torch.Tensor = torch.sum((pts - center_1)**2, dim=-1)
        diff2_sq: torch.Tensor = torch.sum((pts - center_2)**2, dim=-1)

        mode1: torch.Tensor = torch.exp(-diff1_sq / (2.0 * sigma_1**2)) / (2.0 * torch.pi * sigma_1**2)
        mode2: torch.Tensor = torch.exp(-diff2_sq / (2.0 * sigma_2**2)) / (2.0 * torch.pi * sigma_2**2)

        return density * (weight_1 * mode1 + weight_2 * mode2)

    def __len__(self) -> int:
        """Returns the number of realization samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves branch sensor input and initial condition grid target.

        Arguments:
            `idx` (`int`): Sample index.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: Tuple of `(branch_sensor_input, ic_grid_target)`.
        """
        return self.branch_data[idx], self.ic_data[idx]


class BKWClassDataset(Dataset):
    """Dataset generator representing the parametric class of BKW exact solutions.

    ## Description
    Generates non-equilibrium BKW distribution states parameterized by varying
    external relaxation parameter $c_{\\text{ext}} \\in [c_{\\text{min}}, c_{\\text{max}}]$.
    For every sample, the exact analytical solution is accessible at any $(t, v_1, v_2)$ coordinate.
    """

    def __init__(
            self,
            num_samples: int,
            config:      FPLDeepONetConfig,
            seed:        int = 42,
        ) -> None:
        """Initializes the BKW class dataset.

        Arguments:
            `num_samples` (`int`): Number of BKW initial condition functions.
            `config` (`FPLDeepONetConfig`): Problem configuration dictionary.
            `seed` (`int`, default: `42`): Random number generator seed.

        Returns:
            This method returns no values.
        """
        super().__init__()
        self.num_samples: int               = num_samples
        self.config:      FPLDeepONetConfig = config
        self.rng:         np.random.Generator = np.random.default_rng(seed)

        v_max:      float = float(config.get('max_v', float(2.0 * torch.pi)))
        sensor_res: int   = int(config.get('sensor_res', 16))
        sample_v:   int   = int(config.get('sample_v', 32))
        sample_t:   int   = int(config.get('sample_t', 16))
        max_t:      float = float(config.get('max_t', 3.0))

        self.sensors: torch.Tensor = create_sensor_grid(v_max, sensor_res)
        self.t_coord, self.v1_coord, self.v2_coord = create_collocation_coords(max_t, v_max, sample_t, sample_v)

        c_min: float = float(config.get('bkw_coeff_ext_min', 0.1))
        c_max: float = float(config.get('bkw_coeff_ext_max', 0.5))
        self.c_ext_vals: np.ndarray = self.rng.uniform(c_min, c_max, num_samples)

        self.density:   float = float(config.get('density', 0.2))
        self.vhs_coeff: float = float(config.get('vhs_coeff', 0.3125))

        # Precompute sensor evaluations and IC targets
        branch_inputs: List[torch.Tensor] = []
        ic_targets:    List[torch.Tensor] = []

        grid_v1, grid_v2 = torch.meshgrid(self.v1_coord.squeeze(), self.v2_coord.squeeze(), indexing='ij')
        target_pts: torch.Tensor = torch.stack([grid_v1.reshape(-1), grid_v2.reshape(-1)], dim=-1)

        t_zero: torch.Tensor = torch.zeros((1, 1), dtype=torch.float32)

        for i in range(num_samples):
            c_ext: float = float(self.c_ext_vals[i])

            # Sensors at t=0
            f_sensors: torch.Tensor = self.evaluate_bkw(t_zero, self.sensors, c_ext).squeeze()
            branch_inputs.append(f_sensors)

            # Grid at t=0
            f_grid: torch.Tensor = self.evaluate_bkw(t_zero, target_pts, c_ext).reshape(sample_v, sample_v)
            ic_targets.append(f_grid)

        self.branch_data: torch.Tensor = torch.stack(branch_inputs, dim=0)
        self.ic_data:     torch.Tensor = torch.stack(ic_targets, dim=0)
        return None

    def evaluate_bkw(
        self,
        t:     torch.Tensor,
        v_pts: torch.Tensor,
        c_ext: float,
    ) -> torch.Tensor:
        """Evaluates the 2D BKW exact analytical solution $f(t, v)$.

        Arguments:
            `t` (`torch.Tensor`): Time tensor of shape `(N_t, 1)` or scalar tensor.
            `v_pts` (`torch.Tensor`): Velocity tensor of shape `(N_v, 2)`.
            `c_ext` (`float`): External relaxation parameter.

        Returns:
            `torch.Tensor`: Evaluated analytical distribution.
        """
        dim:       int          = 2
        coeff_int: float        = 2.0 * (dim - 1) * self.density * self.vhs_coeff

        # K(t) = 1 - c_ext * exp(-coeff_int * t)
        K_t: torch.Tensor = 1.0 - c_ext * torch.exp(-coeff_int * t)

        speed_sq: torch.Tensor = torch.sum(v_pts**2, dim=-1, keepdim=True)  # (N_v, 1)

        part1: torch.Tensor = self.density * torch.pow(2.0 * torch.pi * K_t, -dim / 2.0)
        part2: torch.Tensor = torch.exp(-speed_sq / (2.0 * K_t))
        part3: torch.Tensor = ((dim + 2.0) / 2.0) - ((dim + speed_sq) / 2.0) / K_t + (speed_sq / 2.0) / (K_t**2)

        return part1 * part2 * part3

    def get_ground_truth_lattice(self, idx: int) -> torch.Tensor:
        """Computes the full 3D ground truth tensor on $(t, v_1, v_2)$ lattice for sample `idx`.

        Arguments:
            `idx` (`int`): Sample index.

        Returns:
            `torch.Tensor`: Ground truth tensor of shape `(N_t, N_v, N_v, 1)`.
        """
        c_ext: float = float(self.c_ext_vals[idx])
        grid_v1, grid_v2 = torch.meshgrid(self.v1_coord.squeeze(), self.v2_coord.squeeze(), indexing='ij')
        v_pts: torch.Tensor = torch.stack([grid_v1.reshape(-1), grid_v2.reshape(-1)], dim=-1)  # (N_v^2, 2)

        n_t: int = self.t_coord.shape[0]
        n_v: int = self.v1_coord.shape[0]

        slices: List[torch.Tensor] = []
        for t_step in range(n_t):
            t_val: torch.Tensor = self.t_coord[[t_step]]
            f_val: torch.Tensor = self.evaluate_bkw(t_val, v_pts, c_ext).reshape(n_v, n_v, 1)
            slices.append(f_val)

        return torch.stack(slices, dim=0)

    def __len__(self) -> int:
        """Returns the number of realization samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves branch sensor input and initial condition grid target.

        Arguments:
            `idx` (`int`): Sample index.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: Tuple of `(branch_sensor_input, ic_grid_target)`.
        """
        return self.branch_data[idx], self.ic_data[idx]


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
