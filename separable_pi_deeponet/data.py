from   typing                  import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from   torch.utils.data        import Dataset


__all__: List[str] = [
    'SeparableDataGenerator',
    'GaussianRandomField1D',
    'GaussianRandomField2D',
    'generate_burgers_test_data',
    'generate_heat_test_data',
    'generate_biot_test_data',
]


##################################################
class SeparableDataGenerator(Dataset):
    """PyTorch Dataset for batching input functions and factorized coordinate tensors in Separable DeepONet.

    ## Description
    Provides indexed access or mini-batch sampling of branch functions $u$ alongside factorized coordinate arrays.
    """

    def __init__(
            self,
            u:          Union[torch.Tensor, np.ndarray],
            trunk_coords: Sequence[Union[torch.Tensor, np.ndarray]],
            s:          Optional[Union[torch.Tensor, np.ndarray]] = None,
            batch_size: Optional[int]                            = None,
        ) -> None:
        """Initializes the dataset generator.

        Arguments:
            `u` (`Union[torch.Tensor, np.ndarray]`): Branch input functions, shape `(N, sensors)`.
            `trunk_coords` (`Sequence[Union[torch.Tensor, np.ndarray]]`): Sequence of 1D coordinate tensors.
            `s` (`Optional[Union[torch.Tensor, np.ndarray]]`, default: `None`): Solution field observations if available.
            `batch_size` (`Optional[int]`, default: `None`): Batch size for sub-sampling branch functions.

        Returns:
            This method returns no values.
        """
        super().__init__()
        self.u: torch.Tensor = torch.as_tensor(u, dtype=torch.float32)
        self.trunk_coords: Tuple[torch.Tensor, ...] = tuple(
            torch.as_tensor(c, dtype=torch.float32).reshape(-1, 1) for c in trunk_coords
        )
        self.s: Optional[torch.Tensor] = torch.as_tensor(s, dtype=torch.float32) if s is not None else None
        self.N: int                    = self.u.shape[0]
        self.batch_size: int           = batch_size if batch_size is not None else self.N
        return None

    def __len__(self) -> int:
        """Returns the number of function instances in the dataset.

        Returns:
            `int`: Total number of samples $N$.
        """
        return self.N

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
        """Retrieves a single sample pair.

        Arguments:
            `idx` (`int`): Index of the function sample.

        Returns:
            `Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]`:
                Tuple containing `(u_sample, trunk_coords)` and optional solution target.
        """
        u_sample: torch.Tensor = self.u[idx]
        s_sample: Optional[torch.Tensor] = self.s[idx] if self.s is not None else None
        return (u_sample, self.trunk_coords), s_sample

    def sample_batch(self) -> Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
        """Randomly samples a mini-batch of functions with the full factorized coordinate grid.

        Returns:
            `Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]`:
                Mini-batch inputs `(u_batch, trunk_coords)` and corresponding solution targets.
        """
        indices: torch.Tensor = torch.randperm(self.N)[:self.batch_size]
        u_batch: torch.Tensor = self.u[indices]
        s_batch: Optional[torch.Tensor] = self.s[indices] if self.s is not None else None
        return (u_batch, self.trunk_coords), s_batch


class GaussianRandomField1D:
    """1D Gaussian Random Field generator for periodic initial conditions.

    ## Description
    Generates periodic smooth 1D random fields via Fourier series with power spectral density:
    $S(k) = \\sigma^2 (\\tau^2 + (2\\pi k)^2)^{-\\gamma}$.
    """

    def __init__(
            self,
            n_points: int   = 101,
            sigma:    float = 25.0,
            tau:      float = 5.0,
            gamma:    float = 4.0,
            seed:     int   = 42,
        ) -> None:
        """Initializes GaussianRandomField1D generator.

        Arguments:
            `n_points` (`int`, default: `101`): Number of grid points on $[0, 1)$.
            `sigma` (`float`, default: `25.0`): Scale amplitude parameter $\\sigma$.
            `tau` (`float`, default: `5.0`): Length scale parameter $\\tau$.
            `gamma` (`float`, default: `4.0`): Regularity decay exponent $\\gamma$.
            `seed` (`int`, default: `42`): Random number generator seed.

        Returns:
            This method returns no values.
        """
        self.n_points: int             = n_points
        self.sigma:    float           = sigma
        self.tau:      float           = tau
        self.gamma:    float           = gamma
        self.rng:      np.random.Generator = np.random.default_rng(seed)
        return None

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Generates `num_samples` realizations of 1D Gaussian random fields.

        Arguments:
            `num_samples` (`int`, default: `1`): Number of realization functions to generate.

        Returns:
            `torch.Tensor`: Tensor of shape `(num_samples, n_points)`.
        """
        k_max: int        = self.n_points // 2
        k:     np.ndarray = np.arange(0, k_max + 1)
        denom: np.ndarray = self.tau**2 + (2.0 * np.pi * k)**2
        power_spectrum: np.ndarray = (self.sigma**2) * (denom**(-self.gamma / 2.0))

        fields: List[np.ndarray] = []
        for _ in range(num_samples):
            xi_r: np.ndarray = self.rng.standard_normal(len(k))
            xi_i: np.ndarray = self.rng.standard_normal(len(k))
            xi_i[0]          = 0.0
            coeffs: np.ndarray = (xi_r + 1j * xi_i) * np.sqrt(power_spectrum)

            full_coeffs: np.ndarray = np.zeros(self.n_points, dtype=complex)
            full_coeffs[:k_max + 1] = coeffs
            if self.n_points % 2 == 0:
                full_coeffs[k_max + 1:] = np.conj(coeffs[1:k_max][::-1])
            else:
                full_coeffs[k_max + 1:] = np.conj(coeffs[1:k_max + 1][::-1])

            field: np.ndarray = np.fft.ifft(full_coeffs).real * self.n_points
            fields.append(field)

        return torch.tensor(np.array(fields), dtype=torch.float32)


class GaussianRandomField2D:
    """2D Gaussian Random Field generator for spatial source terms and parameter fields.

    ## Description
    Generates 2D random fields with power spectral density $P(k) \\sim 1 / |k|^{\\alpha/2}$
    subject to homogeneous Dirichlet or periodic conditions.
    """

    def __init__(
            self,
            size:  Tuple[int, int] = (51, 51),
            alpha: float           = 4.0,
            seed:  int             = 42,
        ) -> None:
        """Initializes 2D Gaussian random field generator.

        Arguments:
            `size` (`Tuple[int, int]`, default: `(51, 51)`): Spatial grid dimensions $(H, W)$.
            `alpha` (`float`, default: `4.0`): Decay power parameter $\\alpha$.
            `seed` (`int`, default: `42`): Random number generator seed.

        Returns:
            This method returns no values.
        """
        self.size:  Tuple[int, int]     = size
        self.alpha: float               = alpha
        self.rng:   np.random.Generator = np.random.default_rng(seed)
        return None

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Generates `num_samples` realizations of 2D Gaussian random fields.

        Arguments:
            `num_samples` (`int`, default: `1`): Number of 2D field realizations.

        Returns:
            `torch.Tensor`: Tensor of shape `(num_samples, H, W)`.
        """
        h, w = self.size
        kx: np.ndarray = np.fft.fftfreq(h)[:, None]
        ky: np.ndarray = np.fft.fftfreq(w)[None, :]
        k_sq: np.ndarray = kx**2 + ky**2
        k_sq[0, 0]       = 1.0
        amplitude: np.ndarray = (k_sq)**(-self.alpha / 4.0)
        amplitude[0, 0]       = 0.0

        fields: List[np.ndarray] = []
        for _ in range(num_samples):
            noise_r: np.ndarray = self.rng.standard_normal(self.size)
            noise_i: np.ndarray = self.rng.standard_normal(self.size)
            noise_complex: np.ndarray = noise_r + 1j * noise_i
            f_spec: np.ndarray  = noise_complex * amplitude
            field: np.ndarray   = np.fft.ifft2(f_spec).real
            # Normalize to zero mean, unit variance
            field = (field - np.mean(field)) / (np.std(field) + 1e-8)
            fields.append(field)

        return torch.tensor(np.array(fields), dtype=torch.float32)


def generate_burgers_test_data(
    u_sol:  Union[torch.Tensor, np.ndarray],
    idx:    int,
    p_test: int = 101,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Generates evaluation grid and test data for a single Burgers' equation sample.

    Arguments:
        `u_sol` (`Union[torch.Tensor, np.ndarray]`): Precomputed solution array `(N, n_t, n_x)`.
        `idx` (`int`): Index of the test instance.
        `p_test` (`int`, default: `101`): Number of evaluation points per axis.

    Returns:
        `Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]`:
            Branch input `u0`, coordinate tuple `(t, x)`, and flattened ground truth solution `s`.
    """
    sol: torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)[idx]
    u0:  torch.Tensor = sol[0, :].reshape(1, -1)
    t:   torch.Tensor = torch.linspace(0.0, 1.0, p_test, dtype=torch.float32).reshape(-1, 1)
    x:   torch.Tensor = torch.linspace(0.0, 1.0, p_test, dtype=torch.float32).reshape(-1, 1)
    s:   torch.Tensor = sol.T.reshape(-1)
    return u0, (t, x), s


def generate_heat_test_data(
    u_sol:  Union[torch.Tensor, np.ndarray],
    c_ref:  Union[torch.Tensor, np.ndarray],
    t_ic:   Union[torch.Tensor, np.ndarray],
    idx:    int,
    p_err:  int = 101,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Generates evaluation grid and test data for a single 4D heat equation sample.

    Arguments:
        `u_sol` (`Union[torch.Tensor, np.ndarray]`): Precomputed temperature solution array.
        `c_ref` (`Union[torch.Tensor, np.ndarray]`): Diffusivity parameter array.
        `t_ic` (`Union[torch.Tensor, np.ndarray]`): Initial temperature field array.
        `idx` (`int`): Index of the test instance.
        `p_err` (`int`, default: `101`): Evaluation resolution per coordinate.

    Returns:
        `Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]`:
            Branch input `u0`, coordinate tuple `(t, x, y, c)`, and ground truth solution.
    """
    u_test: torch.Tensor = torch.as_tensor(t_ic, dtype=torch.float32)[idx].reshape(1, -1)
    c_val:  torch.Tensor = torch.as_tensor(c_ref, dtype=torch.float32)[idx].reshape(-1, 1)
    t:      torch.Tensor = torch.linspace(0.0, 1.0, p_err, dtype=torch.float32).reshape(-1, 1)
    x:      torch.Tensor = torch.linspace(0.0, 1.0, p_err, dtype=torch.float32).reshape(-1, 1)
    y:      torch.Tensor = torch.linspace(0.0, 1.0, p_err, dtype=torch.float32).reshape(-1, 1)
    s_sol:  torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)[idx]
    return u_test, (t, x, y, c_val), s_sol


def generate_biot_test_data(
    u_sol:  Union[torch.Tensor, np.ndarray],
    idx:    int,
    p_test: int = 101,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Generates evaluation grid and test data for a single Biot consolidation sample.

    Arguments:
        `u_sol` (`Union[torch.Tensor, np.ndarray]`): Reference solution tensor of shape `(N, z_steps, t_steps, 2)`.
        `idx` (`int`): Index of the test instance.
        `p_test` (`int`, default: `101`): Number of coordinate points.

    Returns:
        `Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]`:
            Branch input `u0`, coordinate tuple `(t, z)`, and reference solution tensor.
    """
    sol: torch.Tensor = torch.as_tensor(u_sol, dtype=torch.float32)[idx]
    u0:  torch.Tensor = sol[0, :, 0].reshape(1, -1)
    t:   torch.Tensor = torch.linspace(0.0, 1.0, p_test, dtype=torch.float32).reshape(-1, 1)
    z:   torch.Tensor = torch.linspace(0.0, 1.0, p_test, dtype=torch.float32).reshape(-1, 1)
    # Permute from (z, t, 2) to (t, z, 2)
    s_sol: torch.Tensor = sol.permute(1, 0, 2)
    return u0, (t, z), s_sol


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
