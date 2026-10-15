"""# Operator Learning for the Fokker-Planck-Landau (FPL) Equation using Separable PI-DeepONet

-----
This package implements operator learning for the 2-dimensional spatially homogeneous
Fokker-Planck-Landau equation across continuous classes of initial conditions
(Bi-Maxwellian mixtures and analytical BKW states).
"""
from .collision import (
    FPLSpectralCollision,
    compute_conservation_loss,
    compute_moments,
    compute_positivity_loss,
)
from .config    import (
    FPLDeepONetConfig,
    get_default_bimaxwellian_config,
    get_default_bkw_config,
)
from .dataset   import (
    BKWClassDataset,
    BiMaxwellianClassDataset,
    create_collocation_coords,
    create_sensor_grid,
)
from .evaluate  import evaluate_model, plot_comparison
from .train_bimaxwellian import (
    compute_bimaxwellian_loss,
    train_bimaxwellian_routine,
)
from .train_bkw import (
    compute_bkw_loss,
    train_bkw_routine,
)


__all__: list[str] = [
    # Config
    'FPLDeepONetConfig',
    'get_default_bimaxwellian_config',
    'get_default_bkw_config',
    # Data & classes
    'BiMaxwellianClassDataset',
    'BKWClassDataset',
    'create_sensor_grid',
    'create_collocation_coords',
    # Collision & Physics
    'FPLSpectralCollision',
    'compute_moments',
    'compute_conservation_loss',
    'compute_positivity_loss',
    # Training & Evaluation
    'compute_bkw_loss',
    'train_bkw_routine',
    'compute_bimaxwellian_loss',
    'train_bimaxwellian_routine',
    'evaluate_model',
    'plot_comparison',
]
