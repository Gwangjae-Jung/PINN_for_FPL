"""# Separable Physics-Informed Deep Operator Network (Sep-PI-DeepONet) in PyTorch

-----
This package implements the Separable Physics-Informed Deep Operator Network architecture in PyTorch,
designed to break the curse of dimensionality in physics-informed operator learning.

## A. Core Architectures
- `SeparableDeepONet`: Separable PI-DeepONet utilizing 1D factorized sub-networks and tensor rank decomposition.
- `SeparableBranchCNNDeepONet`: Separable DeepONet with 2D CNN branch network for field inputs.
- `DeepONet`: Conventional (Vanilla) Deep Operator Network baseline.

## B. Automatic Differentiation and Gradients
- `derivatives`: Vectorized forward-mode derivatives up to arbitrary orders (`torch.func.jacfwd` + `vmap`).
- `hvp_fwdfwd`: Forward-over-forward Hessian-vector product via `torch.func.jvp`.
- `compute_grad`: Reverse-mode gradient evaluation using `torch.autograd.grad`.

-----
## References
[1] Luis Mandl, Somdatta Goswami, Lena Lambers, and Tim Ricken.
    Separable physics-informed DeepONet: Breaking the curse of dimensionality in physics-informed machine learning.
    Computer Methods in Applied Mechanics and Engineering, Vol. 434, 117586, 2025.
    https://doi.org/10.1016/j.cma.2024.117586

[2] Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park.
    Separable physics-informed neural networks.
    Advances in Neural Information Processing Systems (NeurIPS), 2023.
    https://neurips.cc/virtual/2022/59890

[3] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis.
    Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.
    Nature Machine Intelligence, 3(3):218–229, 2021.
"""
from .autograd import (
    compute_grad,
    derivatives,
    hessian,
    hvp_fwdfwd,
    jacobian,
)
from .data import (
    GaussianRandomField1D,
    GaussianRandomField2D,
    SeparableDataGenerator,
    generate_biot_test_data,
    generate_burgers_test_data,
    generate_heat_test_data,
)
from .helpers import (
    apply_net,
    apply_net_sep,
    mse,
    mse_single,
    relative_l2_error,
)
from .models import (
    BaseModule,
    DeepONet,
    DeepONetConfig,
    SeparableBranchCNNDeepONet,
    SeparableDeepONet,
    setup_deeponet,
    trunk_r_einsum,
)


__all__: list[str] = [
    # Models
    'BaseModule',
    'DeepONet',
    'SeparableDeepONet',
    'SeparableBranchCNNDeepONet',
    'DeepONetConfig',
    'setup_deeponet',
    'trunk_r_einsum',
    # Autograd
    'derivatives',
    'jacobian',
    'hessian',
    'hvp_fwdfwd',
    'compute_grad',
    # Helpers
    'mse',
    'mse_single',
    'relative_l2_error',
    'apply_net',
    'apply_net_sep',
    # Data
    'SeparableDataGenerator',
    'GaussianRandomField1D',
    'GaussianRandomField2D',
    'generate_burgers_test_data',
    'generate_heat_test_data',
    'generate_biot_test_data',
]
