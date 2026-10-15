# Separable Physics-Informed DeepONet (Sep-PI-DeepONet) in PyTorch

A PyTorch reimplementation and modular extension of **Separable Physics-Informed Deep Operator Networks (Sep-PI-DeepONet)**, based on the research by Mandl et al. (CMAME 2025) and inspired by the Separable PINN framework (Cho et al., NeurIPS 2023).

-----
## 1. Overview & Theoretical Background

Physics-Informed Deep Operator Networks (PI-DeepONet) learn nonlinear solution operators $\mathcal{G}: \mathcal{U} \to \mathcal{S}$ mapping infinite-dimensional function spaces to solution spaces using the residual loss of governing partial differential equations (PDEs).

However, conventional PI-DeepONet suffers severely from the **curse of dimensionality**:
- On an $n^d$ discretization of a $d$-dimensional domain with $N$ input functions, conventional PI-DeepONet must evaluate $N \cdot n^d$ forward passes of a single multi-layer perceptron (MLP) trunk network.
- Gradient computation with reverse-mode automatic differentiation (AD) requires evaluating Jacobians of size $N n^d \times N n^d$, rendering problems beyond 2D computationally intractable.

### Key Innovations of Sep-PI-DeepONet:
1. **Factorized Coordinates and 1D Sub-networks**:
   The multidimensional trunk is decomposed into $d$ independent 1D sub-networks $tr^{(j)}: \mathbb{R} \to \mathbb{R}^{r \cdot p \cdot c}$ ($j \in \{1, \dots, d\}$), reducing trunk network evaluations from $\mathcal{O}(N n^d)$ to $\mathcal{O}(nd)$.
2. **Tensor Rank Approximation**:
   Outputs of the $d$ 1D trunk networks are combined via outer product and contracted over tensor rank $r$:
   $$\text{tr}_{i_1, \dots, i_d, \alpha, k} = \sum_{\beta=1}^r \prod_{j=1}^d \text{tr}^{(j)}_{i_j, \beta, \alpha, k}$$
3. **Latent Contraction with Branch Network**:
   The branch network encodes the input function $u$ at sensor locations into latent embedding $br \in \mathbb{R}^{N_u \times p \times c}$. Contracting across latent dimension $p$ gives the final solution:
   $$\mathcal{G}(u)(\zeta)_{m, i_1, \dots, i_d, k} = \sum_{\alpha=1}^p br_{m, \alpha, k} \cdot \text{tr}_{i_1, \dots, i_d, \alpha, k} + b_k$$
4. **Vectorized Forward-Mode Automatic Differentiation**:
   Because each trunk sub-network is strictly 1D, derivatives up to degree $d$ are evaluated in $\mathcal{O}(nd)$ operations using forward-mode AD (`torch.func.jacfwd` and `vmap`). Any mixed partial derivative $\frac{\partial^{|d|}}{\partial \zeta_1^{d_1} \dots \partial \zeta_d^{d_d}} \mathcal{G}(u)(\zeta)$ is then computed via a single contracted Einstein summation without backpropagation or Jacobian construction.

-----
## 2. Directory & Architecture Structure

```
separable_pi_deeponet/
├── __init__.py                     # Package entry point and exports
├── autograd.py                     # Vectorized forward-mode autodiff, hvp_fwdfwd, compute_grad
├── helpers.py                      # Loss metrics (mse, mse_single, relative_l2_error), apply_net
├── data.py                         # SeparableDataGenerator and Gaussian Random Field generators (1D & 2D)
├── models/
│   ├── __init__.py                 # Model package exports
│   ├── base.py                     # BaseModule with parameter counting and summary
│   ├── deeponet.py                 # Conventional (Vanilla) DeepONet baseline
│   ├── separable_deeponet.py       # Separable PI-DeepONet (MLP branch + separable 1D trunks)
│   ├── separable_cnn_deeponet.py   # PI-Sep-CNN-DeepONet (2D CNN branch + separable 1D trunks)
│   └── setup_model.py              # setup_deeponet model factory and DeepONetConfig
├── burgers_separable.py            # 1D Burgers' equation Sep-PI-DeepONet solver
├── burgers_separable_conservative.py # Conservative form Burgers' Sep-PI-DeepONet
├── burgers.py                      # Conventional PI-DeepONet Burgers' baseline
├── heat_separable.py               # 4D Parameterized Heat Equation Sep-PI-DeepONet solver
├── heat.py                         # Conventional PI-DeepONet Heat baseline
├── biot_separable.py               # Biot's Consolidation Theory Sep-PI-DeepONet (2-field system)
├── biot.py                         # Conventional PI-DeepONet Biot baseline
└── poisson_separable_cnn.py        # 2D Poisson with Gaussian Random Field source & CNN branch
```

-----
## 3. Quick Start & Code Example

### Instantiating Separable PI-DeepONet in PyTorch:
```python
import torch
from separable_pi_deeponet import SeparableDeepONet

# Example: 2D problem (t, x), rank r = 50, latent dim p = 50
model = SeparableDeepONet(
    branch_layers = [101, 100, 100, 100, 50],  # 101 sensors -> 50 latent features
    trunk_layers  = [1, 50, 50, 50, 50],        # 1D input -> 50 features per trunk
    n_trunk_vars  = 2,                          # 2 coordinates: (t, x)
    r             = 50,                         # Tensor decomposition rank
    output_dim    = 1,                          # Scalar solution field
    deg_grad      = [1, 2],                     # Max derivative: t order 1, x order 2
)

# Input data
u = torch.randn(32, 101)  # 32 initial condition functions
t = torch.linspace(0, 1, 50).reshape(-1, 1)  # 50 time points
x = torch.linspace(0, 1, 50).reshape(-1, 1)  # 50 spatial points

# Forward pass (precomputes and caches trunk derivatives)
s = model(u, t, x, compute_derivatives=True)  # Shape: (32, 50, 50, 1)

# Instant evaluation of partial derivatives via contracted einsum:
s_t  = model.compute_gradient([1, 0])  # ds/dt
s_x  = model.compute_gradient([0, 1])  # ds/dx
s_xx = model.compute_gradient([0, 2])  # d^2s/dx^2

# Viscous Burgers' PDE residual: s_t + s * s_x - 0.01 * s_xx = 0
res = s_t + s * s_x - 0.01 * s_xx
loss_res = torch.mean(torch.square(res))
```

-----
## 4. Benchmark Problems Included

| Benchmark Script | Governing Equation | Trunk Coordinates | Output Field | Speedup over Vanilla |
|---|---|---|---|---|
| `burgers_separable.py` | $\partial_t u + u \partial_x u - \nu \partial_{xx} u = 0$ | $(t, x)$ ($d=2$) | $u(x, t)$ | ~38x (97.3% faster) |
| `heat_separable.py` | $\partial_t T - \alpha (\partial_{xx} T + \partial_{yy} T) = 0$ | $(t, x, y, \sqrt{\alpha})$ ($d=4$) | $T(x, y, t, \alpha)$ | >100x (2.5h vs. 289h) |
| `biot_separable.py` | $(\lambda+2\mu)u_{zz}-p_z=0$, $u_{tz}-\frac{k}{\rho g}p_{zz}=0$ | $(t, z)$ ($d=2$) | $[u(z, t), p(z, t)]$ ($c=2$) | ~45x (97.8% faster) |
| `poisson_separable_cnn.py` | $k(\partial_{xx} S + \partial_{yy} S) + u(x, y) = 0$ | $(x, y, k)$ ($d=3$) | $S(x, y; k)$ (CNN branch) | Native separable CNN |

-----
## 5. References

1. **Luis Mandl, Somdatta Goswami, Lena Lambers, and Tim Ricken.**  
   *Separable physics-informed DeepONet: Breaking the curse of dimensionality in physics-informed machine learning.*  
   Computer Methods in Applied Mechanics and Engineering, Volume 434, 117586, 2025.  
   [DOI: 10.1016/j.cma.2024.117586](https://doi.org/10.1016/j.cma.2024.117586)

2. **Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park.**  
   *Separable Physics-Informed Neural Networks.*  
   Advances in Neural Information Processing Systems (NeurIPS 2023), 2023.  
   [Link: NeurIPS 2022/2023](https://neurips.cc/virtual/2022/59890)

3. **Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis.**  
   *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.*  
   Nature Machine Intelligence, 3(3):218–229, 2021.

4. **Sifan Wang, Hanwen Wang, and Paris Perdikaris.**  
   *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets.*  
   Science Advances, 7(40):eabi8605, 2021.
