# Part 1 of `opPINN` for approximating the Fokker-Planck-Landau collision operator

**Note**
As for the experiment for the BKW solution, we set $T = 2\pi$, i.e., $R = \pi$.

This directory provides Python scripts which trains neural networks or neural operators to approximate the operator $D$ and $F$ defined in the paper of [opPINN](https://www.sciencedirect.com/science/article/pii/S0021999123001262).
Note that the operators are trained with the coefficient $\Lambda_0$ of the collision operator set to $1$ and the density $\rho_0$ of the input distribution set to $0.2$.
Training PINNs for other values of $\Lambda$ and $\rho$ can be accomplished by multiplying $(\Lambda/\Lambda_0) (\rho/\rho_0)^2$ to the trained neural networks (neural operators).


## Dataset
The dataset used to train and validate the surrogate models for the collision operator consists of the following three types of distribution functions of density $\rho_0 = 0.2$.
1. Maxwellian distribution
2. Sum of two Maxwellian distributions
3. Maxwellian distribution perturbed by a quadratic function