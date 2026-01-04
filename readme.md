# A physics-informed neural network for solving the Fokker-Planck-Landau equation without surrogate model for the collision operator

This repository provides a framework which trains a neural network in the physics-informed manner to solve the Fokker-Planck-Landau (FPL) equation **without surrogate models to approximate the Fokker-Planck-Landau (FPL) collision operator**.
Several studies have provided methodologies to solve the FPL equation using deep learning, and [opPINN](https://doi.org/10.1016/j.jcp.2023.112031) suggests to train two neural networks (or neural operators) to approximate two linear operators comprising the FPL collision operator (which is called *Step 1* in the paper).
In this study, however, we do not conduct the generation of the surrogate model for the collision operator.
Instead, we apply the [fast spectral method](https://www.sciencedirect.com/science/article/pii/S0021999100966129), which is known as a quasilinear numerical method to compute the collision term of a distribution function, to compute the collision term of a given distribution function in the training of a PINN.


# Advantages

Replacing *Step 1* with the numerical method takes the follwing advantages:
1. As *Step 1* is not conducted, the total training time of a PINN can be significantly decreased.
2. PINNs can be trained for initial conditions which do not belong to the training set of the surrogate model for the collision operator.
3. Since the fast spectral method requires $O(N^d \log{N})$ computational complexity and $O(d^2 N^d)$ memory to compute the collision term, upon the architecture of the surrogate model, the fast spectral method might accelerate the training of PINNs.


# Reference

[1] Jae Yong Lee, Juhi Jang, Hyung Ju Hwang, [opPINN: Physics-informed neural network with operator learning to approximate solutions to the Fokker-Planck-Landau equation](https://doi.org/10.1016/j.jcp.2023.112031), Journal of Computational Physics, Volume 480, 2023, 112031, ISSN 0021-9991.

[2] L. Pareschi, G. Russo, G. Toscani, [Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator](https://www.sciencedirect.com/science/article/pii/S0021999100966129), Journal of Computational Physics, Volume 165, Issue 1, 2000, Pages 216-236, ISSN 0021-9991.
