# specPINN: Physics-Informed Neural Networks with Fast Spectral Collision Evaluation for the Fokker–Planck–Landau Equation

<!--
Comment:
- The title is relatively long; consider splitting it into a main title and a subtitle if desired.
- The phrase "without surrogate model for the collision operator" is emphasized in the main text,
  so it could be shortened here for conciseness.
-->

This repository presents **specPINN**, a physics-informed neural network (PINN) framework for solving the **Fokker–Planck–Landau (FPL) equation** by **directly evaluating the collision operator using the fast spectral method**, *without introducing any surrogate model for the collision term*.

<!--
Comment:
- "presents a framework" is more concise than "provides a framework which trains".
- The first paragraph clearly states the main contribution: direct collision evaluation without surrogate models.
-->

Recent studies have proposed deep-learning-based solvers for the FPL equation.
In particular, **opPINN** [1] introduces a two-stage strategy in which two neural networks (or neural operators) are trained to approximate the linear operators composing the FPL collision operator
(referred to as *Step 1* in the original paper).

<img src="./_readme__images/framework_opPINN.png" width="1280px" height="720px" title="The opPINN framework to train the solution to the Fokker-Planck-Landau equation.">

<!--
Comment:
- Summarizing opPINN as a "two-stage strategy" helps position specPINN conceptually.
- Optionally, one could add "prior to training the PINN itself" for additional clarity.
-->

In contrast, **specPINN bypasses this surrogate-modeling step entirely**.
Instead of learning an approximation of the collision operator, we employ the **fast spectral method** [2], a well-established quasi-linear numerical approach, to compute the collision term of the distribution function **on-the-fly during PINN training**.

<img src="./_readme__images/framework_specPINN.png" width="1280px" height="720px" title="The specPINN framework to train the solution to the Fokker-Planck-Landau equation.">


<!--
Comment:
- The phrase "on-the-fly during PINN training" highlights the algorithmic difference and should be retained.
- This naturally motivates later discussions on stability, accuracy, and computational cost.
-->


## Advantages

Replacing *Step 1* in opPINN with a numerical evaluation of the collision operator provides the following advantages:

1. **Reduced training cost**  
   Since no surrogate model for the collision operator is trained, the overall training time of the PINN can be significantly reduced.

2. **Improved generalization with respect to initial conditions**  
   The PINN can be trained for initial distributions that do not belong to, or are far from, the training set used for a surrogate collision model.

3. **Potential computational efficiency**  
   The fast spectral method computes the collision term with $O(N^d \log N)$ computational complexity and $O(d^2 N^d)$ memory usage.
   Depending on the architecture and cost of the surrogate model, this direct evaluation may lead to faster or more stable training.

<!--
Comment:
- Using bold subtitles for each advantage improves readability.
- The phrase "may lead to faster or more stable training" is deliberately cautious and academically appropriate.
- If numerical results are available, consider referencing them explicitly (e.g., "as demonstrated in Section X").
-->

## Experimental results
The following figure shows the relative $L^2_v$ error of the predicted solution over time.
For the case of soft potential, the neural networks trained with the specPINN framework outperform the neural networks trained with the opPINN framework.

<img src="./_readme__images/aggregated_relative_error__epoch5000.png" width="1280px" height="720px" title="The relative $L^2_v$ error of the neural networks trained with the opPINN framework and the specPINN framework.">

## References

[1] Jae Yong Lee, Juhi Jang, Hyung Ju Hwang,  
*[opPINN: Physics-informed neural network with operator learning to approximate solutions to the Fokker–Planck–Landau equation](https://doi.org/10.1016/j.jcp.2023.112031)*, Journal of Computational Physics, Volume 480, 2023, 112031.

[2] L. Pareschi, G. Russo, G. Toscani,  
*[Fast Spectral Methods for the Fokker–Planck–Landau Collision Operator](https://www.sciencedirect.com/science/article/pii/S0021999100966129)*, Journal of Computational Physics, Volume 165, Issue 1, 2000, Pages 216–236.

<!--
Comment:
- Formatting references with line breaks improves readability.
- For a public repository, consider adding a separate "How to Cite" section.
-->
