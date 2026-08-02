# Convex Optimization Projects

This repository contains the comprehensive coursework, mathematical reports, and Python implementations for the **Convex Optimization** course, instructed by **Dr. Mohsen Hooshmand** at IASBS.

The course bridged the gap between abstract mathematical theory (convexity proofs, linear algebra) and practical numerical optimization algorithms used in modern Machine Learning.

## Course Reports
The following reports detail my work across three main assignments, ranging from foundational theory to empirical analysis.

- **[Set 0: Foundations & Applications](./Convex_Optimization_Set_0_Report.pdf)**
  - Covers the **Leontief Input-Output model** and its economic applications.
  - Dimensionality reduction using **PCA** and numerical linear algebra foundations (SVD, Eigen-decomposition).
  - Theoretical analysis of convex/concave functions, **Jensen’s inequality**, and applications in **Soft-Margin SVM** and constrained **K-Means clustering**.

- **[Set 1: Convexity Analysis & First-Order Methods](./Convex_Optimization_Set_1_Report.pdf)**
  - Analytical proofs for the convexity of **Log-Sum-Exp** and **MSE** cost functions (including Hessian analysis).
  - Derivation of **Normal Equations** for Least Squares and implementing efficient solvers for sparse systems.
  - Empirical comparison of optimization algorithms on the **Rosenbrock function**, including Gradient Descent, Momentum, Nesterov, and adaptive methods (Adam, AdaGrad, RMSProp).

- **[Set 2: Experimental Study & Quasi-Newton Methods](./Convex_Optimization_Set_2_Report.pdf)**
  - An in-depth experimental study focusing on the **modified Rosenbrock function** to analyze convergence behavior, stability, and sensitivity to initialization.
  - Comparative analysis of **First-Order Methods** vs. **Quasi-Newton methods (BFGS)**.
  - Visualization of optimization trajectories, "valley" traversal challenges (zigzagging/convergence issues), and Wolfe line search conditions.

## Technical Implementations
The repository includes modular Python implementations of the discussed algorithms:
- **Foundational Solvers:** Least Squares, Normal Equations, Sparse matrix solvers.
- **First-Order Optimizers:** Gradient Descent, Momentum, Nesterov, AdaGrad, AdaDelta, RMSProp, Adam.
- **Quasi-Newton Methods:** BFGS implementation with line search.

All code is implemented using `NumPy` for matrix operations, ensuring clean and readable mathematical transitions from theory to execution.

## Learning Experience
This course was a challenging yet rewarding deep dive into the "why" behind Machine Learning. Learning to identify convex problems and apply rigorous optimization techniques transformed my understanding of model training. I am grateful to Dr. Hooshmand for his instruction, which has been instrumental in my academic growth.

---
*Note: These projects were completed as part of my academic curriculum at IASBS. Due to internet access constraints at the time of the course, they are being uploaded now to maintain a complete record of my technical growth and academic artifacts.*
