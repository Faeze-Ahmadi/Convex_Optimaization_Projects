import numpy as np
from least_squares_normal import least_squares_normal
from least_squares_gradient import least_squares_gradient

A = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4]
], dtype=float)

b = np.array([6, 5, 7, 10], dtype=float)

x_normal = least_squares_normal(A, b)
x_gradient = least_squares_gradient(A, b)

print("\nsolution with Normal Equation:", x_normal)
print("solution with Gradient Descent:", x_gradient)
