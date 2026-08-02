import numpy as np

def least_squares_normal(A, b):
    G = A.T @ A
    c = A.T @ b
    x = np.linalg.solve(G, c)
    return x

A = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4]
], dtype=float)

b = np.array([6, 5, 7, 10], dtype=float)
x = least_squares_normal(A, b)
print("\nsolution x:", x)
