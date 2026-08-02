import numpy as np

def least_squares_gradient(A, b, alpha=0.005, max_iter=5000):
    n, k = A.shape
    x = np.zeros(k)
    for i in range(max_iter):
        r = A @ x - b
        g = A.T @ r
        if np.linalg.norm(g) < 1e-6:
            break

        x = x - 2 * alpha * g
    return x

A = np.array([
    [1,1],
    [1,2],
    [1,3],
    [1,4]
], dtype=float)
b = np.array([6,5,7,10], dtype=float)
x = least_squares_gradient(A,b)
print("\nsolution x:", x)
