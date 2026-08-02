import numpy as np
from rosenbrock import f, grad_f

def momentum(x0, alpha=0.001, beta=0.9, max_iter=1000, tol=1e-6):
    x = x0.copy()
    v = np.zeros_like(x)
    path = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        v = beta * v - alpha * g
        x = x + v
        path.append(x.copy())

    return x, path
