import numpy as np
from rosenbrock import f, grad_f

def rmsprop(x0, lr=0.001, rho=0.9, epsilon=1e-8, max_iter=10000):
    x = np.array(x0, dtype=float)
    Eg = np.zeros_like(x)
    path = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        Eg = rho * Eg + (1 - rho) * (g**2)
        x = x - lr * g / (np.sqrt(Eg) + epsilon)
        path.append(x.copy())

        if np.linalg.norm(g) < 1e-6:
            break

    return x, path
