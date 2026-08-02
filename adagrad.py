import numpy as np
from rosenbrock import f, grad_f

def adagrad(x0, lr=0.1, epsilon=1e-8, max_iter=10000):
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)
    path = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        G += g**2
        adjusted_lr = lr / (np.sqrt(G) + epsilon)
        x = x - adjusted_lr * g
        path.append(x.copy())

        if np.linalg.norm(g) < 1e-6:
            break

    return x, path
