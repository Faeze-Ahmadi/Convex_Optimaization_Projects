import numpy as np
from rosenbrock import grad_f

def nesterov(x0, alpha=0.0003, beta=0.9, max_iter=5000, tol=1e-6):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    path = [x.copy()]

    for i in range(max_iter):
        lookahead = x + beta * v
        g = grad_f(lookahead)
        grad_norm = np.linalg.norm(g)
        if grad_norm > 100:
            g = g * (100.0 / grad_norm)

        if np.linalg.norm(g) < tol:
            break

        v_new = beta * v - alpha * g

        if np.linalg.norm(v_new) > 2:
            v_new = v_new / np.linalg.norm(v_new) * 2

        x_new = x + v_new

        if np.any(np.abs(x_new) > 1e6):
            print("Safeguard triggered: divergence prevented")
            break

        x = x_new
        v = v_new
        path.append(x.copy())
    return x, path
