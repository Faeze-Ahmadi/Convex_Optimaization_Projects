import numpy as np
from rosenbrock import f, grad_f
from line_search import wolfe_line_search, goldstein_line_search

def gradient_descent(x0, max_iter=10000, tol=1e-6, method="wolfe"):
    x = x0.copy()
    path = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break
        d = -g

        if method == "wolfe":
            alpha = wolfe_line_search(f, grad_f, x, d)

        elif method == "goldstein":
            alpha = goldstein_line_search(f, grad_f, x, d)

        else:
            raise ValueError("method must be 'wolfe' or 'goldstein'")

        x = x + alpha * d
        path.append(x.copy())
    return x, path
