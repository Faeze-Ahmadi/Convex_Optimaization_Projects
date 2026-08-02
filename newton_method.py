import numpy as np
from rosenbrock import f, grad_f, hessian_f
from line_search import wolfe_line_search, goldstein_line_search


def newton_method(x0, max_iter=50, tol=1e-6, method="wolfe"):
    x = x0.copy()
    path = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break

        H = hessian_f(x)

        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
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
