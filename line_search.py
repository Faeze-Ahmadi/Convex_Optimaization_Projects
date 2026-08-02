import numpy as np

def wolfe_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9, alpha=1.0, rho=0.5):
    fx = f(x)
    g = grad_f(x)
    gTd = np.dot(g, d)

    while True:
        new_x = x + alpha * d

        if f(new_x) > fx + c1 * alpha * gTd:
            alpha *= rho
            continue

        new_g = grad_f(new_x)

        if np.dot(new_g, d) < c2 * gTd:
            alpha *= rho
            continue

        break
    return alpha

def goldstein_line_search(f, grad_f, x, d, c=0.2, alpha=1.0, rho=0.5, max_iter=50):
    fx = f(x)
    g = grad_f(x)
    gTd = np.dot(g, d)

    for _ in range(max_iter):
        new_x = x + alpha * d
        f_new = f(new_x)

        if f_new > fx + c * alpha * gTd:
            alpha = rho * alpha

        elif f_new < fx + (1 - c) * alpha * gTd:
            alpha = alpha / rho

        else:
            return alpha

        if alpha < 1e-8:
            return alpha
    return alpha

