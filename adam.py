import numpy as np
from rosenbrock import f, grad_f

def adam(x0, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=10000):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]

    for t in range(1, max_iter + 1):
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())

        if np.linalg.norm(g) < 1e-6:
            break

    return x, path
