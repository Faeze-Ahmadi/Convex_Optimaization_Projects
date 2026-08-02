import numpy as np
from rosenbrock import f, grad_f

def adadelta(x0, rho=0.9, epsilon=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    Eg = np.zeros_like(x)
    Edx = np.zeros_like(x)
    path = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        Eg = rho * Eg + (1 - rho) * (g * g)
        rms_g = np.sqrt(Eg + epsilon)
        rms_dx = np.sqrt(Edx + epsilon)

        dx = - (rms_dx / rms_g) * g
        x_new = x + dx
        
        Edx = rho * Edx + (1 - rho) * (dx * dx)

        path.append(x_new.copy())
        x = x_new

        if np.linalg.norm(g) < 1e-6:
            break

    return x, path
