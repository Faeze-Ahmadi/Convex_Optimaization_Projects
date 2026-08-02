import numpy as np

def f(x):
    x1, x2 = x[0], x[1]
    return (2 - x1)**2 + 100 * (x2 - x1**2)**2

def grad_f(x):
    x1, x2 = x[0], x[1]
    df_dx1 = -2*(2 - x1) - 400*x1*(x2 - x1**2)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

def hessian_f(x):
    x1, x2 = x[0], x[1]
    h11 = 2 - 400*(x2 - 3*x1**2)
    h12 = -400 * x1
    h21 = -400 * x1
    h22 = 200
    return np.array([[h11, h12],
                     [h21, h22]])
