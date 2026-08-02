import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# تابع و گرادیان


def f(x):
    return (2 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def grad_f(x):
    df_dx1 = -2*(2 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    df_dx2 = 200*(x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

# پیاده‌سازی الگوریتم‌های مرتبه اول


def momentum(start, lr=0.001, beta=0.9, iters=1000):
    x = np.array(start, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(iters):
        v = beta * v - lr * grad_f(x)
        x += v
        path.append(x.copy())
    return np.array(path), iters


def nesterov(start, lr=0.001, beta=0.9, iters=1000):
    x = np.array(start, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(iters):
        x_lookahead = x + beta * v
        v = beta * v - lr * grad_f(x_lookahead)
        x += v
        path.append(x.copy())
    return np.array(path), iters


def adagrad(start, lr=0.5, epsilon=1e-8, iters=1000):
    x = np.array(start, dtype=float)
    G = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(iters):
        g = grad_f(x)
        G += g**2
        x -= lr / (np.sqrt(G) + epsilon) * g
        path.append(x.copy())
    return np.array(path), iters


def rmsprop(start, lr=0.1, beta=0.9, epsilon=1e-8, iters=1000):
    x = np.array(start, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(iters):
        g = grad_f(x)
        v = beta * v + (1 - beta) * g**2
        x -= lr / (np.sqrt(v) + epsilon) * g
        path.append(x.copy())
    return np.array(path), iters


def adadelta(start, beta=0.9, epsilon=1e-8, iters=1000):
    x = np.array(start, dtype=float)
    E_g2 = np.zeros_like(x)
    E_dx2 = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(iters):
        g = grad_f(x)
        E_g2 = beta * E_g2 + (1 - beta) * g**2
        dx = - (np.sqrt(E_dx2 + epsilon) / np.sqrt(E_g2 + epsilon)) * g
        E_dx2 = beta * E_dx2 + (1 - beta) * dx**2
        x += dx
        path.append(x.copy())
    return np.array(path), iters


def adam(start, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, iters=1000):
    x = np.array(start, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, iters + 1):
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())
    return np.array(path), iters


# تنظیمات و اجرای مقایسه‌ای
print("optimization algorithms simulator")
print("select a starting point:")
print("1. [-1.0,  2.0] (classic difficult start)")
print("2. [ 3.0,  3.0] (far start)")
print("3. [ 0.0, -1.0] (bottom of the valley)")

choice = input("enter 1, 2, or 3 (default is 1): ")
if choice == '2':
    start = [3.0, 3.0]
elif choice == '3':
    start = [0.0, -1.0]
else:
    start = [-1.0, 2.0]

# گرفتن ضریب یادگیری پایه
user_lr = input(
    "enter a base learning rate for adam/momentum (e.g. 0.01, 0.1) [default: 0.1]: ")
try:
    base_lr = float(user_lr)
except ValueError:
    base_lr = 0.1

print(
    f"\nrunning with start point: {start} and base learning rate: {base_lr}...\n")

# اجرای الگوریتم ها (با استفاده از base_lr برای آدام و تنظیم نسبی بقیه)
p_mom, cost_mom = momentum(start, lr=base_lr/200, iters=2000)
p_nes, cost_nes = nesterov(start, lr=base_lr/200, iters=2000)
p_ada, cost_ada = adagrad(start, lr=base_lr*5, iters=2000)
p_rms, cost_rms = rmsprop(start, lr=base_lr/2, iters=2000)
# آدادلتا معمولاً lr نیاز ندارد
p_delta, cost_delta = adadelta(start, iters=2000)
p_adam, cost_adam = adam(start, lr=base_lr, iters=2000)

# شبه نیوتنی (BFGS) که از جستجوی خطی با شرایط Wolfe استفاده میکند
res_bfgs = minimize(f, start, method='BFGS', jac=grad_f,
                    options={'return_all': True})
p_bfgs = np.array(res_bfgs.allvecs)
cost_bfgs = res_bfgs.nfev + res_bfgs.njev


# اجرای الگوریتم ها
p_mom, cost_mom = momentum(start, lr=0.0005, iters=2000)
p_nes, cost_nes = nesterov(start, lr=0.0005, iters=2000)
p_ada, cost_ada = adagrad(start, lr=0.5, iters=2000)
p_rms, cost_rms = rmsprop(start, lr=0.05, iters=2000)
p_delta, cost_delta = adadelta(start, iters=2000)
p_adam, cost_adam = adam(start, lr=0.1, iters=2000)

# شبه نیوتنی (BFGS) که از جستجوی خطی با شرایط Wolfe استفاده میکند
res_bfgs = minimize(f, start, method='BFGS', jac=grad_f,
                    options={'return_all': True})
p_bfgs = np.array(res_bfgs.allvecs)
cost_bfgs = res_bfgs.nfev + res_bfgs.njev

# رسم نمودار
x1 = np.linspace(-2.5, 4.5, 400)
x2 = np.linspace(-1.5, 6.5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = (2 - X1)**2 + 100 * (X2 - X1**2)**2

fig, ax = plt.subplots(figsize=(12, 10))

# تنظیم پس‌زمینه صورتی کم‌رنگ
ax.set_facecolor('#FFF0F5')

# رسم کانتور های خطی
ax.contour(X1, X2, Z, levels=np.logspace(-1, 3.5, 30),
           cmap='viridis', alpha=0.6)

# رسم مسیرها با استایل‌های مختلف برای وضوح بیشتر
ax.plot(p_mom[:, 0], p_mom[:, 1], '--', color='cyan',
        label=f'momentum (cost: {cost_mom})', linewidth=2)
ax.plot(p_nes[:, 0], p_nes[:, 1], '-.', color='magenta',
        label=f'nesterov (cost: {cost_nes})', linewidth=2)
ax.plot(p_ada[:, 0], p_ada[:, 1], ':', color='lime',
        label=f'adagrad (cost: {cost_ada})', linewidth=2)
ax.plot(p_rms[:, 0], p_rms[:, 1], '--', color='orange',
        label=f'rmsprop (cost: {cost_rms})', linewidth=2)
ax.plot(p_delta[:, 0], p_delta[:, 1], '-', color='brown',
        label=f'adadelta (cost: {cost_delta})', linewidth=2)
ax.plot(p_adam[:, 0], p_adam[:, 1], '-', color='blue',
        label=f'adam (cost: {cost_adam})', linewidth=2.5)
ax.plot(p_bfgs[:, 0], p_bfgs[:, 1], '-o', color='black',
        label=f'quasi-newton + wolfe (cost: {cost_bfgs})', markersize=4, linewidth=2)

# نمایش نقطه کمینه سراسری با مثلث قرمز
ax.plot(2, 4, marker='^', color='red', markersize=16,
        markeredgecolor='black', label='global min (2,4)')

# تنظیمات گرافیکی نمودار
ax.set_title("comparison of optimization algorithms",
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
ax.set_xlim(-2.5, 4.5)
ax.set_ylim(-1.5, 6.5)
ax.grid(color='white', linestyle='-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.show()

# چاپ مقایسه هزینه  ها در ترمینال
print(f"cost (gradient/function evaluations):")
print(f"momentum: {cost_mom}, nesterov: {cost_nes}, adagrad: {cost_ada}")
print(f"RMSprop: {cost_rms}, adadelta: {cost_delta}, adam: {cost_adam}")
print(f"quasi-Newton(BFGS): {cost_bfgs}")
