import numpy as np
import matplotlib.pyplot as plt
from rosenbrock import f, grad_f
from gradient_descent import gradient_descent
from newton_method import newton_method
from momentum import momentum
from nesterov import nesterov
from adagrad import adagrad
from adadelta import adadelta
from rmsprop import rmsprop
from adam import adam


def setup_plot_style():
    pink = "#ff6fa3"
    pink2 = "#ff4f8b"
    gray = "#7f7f7f"
    blue_bg = "#eaf4ff"

    plt.rcParams.update({
        "axes.grid": False,
        "axes.facecolor": blue_bg,
        "figure.facecolor": blue_bg,
        "axes.edgecolor": gray,
        "axes.labelcolor": gray,
        "xtick.color": gray,
        "ytick.color": gray,
        "text.color": gray,
        "font.size": 11,
    })
    return pink, pink2, gray


def contour_plot(ax, xlim, ylim, levels=60):
    xs = np.linspace(xlim[0], xlim[1], 300)
    ys = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = f(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = z

    ax.contour(X, Y, Z, levels=levels, cmap="Greys", alpha=0.75)
    return Z


def run_and_plot(optimizer_name, optimizer_fn, x0, plot_cfg, ax, line_style):
    xlim = plot_cfg["xlim"]
    ylim = plot_cfg["ylim"]

    pink, pink2, gray = setup_plot_style()

    x_opt, history = optimizer_fn(x0)

    path = np.array(history)
    final_f = f(path[-1])
    final_gnorm = np.linalg.norm(grad_f(path[-1]))

    ax.plot(
        path[:, 0], path[:, 1],
        color="#ff4f8b",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markerfacecolor="none",
        markeredgecolor="#ff4f8b"
    )

    ax.scatter(
        path[0, 0], path[0, 1],
        color=pink,
        s=45, marker="o",
        edgecolor="white", linewidth=0.8,
        label="start"
    )
    ax.scatter(
        path[-1, 0], path[-1, 1],
        color=pink2,
        s=55, marker="*",
        edgecolor="white", linewidth=0.8,
        label="end"
    )

    ax.set_title(
        f"{optimizer_name} | iters={len(history)-1}\n"
        f"f*={final_f:.4e} , ||∇f||={final_gnorm:.4e}",
        pad=12
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.legend(loc="best", frameon=True)
    return {
        "name": optimizer_name,
        "x_opt": path[-1],
        "iters": len(history) - 1,
        "f_opt": final_f,
        "gnorm": final_gnorm,
    }


def main():
    x0 = np.array([-1.5, 2.0], dtype=float)
    grad_tol = 1e-6

    plot_cfg = {
        "xlim": (-3.0, 3.0),
        "ylim": (-1.0, 6.0),
    }

    optimizers = [
        ("gradient descent (wolfe)", lambda x: gradient_descent(x, method="wolfe"), "-"),
        ("gradient descent (goldstein)",
         lambda x: gradient_descent(x, method="goldstein"), "-"),

        ("newton (wolfe)", lambda x: newton_method(x, method="wolfe"), "--"),
        ("newton (goldstein)", lambda x: newton_method(x, method="goldstein"), "--"),

        ("momentum", momentum, "-."),
        ("nesterov", nesterov, ":"),
        ("adagrad", adagrad, "-"),
        ("adadelta", adadelta, "--"),
        ("rmsprop", rmsprop, "-."),
        ("adam", adam, ":"),
    ]

    summary = []
    for name, fn, ls in optimizers:
        print("_" * 70)
        print(f"running: {name}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_facecolor("#eaf4ff")
        ax.grid(False)

        result = run_and_plot(
            optimizer_name=name,
            optimizer_fn=fn,
            x0=x0,
            plot_cfg=plot_cfg,
            ax=ax,
            line_style=ls
        )

        summary.append(result)

        x_opt = result["x_opt"]
        print(f"{name} finished.")
        print(f"  x_opt = [{x_opt[0]:.8f}, {x_opt[1]:.8f}]")
        print(f"  iters = {result['iters']}")
        print(f"  f(x_opt) = {result['f_opt']:.8e}")
        print(f"  ||grad|| = {result['gnorm']:.8e}")

        plt.tight_layout()
        plt.show()

    print("\n" + "-" * 70)
    print("summary (best by final f):")
    summary_sorted = sorted(summary, key=lambda d: d["f_opt"])
    for rank, r in enumerate(summary_sorted, 1):
        print(
            f"{rank}. {r['name']}: f*={r['f_opt']:.4e}, "
            f"iters={r['iters']}, ||grad||={r['gnorm']:.3e}"
        )


if __name__ == "__main__":
    main()
