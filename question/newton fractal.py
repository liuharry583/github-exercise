import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


def plot_newton_fractal(
    x_c=0.0, y_c=0.0, d=1.0, reso=0.002, title="Newton Fractal for $f(z) = z^3 - 1$"
):
    points = int((2 * d) / reso) + 1

    x = np.linspace(x_c - d, x_c + d, points)
    y = np.linspace(y_c - d, y_c + d, points)

    X, Y = np.meshgrid(x, y)

    Z = X + 1j * Y

    # 处理奇点，避免除以零
    Z[Z == 0] = 1e-10 + 1e-10j

    # 执行牛顿迭代
    iter = 40

    for _ in range(iter):
        Z = Z - (Z**3 - 1) / (3 * Z**2)

    # 确定根的归属
    root1 = 1 + 0j
    root2 = -0.5 + np.sqrt(3) / 2 * 1j
    root3 = -0.5 - np.sqrt(3) / 2 * 1j

    dist1 = np.abs(Z - root1)
    dist2 = np.abs(Z - root2)
    dist3 = np.abs(Z - root3)

    roots_index = np.zeros(Z.shape, dtype=int)

    roots_index[(dist1 < dist2) & (dist1 < dist3)] = 0
    roots_index[(dist2 < dist1) & (dist2 < dist3)] = 1
    roots_index[(dist3 < dist1) & (dist3 < dist2)] = 2

    # 画图
    cmap = ListedColormap(["red", "green", "yellow"])

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(
        roots_index,
        extent=[x_c - d, x_c + d, y_c - d, y_c + d],
        origin="lower",
        cmap=cmap,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Re(z)", fontsize=12)
    plt.ylabel("Im(z)", fontsize=12)

    legend_elements = [
        Patch(facecolor="red", label="Root 1: $1$"),
        Patch(facecolor="green", label="Root 2: $-0.5 + \\frac{\\sqrt{3}}{2}i$"),
        Patch(facecolor="yellow", label="Root 3: $-0.5 - \\frac{\\sqrt{3}}{2}i$"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()


plot_newton_fractal(0.0, 0.0, 1.0, 0.002, "Newton Fractal (Global)")

plot_newton_fractal(
    -0.8, 0.0, 0.25, 0.0005, "Zoom (a): Center (-0.8, 0.0), Half-width 0.25"
)

plot_newton_fractal(
    -0.56, 0.18, 0.1, 0.0002, "Zoom (b): Center (-0.56, 0.18), Half-width 0.1"
)
