import numpy as np
import matplotlib.pyplot as plt
from solvers import rk4_solve


def kapitza_f(u, t, p):
    """
    Equations of motion for the Kapitza pendulum.
    u = [\theta, \dot{\theta}]
    p = {'a': a, 'L': L, 'm': m, 'g': g, 'omega': omega}
    """
    theta = u[0]
    theta_dot = u[1]

    a = p["a"]
    L = p["L"]
    g = p["g"]
    omega = p["omega"]

    # First order system derivatives
    d_theta = theta_dot
    d_theta_dot = -(g / L - (a * omega**2 / L) * np.cos(omega * t)) * np.sin(theta)

    return np.array([d_theta, d_theta_dot])


def run_simulation():
    # Parameters setup
    params = {
        "a": 0.1,
        "L": 1.0,
        "m": 1.0,  # m is fundamentally decoupled from kinematics but maintained in params
        "g": 1.0,
    }

    omegas = [5, 10, 20]
    u0 = np.array([0.8 * np.pi, 0.0])  # Near the inverted position (pi)
    t_span = (0, 15)
    dt = 0.01

    plt.figure(figsize=(10, 6))

    # Solve and plot for each omega
    for omega in omegas:
        params["omega"] = omega
        t_eval, u_eval = rk4_solve(kapitza_f, u0, t_span, dt, params)

        # Extract theta(t)
        theta_t = u_eval[:, 0]

        # Plot normalized to pi for readability
        plt.plot(t_eval, theta_t / np.pi, label=f"$\omega = {omega}$")

    # Plot visual guide for the inverted position
    plt.axhline(
        1.0, color="red", linestyle="--", alpha=0.5, label="Inverted Position ($\pi$)"
    )
    plt.axhline(0.0, color="black", linestyle="-", alpha=0.3)

    plt.title("Kapitza Pendulum: Transition to Stability at Inverted Position")
    plt.xlabel("Time $t$")
    plt.ylabel(r"Angle $\theta(t) / \pi$")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
