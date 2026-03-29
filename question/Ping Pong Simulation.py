import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 参数
G = 10.0
GAMMA = 0.02
A = 0.02
OMEGA = 4 * np.pi

def exact_v(t, t0, v0):
    """Analytical solution for velocity between collisions."""
    return (v0 + G/GAMMA) * np.exp(-GAMMA * (t - t0)) - G/GAMMA

def exact_y(t, t0, y0, v0):
    """Analytical solution for position between collisions."""
    return y0 + (v0 + G/GAMMA) / GAMMA * (1 - np.exp(-GAMMA * (t - t0))) - G/GAMMA * (t - t0)

def paddle_h(t):
    return A * np.sin(OMEGA * t)

def paddle_v(t):
    return A * OMEGA * np.cos(OMEGA * t)

def simulate_pingpong(y0, t_max, dt_eval=0.01):
    """
    Event-driven simulation of the ping-pong ball using exact analytical 
    inter-collision trajectories and Brent's root-finding method.
    """
    t_curr, y_curr, v_curr = 0.0, y0, 0.0
    
    t_history, y_history, h_history = [], [], []
    
    while t_curr < t_max:
        # Step forward to bracket the next collision root
        t_search = t_curr + 1e-4
        step = 0.05
        root_found = False
        
        while t_search <= t_max + step:
            if exact_y(t_search, t_curr, y_curr, v_curr) <= paddle_h(t_search):
                root_found = True
                break
            t_search += step
            
        if not root_found:
            t_arr = np.arange(t_curr, t_max, dt_eval)
            if len(t_arr) > 0:
                t_history.append(t_arr)
                y_history.append(exact_y(t_arr, t_curr, y_curr, v_curr))
                h_history.append(paddle_h(t_arr))
            break
            
        # Bracketed root definition
        def diff(t):
            return exact_y(t, t_curr, y_curr, v_curr) - paddle_h(t)
        
        t_left = max(t_curr + 1e-6, t_search - step)
        t_right = t_search
        
        try:
            t_col = brentq(diff, t_left, t_right)
        except ValueError:
            t_col = t_right # Fallback if precision fails bracketing
        
        # Record trajectory up to collision
        t_arr = np.arange(t_curr, t_col, dt_eval)
        if len(t_arr) > 0:
            t_history.append(t_arr)
            y_history.append(exact_y(t_arr, t_curr, y_curr, v_curr))
            h_history.append(paddle_h(t_arr))
        
        # Exact collision state
        t_history.append([t_col])
        y_history.append([paddle_h(t_col)])
        h_history.append([paddle_h(t_col)])
        
        # Update velocities via collision mechanics
        v_before = exact_v(t_col, t_curr, v_curr)
        vp = paddle_v(t_col)
        v_after = -v_before + 2 * vp
        
        # Anti-chattering safeguard
        v_after = max(v_after, vp + 1e-3)
        
        t_curr, y_curr, v_curr = t_col, paddle_h(t_col), v_after
        
    return np.concatenate(t_history), np.concatenate(y_history), np.concatenate(h_history)

def plot_results():
    # Case 1: Initial height = 0.3
    t1, y1, h1 = simulate_pingpong(y0=0.3, t_max=1000)
    
    # Case 2: Initial height = 1.2 (seeking different long-term mode)
    t2, y2, h2 = simulate_pingpong(y0=1.2, t_max=1000)

    # Plot Configuration
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Early transient for y0=0.3
    mask_early = (t1 >= 0) & (t1 <= 10)
    axs[0].plot(t1[mask_early], y1[mask_early], label='Ball (y0=0.3)', color='blue')
    axs[0].plot(t1[mask_early], h1[mask_early], label='Paddle', color='red', alpha=0.6)
    axs[0].set_title('Case 1: Early Trajectory t ∈ (0, 10)')
    axs[0].set_ylabel('Height y(t)')
    axs[0].legend()
    
    # Plot 2: Steady state for y0=0.3
    mask_late1 = (t1 >= 990) & (t1 <= 1000)
    axs[1].plot(t1[mask_late1], y1[mask_late1], label='Ball (y0=0.3)', color='blue')
    axs[1].plot(t1[mask_late1], h1[mask_late1], label='Paddle', color='red', alpha=0.6)
    axs[1].set_title('Case 1: Steady State Trajectory t ∈ (990, 1000) - Periodic Mode')
    axs[1].set_ylabel('Height y(t)')
    axs[1].legend()

    # Plot 3: Steady state for y0=1.2
    mask_late2 = (t2 >= 990) & (t2 <= 1000)
    axs[2].plot(t2[mask_late2], y2[mask_late2], label='Ball (y0=1.2)', color='green')
    axs[2].plot(t2[mask_late2], h2[mask_late2], label='Paddle', color='red', alpha=0.6)
    axs[2].set_title('Case 2: Steady State Trajectory t ∈ (990, 1000) - Distinct Mode (Higher period/Chaotic)')
    axs[2].set_xlabel('Time t')
    axs[2].set_ylabel('Height y(t)')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()