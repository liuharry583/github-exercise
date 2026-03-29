import numpy as np

def rk4_step(f, u, t, dt, p):
    """
    Performs a single 4th-order Runge-Kutta step.
    
    Parameters:
        f: Callable, the derivative function f(u, t, p).
        u: numpy array, current state vector.
        t: float, current time.
        dt: float, time step size.
        p: tuple/dict, parameters to pass to f.
        
    Returns:
        numpy array, the next state vector.
    """
    k1 = f(u, t, p)
    k2 = f(u + 0.5 * dt * k1, t + 0.5 * dt, p)
    k3 = f(u + 0.5 * dt * k2, t + 0.5 * dt, p)
    k4 = f(u + dt * k3, t + dt, p)
    
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_solve(f, u0, t_span, dt, p):
    """
    Solves an ODE system using the 4th-order Runge-Kutta method.
    
    Parameters:
        f: Callable, the derivative function f(u, t, p).
        u0: numpy array, initial state vector.
        t_span: tuple (t0, tf), start and end times.
        dt: float, time step size.
        p: tuple/dict, parameters to pass to f.
        
    Returns:
        t_eval: numpy array of time points.
        u_eval: numpy array of state vectors at each time point.
    """
    t0, tf = t_span
    t_eval = np.arange(t0, tf, dt)
    
    # Initialize state array
    u_eval = np.zeros((len(t_eval), len(u0)))
    u_eval[0] = u0
    
    u_current = np.array(u0, dtype=float)
    
    for i in range(1, len(t_eval)):
        t_current = t_eval[i-1]
        u_current = rk4_step(f, u_current, t_current, dt, p)
        u_eval[i] = u_current
        
    return t_eval, u_eval