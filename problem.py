# problem.py
import numpy as np


def build_mesh(L: float = 1.0, nel: int = 30) -> np.ndarray:
    """
    Create a 1D uniform mesh for the beam.

    Parameters
    ----------
    L : float, optional
        Total beam length. Default is 1.0.
    nel : int, optional
        Number of finite elements. The number of nodes is nel + 1.
        Default is 30.

    Returns
    -------
    x_nodes : ndarray of shape (nel+1,)
        Nodal coordinates from 0 to L (uniformly spaced).
    """
    nnodes = nel + 1
    x_nodes = np.linspace(0.0, L, nnodes)
    return x_nodes


def build_problem_functions(
    L: float,
    E0: float = 1.0,
    alpha: float = 0.5,
    T: float = 30,
    I0: float = 1.0,
    q0: float = 1.0,
    omega_g: float = 0.8*np.pi,
    Emin_ratio: float = 0.2,
):
    """
    Build callable functions defining time-varying stiffness and external load.

    This module assumes a *clean* model setup:
      - rhoA is constant (handled elsewhere)
      - EI is spatially uniform and depends only on time:
            EI(t) = E(t) * I0
      - The distributed load is a separable space-time function:
            q(x,t) = q0 * sin(pi x / L) * sin(omega_g t)

    Parameters
    ----------
    L : float
        Beam length.
    E0 : float, optional
        Initial Young's modulus scale. Default is 1.0.
    alpha : float, optional
        Linear stiffness-degradation slope parameter in E(t).
        Default is 0.5.
    T : float, optional
        Characteristic time scale for stiffness change in E(t).
    I0 : float, optional
        Second moment of area (constant). Default is 1.0.
    q0 : float, optional
        Load amplitude. Default is 1.0.
    omega_g : float, optional
        Forcing angular frequency (rad/s). Default is 0.8*pi.
    Emin_ratio : float, optional
        Lower bound ratio Emin/E0 used to prevent E(t) from becoming too small.
        Default is 0.2.

    Returns
    -------
    E_fun : callable
        Function E_fun(t) -> float, returning Young's modulus E(t).
    EI_fun : callable
        Function EI_fun(t) -> float, returning bending stiffness EI(t).
    q_fun : callable
        Function q_fun(x, t) -> float, returning distributed load q(x,t).
    omega_g : float
        Forcing angular frequency used in q_fun.
    """
    def E_fun(t: float) -> float:
        Emin = Emin_ratio * E0
        factor = 1.0 - alpha * (t / T)
        return max(E0 * factor, Emin)

    def EI_fun(t: float) -> float:
        return E_fun(t) * I0

    def q_fun(x: float, t: float) -> float:
        return q0 * np.sin(np.pi * x / L) * np.sin(omega_g * t)

    return E_fun, EI_fun, q_fun, omega_g


def build_initial_conditions(
    x_nodes: np.ndarray,
    u0_fun=None,
    v0_fun=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct initial conditions for the full DOF vector.

    The global DOF ordering is assumed to be:
        u_full = [u_0, theta_0, u_1, theta_1, ..., u_N, theta_N]
        v_full = [uDot_0, thetaDot_0, ..., uDot_N, thetaDot_N]

    This helper fills nodal transverse displacement/velocity using u0_fun(x),
    v0_fun(x), and sets all initial rotations and rotation rates to zero.

    Parameters
    ----------
    x_nodes : ndarray of shape (nnodes,)
        Nodal coordinates.
    u0_fun : callable or None, optional
        Function u0_fun(x) -> float specifying initial displacement u(x,0).
        If None, uses 0 everywhere.
    v0_fun : callable or None, optional
        Function v0_fun(x) -> float specifying initial velocity u_t(x,0).
        If None, uses 0 everywhere.

    Returns
    -------
    u0_full : ndarray of shape (2*nnodes,)
        Initial displacement DOF vector [u, theta] at t=0.
    v0_full : ndarray of shape (2*nnodes,)
        Initial velocity DOF vector [uDot, thetaDot] at t=0.
    """
    if u0_fun is None:
        u0_fun = lambda x: 0.0
    if v0_fun is None:
        v0_fun = lambda x: 0.0

    nnodes = len(x_nodes)
    ndof = 2 * nnodes

    u0_full = np.zeros(ndof, dtype=float)
    v0_full = np.zeros(ndof, dtype=float)

    for i, x in enumerate(x_nodes):
        u0_full[2 * i] = float(u0_fun(x))
        v0_full[2 * i] = float(v0_fun(x))
        u0_full[2 * i + 1] = 0.0  # theta
        v0_full[2 * i + 1] = 0.0  # theta_dot

    return u0_full, v0_full
