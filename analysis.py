import numpy as np
from assembly import assemble_K_and_f
from eigs import generalized_eigs_via_solve_safe, find_crossing_time


def build_Khat_ff(x_nodes, free_dofs, ngp=3):
    """
    Construct the reduced stiffness matrix K̂_ff corresponding to unit
    flexural rigidity (EI = 1), restricted to free degrees of freedom.

    This matrix represents the purely geometric contribution of the beam
    and is later scaled by the time-dependent flexural rigidity EI(t).

    Parameters
    ----------
    x_nodes : ndarray of shape (n_nodes,)
        Nodal coordinates along the beam axis.
    free_dofs : array_like of int
        Indices of free degrees of freedom in the full system.
    ngp : int, optional
        Number of Gauss integration points used for element integration
        (default is 3).

    Returns
    -------
    Khat_ff : ndarray of shape (n_free, n_free)
        Reduced stiffness matrix corresponding to EI = 1, restricted to
        free degrees of freedom.
    """
    def q_zero(x, t):
        return 0.0

    K_full_hat, _ = assemble_K_and_f(
        x_nodes, EI_t=1.0, q_fun=q_zero, t=0.0, ngp=ngp
    )
    return K_full_hat[np.ix_(free_dofs, free_dofs)]


def omega1_time_history(t_grid, M_ff, Khat_ff, EI_fun, nmodes=3):
    """
    Compute the time history of the first natural frequency for a beam
    with time-varying flexural rigidity.

    At each time step, the stiffness matrix is constructed as
    K_ff(t) = EI(t) * K̂_ff, and the generalized eigenvalue problem
    is solved to obtain the lowest natural frequencies.

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Discrete time grid.
    M_ff : ndarray of shape (n_free, n_free)
        Mass matrix restricted to free degrees of freedom.
    Khat_ff : ndarray of shape (n_free, n_free)
        Reduced stiffness matrix corresponding to EI = 1.
    EI_fun : callable
        Function EI_fun(t) returning the flexural rigidity at time t.
    nmodes : int, optional
        Number of lowest modes to compute at each time step
        (default is 3).

    Returns
    -------
    omega1 : ndarray of shape (nt,)
        Time history of the first natural frequency.
    """
    nt = len(t_grid)
    omega1 = np.full(nt, np.nan, dtype=float)

    for n, t in enumerate(t_grid):
        K_ff = float(EI_fun(t)) * Khat_ff
        omegas, _ = generalized_eigs_via_solve_safe(
            K_ff, M_ff, nmodes=nmodes
        )
        omega1[n] = omegas[0]

    return omega1


def crossing_time(t_grid, omega1, omega_g):
    """
    Determine the time at which the first natural frequency crosses
    a prescribed excitation frequency.

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Discrete time grid.
    omega1 : ndarray of shape (nt,)
        Time history of the first natural frequency.
    omega_g : float
        Prescribed excitation (forcing) frequency.

    Returns
    -------
    t_cross : float or None
        Estimated time at which omega1(t) = omega_g.
        Returns None if no crossing occurs within the time interval.
    """
    return find_crossing_time(t_grid, omega1, omega_g)
