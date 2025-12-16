# time_integration.py
import numpy as np
from assembly import assemble_K_and_f


def trapezoidal_first_order(
    M_ff: np.ndarray,
    x_nodes: np.ndarray,
    EI_fun,
    q_fun,
    u0: np.ndarray,
    v0: np.ndarray,
    free_dofs: np.ndarray,
    t_grid: np.ndarray,
    ngp: int = 3
):
    """
    Time integration using the trapezoidal rule applied to a first-order system.

    After spatial discretization (FEM), the structural dynamics equation is:
        M ü(t) + K(t) u(t) = f(t)

    We rewrite it as a first-order system by defining velocity v = u̇ and state:
        y(t) = [ u(t) ]
               [ v(t) ]

    Then:
        u̇ = v
        v̇ = - M^{-1} K(t) u + M^{-1} f(t)

    which can be written compactly as:
        ẏ(t) = A(t) y(t) + b(t)

    with:
        A(t) = [ 0      I ]
               [ -M^{-1}K(t)  0 ]
        b(t) = [ 0 ]
               [ M^{-1} f(t) ]

    The trapezoidal rule on ẏ = A(t)y + b(t) gives the implicit update:
        (I - Δt/2 A_{n+1}) y_{n+1}
          = (I + Δt/2 A_n) y_n + Δt/2 (b_n + b_{n+1})

    At each time step, we solve a linear system for y_{n+1}.

    Assumptions
    -----------
    - M_ff is the mass matrix restricted to free DOFs (fixed DOFs removed).
    - EI is spatially uniform and depends only on time: EI(t) (scalar).
    - The distributed load q(x,t) may vary in space and time, and is assembled
      via Gauss quadrature (ngp points) at the element level.

    Parameters
    ----------
    M_ff : ndarray of shape (n_free, n_free)
        Mass matrix restricted to free DOFs.
    x_nodes : ndarray of shape (nnodes,)
        Nodal coordinates of the beam mesh.
    EI_fun : callable
        Function EI_fun(t) -> float returning the scalar bending stiffness EI(t).
    q_fun : callable
        Distributed load function q_fun(x, t) -> float.
    u0 : ndarray of shape (n_free,)
        Initial displacement vector on free DOFs.
    v0 : ndarray of shape (n_free,)
        Initial velocity vector on free DOFs.
    free_dofs : ndarray of shape (n_free,)
        Indices of free DOFs in the *full* system.
        Used here only to restrict K and f assembled in full coordinates.
    t_grid : ndarray of shape (nt,)
        Time grid (assumed uniform spacing).
    ngp : int, optional
        Number of Gauss points for assembling the load vector. Default is 3.

    Returns
    -------
    U : ndarray of shape (nt, n_free)
        Displacement history on free DOFs.
    V : ndarray of shape (nt, n_free)
        Velocity history on free DOFs.
    A_out : ndarray of shape (nt, n_free)
        Acceleration history on free DOFs, computed consistently from:
            a(t) = M^{-1}( f(t) - K(t) u(t) )

    Notes (implementation details)
    ------------------------------
    - This implementation assembles K(t) and f(t) in the full DOF space and then
      restricts them to free DOFs using free_dofs.
    - To avoid re-assembling K and f when computing acceleration afterward,
      K(t) and f(t) (restricted to free DOFs) are stored in K_hist and f_hist.
    """
    dt = float(t_grid[1] - t_grid[0])
    nt = len(t_grid)
    ndof = len(free_dofs)

    # Store solution histories (free DOFs only)
    U = np.zeros((nt, ndof), dtype=float)
    V = np.zeros((nt, ndof), dtype=float)
    U[0, :] = u0
    V[0, :] = v0

    # Small block matrices used to form A(t)
    Iu = np.eye(ndof, dtype=float)
    Z  = np.zeros((ndof, ndof), dtype=float)
    I2 = np.eye(2 * ndof, dtype=float)

    def build_A_b(t: float):
        """
        Construct the state-space matrices A(t) and b(t) from K(t) and f(t).

        Steps:
          1) Evaluate EI(t) (scalar)
          2) Assemble full K(t) and f(t)
          3) Restrict them to free DOFs
          4) Form MinvK and Minvf using linear solves (no explicit inverse)
          5) Build A(t) and b(t) for ẏ = A(t) y + b(t)

        Returns
        -------
        A : ndarray of shape (2*n_free, 2*n_free)
            State matrix at time t.
        b : ndarray of shape (2*n_free,)
            Forcing vector in state-space form at time t.
        K : ndarray of shape (n_free, n_free)
            Stiffness matrix restricted to free DOFs.
        f : ndarray of shape (n_free,)
            Load vector restricted to free DOFs.
        """
        EI_t = float(EI_fun(t))  # evaluate EI(t) here (scalar)

        # Assemble full matrices, then restrict to free DOFs
        K_full, f_full = assemble_K_and_f(x_nodes, EI_t, q_fun, t, ngp=ngp)
        K = K_full[np.ix_(free_dofs, free_dofs)]
        f = f_full[free_dofs]

        # Compute M^{-1}K and M^{-1}f by solving linear systems
        MinvK = np.linalg.solve(M_ff, K)   # (n_free, n_free)
        Minvf = np.linalg.solve(M_ff, f)   # (n_free,)

        # Build A(t) and b(t)
        A = np.block([[Z,      Iu],
                      [-MinvK, Z]])
        b = np.hstack([np.zeros(ndof, dtype=float), Minvf])

        return A, b, K, f

    # ------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------
    # Store K_n and f_n to reuse in acceleration computation
    K_hist = [None] * nt
    f_hist = [None] * nt

    # Precompute at t0
    A_n, b_n, K_n, f_n = build_A_b(t_grid[0])
    K_hist[0] = K_n
    f_hist[0] = f_n

    for n in range(nt - 1):
        t_np1 = t_grid[n + 1]

        # Build A_{n+1}, b_{n+1} and store K_{n+1}, f_{n+1}
        A_np1, b_np1, K_np1, f_np1 = build_A_b(t_np1)
        K_hist[n + 1] = K_np1
        f_hist[n + 1] = f_np1

        # Current state y_n = [u_n; v_n]
        y_n = np.hstack([U[n, :], V[n, :]])

        # Trapezoidal update:
        # (I - dt/2 A_{n+1}) y_{n+1} = (I + dt/2 A_n) y_n + dt/2 (b_n + b_{n+1})
        LHS = I2 - 0.5 * dt * A_np1
        RHS = (I2 + 0.5 * dt * A_n) @ y_n + 0.5 * dt * (b_n + b_np1)

        y_np1 = np.linalg.solve(LHS, RHS)

        # Unpack state into displacement and velocity
        U[n + 1, :] = y_np1[:ndof]
        V[n + 1, :] = y_np1[ndof:]

        # Shift for next step
        A_n, b_n = A_np1, b_np1

    # ------------------------------------------------------------
    # Acceleration recovery (consistent with 2nd-order equation)
    # a(t) = M^{-1}( f(t) - K(t) u(t) )
    # ------------------------------------------------------------
    A_out = np.zeros((nt, ndof), dtype=float)
    for n in range(nt):
        K = K_hist[n]
        f = f_hist[n]
        A_out[n, :] = np.linalg.solve(M_ff, f - K @ U[n, :])

    return U, V, A_out
