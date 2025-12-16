# assembly.py
import numpy as np
from elements import (
    element_mass_matrix,
    element_stiffness_matrix,
    element_load_vector,
)


# ================================================================
#  Assembly of global matrices and vectors
# ================================================================
def assemble_mass_matrix(x_nodes: np.ndarray, rhoA: float) -> np.ndarray:
    """
    Assemble the global mass matrix M.

    This function constructs the global mass matrix by assembling
    the 4×4 Hermite beam element mass matrices over all elements.

    Assumptions
    -----------
    - Mass per unit length rhoA is constant over the entire beam.
    - Each node has two degrees of freedom:
          [u_k, theta_k]
    - Global DOF ordering:
          [u_0, theta_0, u_1, theta_1, ..., u_N, theta_N]

    Parameters
    ----------
    x_nodes : ndarray of shape (nnodes,)
        Nodal coordinates of the 1D mesh.
    rhoA : float
        Mass per unit length (constant).

    Returns
    -------
    M : ndarray of shape (2*nnodes, 2*nnodes)
        Assembled global mass matrix.
    """
    nnodes = len(x_nodes)
    ndof = 2 * nnodes
    M = np.zeros((ndof, ndof), dtype=float)

    for e in range(nnodes - 1):
        # Element domain
        xk, xkp1 = x_nodes[e], x_nodes[e + 1]

        # Element mass matrix (4x4)
        Me = element_mass_matrix(rhoA, xk, xkp1)

        # Global DOF indices for this element
        dofs = [2*e, 2*e + 1, 2*(e+1), 2*(e+1) + 1]

        # Scatter-add element contributions
        for a in range(4):
            ia = dofs[a]
            for b in range(4):
                ib = dofs[b]
                M[ia, ib] += Me[a, b]

    return M


def assemble_stiffness_matrix(x_nodes: np.ndarray, EI_t: float) -> np.ndarray:
    """
    Assemble the global stiffness matrix K(t).

    The global stiffness matrix is obtained by assembling the
    Hermite beam element stiffness matrices.

    Assumptions
    -----------
    - Bending stiffness EI(t) is uniform in space.
    - Time dependence enters only through the scalar EI_t.
    - Euler–Bernoulli beam theory is used.
    - DOF ordering is:
          [u_0, theta_0, u_1, theta_1, ..., u_N, theta_N]

    Parameters
    ----------
    x_nodes : ndarray of shape (nnodes,)
        Nodal coordinates of the 1D mesh.
    EI_t : float
        Bending stiffness EI(t) at the current time.

    Returns
    -------
    K : ndarray of shape (2*nnodes, 2*nnodes)
        Assembled global stiffness matrix at time t.
    """
    nnodes = len(x_nodes)
    ndof = 2 * nnodes
    K = np.zeros((ndof, ndof), dtype=float)

    for e in range(nnodes - 1):
        xk, xkp1 = x_nodes[e], x_nodes[e + 1]

        # Element stiffness matrix (4x4)
        Ke = element_stiffness_matrix(EI_t, xk, xkp1)

        # Global DOF indices
        dofs = [2*e, 2*e + 1, 2*(e+1), 2*(e+1) + 1]

        # Scatter-add
        for a in range(4):
            ia = dofs[a]
            for b in range(4):
                ib = dofs[b]
                K[ia, ib] += Ke[a, b]

    return K


def assemble_load_vector(
    x_nodes: np.ndarray,
    q_fun,
    t: float,
    ngp: int = 3
) -> np.ndarray:
    """
    Assemble the global load vector f(t).

    The load vector is constructed from the distributed load q(x,t)
    by assembling the element load vectors computed via Gauss quadrature.

    Assumptions
    -----------
    - Distributed load q(x,t) may vary in both space and time.
    - Hermite shape functions are used for interpolation.
    - Numerical integration is performed at the element level.

    Parameters
    ----------
    x_nodes : ndarray of shape (nnodes,)
        Nodal coordinates of the mesh.
    q_fun : callable
        Distributed load function q_fun(x, t).
    t : float
        Current time.
    ngp : int, optional
        Number of Gauss points for numerical integration.
        Default is 3.

    Returns
    -------
    f : ndarray of shape (2*nnodes,)
        Assembled global load vector at time t.
    """
    nnodes = len(x_nodes)
    ndof = 2 * nnodes
    f = np.zeros(ndof, dtype=float)

    for e in range(nnodes - 1):
        xk, xkp1 = x_nodes[e], x_nodes[e + 1]

        # Element load vector (4,)
        fe = element_load_vector(q_fun, xk, xkp1, t, ngp)

        # Global DOF indices
        dofs = [2*e, 2*e + 1, 2*(e+1), 2*(e+1) + 1]

        # Scatter-add
        for a in range(4):
            ia = dofs[a]
            f[ia] += fe[a]

    return f


def assemble_K_and_f(
    x_nodes: np.ndarray,
    EI_t: float,
    q_fun,
    t: float,
    ngp: int = 3
):
    """
    Convenience wrapper to assemble both K(t) and f(t).

    This function is provided for clarity and ease of use in
    time-integration routines, where both the stiffness matrix
    and load vector are required at the same time step.

    Parameters
    ----------
    x_nodes : ndarray
        Nodal coordinates.
    EI_t : float
        Bending stiffness EI(t).
    q_fun : callable
        Distributed load function q_fun(x, t).
    t : float
        Current time.
    ngp : int, optional
        Number of Gauss points for load integration.

    Returns
    -------
    K : ndarray
        Global stiffness matrix at time t.
    f : ndarray
        Global load vector at time t.
    """
    K = assemble_stiffness_matrix(x_nodes, EI_t)
    f = assemble_load_vector(x_nodes, q_fun, t, ngp=ngp)
    return K, f
