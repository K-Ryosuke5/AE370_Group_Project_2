# bcs.py
import numpy as np


def get_free_dofs(nnodes: int):
    """
    Determine free and fixed degrees of freedom for a cantilever beam.

    This function applies a clamped (cantilever) boundary condition at
    the left end of the beam (node 0), where both the transverse displacement
    and rotation are fixed:
        u_0 = 0,  theta_0 = 0.

    The global degree-of-freedom (DOF) ordering is assumed to be:
        [u_0, theta_0, u_1, theta_1, ..., u_N, theta_N]

    Parameters
    ----------
    nnodes : int
        Total number of nodes in the mesh.

    Returns
    -------
    free_dofs : ndarray of shape (n_free,)
        Indices of free degrees of freedom (used in reduced systems).
    fixed_dofs : ndarray of shape (2,)
        Indices of fixed degrees of freedom.
        For a cantilever beam, this is always [0, 1].
    """
    ndof = 2 * nnodes

    # Fixed DOFs at the clamped root (node 0)
    fixed_dofs = np.array([0, 1], dtype=int)

    # All remaining DOFs are free
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    return free_dofs, fixed_dofs
