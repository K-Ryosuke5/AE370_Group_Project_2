import numpy as np

def expand_free_to_full(U_free, free_dofs, fixed_dofs, ndof, fixed_value=0.0):
    """
    Expand the solution defined only on free degrees of freedom (DOFs)
    into the full DOF space by inserting prescribed values at fixed DOFs.

    Parameters
    ----------
    U_free : ndarray of shape (nt, n_free)
        Time history of the solution vector restricted to free DOFs,
        where nt is the number of time steps and n_free is the number
        of free degrees of freedom.
    free_dofs : array_like of int
        Indices of free DOFs in the full system.
    fixed_dofs : array_like of int
        Indices of fixed (constrained) DOFs in the full system.
    ndof : int
        Total number of DOFs in the full system.
    fixed_value : float, optional
        Prescribed value assigned to all fixed DOFs (default is 0.0).

    Returns
    -------
    U_full : ndarray of shape (nt, ndof)
        Time history of the displacement vector in the full DOF space,
        with free DOFs filled from U_free and fixed DOFs set to
        fixed_value.
    """
    nt = U_free.shape[0]
    U_full = np.zeros((nt, ndof), dtype=float)
    U_full[:, fixed_dofs] = fixed_value
    U_full[:, free_dofs] = U_free
    return U_full


def nodal_displacements(U_full):
    """
    Extract nodal transverse displacements from the full DOF solution.

    The full DOF ordering is assumed to be
    [u_0, θ_0, u_1, θ_1, ..., u_N, θ_N],
    where u_k is the transverse displacement and θ_k is the rotation
    at node k.

    Parameters
    ----------
    U_full : ndarray of shape (nt, ndof)
        Time history of the solution vector in the full DOF space.

    Returns
    -------
    U_nodes : ndarray of shape (nt, n_nodes)
        Time history of nodal transverse displacements u_k at each node.
    """
    return U_full[:, 0::2]


def tip_displacement(U_nodes):
    """
    Extract the tip (last-node) displacement from nodal displacements.

    Parameters
    ----------
    U_nodes : ndarray of shape (nt, n_nodes)
        Time history of nodal transverse displacements.

    Returns
    -------
    u_tip : ndarray of shape (nt,)
        Time history of the transverse displacement at the beam tip
        (last node).
    """
    return U_nodes[:, -1]
