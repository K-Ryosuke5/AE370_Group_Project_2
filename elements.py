# elements.py
import numpy as np
from quadrature import gauss_points_weights_unit_interval


# ================================================================
#  Hermite shape functions (s in [0,1]) -- displacement interpolation
# ================================================================
def hermite_shape_functions_N(s: float, h: float) -> np.ndarray:
    """
    Evaluate Hermite beam shape functions for displacement interpolation.

    The local coordinate s is defined on the reference interval [0, 1],
    with the physical coordinate given by:
        x = x_k + h * s

    The element has four degrees of freedom ordered as:
        [u_k, theta_k, u_{k+1}, theta_{k+1}]

    The transverse displacement within the element is approximated as:
        u(x) = N(s) · u_e

    where u_e is the vector of element DOFs.

    Parameters
    ----------
    s : float
        Local (reference) coordinate in [0, 1].
    h : float
        Element length (h = x_{k+1} - x_k).

    Returns
    -------
    N : ndarray of shape (4,)
        Hermite shape function vector evaluated at s.
        Note that rotational DOFs are multiplied by h so that
        all entries in N have consistent units.
    """
    H1  = 1.0 - 3.0*s**2 + 2.0*s**3
    H2s = s - 2.0*s**2 + s**3
    H3  = 3.0*s**2 - 2.0*s**3
    H4s = -s**2 + s**3

    # Rotational DOFs are scaled by element length h
    N = np.array([H1, h*H2s, H3, h*H4s], dtype=float)
    return N


# ================================================================
#  Element mass and stiffness matrices
#  (rhoA constant, EI(t) uniform in space)
# ================================================================
def element_mass_matrix(rhoA: float, xk: float, xkp1: float) -> np.ndarray:
    """
    Compute the consistent Hermite beam element mass matrix.

    This implementation assumes:
      - Constant mass per unit length rhoA
      - Euler–Bernoulli beam theory
      - Hermite cubic shape functions

    The resulting matrix corresponds to the analytical expression:
        M_e = rhoA * (h / 420) * [[156,  22h,   54, -13h],
                                 [22h, 4h^2, 13h, -3h^2],
                                 [54,  13h,  156, -22h],
                                 [-13h, -3h^2, -22h, 4h^2]]

    Parameters
    ----------
    rhoA : float
        Mass per unit length (assumed constant).
    xk : float
        Coordinate of the left node of the element.
    xkp1 : float
        Coordinate of the right node of the element.

    Returns
    -------
    Me : ndarray of shape (4, 4)
        Element mass matrix.
    """
    h = xkp1 - xk
    return rhoA * (h / 420.0) * np.array([
        [156.0,    22.0*h,    54.0,   -13.0*h],
        [22.0*h,    4.0*h**2, 13.0*h,  -3.0*h**2],
        [54.0,     13.0*h,   156.0,   -22.0*h],
        [-13.0*h,  -3.0*h**2, -22.0*h,  4.0*h**2]
    ], dtype=float)


def element_stiffness_matrix(EI_t: float, xk: float, xkp1: float) -> np.ndarray:
    """
    Compute the Hermite beam element stiffness matrix.

    Assumptions:
      - Euler–Bernoulli beam theory
      - Bending stiffness EI(t) is uniform in space
      - Time dependence enters only through EI(t)

    The stiffness matrix is given analytically by:
        K_e = (EI(t) / h^3) * [[ 12,  6h, -12,  6h],
                               [ 6h, 4h^2, -6h, 2h^2],
                               [-12, -6h,  12, -6h],
                               [ 6h, 2h^2, -6h, 4h^2]]

    Parameters
    ----------
    EI_t : float
        Bending stiffness EI(t) at the current time.
    xk : float
        Coordinate of the left node of the element.
    xkp1 : float
        Coordinate of the right node of the element.

    Returns
    -------
    Ke : ndarray of shape (4, 4)
        Element stiffness matrix.
    """
    h = xkp1 - xk
    return (EI_t / h**3) * np.array([
        [12.0,      6.0*h,   -12.0,      6.0*h],
        [6.0*h,   4.0*h**2,   -6.0*h,   2.0*h**2],
        [-12.0,    -6.0*h,    12.0,     -6.0*h],
        [6.0*h,   2.0*h**2,   -6.0*h,   4.0*h**2]
    ], dtype=float)


# ================================================================
#  Element load vector (Gauss quadrature)
# ================================================================
def element_load_vector(
    q_fun,
    xk: float,
    xkp1: float,
    t: float,
    ngp: int = 3
) -> np.ndarray:
    """
    Compute the element load vector due to a distributed load q(x,t).

    The element load vector is defined as:
        f_e(t) = ∫_{x_k}^{x_{k+1}} q(x,t) N(x) dx

    where N(x) are the Hermite shape functions. The integral is evaluated
    using Gauss–Legendre quadrature on the reference interval s ∈ [0,1].

    Parameters
    ----------
    q_fun : callable
        Distributed load function q_fun(x, t).
    xk : float
        Coordinate of the left node of the element.
    xkp1 : float
        Coordinate of the right node of the element.
    t : float
        Current time.
    ngp : int, optional
        Number of Gauss points used in the quadrature.
        Default is 3.

    Returns
    -------
    fe : ndarray of shape (4,)
        Element load vector at time t.
    """
    h = xkp1 - xk
    fe = np.zeros(4, dtype=float)

    s_gp, w_gp = gauss_points_weights_unit_interval(ngp)

    for s, w in zip(s_gp, w_gp):
        x = xk + s * h
        q = q_fun(x, t)
        N = hermite_shape_functions_N(s, h)
        fe += q * N * h * w  # dx = h ds

    return fe
