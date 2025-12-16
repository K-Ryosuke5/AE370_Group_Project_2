# quadrature.py
import numpy as np


# ================================================================
#  Gauss–Legendre quadrature on the reference interval [0, 1]
# ================================================================
def gauss_points_weights_unit_interval(ngp: int = 3):
    """
    Return Gauss–Legendre quadrature points and weights on [0, 1].

    This function provides quadrature points and weights suitable for
    numerically evaluating integrals of the form:

        ∫_0^1 f(s) ds ≈ Σ w_i f(s_i)

    Internally, standard Gauss–Legendre points defined on the interval
    [-1, 1] are mapped to the reference element [0, 1] using:

        s = (ξ + 1) / 2,
        w = ŵ / 2,

    where (ξ, ŵ) are the standard Gauss–Legendre points and weights
    on [-1, 1].

    Parameters
    ----------
    ngp : int, optional
        Number of Gauss points.
        Supported values are:
          - ngp = 2 : exact integration of polynomials up to degree 3
          - ngp = 3 : exact integration of polynomials up to degree 5
        Default is 3.

    Returns
    -------
    s : ndarray of shape (ngp,)
        Quadrature points in the reference interval [0, 1].
    w : ndarray of shape (ngp,)
        Corresponding quadrature weights.
    """
    if ngp == 2:
        # Standard Gauss–Legendre rule on [-1, 1]
        xi = np.array(
            [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
            dtype=float
        )
        wi = np.array([1.0, 1.0], dtype=float)

    elif ngp == 3:
        # Higher-order Gauss–Legendre rule on [-1, 1]
        xi = np.array(
            [0.0, -np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0)],
            dtype=float
        )
        wi = np.array([8.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0], dtype=float)

    else:
        raise ValueError("ngp must be 2 or 3")

    # Map [-1, 1] -> [0, 1]
    s = 0.5 * (xi + 1.0)
    w = 0.5 * wi

    return s, w
