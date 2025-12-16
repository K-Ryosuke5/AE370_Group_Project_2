# eigs.py
import numpy as np


def generalized_eigs_via_solve_safe(
    K: np.ndarray,
    M: np.ndarray,
    nmodes: int = 6,
    eps: float = 1e-14
):
    """
    Compute the smallest positive generalized eigenvalues of (K, M) without SciPy.

    We consider the generalized eigenproblem:
        K φ = λ M φ

    For a standard FEM structural dynamics model, M is typically symmetric
    positive definite and K is symmetric positive semidefinite/definite.
    This helper converts the problem to a standard eigenproblem by forming:
        A = M^{-1} K
    and then computing eigenvalues of A.

    Notes
    -----
    - This function uses `np.linalg.eigvals`, which does not exploit symmetry.
      If SciPy were allowed, a symmetric generalized solver would be preferred.
    - Small numerical eigenvalues (|λ| < eps) are treated as zero.
    - Only positive eigenvalues are kept, since ω = sqrt(λ) is real only for λ > 0.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Stiffness matrix (typically symmetric).
    M : ndarray of shape (n, n)
        Mass matrix (should be invertible; typically symmetric positive definite).
    nmodes : int, optional
        Number of smallest positive modes to return. Default is 6.
    eps : float, optional
        Threshold for treating eigenvalues as zero and filtering positivity.
        Default is 1e-14.

    Returns
    -------
    omega_out : ndarray of shape (nmodes,)
        Natural frequencies ω_i = sqrt(λ_i) for the smallest positive eigenvalues.
        If fewer than nmodes positive eigenvalues exist, remaining entries are NaN.
    evals_out : ndarray of shape (nmodes,)
        Corresponding eigenvalues λ_i. Same NaN padding behavior as omega_out.
    """
    omega_out = np.full(nmodes, np.nan, dtype=float)
    evals_out = np.full(nmodes, np.nan, dtype=float)

    # Form A = M^{-1}K by solving M A = K (more stable than explicit inverse)
    A = np.linalg.solve(M, K)

    # Compute eigenvalues of A
    evals = np.linalg.eigvals(A)
    evals = np.real(evals)  # discard tiny imaginary parts from numerical round-off

    # Clean small values and keep only positive eigenvalues
    evals[np.abs(evals) < eps] = 0.0
    evals_pos = evals[evals > eps]
    if evals_pos.size == 0:
        return omega_out, evals_out

    # Sort and take the smallest nmodes
    evals_sorted = np.sort(evals_pos)
    m = min(nmodes, evals_sorted.size)

    evals_out[:m] = evals_sorted[:m]
    omega_out[:m] = np.sqrt(evals_out[:m])

    return omega_out, evals_out


def find_crossing_time(
    t_grid: np.ndarray,
    omega1: np.ndarray,
    omega_g: float
) -> float:
    """
    Estimate the first time when omega1(t) crosses a target frequency omega_g.

    This function searches for the first sign change of:
        diff(t) = omega1(t) - omega_g
    between consecutive time samples. If a sign change exists between
    t_i and t_{i+1}, it returns a linearly interpolated crossing time.

    If no sign change occurs over the provided time window, the function returns
    the time at which |omega1(t) - omega_g| is minimized (closest approach).

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Monotone time grid.
    omega1 : ndarray of shape (nt,)
        Time history of the first natural frequency ω1(t).
    omega_g : float
        Target forcing frequency ω_g.

    Returns
    -------
    t_cross : float
        Estimated crossing time. If no crossing occurs, returns the closest time.
    """
    diff = omega1 - omega_g
    s = np.sign(diff)

    # Find first index where sign changes between samples
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) > 0:
        i = int(idx[0])
        t0, t1 = t_grid[i], t_grid[i + 1]
        d0, d1 = diff[i], diff[i + 1]

        # Linear interpolation: diff(t_cross) = 0
        return float(t0 - d0 * (t1 - t0) / (d1 - d0))

    # No crossing: return closest approach
    i = int(np.argmin(np.abs(diff)))
    return float(t_grid[i])
