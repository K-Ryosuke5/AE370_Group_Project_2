# metrics.py
import numpy as np

def post_rms(t_grid, y, t_cross, delta=0.2):
    """
    Compute RMS of y(t) over the post-crossing interval [t_cross+delta, T].

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Time grid.
    y : ndarray of shape (nt,)
        Signal (e.g., tip displacement).
    t_cross : float or None
        Crossing time. If None, uses the last half of the simulation by default.
    delta : float, optional
        Offset time after crossing to start RMS window (default: 0.2).

    Returns
    -------
    urms_post : float
        RMS value over the post-crossing window.
    t_start : float
        Start time used for RMS window.
    """
    t_grid = np.asarray(t_grid)
    y = np.asarray(y)

    T = float(t_grid[-1])

    if (t_cross is None) or (not np.isfinite(t_cross)):
        # fallback: take second half
        t_start = 0.5 * T
    else:
        t_start = min(float(t_cross) + float(delta), T)

    i0 = int(np.searchsorted(t_grid, t_start, side="left"))
    i0 = min(max(i0, 0), len(t_grid) - 1)

    t_win = t_grid[i0:]
    y_win = y[i0:]

    # RMS via time-average of y^2 using trapezoidal integration
    if len(t_win) < 2:
        return float(np.sqrt(np.mean(y_win**2))), float(t_start)

    integral = np.trapz(y_win**2, t_win)
    urms = np.sqrt(integral / (t_win[-1] - t_win[0]))
    return float(urms), float(t_win[0])


def pre_rms_0_to_cross(t_grid, y, t_cross, delta=0.0):
    """
    Compute RMS of y(t) over the pre-crossing interval [0, t_cross-delta].

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Time grid.
    y : ndarray of shape (nt,)
        Signal (e.g., tip displacement).
    t_cross : float
        Crossing time t*. Must be finite.
    delta : float, optional
        Offset time before crossing to end the RMS window (default: 0.0).
        Use a small positive value if you want to exclude the immediate neighborhood
        of the crossing.

    Returns
    -------
    urms_pre : float
        RMS value over the pre-crossing window.
    t_end : float
        End time used for RMS window.
    """
    t_grid = np.asarray(t_grid)
    y = np.asarray(y)

    if (t_cross is None) or (not np.isfinite(t_cross)):
        return float("nan"), float("nan")

    t_end = float(t_cross) - float(delta)
    t_end = max(min(t_end, float(t_grid[-1])), float(t_grid[0]))

    i1 = int(np.searchsorted(t_grid, t_end, side="right"))  # include t_end
    i1 = min(max(i1, 1), len(t_grid))  # need at least 1 point

    t_win = t_grid[:i1]
    y_win = y[:i1]

    if len(t_win) < 2 or (t_win[-1] <= t_win[0]):
        return float(np.sqrt(np.mean(y_win**2))), float(t_win[-1])

    integral = np.trapz(y_win**2, t_win)
    urms = np.sqrt(integral / (t_win[-1] - t_win[0]))
    return float(urms), float(t_win[-1])
