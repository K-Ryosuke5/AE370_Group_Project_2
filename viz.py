import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_frequency(t_grid, omega1, omega_g, t_cross):
    """
    Plot the time history of the first natural frequency together with
    a prescribed excitation frequency and their crossing time.

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Discrete time grid.
    omega1 : ndarray of shape (nt,)
        Time history of the first natural frequency.
    omega_g : float
        Prescribed excitation (forcing) frequency.
    t_cross : float
        Time at which omega1(t) crosses omega_g.

    Returns
    -------
    None
        A Matplotlib figure is generated showing omega1(t),
        the excitation frequency, and the estimated crossing time.
    """
    plt.figure()
    plt.plot(t_grid, omega1, label=r"$\omega_1(t)$")
    plt.axhline(omega_g, linestyle="--", label=r"$\omega_g$")
    plt.axvline(t_cross, linestyle=":", label=f"t* ≈ {t_cross:.3f}")
    plt.xlabel("time t")
    plt.ylabel("angular frequency [rad/s]")
    plt.title("Instantaneous first natural frequency vs time")
    plt.grid(True)
    plt.legend()


def plot_tip(t_grid, tip_disp, t_cross):
    """
    Plot the time history of the beam tip displacement and mark the
    frequency crossing time.

    Parameters
    ----------
    t_grid : ndarray of shape (nt,)
        Discrete time grid.
    tip_disp : ndarray of shape (nt,)
        Time history of the transverse displacement at the beam tip.
    t_cross : float
        Time at which the natural frequency crosses the excitation
        frequency.

    Returns
    -------
    None
        A Matplotlib figure is generated showing the tip displacement
        as a function of time.
    """
    plt.figure()
    plt.plot(t_grid, tip_disp, label=r"$u(L,t)$")
    plt.axvline(t_cross, linestyle=":", label=f"t* ≈ {t_cross:.3f}")
    plt.xlabel("time t")
    plt.ylabel("tip displacement")
    plt.title("Tip displacement vs time")
    plt.grid(True)
    plt.legend()


def plot_snapshots(x_nodes, U_nodes, t_grid, T_final, dt):
    """
    Plot spatial displacement snapshots at selected times over the
    simulation interval.

    Snapshots are taken at fractions of the final time
    (0, 0.25, 0.5, and 1.0 of T_final).

    Parameters
    ----------
    x_nodes : ndarray of shape (n_nodes,)
        Nodal coordinates along the beam.
    U_nodes : ndarray of shape (nt, n_nodes)
        Time history of nodal transverse displacements.
    t_grid : ndarray of shape (nt,)
        Discrete time grid.
    T_final : float
        Final simulation time.
    dt : float
        Time step size.

    Returns
    -------
    None
        A Matplotlib figure is generated showing displacement profiles
        along the beam at selected times.
    """
    plt.figure()
    nt = len(t_grid)

    for factor in [0.0, 0.25, 0.5, 1.0]:
        t_snapshot = factor * T_final
        idx = min(int(round(t_snapshot / dt)), nt - 1)
        plt.plot(x_nodes, U_nodes[idx, :], label=f"t = {t_grid[idx]:.2f}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Displacement snapshots")
    plt.legend()
    plt.grid(True)


def animate_beam(x_nodes, U_nodes, t_grid, L, every_step=5, interval=30):
    """
    Create an animation of the beam transverse displacement over time.

    Parameters
    ----------
    x_nodes : ndarray of shape (n_nodes,)
        Nodal coordinates along the beam.
    U_nodes : ndarray of shape (nt, n_nodes)
        Time history of nodal transverse displacements.
    t_grid : ndarray of shape (nt,)
        Discrete time grid.
    L : float
        Beam length.
    every_step : int, optional
        Subsampling factor for animation frames (default is 5).
    interval : int, optional
        Delay between frames in milliseconds (default is 30).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object showing the beam deformation as a function
        of time.
    """
    nt = len(t_grid)
    fig, ax = plt.subplots()

    # Reference undeformed configuration
    ax.plot(x_nodes, np.zeros_like(x_nodes), linestyle="--", linewidth=1)
    line, = ax.plot([], [], linewidth=2)

    ax.set_xlim(0.0, L)

    u_min, u_max = float(U_nodes.min()), float(U_nodes.max())
    margin = 0.1 * max(1.0, abs(u_max - u_min))
    ax.set_ylim(u_min - margin, u_max + margin)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")

    frame_indices = np.arange(0, nt, every_step)

    def init():
        line.set_data([], [])
        return (line,)

    def animate_frame(frame_idx):
        n = frame_indices[frame_idx]
        line.set_data(x_nodes, U_nodes[n, :])
        ax.set_title(f"t = {t_grid[n]:.3f}")
        return (line,)

    anim = animation.FuncAnimation(
        fig,
        animate_frame,
        init_func=init,
        frames=len(frame_indices),
        interval=interval,
        blit=True,
    )

    return anim


def plot_alpha_vs_urms(alpha_list, urms_pre_list, urms_post_list):
    """
    Plot alpha vs pre- and post-crossing RMS tip displacement
    on the same figure.

    Parameters
    ----------
    alpha_list : list or ndarray
        List of stiffness reduction parameters alpha.
    urms_pre_list : list or ndarray
        RMS values computed over [0, t*] (pre-crossing).
    urms_post_list : list or ndarray
        RMS values computed over [t*, T] (post-crossing).
    """
    plt.figure()
    plt.plot(alpha_list, urms_pre_list, "o-", label=r"pre-crossing RMS")
    plt.plot(alpha_list, urms_post_list, "o-", label=r"post-crossing RMS")
    plt.xlabel(r"stiffness reduction parameter $\alpha$")
    plt.ylabel(r"RMS tip displacement")
    plt.title(r"$\alpha$ sweep: pre vs post crossing RMS")
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())




def plot_alpha_vs_detuning(alpha_list, detuning_list):
    """
    Plot alpha vs final detuning |omega1(T) - omega_g|.
    """
    plt.figure()
    plt.plot(alpha_list, detuning_list, marker="o")
    plt.xlabel(r"stiffness reduction parameter $\alpha$")
    plt.ylabel(r"$|\omega_1(T) - \omega_g|$")
    plt.title(r"$\alpha$ sweep: final frequency detuning")
    plt.grid(True)

def plot_alpha_vs_omega1_end(alpha_list, omega1_end_list):
    plt.figure()
    plt.plot(alpha_list, omega1_end_list, marker="o")
    plt.xlabel(r"stiffness reduction parameter $\alpha$")
    plt.ylabel(r"$\omega_1(T_{\mathrm{final}})$ [rad/s]")
    plt.title(r"$\alpha$ sweep: final first natural frequency")
    plt.grid(True)

def plot_alpha_vs_E_end(alpha_list, E_end_list):
    plt.figure()
    plt.plot(alpha_list, E_end_list, marker="o")
    plt.xlabel(r"stiffness reduction parameter $\alpha$")
    plt.ylabel(r"$E(T_{\mathrm{final}})$")
    plt.title(r"$\alpha$ sweep: final Young's modulus")
    plt.grid(True)
