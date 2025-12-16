import numpy as np
import matplotlib.pyplot as plt

from settings import DEBUG, NGP
from problem import build_mesh, build_problem_functions, build_initial_conditions
from bc import get_free_dofs
from assembly import assemble_mass_matrix
from time_integrators import trapezoidal_first_order
from postprocess import expand_free_to_full, nodal_displacements, tip_displacement
from analysis import build_Khat_ff, omega1_time_history, crossing_time
from viz import plot_frequency, plot_tip, plot_snapshots, animate_beam


def main():
    """
    Main driver routine for simulating the transverse vibration of a
    cantilever Euler–Bernoulli beam with time-varying flexural rigidity.

    This function performs the complete computational workflow:
      1. Mesh generation and problem setup
      2. Assembly of mass and stiffness-related matrices
      3. Application of boundary conditions
      4. Time integration of the governing equations
      5. Post-processing of displacements
      6. Modal analysis to track the first natural frequency
      7. Visualization of results (plots and animation)

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function produces figures and animations illustrating:
        - The time evolution of the first natural frequency
        - The beam tip displacement response
        - Spatial displacement snapshots
        - An animation of the beam deformation
    """

    # ---- setup ----
    L = 1.0
    nel = 40
    x_nodes = build_mesh(L=L, nel=nel)
    nnodes = len(x_nodes)
    ndof = 2 * nnodes

    rhoA = 1.0

    # ---- time grid ----
    T_final = 30
    dt = 0.005
    t_grid = np.arange(0.0, T_final + dt, dt)

    E_fun, EI_fun, q_fun, omega_g = build_problem_functions(L,alpha = 0.2,T = 30,omega_g= 3.4)

    free_dofs, fixed_dofs = get_free_dofs(nnodes)

    # ---- matrices ----
    M_full = assemble_mass_matrix(x_nodes, rhoA)
    M_ff = M_full[np.ix_(free_dofs, free_dofs)]

    # ---- initial conditions ----
    u0_full, v0_full = build_initial_conditions(x_nodes)
    u0 = u0_full[free_dofs]
    v0 = v0_full[free_dofs]

    # ---- time integration ----
    U_free, V_free, A_free = trapezoidal_first_order(
        M_ff, x_nodes, EI_fun, q_fun, u0, v0, free_dofs, t_grid, ngp=NGP
    )

    # ---- postprocess ----
    U_full = expand_free_to_full(U_free, free_dofs, fixed_dofs, ndof)
    U_nodes = nodal_displacements(U_full)
    tip = tip_displacement(U_nodes)

    # ---- omega1(t) ----
    Khat_ff = build_Khat_ff(x_nodes, free_dofs, ngp=NGP)
    omega1 = omega1_time_history(t_grid, M_ff, Khat_ff, EI_fun)
    t_cross = crossing_time(t_grid, omega1, omega_g)

    # ---- omega1(t) ----
    Khat_ff = build_Khat_ff(x_nodes, free_dofs, ngp=NGP)
    omega1 = omega1_time_history(t_grid, M_ff, Khat_ff, EI_fun)
    t_cross = crossing_time(t_grid, omega1, omega_g)

    if DEBUG:
        i_min = int(np.nanargmin(np.abs(omega1 - omega_g)))
        print("=== Frequency matching diagnostics ===")
        print(f"omega_g              = {omega_g:.6f} rad/s")
        print(f"omega1(t0)           = {omega1[0]:.6f} rad/s at t = {t_grid[0]:.6f}")
        print(f"omega1(t_end)        = {omega1[-1]:.6f} rad/s at t = {t_grid[-1]:.6f}")
        print(f"min |omega1-omega_g|  = {abs(omega1[i_min]-omega_g):.6e} at t = {t_grid[i_min]:.6f}")
        print(f"t_cross (reported)   = {t_cross}")

        # Optional: show sign-change bracket (if it exists)
        s = omega1 - omega_g
        idx = np.where(np.sign(s[:-1]) * np.sign(s[1:]) < 0)[0]
        if len(idx) > 0:
            k = int(idx[0])
            print(f"Sign change between {k} and {k+1}:")
            print(f"  t[{k}]={t_grid[k]:.6f}, omega1={omega1[k]:.6f}")
            print(f"  t[{k+1}]={t_grid[k+1]:.6f}, omega1={omega1[k+1]:.6f}")


        # ---- plots ----
        plot_frequency(t_grid, omega1, omega_g, t_cross)
        plot_tip(t_grid, tip, t_cross)
        plot_snapshots(x_nodes, U_nodes, t_grid, T_final, dt)

    # ---- animation ----
    _anim = animate_beam(x_nodes, U_nodes, t_grid, L)

    plt.show()


if __name__ == "__main__":
    main()
