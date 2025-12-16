import numpy as np
import matplotlib.pyplot as plt

from settings import DEBUG, NGP
from problem import build_mesh, build_problem_functions, build_initial_conditions
from bc import get_free_dofs
from assembly import assemble_mass_matrix
from time_integrators import trapezoidal_first_order
from postprocess import expand_free_to_full, nodal_displacements, tip_displacement
from analysis import build_Khat_ff, omega1_time_history, crossing_time
from metrics import post_rms, pre_rms_0_to_cross
import viz




def main():
    # ---- sweep settings ----
    alpha_list = [0.2,0.25,0.3,0.4,0.5,0.6,0.7]
    T_final = 30.0
    dt = 0.005
    T = T_final          # stiffness changes over whole simulation
    delta_rms = 0.2            # start RMS window t_cross + delta

    # ---- setup ----
    L = 1.0
    nel = 40
    x_nodes = build_mesh(L=L, nel=nel)
    nnodes = len(x_nodes)
    ndof = 2 * nnodes

    rhoA = 1.0
    free_dofs, fixed_dofs = get_free_dofs(nnodes)

    # ---- time grid ----
    t_grid = np.arange(0.0, T_final + dt, dt)

    # ---- matrices (build once) ----
    M_full = assemble_mass_matrix(x_nodes, rhoA)
    M_ff = M_full[np.ix_(free_dofs, free_dofs)]
    Khat_ff = build_Khat_ff(x_nodes, free_dofs, ngp=NGP)

    # ---- initial conditions (build once) ----
    u0_full, v0_full = build_initial_conditions(x_nodes)
    u0 = u0_full[free_dofs]
    v0 = v0_full[free_dofs]

    # ---- choose omega_g once (keep forcing consistent across alpha) ----
    # If EI_fun(t=0)=EI0 is the same for all alpha, then omega1(0) is the same.
    # So define omega_g from the alpha=0 reference.
    # (Implement omega_g inside build_problem_functions, or do it explicitly there.)

    # Example: set omega_g from a reference call
    # First call with alpha=0 to get omega_g based on your preferred rule inside problem.py
    E_fun, EI_fun_ref, q_fun_ref, omega_g = build_problem_functions(
        L, alpha=0.0, T=T, q0=0.05, omega_g=3.2
    )
    # If you want omega_g = 0.95*omega1(0), compute omega1(0) once and overwrite omega_g:
    # from eigs import generalized_eigs_via_solve_safe
    # omegas0, _ = generalized_eigs_via_solve_safe(float(EI_fun_ref(0.0))*Khat_ff, M_ff, nmodes=3)
    # omega_g = 0.95 * omegas0[0]

    urms_post_list = []
    urms_pre_list = []
    detuning_end_list = []
    omega1_end_list = []
    E_end_list = []



    for alpha in alpha_list:
        # build functions for this alpha (EI changes, omega_g fixed)
        E_fun, EI_fun, q_fun, _ = build_problem_functions(
            L, alpha=alpha, T=T, q0=0.05, omega_g=omega_g
        )

        # ---- time integration ----
        U_free, V_free, A_free = trapezoidal_first_order(
            M_ff, x_nodes, EI_fun, q_fun, u0, v0, free_dofs, t_grid, ngp=NGP
        )

        # ---- tip displacement ----
        U_full = expand_free_to_full(U_free, free_dofs, fixed_dofs, ndof)
        U_nodes = nodal_displacements(U_full)
        tip = tip_displacement(U_nodes)

        # ---- omega1(t) and crossing ----
        omega1 = omega1_time_history(t_grid, M_ff, Khat_ff, EI_fun)
        t_cross = crossing_time(t_grid, omega1, omega_g)

        # final values at T_final
        omega1_end_list.append(float(omega1[-1]))
        E_end_list.append(float(E_fun(t_grid[-1])))

        if DEBUG:
            print(
                f"alpha={alpha:.3f}, "
                f"omega1(T)={omega1[-1]:.6f}, "
                f"E(T)={E_fun(t_grid[-1]):.6f}"
            )
        

        # ---- metric: post-crossing RMS ----
        urms_post, t_start = post_rms(t_grid, tip, t_cross, delta=delta_rms)
        urms_pre, _ = pre_rms_0_to_cross(t_grid, tip, t_cross)
        urms_post_list.append(urms_post)
        urms_pre_list.append(urms_pre)
        delta_end = abs(omega1[-1] - omega_g)
        detuning_end_list.append(delta_end)


        if DEBUG:
            print(f"[alpha={alpha:.3f}] t_cross={t_cross}, urms_post={urms_post:.6e} (from t={t_start:.3f})")

    # ---- plot alpha vs metric ----

    viz.plot_alpha_vs_urms(alpha_list, urms_pre_list, urms_post_list)

    viz.plot_alpha_vs_detuning(alpha_list, detuning_end_list)

    viz.plot_alpha_vs_omega1_end(alpha_list, omega1_end_list)
    viz.plot_alpha_vs_E_end(alpha_list, E_end_list)


    plt.show()

if __name__ == "__main__":
    main()
