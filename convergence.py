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

# ------------------------------------------------
#  Solver wrapper for convergence tests
# ------------------------------------------------
def solve_beam(nel, dt, T_final,
              L=1.0,
              alpha=0.5,
              T=2,
              q0=0.05,
              omega_g=2*np.pi,
              rhoA=1.0):


    if T is None:
        T = T_final

    # mesh
    x_nodes = build_mesh(L=L, nel=nel)
    nnodes = len(x_nodes)
    ndof = 2 * nnodes
    free_dofs, fixed_dofs = get_free_dofs(nnodes)

    # time
    t_grid = np.arange(0.0, T_final + dt, dt)

    # matrices
    M_full = assemble_mass_matrix(x_nodes, rhoA)
    M_ff = M_full[np.ix_(free_dofs, free_dofs)]
    Khat_ff = build_Khat_ff(x_nodes, free_dofs, ngp=NGP)

    # IC
    u0_full, v0_full = build_initial_conditions(x_nodes)
    u0 = u0_full[free_dofs]
    v0 = v0_full[free_dofs]

    # problem functions
    E_fun, EI_fun, q_fun, _ = build_problem_functions(
        L, alpha=alpha, T=T, q0=q0, omega_g=omega_g
    )

    # integrate
    U_free, V_free, A_free = trapezoidal_first_order(
        M_ff, x_nodes, EI_fun, q_fun, u0, v0, free_dofs, t_grid, ngp=NGP
    )

    # tip
    U_full = expand_free_to_full(U_free, free_dofs, fixed_dofs, ndof)
    U_nodes = nodal_displacements(U_full)
    tip = tip_displacement(U_nodes)

    return t_grid, tip


# ================================================================
#  Convergence tests (member's code adapted)
# ================================================================

def run_spatial_convergence():
    print("Spatial convergence test")
    L = 1.0
    T_final = 0.1
    dt = 5.0e-4
    nel_list = [20, 40, 80, 160]

    nel_ref = nel_list[-1]
    t_ref, tip_ref = solve_beam(nel_ref, dt, T_final, L=L)
    u_ref_final = tip_ref[-1]

    hs, errors = [], []
    for nel in nel_list[:-1]:
        t, tip = solve_beam(nel, dt, T_final, L=L)
        u_final = tip[-1]
        h = L / nel
        err = abs(u_final - u_ref_final)
        hs.append(h)
        errors.append(err)
        print(f"nel = {nel:4d}, h = {h:.5e}, |u - u_ref| = {err:.5e}")

    rates = []
    for i in range(len(errors) - 1):
        p = np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        rates.append(p)

    for i, p in enumerate(rates):
        print(f"Estimated spatial rate between nel = {nel_list[i]} and {nel_list[i+1]}: p = {p:.3f}")

    plt.figure()
    plt.loglog(hs, errors, "o-", label="spatial error")
    h0, e0 = hs[0], errors[0]
    ref_line = e0 * (np.array(hs) / h0) ** 4
    plt.loglog(hs, ref_line, "--", label="reference slope p = 4")
    plt.xlabel("h")
    plt.ylabel("error in tip displacement at T_final")
    plt.title("Spatial convergence")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


def run_temporal_convergence():
    print("Temporal convergence test")
    T_final = 2.0
    nel = 80
    dt_list = [0.02, 0.01, 0.005, 0.0025]

    dt_ref = dt_list[-1]
    t_ref, tip_ref = solve_beam(nel, dt_ref, T_final)
    u_ref_final = tip_ref[-1]

    dt_vals, errors = [], []
    for dt in dt_list[:-1]:
        t, tip = solve_beam(nel, dt, T_final)
        u_final = tip[-1]
        err = abs(u_final - u_ref_final)
        dt_vals.append(dt)
        errors.append(err)
        print(f"dt = {dt:.5e}, |u - u_ref| = {err:.5e}")

    rates = []
    for i in range(len(errors) - 1):
        p = np.log(errors[i] / errors[i + 1]) / np.log(dt_vals[i] / dt_vals[i + 1])
        rates.append(p)

    for i, p in enumerate(rates):
        print(f"Estimated temporal rate between dt = {dt_list[i]} and {dt_list[i+1]}: p = {p:.3f}")

    plt.figure()
    plt.loglog(dt_vals, errors, "o-", label="temporal error")
    dt0, e0 = dt_vals[0], errors[0]
    ref_line = e0 * (np.array(dt_vals) / dt0) ** 2
    plt.loglog(dt_vals, ref_line, "--", label="reference slope p = 2")
    plt.xlabel("dt")
    plt.ylabel("error in tip displacement at T_final")
    plt.title("Temporal convergence")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


def main():
    run_spatial_convergence()
    run_temporal_convergence()


if __name__ == "__main__":
    main()
