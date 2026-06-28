import os
from dolfin import *
import numpy as np
import time
import argparse

def run_henry(
    D=0.57024,
    Q_in=5.7024,  # m^3/d per metre width  (scenario a: 5.7024, scenario b: 2.851)
    T=0.5,
    dt=0.001,
    Nx=80,
    Ny=40,
    K=864.0,
    phi=0.35,
    name=None,
    notes="",
    npz_export_frames=None,
    auto_terminate=False,
    ss_tol=1e-4,
):
    """
    Run a single Henry Problem simulation.

    Parameters
    ----------
    D : float
        Diffusion coefficient (m^2/d).
    Q_in : float
        Total freshwater inflow rate (m^3/d per metre width).
        Classic Henry values: 5.7024 (scenario a), 2.851 (scenario b).
    T : float
        Simulation end time (days). Ignored when auto_terminate=True
        (the loop ceiling is then T_auto = 10,000 d).
    dt : float
        Time step size (days). Default 0.001 d.
    Nx, Ny : int
        Mesh subdivisions in x and y.
    K : float
        Hydraulic conductivity (m/d).
    phi : float
        Porosity (dimensionless).
    name : str or None
        Base name for the output folder (timestamp is always prepended).
        If None, defaults to "henry_Q{Q_in:.4g}". Override in SWEEP_CONFIGS
        to use any label you like.
    notes : str
        Free-text note written to the summary file.
    npz_export_frames : list of int or None
        Specific step indices to export .npz arrays.
        Defaults to [0, n_steps] when auto_terminate=False,
        or [0] when auto_terminate=True (the final frame is always exported).
    auto_terminate : bool
        If True, run until the relative per-step change in total salt mass
        drops below ss_tol, up to a hard ceiling of T_auto = 10,000 days.
        If False, run for exactly n_steps = round(T / dt) steps.
    ss_tol : float
        Relative change tolerance for steady-state detection.
        Only used when auto_terminate=True. Default 1e-4.

    Returns
    -------
    dict
        Keys: sim_name, D, Q_in, toe_length, mixing_zone_width,
              total_salt_mass, ss_reached, ss_step, ss_time, elapsed_time.
    """
    # T_auto: hard ceiling for auto-terminate mode (days)
    T_auto = 10000.0

    # n_steps derived from T and dt (used for npz defaults and reporting)
    n_steps = int(round(T / dt))

    if npz_export_frames is None:
        if auto_terminate:
            npz_export_frames = [0]   # final frame always exported on exit
        else:
            npz_export_frames = [0, n_steps]

    # ==============================
    # Naming & Output Directories
    # ==============================
    current_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    if name is None:
        name = f"henry_Q{Q_in:.4g}"
    sim_name = f"{current_time_str}_{name}"
    output_dir = sim_name
    os.makedirs(output_dir, exist_ok=True)

    npz_h_dir = os.path.join(output_dir, f"{sim_name}_h")
    npz_p_dir = os.path.join(output_dir, f"{sim_name}_p")
    npz_c_dir = os.path.join(output_dir, f"{sim_name}_c")
    os.makedirs(npz_h_dir, exist_ok=True)
    os.makedirs(npz_p_dir, exist_ok=True)
    os.makedirs(npz_c_dir, exist_ok=True)

    h_filename       = os.path.join(output_dir, sim_name + "_h")
    p_filename       = os.path.join(output_dir, sim_name + "_p")
    c_filename       = os.path.join(output_dir, sim_name + "_c")
    summary_filename = os.path.join(output_dir, f"{sim_name}_summary.txt")

    # ==============================
    # Domain & Mesh
    # ==============================
    Lx, Ly = 2.0, 1.0
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)

    # ==============================
    # Physical Parameters
    # ==============================
    rho_f   = 1000.0
    rho_s   = 1024.5
    c_fresh = 0.0
    c_salt  = 35.0 / rho_s
    beta    = (rho_s - rho_f) / c_salt

    q_in = Q_in / Ly

    p_atm_waiwera = 101325.0
    g = 9.81

    # ==============================
    # Time Stepping
    # ==============================
    if auto_terminate:
        n_steps_max = int(T_auto / dt)
    else:
        n_steps_max = n_steps

    # Startup diagnostics
    _h_elem       = min(Lx / Nx, Ly / Ny)
    _dt_crit_diff = _h_elem**2 / (2.0 * phi * D + 1e-20)
    print(f"\n  INFO  Mesh: {Nx}x{Ny}  |  element size: h = {_h_elem:.5f} m")
    print(f"  INFO  Diffusion coeff D = {D:.5f} m^2/d")
    print(f"  INFO  dt = {dt:.6f} d  |  diffusive stability limit: dt_crit ~ {_dt_crit_diff:.5f} d")
    if auto_terminate:
        print(f"  INFO  Auto-terminate ON: ss_tol = {ss_tol:.2e} | "
              f"T_auto = {T_auto:.0f} d (max {n_steps_max} steps)")
    else:
        print(f"  INFO  Fixed run: T = {T} d  |  n_steps = {n_steps}")
    print()

    # ==============================
    # Function Spaces
    # ==============================
    H = FunctionSpace(mesh, "CG", 1)
    C = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    h_sol  = Function(H)
    c_new  = Function(C)
    c_old  = Function(C)
    v_proj = Function(V)

    p = Function(H)
    p.rename("Absolute_Pressure", "Absolute Pressure (Pa)")

    h_trial = TrialFunction(H)
    h_test  = TestFunction(H)
    c_trial = TrialFunction(C)
    c_test  = TestFunction(C)

    # ==============================
    # Density Helpers
    # ==============================
    def density(conc):
        return rho_f + beta * conc

    def density_ratio(conc):
        return density(conc) / rho_f

    # ==============================
    # Boundary Conditions
    # ==============================
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], Lx) and on_boundary

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Left().mark(boundaries, 1)
    Right().mark(boundaries, 2)

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    n  = FacetNormal(mesh)

    p_atm_fenics = 0.0
    z_sea  = Ly
    h_right = Expression(
        "(p_atm/(rho_f*g)) + (rho_s/rho_f)*(z_sea - x[1]) + x[1]",
        p_atm=p_atm_fenics, rho_f=rho_f, rho_s=rho_s,
        g=g, z_sea=z_sea, degree=1,
    )

    delx     = Lx / Nx
    ghb_cond = K / (0.5 * delx)
    bcs_h    = []

    bc_c_left = DirichletBC(C, Constant(c_fresh), boundaries, 1)
    bcs_c     = [bc_c_left]
    c_hat_in  = Constant(c_salt)

    # ==============================
    # Initial Condition
    # ==============================
    c_old.interpolate(Constant(c_fresh))

    # ==============================
    # Output Files
    # ==============================
    xdmf_h  = XDMFFile(h_filename  + ".xdmf")
    xdmf_p  = XDMFFile(p_filename  + ".xdmf")
    xdmf_c  = XDMFFile(c_filename  + ".xdmf")

    for xf in (xdmf_h, xdmf_p, xdmf_c):
        xf.parameters["flush_output"] = True
        xf.parameters["functions_share_mesh"] = True

    # ==============================
    # Time Loop
    # ==============================
    t        = 0.0
    step_idx = 0
    start_time = time.time()

    # Steady-state tracking
    ss_reached   = False
    ss_step      = None
    ss_time      = None
    mass_prev    = None   # total salt mass from previous step (for ss check)

    h_mesh  = CellDiameter(mesh)
    z_coord = SpatialCoordinate(mesh)[1]

    def save_npz(step):
        h_nodal = h_sol.vector().get_local()
        p_nodal = p.vector().get_local()
        h_dof   = h_sol.function_space().tabulate_dof_coordinates()
        c_nodal = c_new.vector().get_local()
        c_dof   = c_new.function_space().tabulate_dof_coordinates()
        np.savez_compressed(
            os.path.join(npz_h_dir, f"{sim_name}_h_{step}.npz"),
            x=h_dof[:, 0], y=h_dof[:, 1], u=h_nodal)
        np.savez_compressed(
            os.path.join(npz_p_dir, f"{sim_name}_p_{step}.npz"),
            x=h_dof[:, 0], y=h_dof[:, 1], u=p_nodal)
        np.savez_compressed(
            os.path.join(npz_c_dir, f"{sim_name}_c_{step}.npz"),
            x=c_dof[:, 0], y=c_dof[:, 1], u=c_nodal)

    # Export initial condition (frame 0)
    c_new.assign(c_old)
    p_expr = p_atm_waiwera + rho_f * g * (h_sol - z_coord)
    p.assign(project(p_expr, H))

    if step_idx in npz_export_frames:
        save_npz(step_idx)

    xdmf_h.write(h_sol, t)
    xdmf_p.write(p, t)
    xdmf_c.write(c_new, t)

    while step_idx < n_steps_max:
        t        += dt
        step_idx += 1

        # --- 1. Flow ---
        a_h = (K * dot(grad(h_trial), grad(h_test)) * dx
               + ghb_cond * h_trial * h_test * ds(2))
        L_h = (-K * (density_ratio(c_old) - 1.0) * Dx(h_test, 1) * dx
               + q_in * h_test * ds(1)
               + ghb_cond * h_right * h_test * ds(2))
        solve(a_h == L_h, h_sol, bcs_h)

        # --- 2. Darcy Velocity ---
        v_expr = -K * (grad(h_sol) + (density_ratio(c_old) - 1.0) * as_vector((0.0, 1.0)))
        v_proj.assign(project(v_expr, V))

        # --- 3. Transport (SUPG stabilised) ---
        vn    = dot(v_proj, n)
        vnorm = sqrt(dot(v_proj, v_proj) + 1e-14)
        Pe_local = vnorm * h_mesh / (2.0 * phi * D + 1e-20)
        # Optimal SUPG tau: xi(Pe) * h/(2|u|)  [Franca & Frey, 1992]
        #   xi = Pe/3  for Pe < 3  (diffusion-dominated: small stabilisation)
        #   xi = 1     for Pe >= 3 (advection-dominated: full upwind)
        xi  = conditional(lt(Pe_local, 3.0), Pe_local / 3.0, Constant(1.0))
        tau = xi * h_mesh / (2.0 * vnorm + 1e-14)

        a_c = (
            (phi / dt) * c_trial * c_test * dx
            + phi * D * dot(grad(c_trial), grad(c_test)) * dx
            - c_trial * dot(v_proj, grad(c_test)) * dx
            + conditional(ge(vn, 0.0), vn, 0.0) * c_trial * c_test * ds(2)
        )
        a_c += tau * ((phi / dt) * c_trial + dot(v_proj, grad(c_trial))) \
               * dot(v_proj, grad(c_test)) * dx

        L_c = (
            (phi / dt) * c_old * c_test * dx
            - conditional(lt(vn, 0.0), vn * c_hat_in, 0.0) * c_test * ds(2)
        )
        L_c += tau * (phi / dt) * c_old * dot(v_proj, grad(c_test)) * dx

        solve(a_c == L_c, c_new, bcs_c)

        # Concentration clipping + BC re-enforcement
        c_array = c_new.vector().get_local()
        c_array = np.clip(c_array, 0.0, float(c_salt))
        c_new.vector().set_local(c_array)
        c_new.vector().apply("insert")
        for bc in bcs_c:
            bc.apply(c_new.vector())

        # --- 4. Pressure field ---
        p_expr = p_atm_waiwera + rho_f * g * (h_sol - z_coord)
        p.assign(project(p_expr, H))

        # Diagnostics
        flux_left  = assemble(dot(v_proj, n) * ds(1))
        flux_right = assemble(dot(v_proj, n) * ds(2))
        c_mean     = assemble(c_new * dx) / (Lx * Ly)
        CFL = v_proj.vector().norm("linf") * dt / min(Lx / Nx, Ly / Ny)
        print(f"t={t:.4f} (Step {step_idx}/{n_steps_max}) | "
              f"flux_left={flux_left:.4f} | flux_right={flux_right:.4f} | "
              f"c_mean={c_mean:.4f} | CFL={CFL:.3f}")

        xdmf_h.write(h_sol, t)
        xdmf_p.write(p, t)
        xdmf_c.write(c_new, t)

        if step_idx in npz_export_frames:
            save_npz(step_idx)

        # --- 5. Steady-state check (auto_terminate mode) ---
        if auto_terminate:
            mass_current = assemble(phi * rho_s * c_new * dx)
            if mass_prev is not None:
                rel_change = abs(mass_current - mass_prev) / max(abs(mass_prev), 1e-20)
                print(f"  [SS check] mass={mass_current:.6f} kg/m | "
                      f"rel_change={rel_change:.3e} (tol={ss_tol:.2e})")
                if rel_change < ss_tol:
                    ss_reached = True
                    ss_step    = step_idx
                    ss_time    = t
                    print(f"  [SS] Steady state reached at t={t:.4f} d "
                          f"(step {step_idx}). Terminating.")
                    # Always save the final frame on early exit
                    save_npz(step_idx)
                    c_old.assign(c_new)
                    break
            mass_prev = mass_current

        c_old.assign(c_new)

    for xf in (xdmf_h, xdmf_p, xdmf_c):
        xf.close()

    end_time     = time.time()
    elapsed_time = end_time - start_time

    # ==============================
    # Post-Simulation Statistics
    # ==============================
    print("\n--- Final Model Statistics ---")

    # Extract bottom-boundary node values (y ≈ 0)
    coords       = mesh.coordinates()
    c_vals       = c_new.compute_vertex_values(mesh)
    bottom_mask  = np.isclose(coords[:, 1], 0.0)
    x_bottom     = coords[bottom_mask, 0]
    c_bottom     = c_vals[bottom_mask]

    # Sort by x so interpolation is well-behaved
    sort_order   = np.argsort(x_bottom)
    x_bottom     = x_bottom[sort_order]
    c_bottom     = c_bottom[sort_order]

    # --- Stat 1: Toe length (50% isoline on bottom boundary) ---
    target_50 = 0.5 * c_salt
    toe_length = float(x_bottom[np.abs(c_bottom - target_50).argmin()])
    print(f"Toe length (50% isoline on bottom): x = {toe_length:.4f} m")

    # --- Stat 2: Mixing zone width (distance between 10% and 90% isolines) ---
    target_10 = 0.10 * c_salt
    target_90 = 0.90 * c_salt
    x_10 = float(x_bottom[np.abs(c_bottom - target_10).argmin()])
    x_90 = float(x_bottom[np.abs(c_bottom - target_90).argmin()])
    mixing_zone_width = abs(x_10 - x_90)
    print(f"Mixing zone width (10%-90% isolines on bottom): {mixing_zone_width:.4f} m")
    print(f"  (x_10%={x_10:.4f} m, x_90%={x_90:.4f} m)")

    # --- Stat 3: Total salt mass (kg/m) ---
    # mass = integral(phi * rho_s * c) dV  [units: kg per metre of aquifer depth]
    total_salt_mass = assemble(phi * rho_s * c_new * dx)
    print(f"Total salt mass: {total_salt_mass:.4f} kg/m")
    print("------------------------------\n")

    # ==============================
    # Summary File
    # ==============================
    with open(summary_filename, "w") as f:
        f.write("=========================================\n")
        f.write("    HENRY PROBLEM SIMULATION SUMMARY\n")
        f.write("=========================================\n\n")
        f.write("--- Parameters ---\n")
        f.write(f"Date/Time: {current_time_str}\n")
        f.write(f"Simulation Name: {sim_name}\n")
        f.write(f"Notes: {notes}\n")
        f.write(f"Domain: {Lx} m x {Ly} m (Nx={Nx}, Ny={Ny})\n")
        f.write(f"Hydraulic Conductivity (K): {K} m/d\n")
        f.write(f"Porosity (phi): {phi}\n")
        f.write(f"Diffusion Coefficient (D): {float(D):.5f} m^2/d\n")
        f.write(f"Freshwater/Seawater Conc: {c_fresh} / {c_salt}\n")
        f.write(f"Densities (rho_f / rho_s): {rho_f} / {rho_s} kg/m^3\n")
        f.write(f"Equivalent Waiwera Atmospheric Pressure (P_atm): {p_atm_waiwera} Pa\n")
        f.write(f"Total Inflow (Q_in): {Q_in} m^3/d\n")
        f.write(f"Simulation Time (T): {T} d  |  dt = {dt:.6f} d  |  n_steps = {n_steps}\n")
        if auto_terminate:
            f.write(f"Auto-terminate: ON (ss_tol={ss_tol:.2e}, T_auto={T_auto:.0f} d)\n")
        f.write("\n")
        f.write("--- Final Model Statistics ---\n")
        f.write(f"Total Steps Completed: {step_idx} of {n_steps_max} max\n")
        f.write(f"Simulation End Time: {t:.4f} d\n")
        if auto_terminate:
            f.write(f"Steady State Reached: {'YES' if ss_reached else 'NO (T_auto reached)'}\n")
            if ss_reached:
                f.write(f"  Steady state at step {ss_step}, t = {ss_time:.4f} d\n")
        f.write(f"Toe length (50% isoline on bottom boundary): {toe_length:.4f} m\n")
        f.write(f"Mixing zone width (10%-90% isolines on bottom): {mixing_zone_width:.4f} m\n")
        f.write(f"  x at 10% isoline: {x_10:.4f} m\n")
        f.write(f"  x at 90% isoline: {x_90:.4f} m\n")
        f.write(f"Total salt mass (phi * rho_s * c integrated): {total_salt_mass:.4f} kg/m\n")
        f.write(f"Simulation Time Elapsed: {elapsed_time:.2f} seconds\n")

    print(f"Simulation complete. Summary saved to: {summary_filename}")
    print(f"Simulation took {elapsed_time:.2f} s ({elapsed_time/60:.2f} min)")

    return {
        "sim_name":          sim_name,
        "D":                 D,
        "Q_in":              Q_in,
        "toe_length":        toe_length,
        "mixing_zone_width": mixing_zone_width,
        "x_10":              x_10,
        "x_90":              x_90,
        "total_salt_mass":   total_salt_mass,
        "ss_reached":        ss_reached,
        "ss_step":           ss_step,
        "ss_time":           ss_time,
        "elapsed_time":      elapsed_time,
    }

# ==============================
# CLI Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single Henry Problem simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--D",     type=float, default=0.57024,
                        help="Diffusion coefficient (m^2/d)")
    parser.add_argument("--Q_in",  type=float, default=5.7024,
                        help="Freshwater inflow rate (m^3/d per m width). "
                             "Classic values: 5.7024 (scenario a), 2.851 (scenario b)")
    parser.add_argument("--T",     type=float, default=0.5,
                        help="Simulation end time (days)")
    parser.add_argument("--dt",    type=float, default=0.001,
                        help="Time step size (days)")
    parser.add_argument("--Nx",    type=int,   default=80,
                        help="Mesh subdivisions in x")
    parser.add_argument("--Ny",    type=int,   default=40,
                        help="Mesh subdivisions in y")
    parser.add_argument("--K",     type=float, default=864.0,
                        help="Hydraulic conductivity (m/d)")
    parser.add_argument("--phi",   type=float, default=0.35,
                        help="Porosity")
    parser.add_argument("--notes", type=str,   default="",
                        help="Free-text note for the summary file")
    args = parser.parse_args()

    run_henry(
        D=args.D,
        Q_in=args.Q_in,
        T=args.T,
        dt=args.dt,
        Nx=args.Nx,
        Ny=args.Ny,
        K=args.K,
        phi=args.phi,
        notes=args.notes,
    )