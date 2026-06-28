import os
from dolfin import *
import numpy as np
import time
import argparse

def run_island_single(
    D=0.57024,
    K=25.0,     # hydraulic conductivity (m/d)  (Holocene: 25.0, Pleistocene: 500.0)
    phi=0.30,   # porosity                       (Holocene: 0.30, Pleistocene: 0.20)
    T=3650.0,
    dt=0.365,
    dt_min=1e-4,
    dt_max=5.0,
    cfl_target=0.8,
    dc_target=0.01,
    Nx=100,
    Ny=50,
    name=None,
    notes="",
    npz_export_frames=None,
    auto_terminate=False,
    ss_tol=1e-4,
):
    """
    Run an island freshwater lens simulation (single homogeneous layer).

    Solves density-driven flow and transport on a 2D vertical half-island
    cross-section with a single uniform geological layer.

    Domain: 500 m (x) x 50 m (z), where z=0 is the aquifer base and z=50
    is the land surface.

    Boundary Conditions:
      Left   (x=0):     Symmetry – no flow, no salt flux.
      Right  (x=500):   Hydrostatic seawater pressure (GHB), seawater
                         salinity on inflow.
      Top    (z=50):    Rainfall recharge (2 m/yr freshwater).
      Bottom (z=0):     No flow.

    Initial Condition:  Fully saline aquifer.

    Parameters
    ----------
    D : float
        Dispersion coefficient (m^2/d).
    K : float
        Hydraulic conductivity (m/d).
        Classic island values: 25.0 (Holocene), 500.0 (Pleistocene).
    phi : float
        Porosity (dimensionless).
        Classic island values: 0.30 (Holocene), 0.20 (Pleistocene).
    T : float
        Simulation end time (days). Ignored when auto_terminate=True
        (the loop ceiling is then T_auto = 10,000 d).
    dt : float
        Time step size (days). Default 0.365 d (~1 step/day).
    Nx, Ny : int
        Mesh subdivisions in x and y.
    name : str or None
        Base name for the output folder (timestamp is always prepended).
        If None, defaults to "island_single_K{K:.0f}".
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
    dt_min : float
        Minimum allowed adaptive time step size (days).
    dt_max : float
        Maximum allowed adaptive time step size (days).
    cfl_target : float
        Target CFL number for adaptive time stepping.
    dc_target : float
        Target concentration change metric for adaptive stepping.

    Returns
    -------
    dict
        Keys: sim_name, D, K, phi, depth_50, depth_10, mixing_zone_width,
              total_salt_mass, ss_reached, ss_step, ss_time, elapsed_time.
    """
    # T_auto: hard ceiling for auto-terminate mode (days)
    T_auto = 10000.0

    # n_steps derived from the initial dt (used for npz defaults and reporting)
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
        name = f"island_single_K{K:.0f}"
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
    Lx = 500.0   # half-island width (m)
    Ly =  50.0   # aquifer depth (m); y=0 is base, y=50 is land surface

    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)

    # ==============================
    # Physical Parameters
    # ==============================
    rho_f   = 1000.0        # freshwater density (kg/m^3)
    rho_s   = 1024.5        # seawater density  (kg/m^3)
    c_fresh = 0.0
    c_salt  = 35.0 / rho_s  # mass fraction
    beta    = (rho_s - rho_f) / c_salt

    # Recharge: 2 m/year expressed as Darcy velocity (m/d)
    q_recharge = 2.0 / 365.25

    p_atm_waiwera = 101325.0
    g = 9.81

    # ==============================
    # Spatially Varying Fields (UFL)
    # ==============================
    x_coord, z_coord_sym = SpatialCoordinate(mesh)

    K_field   = Constant(K)
    phi_field = Constant(phi)

    # ==============================
    # Time Stepping
    # ==============================
    if auto_terminate:
        n_steps_max = int(T_auto / dt)
    else:
        n_steps_max = n_steps

    _h_elem       = min(Lx / Nx, Ly / Ny)
    _dt_crit_diff = _h_elem**2 / (2.0 * phi * D + 1e-20)
    print(f"\n  INFO  Mesh: {Nx}x{Ny}  |  element size: h = {_h_elem:.5f} m")
    print(f"  INFO  Domain: {Lx} m x {Ly} m")
    print(f"  INFO  Layer properties: K = {K} m/d, phi = {phi}")
    print(f"  INFO  Recharge = {q_recharge:.6f} m/d  (2 m/yr)")
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
    V = VectorFunctionSpace(mesh, "DG", 0)

    h_sol  = Function(H)
    c_new  = Function(C)
    c_old  = Function(C)
    c_old_backup = Function(C)
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

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], Ly) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Left().mark(boundaries, 1)
    Right().mark(boundaries, 2)
    Top().mark(boundaries, 3)
    Bottom().mark(boundaries, 4)

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    n  = FacetNormal(mesh)

    # --- Flow BCs ---
    p_atm_fenics = 0.0
    z_sea = Ly
    h_right = Expression(
        "(p_atm/(rho_f*g)) + (rho_s/rho_f)*(z_sea - x[1]) + x[1]",
        p_atm=p_atm_fenics, rho_f=rho_f, rho_s=rho_s,
        g=g, z_sea=z_sea, degree=1,
    )

    delx     = Lx / Nx
    ghb_cond = K_field / (0.5 * delx)
    bcs_h    = []

    # --- Transport BCs ---
    bcs_c    = []
    c_hat_in = Constant(c_salt)

    # ==============================
    # Initial Condition – fully saline
    # ==============================
    c_old.interpolate(Constant(c_salt))

    # ==============================
    # Output Files
    # ==============================
    xdmf_h = XDMFFile(h_filename + ".xdmf")
    xdmf_p = XDMFFile(p_filename + ".xdmf")
    xdmf_c = XDMFFile(c_filename + ".xdmf")

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
    ss_reached = False
    ss_step    = None
    ss_time    = None
    mass_prev  = None

    h_mesh = CellDiameter(mesh)
    dx_min = min(Lx / Nx, Ly / Ny)

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
    p_expr = p_atm_waiwera + rho_f * g * (h_sol - z_coord_sym)
    p.assign(project(p_expr, H))

    if step_idx in npz_export_frames:
        save_npz(step_idx)

    xdmf_h.write(h_sol, t)
    xdmf_p.write(p, t)
    xdmf_c.write(c_new, t)

    T_final = T_auto if auto_terminate else T

    while t < T_final:
        if t + dt > T_final:
            dt = T_final - t

        step_idx += 1
        t        += dt

        c_old_backup.assign(c_old)

        # --- 1. Flow ---
        a_h = (K_field * dot(grad(h_trial), grad(h_test)) * dx
               + ghb_cond * h_trial * h_test * ds(2))
        L_h = (-K_field * (density_ratio(c_old) - 1.0) * Dx(h_test, 1) * dx
               + q_recharge * h_test * ds(3)
               + ghb_cond * h_right * h_test * ds(2))
        solve(a_h == L_h, h_sol, bcs_h)

        # --- 2. Darcy Velocity ---
        v_expr = -K_field * (grad(h_sol)
                             + (density_ratio(c_old) - 1.0) * as_vector((0.0, 1.0)))
        v_proj.assign(project(v_expr, V))

        # --- 3. Transport (SUPG stabilised) ---
        vn    = dot(v_proj, n)
        vnorm = sqrt(dot(v_proj, v_proj) + 1e-14)
        Pe_local = vnorm * h_mesh / (2.0 * phi_field * D + 1e-20)
        xi  = conditional(lt(Pe_local, 3.0), Pe_local / 3.0, Constant(1.0))
        tau = xi * h_mesh / (2.0 * vnorm + 1e-14)

        a_c = (
            (phi_field / dt) * c_trial * c_test * dx
            + phi_field * D * dot(grad(c_trial), grad(c_test)) * dx
            - c_trial * dot(v_proj, grad(c_test)) * dx
            + conditional(ge(vn, 0.0), vn, 0.0) * c_trial * c_test * ds(2)
            + conditional(ge(vn, 0.0), vn, 0.0) * c_trial * c_test * ds(3)
        )
        a_c += tau * ((phi_field / dt) * c_trial + dot(v_proj, grad(c_trial))) \
               * dot(v_proj, grad(c_test)) * dx

        L_c = (
            (phi_field / dt) * c_old * c_test * dx
            - conditional(lt(vn, 0.0), vn * c_hat_in, 0.0) * c_test * ds(2)
        )
        L_c += tau * (phi_field / dt) * c_old * dot(v_proj, grad(c_test)) * dx

        solve(a_c == L_c, c_new, bcs_c)

        c_array = c_new.vector().get_local()
        c_array = np.clip(c_array, 0.0, float(c_salt))
        c_new.vector().set_local(c_array)
        c_new.vector().apply("insert")
        for bc in bcs_c:
            bc.apply(c_new.vector())

        p_expr = p_atm_waiwera + rho_f * g * (h_sol - z_coord_sym)
        p.assign(project(p_expr, H))

        flux_left   = assemble(dot(v_proj, n) * ds(1))
        flux_right  = assemble(dot(v_proj, n) * ds(2))
        flux_top    = assemble(dot(v_proj, n) * ds(3))
        flux_bottom = assemble(dot(v_proj, n) * ds(4))
        c_mean      = assemble(c_new * dx) / (Lx * Ly)
        CFL = v_proj.vector().norm("linf") * dt / dx_min
        c_diff_local = np.abs(c_array - c_old_backup.vector().get_local())
        max_dc = np.max(c_diff_local) / float(c_salt)

        if CFL > 1.2:
            print(f"  [STEP REJECTED] Rollback triggered -> CFL: {CFL:.2f} (Max: 1.2 allowed)")
            t        -= dt
            step_idx -= 1
            dt        = max(dt_min, dt * 0.5)
            c_old.assign(c_old_backup)
            continue

        dt_factor = cfl_target / max(CFL, 1e-6)
        dt_factor = max(0.5, min(1.5, dt_factor))
        dt_next = max(dt_min, min(dt_max, dt * dt_factor))

        print(f"t={t:.2f} d  ({t/365.25:.3f} yr)  Step {step_idx} | "
              f"dt: {dt:.4f} d -> next: {dt_next:.4f} d | "
              f"CFL: {CFL:.2f} | max_dc: {max_dc:.3f} | c_mean: {c_mean:.6f}")

        xdmf_h.write(h_sol, t)
        xdmf_p.write(p, t)
        xdmf_c.write(c_new, t)

        if not auto_terminate and step_idx in npz_export_frames:
            save_npz(step_idx)

        if auto_terminate:
            mass_current = assemble(phi_field * rho_s * c_new * dx)
            if mass_prev is not None:
                rel_change = abs(mass_current - mass_prev) / max(abs(mass_prev), 1e-20)
                print(f"  [SS check] mass={mass_current:.6f} kg/m | "
                      f"rel_change={rel_change:.3e} (tol={ss_tol:.2e})")
                if rel_change < ss_tol:
                    ss_reached = True
                    ss_step    = step_idx
                    ss_time    = t
                    print(f"  [SS] Steady state reached at t={t:.2f} d "
                          f"(step {step_idx}). Terminating.")
                    save_npz(step_idx)
                    break
            mass_prev = mass_current

        dt = dt_next
        c_old.assign(c_new)

    for xf in (xdmf_h, xdmf_p, xdmf_c):
        xf.close()

    end_time     = time.time()
    elapsed_time = end_time - start_time

    # ==============================
    # Post-Simulation Statistics
    # ==============================
    print("\n--- Final Model Statistics ---")

    coords      = mesh.coordinates()
    c_vals      = c_new.compute_vertex_values(mesh)
    left_mask   = np.isclose(coords[:, 0], 0.0)
    y_left      = coords[left_mask, 1]
    c_left      = c_vals[left_mask]

    sort_order = np.argsort(y_left)
    y_left     = y_left[sort_order]
    c_left     = c_left[sort_order]

    # --- Stat 1: Depth_50 (50% isoline on left boundary) ---
    depth_50 = float(y_left[np.abs(c_left - 0.5 * c_salt).argmin()])
    print(f"Depth_50 (50% isoline on left boundary): y = {depth_50:.4f} m")

    # --- Stat 2: Mixing zone width (depth_10 - depth_50 on left boundary) ---
    depth_10 = float(y_left[np.abs(c_left - 0.10 * c_salt).argmin()])
    mixing_zone_width = abs(depth_10 - depth_50)
    print(f"Mixing zone width (depth_10 - depth_50 on left boundary): {mixing_zone_width:.4f} m")
    print(f"  (depth_10={depth_10:.4f} m, depth_50={depth_50:.4f} m)")

    # --- Stat 3: Total salt mass (kg/m) ---
    # mass = integral(phi * rho_s * c) dV  [units: kg per metre of aquifer depth]
    total_salt_mass = assemble(phi_field * rho_s * c_new * dx)
    print(f"Total salt mass: {total_salt_mass:.4f} kg/m")
    print("------------------------------\n")

    # ==============================
    # Summary File
    # ==============================
    with open(summary_filename, "w") as f:
        f.write("=============================================\n")
        f.write("    ISLAND SINGLE-LAYER SIMULATION SUMMARY\n")
        f.write("=============================================\n\n")
        f.write("--- Parameters ---\n")
        f.write(f"Date/Time: {current_time_str}\n")
        f.write(f"Simulation Name: {sim_name}\n")
        f.write(f"Notes: {notes}\n")
        f.write(f"Domain: {Lx} m x {Ly} m (Nx={Nx}, Ny={Ny})\n")
        f.write(f"Layer Properties: K = {K} m/d, phi = {phi}\n")
        f.write(f"Diffusion Coefficient (D): {float(D):.5f} m^2/d\n")
        f.write(f"Recharge Rate: {q_recharge:.6f} m/d  (2 m/yr)\n")
        f.write(f"Freshwater/Seawater Conc: {c_fresh} / {c_salt}\n")
        f.write(f"Densities (rho_f / rho_s): {rho_f} / {rho_s} kg/m^3\n")
        f.write(f"Equivalent Waiwera Atmospheric Pressure (P_atm): {p_atm_waiwera} Pa\n")
        f.write(f"Simulation Time (T): {T} d  |  dt = {dt:.6f} d  |  n_steps = {n_steps}\n")
        if auto_terminate:
            f.write(f"Auto-terminate: ON (ss_tol={ss_tol:.2e}, T_auto={T_auto:.0f} d)\n")
        f.write("\n")
        f.write("--- Boundary Conditions ---\n")
        f.write("Left   (x=0):   Symmetry – no flow, no salt flux\n")
        f.write("Right  (x=500): Hydrostatic seawater GHB + ambient salt\n")
        f.write("Top    (z=50):  Rainfall recharge (2 m/yr freshwater)\n")
        f.write("Bottom (z=0):   No flow\n\n")
        f.write("--- Final Model Statistics ---\n")
        f.write(f"Total Steps Completed: {step_idx} of {n_steps_max} max\n")
        f.write(f"Simulation End Time: {t:.2f} d  ({t/365.25:.3f} yr)\n")
        if auto_terminate:
            f.write(f"Steady State Reached: {'YES' if ss_reached else 'NO (T_auto reached)'}\n")
            if ss_reached:
                f.write(f"  Steady state at step {ss_step}, t = {ss_time:.2f} d\n")
        f.write(f"Depth_50 (50% isoline on left boundary): {depth_50:.4f} m\n")
        f.write(f"Mixing zone width (depth_10 - depth_50 on left boundary): {mixing_zone_width:.4f} m\n")
        f.write(f"  depth_10: {depth_10:.4f} m\n")
        f.write(f"  depth_50: {depth_50:.4f} m\n")
        f.write(f"Total salt mass (phi * rho_s * c integrated): {total_salt_mass:.4f} kg/m\n")
        f.write(f"Simulation Time Elapsed: {elapsed_time:.2f} seconds\n")

    print(f"Simulation complete. Summary saved to: {summary_filename}")
    print(f"Simulation took {elapsed_time:.2f} s ({elapsed_time/60:.2f} min)")

    return {
        "sim_name":          sim_name,
        "D":                 D,
        "K":                 K,
        "phi":               phi,
        "depth_50":          depth_50,
        "depth_10":          depth_10,
        "mixing_zone_width": mixing_zone_width,
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
        description="Run an island freshwater lens simulation (single layer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--D",              type=float, default=0.57024,
                        help="Diffusion coefficient (m^2/d)")
    parser.add_argument("--K",              type=float, default=25.0,
                        help="Hydraulic conductivity (m/d). "
                             "Classic: 25.0 (Holocene), 500.0 (Pleistocene)")
    parser.add_argument("--phi",            type=float, default=0.30,
                        help="Porosity. Classic: 0.30 (Holocene), 0.20 (Pleistocene)")
    parser.add_argument("--T",              type=float, default=3650.0,
                        help="Simulation end time (days)")
    parser.add_argument("--dt",             type=float, default=0.365,
                        help="Initial time step size (days)")
    parser.add_argument("--dt_min",         type=float, default=1e-4,
                        help="Minimum allowed time step size (days)")
    parser.add_argument("--dt_max",         type=float, default=5.0,
                        help="Maximum allowed time step size (days)")
    parser.add_argument("--cfl_target",     type=float, default=0.8,
                        help="Target CFL number for adaptive stepping")
    parser.add_argument("--dc_target",      type=float, default=0.01,
                        help="Target concentration change metric for adaptive stepping")
    parser.add_argument("--Nx",             type=int,   default=100,
                        help="Mesh subdivisions in x")
    parser.add_argument("--Ny",             type=int,   default=50,
                        help="Mesh subdivisions in y")
    parser.add_argument("--auto_terminate", action="store_true",
                        help="Enable steady-state auto-termination")
    parser.add_argument("--ss_tol",         type=float, default=1e-4,
                        help="Relative salt-mass change tolerance for auto-termination")
    parser.add_argument("--notes",          type=str,   default="",
                        help="Free-text note for the summary file")
    args = parser.parse_args()

    run_island_single(
        D=args.D,
        K=args.K,
        phi=args.phi,
        T=args.T,
        dt=args.dt,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        cfl_target=args.cfl_target,
        dc_target=args.dc_target,
        Nx=args.Nx,
        Ny=args.Ny,
        auto_terminate=args.auto_terminate,
        ss_tol=args.ss_tol,
        notes=args.notes,
    )
