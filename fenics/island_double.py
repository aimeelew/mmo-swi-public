import os
from dolfin import *
import numpy as np
import time
import argparse

def run_island_double(
    D=0.57024, # m2 per day
    T=365.25*20, # 20 years
    dt=0.1, # default time step size (days)
    dt_min=1e-4, # smallest time step allowed (days)
    dt_max=30.0, # largest time step allowed (days) = 1 month
    cfl_target=0.8, # Target Courant number
    Nx=100,
    Ny=50,
    K_upper=25.0,
    K_lower=500.0,
    phi_upper=0.30,
    phi_lower=0.20,
    thurber_depth=12.0,
    q_recharge=None,
    name=None,
    notes="",
    auto_terminate=False,
    ss_tol=1e-4,
    output_frequency=100,
):
    """
    Run an island freshwater lens simulation with Adaptive Time Stepping.
    """
    T_auto = 10000.0 # Maximum simulation time for auto-termination (days)
    print(f"  INFO  Output frequency: every {output_frequency} steps (first and final always exported)")

    # ==============================
    # Naming & Output Directories
    # ==============================
    current_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    if name is None:
        name = "island_double"
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
    Lx = 500.0   
    Ly =  50.0   
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)

    # ==============================
    # Physical Parameters
    # ==============================
    rho_f   = 1000.0        
    rho_s   = 1024.5        
    c_fresh = 0.0
    c_salt  = 35.0 / rho_s  
    beta    = (rho_s - rho_f) / c_salt
    z_thurber = Ly - thurber_depth

    if q_recharge is None:
        q_recharge = 2.0 / 365.25

    p_atm_waiwera = 101325.0
    g = 9.81

    # ==============================
    # Spatially Varying Fields
    # ==============================
    x_coord, z_coord_sym = SpatialCoordinate(mesh)
    K_field   = conditional(ge(z_coord_sym, z_thurber), Constant(K_upper), Constant(K_lower))
    phi_field = conditional(ge(z_coord_sym, z_thurber), Constant(phi_upper), Constant(phi_lower))

    # ==============================
    # Function Spaces
    # ==============================
    H = FunctionSpace(mesh, "CG", 1)
    C = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "DG", 0)

    h_sol  = Function(H)
    c_new  = Function(C)
    c_old  = Function(C)
    c_old_backup = Function(C)  # Allocated for state rollbacks
    c_clipped = Function(C)     
    v_proj = Function(V)

    p = Function(H)
    p.rename("Absolute_Pressure", "Absolute Pressure (Pa)")

    h_trial = TrialFunction(H)
    h_test  = TestFunction(H)
    c_trial = TrialFunction(C)
    c_test  = TestFunction(C)

    def density(conc):
        return rho_f + beta * conc

    def density_ratio(conc):
        return density(conc) / rho_f

    # ==============================
    # Boundary Conditions
    # ==============================
    class Left(SubDomain):
        def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary
    class Right(SubDomain):
        def inside(self, x, on_boundary): return near(x[0], Lx) and on_boundary
    class Top(SubDomain):
        def inside(self, x, on_boundary): return near(x[1], Ly) and on_boundary
    class Bottom(SubDomain):
        def inside(self, x, on_boundary): return near(x[1], 0.0) and on_boundary

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Left().mark(boundaries, 1)
    Right().mark(boundaries, 2)
    Top().mark(boundaries, 3)
    Bottom().mark(boundaries, 4)

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh)

    n  = FacetNormal(mesh)

    p_atm_fenics = 0.0
    z_sea = Ly
    h_right = Expression(
        "(p_atm/(rho_f*g)) + (rho_s/rho_f)*(z_sea - x[1]) + x[1]",
        p_atm=p_atm_fenics, rho_f=rho_f, rho_s=rho_s, g=g, z_sea=z_sea, degree=1,
    )

    bcs_h    = [DirichletBC(H, h_right, boundaries, 2)]
    bcs_c    = []

    D_constant = Constant(D)
    q_constant = Constant(q_recharge)
    dt_constant = Constant(dt)
    c_hat_in = Constant(c_salt)

    c_old.interpolate(Constant(c_salt))

    solver_parameters = {
        "linear_solver": "bicgstab", 
        "preconditioner": "ilu"
        }

    # ==============================
    # Output Management
    # ==============================
    xdmf_h = XDMFFile(h_filename + ".xdmf")
    xdmf_p = XDMFFile(p_filename + ".xdmf")
    xdmf_c = XDMFFile(c_filename + ".xdmf")

    for xf in (xdmf_h, xdmf_p, xdmf_c):
        xf.parameters["flush_output"] = True
        xf.parameters["functions_share_mesh"] = True

    t          = 0.0
    step_idx   = 0
    start_time = time.time()
    ss_reached = False
    ss_step    = None
    ss_time    = None
    last_xdmf_step = 0

    h_mesh = CellDiameter(mesh)
    dx_min = min(Lx / Nx, Ly / Ny)

    def save_npz(step):
        h_nodal = h_sol.vector().get_local()
        p_nodal = p.vector().get_local()
        h_dof   = h_sol.function_space().tabulate_dof_coordinates()
        c_nodal = c_new.vector().get_local()
        c_dof   = c_new.function_space().tabulate_dof_coordinates()
        np.savez_compressed(os.path.join(npz_h_dir, f"{sim_name}_h_{step}.npz"), x=h_dof[:, 0], y=h_dof[:, 1], u=h_nodal)
        np.savez_compressed(os.path.join(npz_p_dir, f"{sim_name}_p_{step}.npz"), x=h_dof[:, 0], y=h_dof[:, 1], u=p_nodal)
        np.savez_compressed(os.path.join(npz_c_dir, f"{sim_name}_c_{step}.npz"), x=c_dof[:, 0], y=c_dof[:, 1], u=c_nodal)

    c_new.assign(c_old)
    p.assign(project(p_atm_waiwera + rho_f * g * (h_sol - z_coord_sym), H))
    save_npz(step_idx) # Save the first frame

    xdmf_h.write(h_sol, t)
    xdmf_p.write(p, t)
    xdmf_c.write(c_new, t)

    T_final = T

    # ==============================
    # Main Adaptive Time Loop
    # ==============================
    while t < T_final:
        # Final step projection safety check
        if t + dt > T_final:
            dt = T_final - t
        dt_constant.assign(dt)

        step_idx += 1
        t        += dt

        # Snapshot current clean state for rollback capacity
        c_old_backup.assign(c_old)

        # --- 0. Strict Density Decoupling ---
        c_clipped.assign(c_old)
        c_clip_arr = c_clipped.vector().get_local()
        np.clip(c_clip_arr, 0.0, float(c_salt), out=c_clip_arr)
        c_clipped.vector().set_local(c_clip_arr)
        c_clipped.vector().apply("insert")

        # --- 1. Flow ---
        a_h = (K_field * dot(grad(h_trial), grad(h_test)) * dx)
        L_h = (-K_field * (density_ratio(c_clipped) - 1.0) * Dx(h_test, 1) * dx
               + q_constant * h_test * ds(3))
        solve(a_h == L_h, h_sol, bcs_h)

        # --- 2. Darcy Velocity ---
        v_expr = -K_field * (grad(h_sol) + (density_ratio(c_clipped) - 1.0) * as_vector((0.0, 1.0)))
        v_proj.assign(project(v_expr, V))

        # =========================================================
        # --- 3. Transport (SUPG & Discontinuity Capturing) ---
        # =========================================================
        vn    = dot(v_proj, n)
        vnorm = sqrt(dot(v_proj, v_proj) + 1e-14)
        
        # Grid scale calculation
        Pe_local = vnorm * h_mesh / (2.0 * phi_field * D_constant + 1e-20)
        xi  = conditional(lt(Pe_local, 3.0), Pe_local / 3.0, Constant(1.0))
        tau = xi * h_mesh / (2.0 * vnorm + 1e-14)

        # Explicit Inflow Concentration Definitions
        c_sea_in      = Constant(c_salt)
        c_recharge_in = Constant(0.0)  # Pure freshwater recharge

        # Base Advection-Diffusion Bilinear Form
        a_c = (
            (phi_field / dt_constant) * c_trial * c_test * dx
            + phi_field * D_constant * dot(grad(c_trial), grad(c_test)) * dx
            + dot(v_proj, grad(c_trial)) * c_test * dx
            # Inflow boundaries must be added to LHS matrix diagonal (vn < 0)
            - conditional(lt(vn, 0.0), vn, 0.0) * c_trial * c_test * ds(2)   # <-- ADDED FOR INFLOW RIGHT
            - conditional(lt(vn, 0.0), vn, 0.0) * c_trial * c_test * ds(3)   # <-- ADDED FOR INFLOW TOP
        )
        
        # Base Linear Form
        L_c = (
            (phi_field / dt_constant) * c_old * c_test * dx
            # Inflow handles (vn < 0)
            - conditional(lt(vn, 0.0), vn * c_sea_in, 0.0) * c_test * ds(2)
            - conditional(lt(vn, 0.0), vn * c_recharge_in, 0.0) * c_test * ds(3)
        )

        # --- SUPG Stabilization Term ---
        a_c += tau * ((phi_field / dt_constant) * c_trial + dot(v_proj, grad(c_trial))) \
               * dot(v_proj, grad(c_test)) * dx
        L_c += tau * (phi_field / dt_constant) * c_old * dot(v_proj, grad(c_test)) * dx

        # --- Discontinuity Capturing (DC) Shock-Stabilization ---
        # Evaluated semi-implicitly using c_old to preserve linearity of the system
        grad_c_old = grad(c_old)
        norm_grad_c_old = sqrt(dot(grad_c_old, grad_c_old) + 1e-12)
        
        # Strong residual of the transport equation at the previous step
        R_old = dot(v_proj, grad(c_old))
        
        # Viscosity parameter proportional to residual divided by gradient norm
        tau_DC = conditional(gt(norm_grad_c_old, 1e-5), abs(R_old) / norm_grad_c_old, 0.0)
        nu_DC  = 1.5 * h_mesh * conditional(lt(tau_DC, vnorm), tau_DC, vnorm)
        
        # Add orthogonal/crosswind dissipation to bilinear form
        a_c += nu_DC * dot(grad(c_trial), grad(c_test)) * dx

        solve(a_c == L_c, c_new, bcs_c, solver_parameters=solver_parameters)

        # Nodal Clipping bounds verification
        c_array = c_new.vector().get_local()
        c_array = np.clip(c_array, 0.0, float(c_salt))
        c_new.vector().set_local(c_array)
        c_new.vector().apply("insert")

        # =========================================================
        # --- 4. STEP EVALUATION & ADAPTIVE STEPPING METRICS ---
        # =========================================================
        current_v_max = v_proj.vector().norm("linf")
        CFL = current_v_max * dt / dx_min
        
        # Calculate maximum concentration delta for logging purposes only
        c_diff_local = np.abs(c_array - c_old_backup.vector().get_local())
        max_dc = np.max(c_diff_local) / float(c_salt)

        # HARD REJECTION THRESHOLDS (Controlled entirely by CFL)
        if CFL > 1.2:
            print(f"  [STEP REJECTED] Rollback triggered -> CFL: {CFL:.2f} (Max: 1.2 allowed)")
            t        -= dt
            step_idx -= 1
            dt        = max(dt_min, dt * 0.5)
            c_old.assign(c_old_backup)
            continue  # Re-run loop step with halved dt

        # SMOOTH ADAPTATION FOR CONTINUOUS TRACKING
        # Dt factor scales based on how close we are to our target CFL (e.g., 0.8)
        dt_factor = cfl_target / max(CFL, 1e-6)
        
        # Dampen adjustment rate change to keep time-marching smooth (0.5x to 1.5x)
        dt_factor  = max(0.5, min(1.5, dt_factor))
        dt_next    = max(dt_min, min(dt_max, dt * dt_factor))

        # --- Update accepted path fields ---
        p.assign(project(p_atm_waiwera + rho_f * g * (h_sol - z_coord_sym), H))

        flux_left   = assemble(dot(v_proj, n) * ds(1))
        flux_right  = assemble(dot(v_proj, n) * ds(2))
        flux_top    = assemble(dot(v_proj, n) * ds(3))
        flux_bottom = assemble(dot(v_proj, n) * ds(4))
        c_mean      = assemble(c_new * dx) / (Lx * Ly)
        
        print(f"t={t:.2f} d | Step {step_idx} | dt: {dt:.4f} d -> next: {dt_next:.4f} d | "
              f"CFL: {CFL:.2f} | max_dc: {max_dc:.3f} | c_mean: {c_mean:.5f}")

        if step_idx % output_frequency == 0:
            xdmf_h.write(h_sol, t)
            xdmf_p.write(p, t)
            xdmf_c.write(c_new, t)
            last_xdmf_step = step_idx

        # --- 5. Steady-State Metric Termination Tracking ---
        # Calculate total salt mass at the current and previous time steps
        mass_current = assemble(phi_field * rho_s * c_new * dx)
        mass_prev    = assemble(phi_field * rho_s * c_old_backup * dx)
        
        # Compute relative change and scale by time step (change rate per day)
        rel_change = abs(mass_current - mass_prev) / max(abs(mass_prev), 1e-20)
        mass_change_rate_per_day = rel_change / dt
        
        # Check steady-state criterion (computed regardless of auto_terminate flag)
        if mass_change_rate_per_day < ss_tol:
            ss_reached = True
            ss_step    = step_idx
            ss_time    = t
            print(f"\n  [SS] Steady state reached at t={t:.2f} d (Step {step_idx}).")
            print(f"  Current salt mass change rate: {mass_change_rate_per_day:.2e} / day (Tol: {ss_tol:.2e})")
            # Only break the loop if auto_terminate is enabled
            if auto_terminate:
                break

        # Prepare fields for the next iteration
        dt = dt_next
        c_old.assign(c_new)

    print(f"\n[EXPORT] Exporting final frame (Step {step_idx}) to NPZ arrays...")
    save_npz(step_idx)

    if last_xdmf_step != step_idx:
        xdmf_h.write(h_sol, t)
        xdmf_p.write(p, t)
        xdmf_c.write(c_new, t)

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
    
    # --- Left Boundary Mask (x = 0) ---
    left_mask   = np.isclose(coords[:, 0], 0.0)
    y_left      = coords[left_mask, 1]
    c_left      = c_vals[left_mask]

    # --- Bottom Boundary Mask (z = 0) ---
    bottom_mask = np.isclose(coords[:, 1], 0.0)
    x_bottom_nodes = coords[bottom_mask, 0]
    c_bottom_nodes = c_vals[bottom_mask]

    # --- Stat 1: Depth_50 (50% isoline on left boundary) ---
    depth_50 = float(y_left[np.abs(c_left - 0.5 * c_salt).argmin()])
    print(f"Depth_50 (50% isoline on left boundary): y = {depth_50:.4f} m")

    # --- Stat 2: Mixing zone width (depth_10 - depth_50 on left boundary) ---
    depth_10 = float(y_left[np.abs(c_left - 0.10 * c_salt).argmin()])
    mixing_zone_width = abs(depth_10 - depth_50)
    print(f"Mixing zone width (depth_10 - depth_50 on left boundary): {mixing_zone_width:.4f} m")
    print(f"  (depth_10={depth_10:.4f} m, depth_50={depth_50:.4f} m)")

    # --- Stat 3: x_bottom (50% isoline on bottom boundary) ---
    x_bottom = float(x_bottom_nodes[np.abs(c_bottom_nodes - 0.5 * c_salt).argmin()])
    print(f"x_bottom (50% isoline on bottom boundary): x = {x_bottom:.4f} m")

    # --- Stat 4: Total salt mass (kg/m) ---
    total_salt_mass = assemble(phi_field * rho_s * c_new * dx)
    print(f"Total salt mass: {total_salt_mass:.4f} kg/m")

    # --- Stat 5: Percentage of cells between 10% and 90% seawater ---
    C_DG0 = FunctionSpace(mesh, "DG", 0)
    c_dg0 = interpolate(c_new, C_DG0)
    c_dg0_vals = c_dg0.vector().get_local()
    lower_bound = 0.10 * c_salt
    upper_bound = 0.90 * c_salt
    in_range = (c_dg0_vals >= lower_bound) & (c_dg0_vals <= upper_bound)
    pct_cells = np.sum(in_range) / len(c_dg0_vals) * 100.0
    print(f"Percentage of cells with 10% to 90% seawater: {pct_cells:.2f}%")
    print("------------------------------\n")

    # ==============================
    # Summary File
    # ==============================
    with open(summary_filename, "w") as f:
        f.write("=============================================\n")
        f.write("    ISLAND DOUBLE-LAYER SIMULATION SUMMARY\n")
        f.write("=============================================\n\n")
        f.write("--- Parameters ---\n")
        f.write(f"Date/Time: {current_time_str}\n")
        f.write(f"Simulation Name: {sim_name}\n")
        f.write(f"Notes: {notes}\n")
        f.write(f"Domain: {Lx} m x {Ly} m (Nx={Nx}, Ny={Ny})\n")
        f.write(f"Thurber Discontinuity: z = {thurber_depth} m \n")
        f.write(f"Upper Layer (Holocene):    K = {K_upper} m/d, phi = {phi_upper}\n")
        f.write(f"Lower Layer (Pleistocene): K = {K_lower} m/d, phi = {phi_lower}\n")
        f.write(f"Diffusion Coefficient (D): {float(D):.5f} m^2/d\n")
        f.write(f"Recharge Rate: {q_recharge:.6f} m/d\n")
        f.write(f"Freshwater/Seawater Conc: {c_fresh} / {c_salt}\n")
        f.write(f"Densities (rho_f / rho_s): {rho_f} / {rho_s} kg/m^3\n")
        f.write(f"Equivalent Waiwera Atmospheric Pressure (P_atm): {p_atm_waiwera} Pa\n")
        f.write(f"Simulation Time (T): {T} d  |  dt = {dt:.6f} d  |  n_steps = {step_idx}\n")
        if auto_terminate:
            f.write(f"Auto-terminate: ON (ss_tol={ss_tol:.2e}, T_auto={T_auto:.0f} d)\n")
        else:
            f.write(f"Auto-terminate: OFF (ss_tol={ss_tol:.2e}; metrics computed but not enforced)\n")
        f.write("\n")
        f.write("--- Boundary Conditions ---\n")
        f.write("Left   (x=0):   Symmetry – no flow, no salt flux\n")
        f.write("Right  (x=500): Hydrostatic Dirichlet pressure + ambient salt\n")
        f.write("Top    (z=50):  Rainfall recharge (Q_in m/yr freshwater)\n")
        f.write("Bottom (z=0):   No flow\n\n")
        f.write("--- Final Model Statistics ---\n")
        f.write(f"Total Steps Completed: {step_idx}\n")
        f.write(f"Simulation End Time: {t:.2f} d  ({t/365.25:.3f} yr)\n")
        f.write(f"Steady State Reached: {'YES' if ss_reached else 'NO'}\n")
        if ss_reached:
            f.write(f"  Steady state at step {ss_step}, t = {ss_time} d\n")
        f.write(f"x_50 (50% isoline on bottom boundary): {x_bottom:.6f} m\n")
        f.write(f"Depth_50 (50% isoline on left boundary): {Ly - depth_50:.4f} m\n")
        f.write(f"Mixing zone width (depth_10 - depth_50 on left boundary): {mixing_zone_width:.4f} m\n")
        f.write(f"  depth_10: {Ly - depth_10:.4f} m\n")
        f.write(f"  depth_50: {Ly - depth_50:.4f} m\n")
        f.write(f"Total salt mass (phi * rho_s * c integrated): {total_salt_mass:.4f} kg/m\n")
        f.write(f"Percentage of cells with 10% to 90% seawater: {pct_cells:.2f}%\n")
        f.write(f"Simulation Time Elapsed: {elapsed_time:.2f} seconds\n")

    print(f"Simulation complete. Summary saved to: {summary_filename}")
    print(f"Simulation took {elapsed_time:.2f} s ({elapsed_time/60:.2f} min)")

    return {
        "sim_name":              sim_name,
        "D":                     D,
        "depth_50":              Ly - depth_50,
        "depth_10":              Ly - depth_10,
        "x_50":                  x_bottom,
        "mixing_zone_width":     mixing_zone_width,
        "total_salt_mass":       total_salt_mass,
        "pct_cells_mixing_zone": pct_cells,
        "ss_reached":            ss_reached,
        "ss_step":               ss_step,
        "ss_time":               ss_time,
        "elapsed_time":          elapsed_time,
    }

# ==============================
# CLI Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an island freshwater lens simulation (double-layer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--D",              type=float, default=0.57024,
                        help="Diffusion coefficient (m^2/d)")
    parser.add_argument("--T",              type=float, default=7305.0,
                        help="Simulation end time (days)")
    parser.add_argument("--dt",             type=float, default=0.1,
                        help="Time step size (days)")
    parser.add_argument("--Nx",             type=int,   default=100,
                        help="Mesh subdivisions in x")
    parser.add_argument("--Ny",             type=int,   default=50,
                        help="Mesh subdivisions in y")
    parser.add_argument("--K_upper",        type=float, default=25.0,
                        help="Hydraulic conductivity in the upper layer (m/d)")
    parser.add_argument("--K_lower",        type=float, default=500.0,
                        help="Hydraulic conductivity in the lower layer (m/d)")
    parser.add_argument("--phi_upper",      type=float, default=0.30,
                        help="Porosity in the upper layer")
    parser.add_argument("--phi_lower",      type=float, default=0.20,
                        help="Porosity in the lower layer")
    parser.add_argument("--thurber_depth",  type=float, default=12.0,
                        help="Depth of Thurber discontinuity below surface (m)")
    parser.add_argument("--q_recharge",     type=float, default=2.0/365.25,
                        help="Recharge rate (m/d)")
    parser.add_argument("--auto_terminate", action="store_true",
                        help="Enable steady-state auto-termination")
    parser.add_argument("--ss_tol",         type=float, default=1e-4,
                        help="Relative salt-mass change tolerance for auto-termination")
    parser.add_argument("--output_frequency", type=int, default=100,
                        help="Write XDMF output every n steps (first/final always exported)")
    parser.add_argument("--notes",          type=str,   default="",
                        help="Free-text note for the summary file")
    parser.add_argument("--name",          type=str,   default="",
                        help="Simulation name (optional).")
    args = parser.parse_args()

    run_island_double(
        D=args.D,
        T=args.T,
        dt=args.dt,
        Nx=args.Nx,
        Ny=args.Ny,
        K_upper=args.K_upper,
        K_lower=args.K_lower,
        phi_upper=args.phi_upper,
        phi_lower=args.phi_lower,
        thurber_depth=args.thurber_depth,
        q_recharge=args.q_recharge,
        auto_terminate=args.auto_terminate,
        ss_tol=args.ss_tol,
        output_frequency=args.output_frequency,
        notes=args.notes,
        name=args.name
    )