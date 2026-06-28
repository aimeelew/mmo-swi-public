import json
import layermesh.mesh as lm
import os
import argparse

# =========================================================
# This script models an island freshwater lens, adapting the pipeline 
# for a 2D vertical half-island cross-section ($Lx=500.0, Lz=50.0$).
#
# - Mesh Generation:
#      Uses a larger 500x50m domain.
# - Initialization & Boundary Conditions:
#      Initialized as a fully saline  
#      aquifer. Freshwater enters uniformly from the top boundary as rainfall 
#      recharge (2 m/year). The right and bottom boundaries are set to ocean 
#      hydrostatic seawater pressure.
# - Geology:
#      Models two distinct rock layers separated by the Thurber 
#      discontinuity at a 12m depth: the upper Holocene aquifer and the lower 
#      Pleistocene aquifer.
# =========================================================

def generate_waiwera_models(sim_name, output_folder, overrides=None):
    os.makedirs(output_folder, exist_ok=True)
    if overrides is None:
        overrides = {}
    elif isinstance(overrides, str):
        overrides = json.loads(overrides)
    # =========================================================
    # 1. Physical Parameters
    # =========================================================
    Lx, Ly, Lz = 500.0, 1.0, 50.0
    Nx = overrides.get('Nx', 100)
    Ny = overrides.get('Ny', 1)
    Nz = overrides.get('Nz', 50)

    T_years = overrides.get('T_years', 10.0)
    T_sec = T_years * 365.25 * 24.0 * 3600.0
    dt_init = overrides.get('dt_init', 86400)  # default initial time step in seconds

    mu = 1e-3           # dynamic viscosity of water in Pa.s
    rho_f = 1000.0      # density freshwater in kg/m^3
    rho_s = 1024.5      # density saltwater in kg/m^3
    c_salt = 35.0 / rho_s  # salt mass fraction

    p_atm = 1.01325e5     # atmospheric pressure in Pa
    g = 9.81            # gravity in m/s^2

    # ---- Thurber discontinuity depth (m below surface) ----
    thurber_depth = overrides.get('thurber_depth', 12.0)  # depth of interface between upper and lower layers

    # Upper Layer: Holocene Aquifer (0 to 12 m depth)
    k_upper  = 2.95e-11   # permeability equiv. to K=25 m/day
    phi_upper = 0.30      # porosity

    # Lower Layer: Pleistocene Aquifer (12 to 50 m depth)
    k_lower  = 5.90e-10   # permeability equiv. to K=500 m/day
    phi_lower = 0.20      # porosity

    # Recharge
    Q_in = 2.0           # uniform freshwater recharge in m/year applied to every top-surface cell

    # Apply overrides
    k_upper = overrides.get('k_upper', k_upper)
    phi_upper = overrides.get('phi_upper', phi_upper)
    k_lower = overrides.get('k_lower', k_lower)
    phi_lower = overrides.get('phi_lower', phi_lower)
    Q_in = overrides.get('Q_in', Q_in)

    recharge_m_per_s = Q_in / (365.25 * 24.0 * 3600.0)

    # =========================================================
    # 2. Mesh
    # =========================================================
    dx = [Lx / Nx] * Nx
    dy = [Ly / Ny] * Ny
    dz = [Lz / Nz] * Nz

    mesh = lm.mesh(rectangular=(dx, dy, dz))
    mesh.write(os.path.join(output_folder, sim_name + "_mesh.h5"))
    mesh.export(os.path.join(output_folder, sim_name + "_mesh.exo"))

    # =========================================================
    # 3. Build IC + Boundary Lists + Layer Cell Classification
    # =========================================================
    initial_primary   = []
    ocean_boundaries  = []
    bottom_boundaries = []
    top_cells         = []
    upper_cells       = []   # Holocene layer (depth < thurber_depth)
    lower_cells       = []   # Pleistocene layer (depth >= thurber_depth)

    for c in mesh.cell:
        x, y, z = c.centroid
        depth = -z  # z=0 is top surface; depth increases downward

        # --- Initial condition: fully saline aquifer ---
        p = p_atm + rho_s * g * depth
        initial_primary.append([p, 20.0, c_salt])

        # --- Layer classification by Thurber discontinuity ---
        if depth < thurber_depth:
            upper_cells.append(c.index)
        else:
            lower_cells.append(c.index)

        # --- Right boundary: ocean (hydrostatic seawater, full salinity) ---
        if x > Lx - dx[-1]:
            p_ocean = p_atm + rho_s * g * depth
            ocean_boundaries.append({
                "primary": [p_ocean, 20.0, c_salt],
                "region": 1,
                "faces": [{"cells": [c.index], "normal": [1.0, 0.0, 0.0]}]
            })

        # --- Bottom boundary: hydrostatic seawater (deep saline support) ---
        if depth > Lz - dz[-1]:
            p_bottom = p_atm + rho_s * g * depth
            bottom_boundaries.append({
                "primary": [p_bottom, 20.0, c_salt],
                "region": 1,
                "faces": [{"cells": [c.index], "normal": [0.0, 0.0, -1.0]}]
            })

        # --- Top boundary cells: rainfall recharge ---
        if depth < dz[0]:
            top_cells.append(c.index)

    # =========================================================
    # 4. Recharge source rate
    # =========================================================
    # Volumetric recharge flux per cell = recharge velocity * cell face area (dx * dy)
    # This distributes the 2 m/year rainfall uniformly over all top-surface cells.
    cell_face_area = dx[0] * dy[0]                            # m^2 per top cell
    recharge_rate_per_cell = recharge_m_per_s * cell_face_area * rho_f  # kg/s per cell

    # =========================================================
    # 5. JSON Model Setup
    # =========================================================
    model = {
        "title": "Island Freshwater Lens (Half-Island, Thurber Two-Layer)",
        "notes": overrides.get("notes", ""),
        "mesh": {"filename": sim_name + "_mesh.exo"},
        "logfile": False,
        "eos": {"name": "wse"},
        "gravity": g,
        "params": {
            "T_years": T_years,
            "T_sec": T_sec,
            "k_upper": k_upper,
            "phi_upper": phi_upper,
            "k_lower": k_lower,
            "phi_lower": phi_lower,
            "Q_in": Q_in,
            "thurber_depth": thurber_depth
        },
        "rock": {
            "types": [
                {
                    # Upper Holocene aquifer (0–12 m depth)
                    "name": "holocene",
                    "porosity": phi_upper,
                    "permeability": k_upper,
                    "density": 2650.0,
                    "cells": upper_cells
                },
                {
                    # Lower Pleistocene aquifer (12–50 m depth)
                    "name": "pleistocene",
                    "porosity": phi_lower,
                    "permeability": k_lower,
                    "density": 2650.0,
                    "cells": lower_cells
                }
            ]
        },
        # Initial condition: fully saline aquifer (hydrostatic seawater pressure)
        "initial": {"primary": initial_primary},
        # Boundary conditions: ocean face (right)
        # Left boundary (symmetry plane) and bottom are implicitly no-flow by default.
        "boundaries": ocean_boundaries,
        # Freshwater recharge uniformly applied to the top surface
        "source": [{
                    "rate": recharge_rate_per_cell,
                    "enthalpy": 84e3,
                    "component": "water",
                    "cells": top_cells
                }],
        "time": {
            "start": 0,
            "stop": T_sec,
            "step": {
                "size": dt_init,
                "adapt": {
                    "on": True
                },
                "maximum": {
                    "size": 86400 * 90,   # max 90-day timestep
                    "number": 500000
                }
            },
        },
        "output": {
            "fields": {"fluid": ["pressure", "liquid_salt_mass_fraction"]},
            "frequency": 0,  # 0 means only write the starting state and final state
            "filename": sim_name + "_output.h5",
        }
    }

    json_path = os.path.join(output_folder, f"{sim_name}.json")
    with open(json_path, "w") as f:
        json.dump(model, f, indent=4)

    print("SUCCESS: Island Lens setup generated")
    print(f"   - Upper (Holocene) layer:    {len(upper_cells)} cells  (k={k_upper:.2e} m², phi={phi_upper})")
    print(f"   - Lower (Pleistocene) layer: {len(lower_cells)} cells  (k={k_lower:.2e} m², phi={phi_lower})")
    print(f"   - Top recharge cells:        {len(top_cells)}  cells  ({Q_in} m/yr, {recharge_rate_per_cell:.4e} kg/s per cell)")
    print(f"   - Right ocean BCs:           {len(ocean_boundaries)} faces")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", required=True)
    parser.add_argument("--overrides", default=None, help="JSON string of parameter overrides")
    args = parser.parse_args()

    # Pass sim_name for both the file prefix and the output folder
    generate_waiwera_models(args.sim_name, args.sim_name, args.overrides)