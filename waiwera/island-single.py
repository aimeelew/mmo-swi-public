import json
import layermesh.mesh as lm
import os
import argparse

# ========================================================================  
# This script simulates a freshwater lens using a single, uniform rock layer.
# It was used mainly to test the island geometry and boundary conditions 
# before progressing to the dual-aquifer model.
#
# - Overrides Support:
#     Hardcoded scenario switching has been removed in favor of explicit 
#     parameter overrides. By default, it uses Holocene aquifer parameters 
#     (Scenario A), but it supports dynamic JSON overrides for rock 
#     permeability (`k_rock`), porosity (`phi_rock`), and surface recharge 
#     (`Q_in`).
# ========================================================================

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

    T_days = overrides.get('T_days', 20.0 * 365.25)
    T_sec = T_days * 24.0 * 3600.0

    mu = 1e-3           # dynamic viscosity of water in Pa.s
    rho_f = 1000.0      # density freshwater in kg/m^3
    rho_s = 1024.5      # density saltwater in kg/m^3
    c_salt = 35.0 / rho_s  # salt mass fraction

    p_atm = 1.013e5     # atmospheric pressure in Pa
    g = 9.81            # gravity in m/s^2

    # Default parameters (Holocene Aquifer / Scenario A)
    rock_name = "single_layer"
    k_rock  = 2.95e-11   # permeability equiv. to K=25 m/day (Pleistocene uses 5.90e-10)
    phi_rock = 0.30      # porosity (Pleistocene uses 0.20)
    Q_in = 2.0           # uniform freshwater recharge in m/year applied to every top-surface cell
    
    # Apply overrides
    k_rock = overrides.get('k_rock', k_rock)
    phi_rock = overrides.get('phi_rock', phi_rock)
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
    all_cells         = []   # Single rock layer

    for c in mesh.cell:
        x, y, z = c.centroid
        depth = -z  # z=0 is top surface; depth increases downward

        # --- Initial condition: fully saline aquifer ---
        p = p_atm + rho_s * g * depth
        initial_primary.append([p, 20.0, c_salt])

        # --- Layer classification ---
        all_cells.append(c.index)

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
        "title": f"Island Freshwater Lens (Half-Island, Single-Layer)",
        "notes": overrides.get("notes", ""),
        "mesh": {"filename": sim_name + "_mesh.exo"},
        "eos": {"name": "wse"},
        "gravity": g,
        "rock": {
            "types": [
                {
                    "name": rock_name,
                    "porosity": phi_rock,
                    "permeability": k_rock,
                    "density": 2650.0,
                    "cells": all_cells
                }
            ]
        },
        # Initial condition: fully saline aquifer (hydrostatic seawater pressure)
        "initial": {"primary": initial_primary},
        # Boundary conditions: ocean face (right) + deep saline support (bottom)
        # Left boundary (symmetry plane) is implicitly no-flow by default.
        "boundaries": ocean_boundaries + bottom_boundaries,
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
                "size": 86400,   # 1 day — appropriate starting scale for a decade-scale run
                "adapt": {
                    "on": True
                },
                "maximum": {
                    "size": 86400 * 90,   # max 90-day timestep
                    "number": 100000
                }
            },
        },
        "output": {
            "frequency": 1,
            "filename": sim_name + "_output.h5"
        }
    }

    json_path = os.path.join(output_folder, f"{sim_name}.json")
    with open(json_path, "w") as f:
        json.dump(model, f, indent=4)

    print(f"SUCCESS: Island Lens (Single-Layer) setup generated")
    print(f"   - Rock type:                 {rock_name}")
    print(f"   - Cells:                     {len(all_cells)} cells  (k={k_rock:.2e} m², phi={phi_rock})")
    print(f"   - Top recharge cells:        {len(top_cells)}  cells  ({Q_in} m/yr, {recharge_rate_per_cell:.4e} kg/s per cell)")
    print(f"   - Right ocean BCs:           {len(ocean_boundaries)} faces")
    print(f"   - Bottom ocean BCs:          {len(bottom_boundaries)} faces")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", required=True)
    parser.add_argument("--overrides", default=None, help="JSON string of parameter overrides")
    args = parser.parse_args()

    # Pass sim_name for both the file prefix and the output folder
    generate_waiwera_models(args.sim_name, args.sim_name, args.overrides)