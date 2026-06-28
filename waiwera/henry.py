import json
import layermesh.mesh as lm
import os
import argparse
import sys

# =========================================================
# This script is responsible for constructing the environment and 
# compiling the execution parameters into a Waiwera-compatible JSON file.
#
# - Mesh Generation:
#      Uses the `layermesh` library to build a unified rectangular hex grid. 
#      Although acting as a 2D slice, it is generated as a `[80, 1, 40]` 3D 
#      mesh ($Lx=2.0$, $Lz=1.0$).
# - Initialization:
#      Computes precise hydrostatic pressures relative to the absolute $z=0$ 
#      top-plane geometry (depth $=-z$).
# - Boundary Conditions:
#      Calculates the heavier ocean hydrostatic forces applied as Dirichlet 
#      boundaries to the active seaward `faces` and calculates constant 
#      fresh-water mass injection rates uniformly applied along the landward boundary.
# - Physical Output:
#      Generates `<sim_name>_mesh.exo` and `<sim_name>.json`.
# =========================================================
#

def generate_waiwera_models(sim_name, output_folder, overrides=None):
    os.makedirs(output_folder, exist_ok=True)
    if overrides is None:
        overrides = {}
    elif isinstance(overrides, str):
        overrides = json.loads(overrides)
    # =========================================================
    # 1. Physical Parameters
    # =========================================================
    Lx, Ly, Lz = 2.0, 1.0, 1.0
    Nx = overrides.get('Nx', 80)
    Ny = overrides.get('Ny', 1)
    Nz = overrides.get('Nz', 40)

    T_days = overrides.get('T_days', 0.5)
    T_sec = T_days * 24.0 * 3600.0

    mu = 1e-3           # dymamic viscosity of water in Pa.s
    rho_f = 1000.0      # density freshwater in kg/m^3
    rho_s = 1024.5      # density saltwater in kg/m^3
    c_salt = 35.0 / rho_s # salt mass fraction

    p_atm = 1.01325e5   # atmospheric pressure in Pa
    g = 9.81            # gravity in m/s^2

    # Default parameters (Scenario A)
    K = 864.0 / (24.0 * 3600.0)     # hydraulic conductivity in m/s (converted from 864 m/day)
    phi = 0.35                      # porosity
    Q_in = 5.7024                   # source flux 
                                    # (Scenario A uses 5.7024 m3/day, Scenario B uses 2.851 m3/day)
    
    # Apply overrides
    K = overrides.get('K', K)
    phi = overrides.get('phi', phi)
    Q_in = overrides.get('Q_in', Q_in)

    k = K * mu / (rho_f * g)        # hydraulic permeability
    Q = Q_in / (24.0 * 3600.0)      # m/s
    
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
    # 3. Build IC + Boundary Lists
    # =========================================================
    initial_primary = []
    ocean_boundaries = []
    inflow_faces = []

    for c in mesh.cell:
        x, y, z = c.centroid
        depth = -z  # Top surface is at z=0, so depth is negative z

        # Uniform initial guess (freshwater everywhere)
        p = p_atm + rho_f * g * depth
        initial_primary.append([p, 20.0, 0.0])

        # Right boundary (ocean)
        if x > Lx - dx[-1]:
            p_ocean = p_atm + rho_s * g * depth
            ocean_boundaries.append({
                "primary": [p_ocean, 20.0, c_salt],
                "region": 1,
                "faces": [{"cells": [c.index], "normal": [1.0, 0.0, 0.0]}]
            })

        # Left boundary (inflow face)
        if x < dx[0]:
            inflow_faces.append(c.index)

    # =========================================================
    # 4. JSON Model Setup
    # =========================================================
    model = {
        "title": "Henry Transient Problem",
        "notes": overrides.get("notes", ""),
        "mesh": {"filename": sim_name + "_mesh.exo"},
        "eos": {"name": "wse"},
        "gravity": g,
        "rock": {
            "types": [{
                "name": "sand",
                "porosity": phi,
                "permeability": k,
                "density": 2650.0,
                "cells": list(range(len(mesh.cell)))
            }]
        },
        # CORRECTED: Direct initialization with the freshwater hydrostatic array
        "initial": {"primary": initial_primary},
        "boundaries": ocean_boundaries,
        "source": [{
            "rate": Q * rho_f / len(inflow_faces),
            "enthalpy": 84e3,
            "component": "water",
            "cells": inflow_faces
        }],
        "time": {
            "start": 0,
            "stop": T_sec,
            "step": {
                "size": 8.64,
                "adapt": {
                    "on": True
                },
                "maximum": {
                    "size": 1000,  
                    "number": 5000   
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

    print("Henry setup generated")
    print(f"   - Assigned {len(inflow_faces)} cell boundaries to freshwater inflow.")
    print(f"   - Assigned {len(ocean_boundaries)} cell boundaries for the hydrostatic ocean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", required=True)
    parser.add_argument("--overrides", default=None, help="JSON string of parameter overrides")
    args = parser.parse_args()

    # Pass sim_name for both the file prefix and the output folder
    generate_waiwera_models(args.sim_name, args.sim_name, args.overrides)