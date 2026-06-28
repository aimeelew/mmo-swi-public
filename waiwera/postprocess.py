import os
import h5py
import numpy as np
import json
import sys

# ========================================================================
# Standalone module that calculates simulation statistics from the Waiwera 
# output.
#
# - Model-Agnostic Statistics:
#     Calculates standardized metrics at the final time step: 
#       (1) Toe Length (50% salt isoline position on bottom boundary),
#       (2) Lens Depth (50% salt isoline position on left boundary),
#       (3) Mixing Zone Width (10% to 90% isoline width), 
#       (4) Total salt mass (accounting for rock porosity), 
#       (5) Steady state convergence time (time at which total salt mass 
#           change drops below 0.01%).
# - Outputs:
#     Saves the output as a plain-text `<sim_name>_summary.txt` file inside the 
#     simulation directory.
# ========================================================================

def find_contour_x(x_coords, s_values, target):
    """
    Finds the exact X coordinate where s_values crosses target by linear interpolation.
    Assumes x_coords and s_values are ordered from left (freshwater) to right (saltwater).
    """
    for i in range(len(s_values) - 1):
        s1, s2 = s_values[i], s_values[i+1]
        x1, x2 = x_coords[i], x_coords[i+1]
        if (s1 <= target <= s2) or (s1 >= target >= s2):
            if s1 == s2:
                return x1
            return x1 + (target - s1) / (s2 - s1) * (x2 - x1)
    # Fallback to closest cell if no crossing found
    idx = np.abs(s_values - target).argmin()
    return x_coords[idx]

def analyze_results(sim_name, output_folder):
    h5_path = os.path.join(output_folder, f"{sim_name}_output.h5")
    json_path = os.path.join(output_folder, f"{sim_name}.json")
    
    if not os.path.exists(h5_path):
        print(f"Output file {h5_path} not found. Run simulation first.")
        return
        
    if not os.path.exists(json_path):
        print(f"Configuration file {json_path} not found.")
        return

    # Load JSON to extract porosity per cell
    with open(json_path, 'r') as jf:
        model_data = json.load(jf)
        
    # Standard constants
    rho_s = 1024.5
    rho_f = 1000.0
    c_salt = 35.0 / rho_s

    try:
        with h5py.File(h5_path, 'r') as f:
            t_array = f['time'][:].flatten()
            
            # Cell index to map global parallel to natural
            if 'cell_index' in f:
                cell_index = f['cell_index'][:].flatten()
            else:
                cell_index = slice(None)

            c_global = f['cell_fields']['cell_geometry_centroid'][()]
            c = c_global[cell_index]
            
            v_global = f['cell_fields']['cell_geometry_volume'][()]
            v_cells = v_global[cell_index]
            
            unique_x = np.sort(np.unique(np.round(c[:, 0], 5)))
            unique_z = np.sort(np.unique(np.round(c[:, 2], 5)))
            Nx = len(unique_x)
            Nz = len(unique_z)
            num_cells = Nx * Nz
            
            # Map porosity from rock types
            phi_array = np.zeros(num_cells)
            for rock in model_data.get('rock', {}).get('types', []):
                rock_porosity = rock.get('porosity', 0.0)
                rock_cells = rock.get('cells', [])
                if rock_cells:
                    phi_array[rock_cells] = rock_porosity
            
            # 1 & 2: Calculate left-boundary depth positions for 50% and 10% salinity
            s_final_global = f['cell_fields']['fluid_liquid_salt_mass_fraction'][-1]
            s_final = s_final_global[cell_index]
            
            # Build 2D mapping dictionary to robustly reconstruct layout without ordering assumptions
            x_to_ix = {float(np.round(val, 5)): idx for idx, val in enumerate(unique_x)}
            z_to_iz = {float(np.round(val, 5)): idx for idx, val in enumerate(unique_z)}
            
            s_final_2d = np.zeros((Nz, Nx))
            c_x_2d = np.zeros((Nz, Nx))
            c_z_2d = np.zeros((Nz, Nx))
            
            for j in range(len(s_final)):
                xj = float(np.round(c[j, 0], 5))
                zj = float(np.round(c[j, 2], 5))
                ix = x_to_ix[xj]
                iz = z_to_iz[zj]
                s_final_2d[iz, ix] = s_final[j]
                c_x_2d[iz, ix] = c[j, 0]
                c_z_2d[iz, ix] = c[j, 2]

            # Infer model run before calculating metrics
            title = model_data.get("title", "").lower()
            if "henry" in title or "henry" in sim_name:
                model_run = "henry"
            elif "single-layer" in title or "island-single" in sim_name:
                model_run = "island-single"
            elif "two-layer" in title or "island-double" in sim_name:
                model_run = "island-double"
            else:
                model_run = "unknown"

            # Targets
            target_50 = 0.5 * c_salt
            target_10 = 0.1 * c_salt

            # Left boundary values (closest column at minimum x)
            s_left = s_final_2d[:, 0]
            z_left = c_z_2d[:, 0]
            z_50_left = find_contour_x(z_left, s_left, target_50)
            z_10_left = find_contour_x(z_left, s_left, target_10)
            mixing_zone_width_left = abs(z_10_left - z_50_left)

            # Bottom boundary values (closest row at minimum z)
            s_bottom = s_final_2d[0, :]
            x_bottom = c_x_2d[0, :]
            x_50_bottom = find_contour_x(x_bottom, s_bottom, target_50)
            x_10_bottom = find_contour_x(x_bottom, s_bottom, target_10)
            x_90_bottom = find_contour_x(x_bottom, s_bottom, 0.9 * c_salt)
            mixing_zone_width_bottom = abs(x_90_bottom - x_10_bottom)

            # 3: Percentage of cells above 50% seawater concentration
            cells_over_50 = np.count_nonzero(s_final > target_50)
            percent_cells_over_50 = 100.0 * cells_over_50 / len(s_final) if len(s_final) > 0 else 0.0

            # 4: Percentage of cells between 10% and 90% seawater concentration (inclusive)
            cells_between_10_90 = np.count_nonzero((s_final >= target_10) & (s_final <= 0.9 * c_salt))
            percent_cells_between_10_90 = 100.0 * cells_between_10_90 / len(s_final) if len(s_final) > 0 else 0.0

            # 5: Total salt mass and Steady state time
            masses = []
            for i in range(len(t_array)):
                s_global = f['cell_fields']['fluid_liquid_salt_mass_fraction'][i]
                s = s_global[cell_index]
                
                # Approximate density
                rho = rho_f + (rho_s - rho_f) * (s / c_salt)
                mass = np.sum(s * phi_array * v_cells * rho)
                masses.append(mass)

            # Timescale-normalized relative change tolerance
            # Rel. change < 0.01% (1e-4) over a characteristic step (T_stop / 100)
            T_stop = t_array[-1]
            tol = 1e-2 / T_stop

            steady_state_time = None
            for i in range(len(t_array) - 1, 0, -1):
                dt = t_array[i] - t_array[i-1]
                if dt > 0:
                    denom = masses[i-1] if masses[i-1] > 0 else masses[-1]
                    if denom > 0:
                        step_rate = abs(masses[i] - masses[i-1]) / (denom * dt)
                    else:
                        step_rate = 0.0
                    if step_rate > tol:
                        # Steady state breached. Next step is when it stabilized.
                        if i < len(t_array) - 1:
                            steady_state_time = t_array[i]
                        break
            else:
                # Never breached, steady from start
                steady_state_time = t_array[0]

            total_salt_mass = masses[-1]

            # Extract simulation parameters for summary
            params = model_data.get("params", {})
            T_sec_param = params.get("T_sec", model_data.get("time", {}).get("stop"))
            T_days_param = params.get("T_days", T_sec_param / 86400.0 if T_sec_param is not None else None)
            k_upper = params.get("k_upper")
            phi_upper = params.get("phi_upper")
            k_lower = params.get("k_lower")
            phi_lower = params.get("phi_lower")
            recharge = params.get("Q_in")
            thurber_depth = params.get("thurber_depth")
            if k_upper is None or phi_upper is None or k_lower is None or phi_lower is None:
                for rock in model_data.get('rock', {}).get('types', []):
                    name = rock.get('name', '').lower()
                    if 'upper' in name or 'holocene' in name:
                        k_upper = k_upper if k_upper is not None else rock.get('permeability')
                        phi_upper = phi_upper if phi_upper is not None else rock.get('porosity')
                    elif 'lower' in name or 'pleistocene' in name:
                        k_lower = k_lower if k_lower is not None else rock.get('permeability')
                        phi_lower = phi_lower if phi_lower is not None else rock.get('porosity')

            # Calculate physical dimensions
            dx = (unique_x[-1] - unique_x[0]) / (Nx - 1) if Nx > 1 else 0.0
            dz = (unique_z[-1] - unique_z[0]) / (Nz - 1) if Nz > 1 else 0.0
            Lx = unique_x[-1] - unique_x[0] + dx
            Lz = unique_z[-1] - unique_z[0] + dz

            # Output to text file
            summary_path = os.path.join(output_folder, f"{sim_name}_summary.txt")
            with open(summary_path, 'w') as out:
                out.write("=========================================\n")
                out.write(f" Simulation Summary: {sim_name}\n")
                out.write("=========================================\n\n")
                out.write("PARAMETERS USED:\n")
                out.write(f"  Model run: {model_run}\n")
                out.write(f"  Grid physical dimensions (Lx, Lz): {Lx:.2f} m, {Lz:.2f} m\n")
                out.write(f"  Grid (Nx, Nz): {Nx}, {Nz}\n")
                out.write(f"  Simulation end time: {T_stop:.1f} s / {T_stop / 86400.0:.4f} days\n")
                out.write(f"  Average Porosity: {np.mean(phi_array):.3f}\n")
                
                # Write Hydraulic Conductivity for each rock type
                g_val = model_data.get('gravity', 9.81)
                for rock in model_data.get('rock', {}).get('types', []):
                    rock_name = rock.get('name', 'unknown')
                    perm_raw = rock.get('permeability', 0.0)
                    if isinstance(perm_raw, (int, float)):
                        k_val = perm_raw
                    elif isinstance(perm_raw, list) and len(perm_raw) > 0:
                        k_val = perm_raw[0]
                    else:
                        k_val = 0.0
                    K_val = k_val * rho_f * g_val / 1e-3
                    K_day = K_val * 86400.0
                    out.write(f"  Hydraulic conductivity ({rock_name}): {K_val:.5e} m/s ({K_day:.4f} m/day)\n")
                    
                out.write(f"  Seawater density: {rho_s} kg/m^3\n")
                out.write(f"  Max Salt mass fraction: {c_salt:.5f}\n")
                if T_days_param is not None:
                    out.write(f"  Total simulation time: {T_days_param:.2f} days / {T_sec_param:.1f} s\n")
                if k_upper is not None:
                    out.write(f"  Permeability upper: {k_upper:.4e} m^2\n")
                if phi_upper is not None:
                    out.write(f"  Phi upper: {phi_upper:.3f}\n")
                if k_lower is not None:
                    out.write(f"  Permeability lower: {k_lower:.4e} m^2\n")
                if phi_lower is not None:
                    out.write(f"  Phi lower: {phi_lower:.3f}\n")
                if recharge is not None:
                    out.write(f"  Recharge: {recharge:.4f} m/year\n")
                if thurber_depth is not None:
                    out.write(f"  Thurber depth: {thurber_depth:.4f} m\n")
                out.write("\n")
                user_notes = model_data.get("notes", "")
                if user_notes:
                    out.write(f"  Notes: {user_notes}\n")
                out.write("\n")
                out.write("FINAL STATISTICS:\n")
                out.write(f"  1. Depth at 50% salinity along left boundary: {z_50_left:.4f} m\n")
                out.write(f"  2. Distance at 50% salinity along bottom boundary: {x_50_bottom:.4f} m\n")
                out.write(f"  3. Percentage of cells above 50% seawater concentration: {percent_cells_over_50:.4f}%\n")
                out.write(f"  4. Percentage of cells between 10% and 90% seawater concentration: {percent_cells_between_10_90:.4f}%\n")
                out.write(f"  5. Total salt mass: {total_salt_mass:.4f} kg\n")
                if steady_state_time is not None:
                    out.write(f"  6. Steady state time (<0.01% mass change): {steady_state_time:.1f} s / {steady_state_time / 86400.0:.4f} days\n")
                else:
                    out.write(f"  6. Steady state time (<0.01% mass change): Not reached\n")
                    
            print(f"Summary saved to {summary_path}")

    except Exception as e:
        print(f"Failed to calculate statistics: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()
    analyze_results(args.sim_name, args.output_folder)
