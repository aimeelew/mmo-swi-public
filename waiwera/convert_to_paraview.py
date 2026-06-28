import h5py
import meshio
import numpy as np
import os
import argparse

def export_to_paraview(sim_name, output_dir):
    h5_file = os.path.join(output_dir, f"{sim_name}_output.h5")
    mesh_file = os.path.join(output_dir, f"{sim_name}_mesh.exo")
    out_prefix = f"{sim_name}_paraview"

    print(f"Reading mesh from '{mesh_file}'...")
    try:
        mesh = meshio.read(mesh_file)
    except FileNotFoundError:
        print(f"Error: Mesh file '{mesh_file}' not found.")
        return
        
    print(f"Reading Waiwera output from '{h5_file}'...")
    try:
        f = h5py.File(h5_file, "r")
    except FileNotFoundError:
        print(f"Error: Output file '{h5_file}' not found. Did you run Waiwera?")
        return

    # --- SAFETY CHECK ---
    if "time" not in f:
        print("Error: No 'time' dataset found in the HDF5 file.")
        print("   This means the Waiwera simulation crashed before writing any results.")
        print("   Please check your terminal output from when you ran 'waiwera henry_waiwera.json' for errors.")
        f.close()
        return
    # --------------------

    # Extract time array and cell mapping
    times = f["time"][:].flatten() # Flattens the 2D column into a simple 1D list
    cell_index = f["cell_index"][:].flatten() # .flatten() ensures 1D shape to prevent indexing errors
    num_cells = len(cell_index)
    
    # Identify the time-varying thermodynamic fields (ignore static geometry)
    fields_to_extract = []
    for field in f["cell_fields"].keys():
        dataset = f["cell_fields"][field]
        # Check if the shape matches (num_timesteps, num_cells)
        if len(dataset.shape) == 2 and dataset.shape[0] == len(times) and dataset.shape[1] == num_cells:
            fields_to_extract.append(field)
            
    print(f"Exporting time-varying fields: {fields_to_extract}")
    print(f"Exporting {len(times)} timesteps...")

    pvd_content = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>'
    ]

    # Create a subfolder to prevent cluttering the root output directory
    vtu_dir = os.path.join(output_dir, "vtu")
    os.makedirs(vtu_dir, exist_ok=True)

    # Process each timestep
    for i, t in enumerate(times):
        step_cell_data = {}
        
        for field in fields_to_extract:
            data_at_t = f["cell_fields"][field][i, :]
            
            # Map the data back to the original Exodus mesh order (natural ordering)
            # data_at_t is in global (MPI partitioned) order. cell_index maps natural -> global.
            ordered_data = data_at_t[cell_index]
            step_cell_data[field] = [ordered_data]
            
        mesh.cell_data = step_cell_data
        
        vtu_filename = f"{out_prefix}_{i:04d}.vtu"
        mesh.write(os.path.join(vtu_dir, vtu_filename))
        
        pvd_content.append(f'    <DataSet timestep="{t}" group="" part="0" file="vtu/{vtu_filename}"/>')
        
        if i % 10 == 0 or i == len(times) - 1:
            print(f"  - Processed step {i+1}/{len(times)} (t = {t:.1f} s)")

    pvd_content.append('  </Collection>')
    pvd_content.append('</VTKFile>')

    pvd_path = os.path.join(output_dir, f"{out_prefix}.pvd")
    with open(pvd_path, "w") as pvd_file:
        pvd_file.write("\n".join(pvd_content))

    f.close()
    print(f"Export complete! Open '{pvd_path}' in ParaView to view the time series.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()
    
    export_to_paraview(args.sim_name, args.output_folder)