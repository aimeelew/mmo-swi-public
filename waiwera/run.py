"""
Waiwera Saltwater Intrusion Simulation Pipeline

Usage:
------
1. Single Simulation Run:
   python run.py --model henry
   python run.py --model island-single
   python run.py --model island-double

2. Queue Batch Execution via JSON Config:
   python run.py --config queue.json

Configuration File Structure (queue.json):
-----------------------------------------
The configuration file should contain a JSON list of run configurations. 
For example:
[
    {
        "model": "henry",
        "sim_name": "custom_henry_run",
        "overrides": {
            "Q_in": 2.851,
            "phi": 0.35
        }
    },
    {
        "model": "island-single",
        "sim_name": "lens_single_run",
        "overrides": {
            "Q_in": 1.5,
            "phi_rock": 0.20
        }
    }
]

Supported Models and Parameter Overrides:
----------------------------------------

1. Henry Problem ('henry')
   Simulates classic 2D Henry saltwater intrusion problem.
   Overrides:
   - K: Hydraulic conductivity in m/s (default: 864.0 / 86400.0 = 0.01)
   - phi: System porosity (default: 0.35)
   - Q_in: Freshwater source flux in m3/day (default: 5.7024, Scenario B is 2.851)
   - Nx: Grid subdivisions in X direction (default: 80)
   - Ny: Grid subdivisions in Y direction (default: 1)
   - Nz: Grid subdivisions in Z direction (default: 40)
   - T_days: Total simulation duration in days (default: 0.5)

2. Island Single Layer ('island-single')
   Simulates freshwater lens inside a single rock layer.
   Overrides:
   - k_rock: Rock permeability in m2 (default: 2.95e-11 (Holocene), Pleistocene is 5.90e-10)
   - phi_rock: Rock porosity (default: 0.30 (Holocene), Pleistocene is 0.20)
   - Q_in: Uniform freshwater surface recharge in m/year (default: 2.0)
   - Nx: Grid subdivisions in X direction (default: 100)
   - Ny: Grid subdivisions in Y direction (default: 1)
   - Nz: Grid subdivisions in Z direction (default: 50)
   - T_days: Total simulation duration in days (default: 7305.0 or 20 years)

3. Island Double Layer ('island-double')
   Simulates freshwater lens in two layers separated by the Thurber discontinuity.
   Overrides:
   - k_upper: Upper layer (Holocene) rock permeability in m2 (default: 2.95e-11)
   - phi_upper: Upper layer (Holocene) rock porosity (default: 0.30)
   - k_lower: Lower layer (Pleistocene) rock permeability in m2 (default: 5.90e-10)
   - phi_lower: Lower layer (Pleistocene) rock porosity (default: 0.20)
   - Q_in: Uniform freshwater surface recharge in m/year (default: 2.0)
   - Nx: Grid subdivisions in X direction (default: 100)
   - Ny: Grid subdivisions in Y direction (default: 1)
   - Nz: Grid subdivisions in Z direction (default: 50)
   - T_days: Total simulation duration in days (default: 7305.0 or 20 years)

Pipeline Flags:
---------------
--config: Path to a JSON list of configurations (runs sequentially).
--model: Model type to run if --config is omitted ('henry', 'island-single', 'island-double').
--skip-sim: Skip the actual Docker simulation. Useful for debugging mesh or script generation.
--skip-paraview: Skip converting Exodus/HDF5 data to ParaView format (.vtu/.pvd).
--cores: Parallel execution core count (default: 8).
--paraview-frames: Number of evenly spaced checkpoint frames saved (default: 100).
--open-paraview: Automatically launch ParaView with the generated PVD file (Windows only).
"""

import os
import subprocess
import argparse
import sys
import time
import json
from datetime import datetime
import h5py
import numpy as np
from postprocess import analyze_results

def extract_npz(sim_name, output_folder, percentages):
    h5_path = os.path.join(output_folder, f"{sim_name}_output.h5")
    
    if not os.path.exists(h5_path):
        print(f"Could not find {h5_path}")
        return
        
    try:
        with h5py.File(h5_path, 'r') as f:
            t_array = f['time'][:].flatten()
            total_time = t_array[-1]
            
            # Use cell_index to map parallel output back to natural mesh order
            if 'cell_index' in f:
                cell_index = f['cell_index'][:].flatten()
            else:
                cell_index = slice(None)
            
            # Dynamically infer dimensions from centroids (in natural order)
            c_global = f['cell_fields']['cell_geometry_centroid'][()]
            c = c_global[cell_index]
            Nx = len(np.unique(np.round(c[:, 0], 5)))
            Nz = len(np.unique(np.round(c[:, 2], 5)))
            
            for pct in percentages:
                target_t = pct * total_time
                # Find closest index
                idx = (np.abs(t_array - target_t)).argmin()
                
                # Extract and map to natural order
                p = f['cell_fields']['fluid_pressure'][idx][cell_index]
                s = f['cell_fields']['fluid_liquid_salt_mass_fraction'][idx][cell_index]
                
                p_2d = p.reshape((Nz, Nx))
                s_2d = s.reshape((Nz, Nx))
                
                pct_str = f"{pct*100:g}pct"
                npz_press_path = os.path.join(output_folder, f"{sim_name}_{pct_str}_pressure.npz")
                npz_salt_path = os.path.join(output_folder, f"{sim_name}_{pct_str}_salt_mass_fraction.npz")
                
                np.savez_compressed(npz_press_path, pressure=p_2d)
                np.savez_compressed(npz_salt_path, salt_mass_fraction=s_2d)
                print(f"Extracted data at t={t_array[idx]:.2f}s ({pct_str}) -> {os.path.basename(npz_press_path)}")
    except Exception as e:
        print(f"Failed to extract NPZ: {e}")

def run_pipeline():
    parser = argparse.ArgumentParser(description="End-to-End Execution for Waiwera Saltwater Intrusion Simulations")
    # Model selection
    parser.add_argument("--model", choices=["henry", "island-double", "island-single"], default="henry",
                        help="Model to run: 'henry' (Henry problem), 'island-double' (Island double layer), or 'island-single' (Island single layer)")
    # Customization options
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON file containing a list of run configurations")
    parser.add_argument("--skip-sim", action="store_true", help="Skip the Docker Waiwera simulation step")
    parser.add_argument("--skip-paraview", action="store_true", help="Skip ParaView export step")
    parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores for parallel execution")
    parser.add_argument("--paraview-frames", type=int, default=100, help="Number of evenly spaced frames for ParaView animation")
    parser.add_argument("--open-paraview", action="store_true", default=True, help="Automatically open the .pvd file in ParaView")
    
    # Optional override if you need to re-run a specific existing timestamp
    parser.add_argument("--timestamp", default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 
                        help="Timestamp (defaults to current time: YYYY-MM-DD_HH-MM-SS)")
    
    args = parser.parse_args()
    
    # =========================================================
    # Interactive Prompt
    # (Only triggers if NO arguments were passed via the command line)
    # =========================================================
    if len(sys.argv) == 1:
        print("No CLI arguments detected.")
        use_default = input("Do you want to use the default configuration? (Y/n): ").strip().lower()
        
        if use_default == 'n':
            print("\n--- Custom Configuration ---")
            
            # Model prompt
            model_choice = input(f"Enter model to run (henry/island-double/island-single) [default: {args.model}]: ").strip().lower()
            if model_choice in ['henry', 'island-double', 'island-single']:
                args.model = model_choice
            
            # Skip simulation prompt
            skip_s = input("Skip Docker simulation? (y/N) [default: N]: ").strip().lower()
            if skip_s == 'y':
                args.skip_sim = True
            
            # Skip ParaView prompt
            skip_p = input("Skip ParaView export? (y/N) [default: N]: ").strip().lower()
            if skip_p == 'y':
                args.skip_paraview = True
                
            # ParaView Frames prompt
            if not args.skip_paraview:
                pv_in = input(f"Enter number of evenly spaced frames for ParaView animation [default: {args.paraview_frames}]: ").strip()
                if pv_in.isdigit():
                    args.paraview_frames = int(pv_in)
                    
            # Cores prompt
            cores_in = input(f"Enter number of cores for parallel execution [default: {args.cores}]: ").strip()
            if cores_in.isdigit():
                args.cores = int(cores_in)
            
            # Open ParaView prompt (only ask if we are actually generating the file)
            if not args.skip_paraview:
                open_p = input("Automatically open .pvd in ParaView when finished? (y/N) [default: y]: ").strip().lower()
                if open_p == 'y':
                    args.open_paraview = True
                elif open_p == 'n':
                    args.open_paraview = False
            
                    
            print("----------------------------\n")

    # Resolve the directory that run.py lives in, so relative script paths
    # (island.py, henry.py, convert_to_paraview.py) work regardless of launch CWD.
    HERE = os.path.dirname(os.path.abspath(__file__))

    configs = []
    if args.config:
        # Resolve config path relative to the script directory so users can pass
        # just the filename from any CWD. This ensures the JSON's `sim_name`
        # entries are actually loaded instead of falling back to timestamped names.
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(HERE, config_path)

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configs = json.load(f)
        else:
            print(f"Warning: config file '{args.config}' not found (checked {config_path}). Using default config.")

    if not configs:
        configs = [{
            "model": args.model,
            "overrides": {}
        }]
    
    # Timekeeping: record wall-clock start and per-step durations
    pipeline_start = time.perf_counter()
    wall_start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for idx, config in enumerate(configs):
        model_name = config.get("model", "henry")
        overrides = config.get("overrides", {})
        config_sim_name = config.get("sim_name") or overrides.get("sim_name")

        if config_sim_name:
            sim_name = config_sim_name
        elif len(configs) > 1:
            sim_name = f"{args.timestamp}_{model_name}_{idx}"
        else:
            sim_name = f"{args.timestamp}_{model_name}"
        
        output_folder = os.path.join(HERE, sim_name)
        step_times = {}

        print(f"\n{'='*50}")
        print(f"Starting Pipeline: {sim_name} ({idx+1}/{len(configs)})")
        print(f"{'='*50}")
        
        # --- STEP 1: Generate JSON & Mesh ---
        print("\n[1/5] Generating JSON & Mesh...")
        _t0 = time.perf_counter()
        model_script = os.path.join(HERE, f"{model_name}.py")
        gen_cmd = [sys.executable, model_script, "--sim_name", sim_name]
        if overrides:
            gen_cmd.extend(["--overrides", json.dumps(overrides)])
        subprocess.run(gen_cmd, check=True, cwd=HERE)
        step_times["[1/5] JSON & Mesh generation"] = time.perf_counter() - _t0
        
        # --- STEP 1.5: Optimize Output Control and Checkpoints ---
        json_path = os.path.join(output_folder, f"{sim_name}.json")
        if os.path.exists(json_path) and not args.skip_sim:
            with open(json_path, 'r') as f:
                model_data = json.load(f)
            
            if "output" not in model_data:
                model_data["output"] = {}
            
            if args.skip_paraview:
                # Instead of deleting checkpoints completely, inject a lightweight set 
                # of checkpoints (e.g., 100 frames) strictly for steady-state analysis.
                time_stop = model_data.get("time", {}).get("stop", 0)
                if time_stop > 0:
                    analysis_times = np.linspace(0, time_stop, 100) # 100 frames is plenty
                    checkpoints = sorted(list(set([t for t in analysis_times if 0.0 < t < time_stop])))
                    model_data["output"]["checkpoint"] = {"time": checkpoints}
                print("   [+] Injected 100 lightweight checkpoints for steady-state analysis.")
            else:
                # If we WANT ParaView, set up checkpoints based on user-specified frame count
                time_stop = model_data.get("time", {}).get("stop", 0)
                if time_stop > 0 and args.paraview_frames > 0:
                    pv_times = np.linspace(0, time_stop, args.paraview_frames)
                    checkpoints = sorted(list(set([t for t in pv_times if 0.0 < t < time_stop])))
                    if checkpoints:
                        model_data["output"]["checkpoint"] = {"time": checkpoints}
                    print(f"   [+] Set up {len(checkpoints)} paraview checkpoints from {args.paraview_frames} requested frames.")
            
            with open(json_path, 'w') as f:
                json.dump(model_data, f, indent=4)
        
        # --- STEP 2: Run Docker Simulation ---
        if not args.skip_sim:
            print("\n[2/5] Running Waiwera Simulation via Docker...")
            target_dir = os.path.abspath(output_folder)
            
            if args.cores > 1:
                docker_cmd = [
                    "docker", "run", "--rm", 
                    "-v", f"{target_dir}:/work", 
                    "-w", "/work", 
                    "waiwera/waiwera:latest", 
                    "mpiexec", "-np", str(args.cores), "waiwera", f"{sim_name}.json"
                ]
            else:
                docker_cmd = [
                    "docker", "run", "--rm", 
                    "-v", f"{target_dir}:/work", 
                    "-w", "/work", 
                    "waiwera/waiwera:latest", 
                    "waiwera", f"{sim_name}.json"
                ]
            
            print(f"Executing: {' '.join(docker_cmd)}")
            _t0 = time.perf_counter()
            try:
                subprocess.run(docker_cmd, check=True)
                step_times["[2/5] Waiwera simulation"] = time.perf_counter() - _t0
                print("Simulation complete.")
            except subprocess.CalledProcessError as e:
                step_times["[2/5] Waiwera simulation"] = time.perf_counter() - _t0
                print(f"Failed to run simulation: {e}")
                continue
        else:
            step_times["[2/5] Waiwera simulation"] = None
            print("\n[2/5] Skipping Simulation Step (--skip-sim passed)")

        # --- STEP 3: Export to ParaView ---
        if not args.skip_paraview:
            print("\n[3/5] Exporting to ParaView...")
            _t0 = time.perf_counter()
            subprocess.run([
                sys.executable, os.path.join(HERE, "convert_to_paraview.py"),
                "--sim_name", sim_name,
                "--output_folder", output_folder
            ], check=True, cwd=HERE)
            step_times["[3/5] ParaView export"] = time.perf_counter() - _t0
            
            if args.open_paraview:
                pvd_path = os.path.abspath(os.path.join(output_folder, f"{sim_name}_paraview.pvd"))
                print(f"\n[3/5] Opening '{pvd_path}'...")
                try:
                    os.startfile(pvd_path)
                except AttributeError:
                    print("Error: os.startfile is only available on Windows.")
                except Exception as e:
                    print(f"Could not automatically open ParaView: {e}")
        else:
            step_times["[3/5] ParaView export"] = None
            print("\n[3/5] Skipping ParaView Export (--skip-paraview passed)")
            
        # --- STEP 4: Extract NPZ Data ---
        print("\n[4/5] Extracting simulation data to .npz...")
        _t0 = time.perf_counter()
        extract_npz(sim_name, output_folder, [0.0, 1.0])
        step_times["[4/5] NPZ extraction"] = time.perf_counter() - _t0

        # --- STEP 5: Generate Summary Text File ---
        print("\n[5/5] Generating Summary Text File...")
        _t0 = time.perf_counter()
        analyze_results(sim_name, output_folder)
        step_times["[5/5] Summary generation"] = time.perf_counter() - _t0

        # Timing Summary for this run
        def _fmt(seconds):
            if seconds is None: return "  (skipped)"
            s = int(seconds)
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            if h > 0: return f"{h}h {m:02d}m {sec:02d}s"
            elif m > 0: return f"{m}m {sec:02d}s"
            else: return f"{sec}s ({seconds:.2f}s)"

        print(f"\n{'='*50}")
        print(f"Timing Summary - {sim_name}")
        print(f"{'='*50}")
        for step, elapsed in step_times.items():
            print(f"  {step:<34} {_fmt(elapsed):>10}")
        print(f"{'='*50}")

    total_elapsed = time.perf_counter() - pipeline_start
    wall_end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n{'='*50}")
    print(f"Pipeline Complete. Processed {len(configs)} configurations. Total elapsed time: {_fmt(total_elapsed)}")
    print(f"  Started:  {wall_start_str}")
    print(f"  Finished: {wall_end_str}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_pipeline()