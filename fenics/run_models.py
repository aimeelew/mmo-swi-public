import os
import argparse
import importlib
import time
os.environ["PKG_CONFIG_PATH"] = "/home/aimee/miniconda3/envs/fenicsproject/lib/pkgconfig"

# ==============================================================================
# Sweep Configuration
# ==============================================================================
#
# SWEEP_CONFIGS is a dictionary that defines which simulations to run and in
# what order. Each top-level key is a model name (e.g. "henry", "island_single",
# "island_double") and maps to a LIST of configuration dictionaries. Each dict
# in the list is one simulation run; the runner executes them sequentially and
# collects the results into a summary table and CSV at the end.
#
# HOW TO USE
# ----------
# Each configuration dict contains only the parameters you want to OVERRIDE
# from the function defaults. Any parameter not listed will use its default
# value as defined in the corresponding run_*() function. For example, if you
# only need to vary Q_in you can write just {"Q_in": 2.851} and every other
# parameter (D, K, phi, dt, ...) stays at its default.
#
# ==============================================================================
# Model Default Reference: henry
# ==============================================================================
#   "Nx":             80        # mesh cells in x (horizontal)
#   "Ny":             40        # mesh cells in y (vertical)
#   "T":              0.5       # simulation end time (days); ignored if auto_terminate=True
#   "dt":             0.001     # time step size (days)
#   "auto_terminate": False     # if True: run until salt-mass change < ss_tol (up to T_auto)
#   "ss_tol":         1e-4      # steady-state relative tolerance (only used when auto_terminate=True)
#   "D":              0.57024   # diffusion/dispersion coefficient (m^2/d)
#   "Q_in":           5.7024   # freshwater inflow rate (m^3/d per m width)
#                               #   standard Henry a: 5.7024,  low-inflow Henry b: 2.851
#   "K":              864.0     # hydraulic conductivity (m/d)
#   "phi":            0.35      # porosity (dimensionless)
#   "name":           None      # base folder name; None → auto "henry_Q{Q_in:.4g}"
#   "notes":          ""        # free-text note written to the summary file
#
# EXAMPLE — a simple Q_in sweep for the henry model:
#   "henry": [
#       {"Q_in": 5.7024, "notes": "Henry scenario a"},
#       {"Q_in": 4.0,    "notes": "Intermediate inflow"},
#       {"Q_in": 2.851,  "notes": "Henry scenario b"},
#   ]
# ==============================================================================
# Model Default Reference: island_single
# ==============================================================================
# Function: run_island_single() in island-single.py
#
#   "D":              0.57024   # dispersion coefficient (m^2/d)
#   "K":              25.0      # hydraulic conductivity (m/d)
#                               #   Holocene: 25.0,  Pleistocene: 500.0
#   "phi":            0.30      # porosity
#                               #   Holocene: 0.30,  Pleistocene: 0.20
#   "T":              3650.0    # simulation end time (days, ~10 years)
#   "dt":             0.365     # time step size (days, ~1 step/day)
#   "Nx":             100       # mesh cells in x
#   "Ny":             50        # mesh cells in y
#   "auto_terminate": False     # steady-state auto-termination
#   "ss_tol":         1e-4      # relative tolerance for auto-termination
#   "name":           None      # None → auto "island_single_K{K:.0f}"
#   "notes":          ""        # free-text note
#
# ==============================================================================
# Model Default Reference: island_double
# ==============================================================================
# Function: run_island_double() in island-double.py
# Layer properties are hardcoded (Thurber discontinuity at 12 m depth):
#   Upper (Holocene):    K = 25 m/d,  phi = 0.30
#   Lower (Pleistocene): K = 500 m/d,  phi = 0.20
#
#   "D":              0.57024   # diffusion coefficient (m^2/d)
#   "T":              3650.0    # simulation end time (days, ~10 years)
#   "dt":             0.365     # time step size (days, ~1 step/day)
#   "Nx":             100       # mesh cells in x
#   "Ny":             50        # mesh cells in y
#   "K_upper":        25.0      # hydraulic conductivity in upper layer (m/d)
#   "K_lower":        500.0     # hydraulic conductivity in lower layer (m/d)
#   "phi_upper":      0.30      # porosity in upper layer
#   "phi_lower":      0.20      # porosity in lower layer
#   "q_recharge":     2.0/365.25 # recharge rate (m/d)
#   "notes":          ""        # free-text note
#
# ==============================================================================
SWEEP_CONFIGS = {
   # List of configurations for models.
   "henry": [
      {"Q_in": 5.7024, "notes": "Henry scenario a"},
      {"Q_in": 2.851,  "notes": "Henry scenario b"},
  ]
}
# ==============================
# Run Sweep
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model parameter sweep runner (run_models.py)")
    parser.add_argument("--model", type=str, nargs="*", default=["all"], 
                        help="Which models to run sweeps for (e.g., henry island_single). Default: runs all configs present in SWEEP_CONFIGS")
    args = parser.parse_args()

    models_to_run = args.model
    if "all" in models_to_run:
        models_to_run = list(SWEEP_CONFIGS.keys())

    if not models_to_run:
        banner = "=" * 40
        print(banner)
        print("  PARAMETER SWEEP")
        print("  0 simulation(s) queued")
        print(banner)
        print("No configurations found in SWEEP_CONFIGS.")
        exit(0)

    grand_start = time.time()

    for model_name in models_to_run:
        configs = SWEEP_CONFIGS.get(model_name, [])
        n_sims = len(configs)

        sweep_title = f"  {model_name.upper()} PARAMETER SWEEP  |  {n_sims} simulation(s) queued"
        banner = "=" * len(sweep_title)
        print("\n" + banner)
        print(sweep_title)
        print(banner)

        if n_sims == 0:
            print(f"No configurations found for model '{model_name}'.")
            continue

        if model_name == "henry":
            henry_mod = importlib.import_module("henry")
            run_func = henry_mod.run_henry
        elif model_name == "island_single":
            island_single_mod = importlib.import_module("island_single")
            run_func = island_single_mod.run_island_single
        elif model_name == "island_double":
            island_double_mod = importlib.import_module("island_double")
            run_func = island_double_mod.run_island_double
        else:
            print(f"Unknown model: {model_name}")
            continue

        sweep_start = time.time()
        results = []
        
        for i, config in enumerate(configs, 1):
            config_line = f"  [{i}/{n_sims}]  Config: {config}"
            banner = "=" * len(config_line)
            result = run_func(**config)
            results.append(result)
            print(f"\n{banner}")
            print(config_line)
            print(banner)

        sweep_end = time.time()
        total_time = sweep_end - sweep_start

        print(f"\nTotal sweep time for {model_name}: {total_time:.1f} s ({total_time / 60:.1f} min)")

    grand_end   = time.time()
    grand_total = grand_end - grand_start
    n_total     = sum(len(SWEEP_CONFIGS.get(m, [])) for m in models_to_run)
    print(f"\n{'='*60}")
    print(f"  ALL SWEEPS COMPLETE")
    print(f"  Models run : {', '.join(models_to_run)}")
    print(f"  Total sims : {n_total}")
    print(f"  Total time : {grand_total:.1f} s  ({grand_total/60:.1f} min)")
    print(f"{'='*60}\n")
