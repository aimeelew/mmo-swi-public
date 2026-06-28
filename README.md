# mmo-swi-public

## Physical Background

This repository contains 2D density-driven saltwater intrusion models. Because seawater is denser than freshwater, a buoyancy-driven circulation flow develops when the two interact in coastal aquifers. These models explore this phenomenon through two main scenarios:

- **The Henry Problem**: A classic benchmark modeling a 2D vertical cross-section of a coastal aquifer. Fresh inland groundwater flows steadily toward the sea and interacts with a stationary seawater boundary, forming a denser "saltwater wedge" that intrudes landward along the aquifer base while lighter freshwater flows over it.
- **Island Freshwater Lens**: Models a 2D vertical half-island cross-section where a freshwater lens forms due to surface rainfall recharge over a saline aquifer. This is simulated in two variations:
  - **Single-Layer**: A uniform rock layer.
  - **Double-Layer**: Features two distinct geological layers separated by the Thurber discontinuity at a 12m depth (a Holocene aquifer layered above a more permeable Pleistocene aquifer).

## Repository Structure

The models are implemented using two different simulation engines, separated into their respective folders:

- **`fenics/`**: Contains FEniCS-based finite element models simulating the Henry problem, the single-layer half-island, and the double-layer half-island. More documentation [here](https://fenicsproject.org/).
- **`waiwera/`**: Contains Waiwera (a multi-phase geothermal and groundwater simulator) models simulating the same three scenarios utilising the `wse` module (Water, Salt, Energy Equation of State). More documentation [here](https://waiwera.readthedocs.io/en/latest/) and [here](https://waiwera.github.io/).

## Requirements and Prerequisites

- **Python Dependencies**: Each folder contains a requirements file (`fenics_requirements.txt` and `waiwera_requirements.txt`) listing the necessary third-party dependencies outside the Python standard library.
- **Docker (Waiwera only)**: The Waiwera models were built to run using Docker Desktop in Windows. You must have Docker Desktop installed and running before executing the Waiwera scripts.

## Running the Models

To set up model parameters and execute the simulations, use the designated entry point scripts in each folder:

- **FEniCS Models**: Use the `run_models.py` script located in the `fenics` folder.
- **Waiwera Models**: Use the `run.py` script located in the `waiwera` folder.

Both entry scripts act as orchestrators to configure specific model parameters and run the simulations.

## Data Folder Layout

The `data/` directory contains outputs from numerous configured simulations. Inside its sub-folders, you will find:
- **NPZ Files**: NumPy array files containing the final concentration distributions at the end of each simulation.
- **CSV Files**: Tabulated metrics and summary statistics for each simulation run.
