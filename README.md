# Monoenergetic 1D Diffusion Solver for Eigenvalue Problems

## Overview

This **class** project implements a 1D neutron diffusion solver for both **fixed source** and **fission source** problems in **heterogeneous media**. It uses the **finite volume method (FVM)** for spatial discretization and a **memory-optimized Gauss-Seidel iterative solver** for efficient matrix solution. The code supports **vacuum** and **reflective** boundary conditions and visualizes results such as neutron flux and residuals per iteration.

---

## Features

- Handles both **fixed** and **fission** source problems
- User-defined **boundary conditions** (vacuum or reflective)
- Support for **heterogeneous materials**
- **Adaptive mesh** refinement for vacuum boundaries
- Multiple iterative solvers:
  - Gauss-Seidel (optimized for sparse matrix)
  - Jacobi (regular, parallelized)
- Robust **user input validation**
- Optional **write-to-file** logging for reproducibility
- **Visualization** of neutron flux and convergence behavior

![alt text](https://github.com/AyubSherif/ReactorPhysicsNE5002/blob/main/1D_Diffusion_Solver.png)

![alt text](https://github.com/AyubSherif/ReactorPhysicsNE5002/blob/main/Iterative%20Solvers%20Comparision.png)

---

## File Structure

- `final_project.py`: Main program logic and user interface.
- `iterative_solvers.py`: Contains all iterative solvers (Gauss-Seidel, Jacobi, etc.).
- `Final Project Report.pdf`: Detailed technical report explaining mathematical formulation, implementation, and results.
- `data.txt`: Automatically generated log file storing inputs and results.

---

## Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib