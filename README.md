# Vlasov–Poisson Mirror Confinement Model (`vp-mirror_model`)

`vp-mirror_model` is a GPU-accelerated, fully differentiable kinetic simulation and optimization framework written in **JAX** and **Equinox**[cite: 1, 2, 3]. It models 1D2V kinetic plasma dynamics in magnetic mirror geometries by solving the non-linear Vlasov–Poisson system with magnetic moment ($\mu$) force coupling[cite: 1, 3]:

$$\frac{\partial f_s}{\partial t} + v \frac{\partial f_s}{\partial z} + \frac{1}{m_s} \left( q_s E - \mu \frac{\partial B}{\partial z} \right) \frac{\partial f_s}{\partial v} = 0$$

The solver leverages a **conservative semi-Lagrangian scheme** using monotonic PCHIP shape-preserving spline interpolations (via `interpax`) to guarantee exact mass preservation and wiggle-free distribution functions[cite: 1, 3].

---

## Key Features

* **High-Order Numerical Solvers:**
  * **Single-species solver (`jax_mirror_solver.py`):** Fast kinetic kinetic transport for simplified single-species setups[cite: 2, 3].
  * **Multi-species solver (`jax_mirror_solver_full.py`):** Fully coupled kinetic multi-species (electron & ion) solver with self-consistent Poisson field solve[cite: 1, 2].
* **Neural Magnetic Field Optimization:**
  * Neural network parametrization of $B(z)$ using Equinox MLPs[cite: 2].
  * End-to-end automatic differentiation via JAX & Optax to discover optimal magnetic mirror shapes that maximize plasma particle retention[cite: 2].
* **Mass-Conservative Transport:**
  * 1D cumulative distribution function (CDF) mapping using `interpax.PchipInterpolator` prevents unphysical oscillations and numerical loss[cite: 1, 3].
* **Flexible Execution Backends:**
  * Memory-efficient JAX `lax.scan` time-stepping routines for single GPU execution[cite: 1, 3].
  * Hybrid GPU-to-CPU stream processing for memory-limited hardware (`run_forward_hybrid`).

---

## Directory Structure

```text
vp-mirror_model/
├── vp_solver/                  # Core library modules
│   ├── jax_mirror_solver.py       # Single-species mesh & Vlasov-Poisson solver
│   ├── jax_mirror_solver_full.py  # Multi-species mesh & Vlasov-Poisson solver
│   ├── utils_mirror.py            # Neural B-field MLP, distribution initializers, & loss functions
│   └── utils_plots.py             # Plotting functions
├── examples/                   # Demonstration and research notebooks
│   ├── simulation_single.ipynb    # Single-species forward kinetic simulation
│   ├── simulation_multi.ipynb     # Multi-species forward kinetic simulation
│   ├── optimize_single.ipynb      # B-field optimization for single-species confinement
│   └── optimize_multi.ipynb       # B-field optimization for multi-species confinement
├── tests/                      # Pytest unit and integration test suite
├── pyproject.toml              # Build & dependency management file
└── README.md                   # Project documentation
