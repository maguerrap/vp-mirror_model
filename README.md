# Vlasov–Poisson Mirror Solver

A solver for drift-kinetic plasma confinement using **magnetic mirrors** with a conservative semi-Lagrangian scheme, built with [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox) for high-performance computing and automatic differentiation.

This project enables forward simulation of multi-species kinetic plasma dynamics and optimization over external magnetic field profiles $|\mathbf{B}(z)|$. These numerical experiments support the paper [Kinetic Optimization of Magnetic Mirror Confinement: Beyond Classical Loss-Cone Theory](https://arxiv.org/abs/2607.26479).

---

## Mathematical Model

We solve the **drift-kinetic Vlasov–Poisson system** in spatial dimension $z$ and velocity dimensions $(v, \mu)$ under an external magnetic field $|\mathbf{B}(z)|$:

$$
\begin{cases} \partial_{t} f_s + v\partial_{z} f_s + \left( \frac{q_s}{m_s}E(t,z) - \mu \partial_z \vert{}\mathbf{B}(z)\vert{} \right)\partial_{v} f_s = 0 , \\
E(t,z) = -\partial_{z} \phi(t,z) ,\\ 
-\partial_{zz}\phi + \partial_{z}\phi \frac{\partial_z \vert{}\mathbf{B}(z)\vert{}}{\vert{}\mathbf{B}(z)\vert{}} = 2\pi \vert{}\mathbf{B}(z)\vert{} \iint \left( f_i - f_e \right) \mathrm{d}v \mathrm{d}\mu .
\end{cases}
$$

where:
- $f_s(t, z, v, \mu)$ is the distribution function for plasma species $s \in \lbrace e, i\rbrace$ (electrons and ions),
- $E(t, z)$ is the self-consistent electric field and $\phi(t, z)$ is the electric potential,
- $|\mathbf{B}(z)|$ is the external magnetic field profile acting as a mirror control,
- $q_s$ and $m_s$ are the charge and mass for species $s$,
- $\mu$ is the magnetic moment (first adiabatic invariant),
- $z \in [-L_z, L_z]$, $v \in [-L_v, L_v]$, and $\mu \in [0, \mu_{\max}]$.

---

## Features

- ⚡ Conservative semi-Lagrangian solver with Strang operator splitting  
- 🔁 Built with [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox) for GPU/TPU acceleration and automatic differentiation  
- 🧲 Pre-trained neural magnetic field profiles (`.eqx`) for single- and multi-species mirror regimes  
- ⚡ Self-consistent Poisson solver adapted for non-uniform magnetic flux tube geometry  
- 📊 Visualization utilities for kinetic phase space distributions, charge densities, and field dynamics  
- 📓 Example Jupyter notebooks for reproducible experiments  

---

## Installation

### Requirements
- Python **3.12+**
- [pip](https://pip.pypa.io/en/stable/) for package management  
- NVIDIA GPU with recent drivers (recommended for JAX GPU execution)

### CPU Version (default)
To install the solver with CPU JAX:

```bash
pip install git+https://github.com/maguerrap/vp-mirror_model.git@main
```

### GPU Version (recommended)
For GPU acceleration, first install JAX with CUDA support by following the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html).
Then install the solver with the command above.

## Usage

```bash
from vp_solver.jax_mirror_solver import Mesh, VlasovPoissonSolver # for single-species
from vp_solver.jax_mirror_solver import MeshFull, VlasovPoissonSolverFull # for multi-species
```

### Examples & Pre-trained models

We provide pre-trained neural network profiles and Jupyter demonstration notebooks in the `examples/` directory.

#### Directory Structure

```text
examples/
├── trained_bfields/
│   ├── bfield_single_species.eqx    # Pre-trained profile for single-species regime
│   └── bfield_multi_species.eqx     # Pre-trained profile for multi-species regime
├── VP_mirror_E_effect.ipynb         # Single-species E-field effect
├── VP_mirror_opt.ipynb              # Single-species confinement optimization routine
├── VP_mirror_E_effect_full.ipynb    # Multi-species E-field effect
└── VP_mirror_opt_full.ipynb         # Multi-species confinement optimization routine
```

In these notebooks, we:

- Run forward simulations of the Vlasov–Poisson system with and without the presence of an electric field $E(t,z)$.
- Use [Optax](https://github.com/google-deepmind/optax/tree/main) to solve a PDE-constrained optimization problem:
    - The goal is to design the magnetic profile  $|\mathbf{B}(z)|$ that maximizes plasma confinement.

To run them, launch Jupyter:

```bash
jupyter notebook examples/
```

Or if Google Colab is preferred one can access them below:

#### 1. Electric Field Effect Study for Single-species
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maguerrap/vp-mirror_model/blob/main/examples/VP_mirror_E_effect.ipynb)

* **File:** `examples/VP_mirror_E_effect.ipynb`
* **Focus:** Understand self-consistent electric fields $E(t,z)$ in a single-species regime ($s=e$).

---

#### 2. Electric Field Effect Study for Multi-species
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maguerrap/vp-mirror_model/blob/main/examples/VP_mirror_E_effect_full.ipynb)

* **File:** `examples/VP_mirror_E_effect_full.ipynb`
* **Focus:** Understand self-consistent electric fields $E(t,z)$ in a multi-species regime ($s=e, i$).

---

#### 3. Magnetic Field Control & Optimization for Single-species
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maguerrap/vp-mirror_model/blob/main/examples/VP_mirror_opt.ipynb)

* **File:** `examples/VP_mirror_opt.ipynb`
* **Focus:** Uses Optax and Equinox to perform PDE-constrained optimization over neural magnetic field profiles for single-species regime ($s=e$).

---

#### 4. Magnetic Field Control & Optimization for Multi-species
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maguerrap/vp-mirror_model/blob/main/examples/VP_mirror_opt_full.ipynb)

* **File:** `examples/VP_mirror_opt_full.ipynb`
* **Focus:** Uses Optax and Equinox to perform PDE-constrained optimization over neural magnetic field profiles for multi-species regime ($s=e, i$).

---

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/maguerrap/vp-mirror_model/blob/main/LICENSE).
