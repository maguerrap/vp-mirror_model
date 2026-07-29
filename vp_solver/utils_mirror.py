from typing import Callable

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from .jax_mirror_solver import VlasovPoissonSolver, Mesh
from .jax_mirror_solver_full import VlasovPoissonSolverFull, MeshFull

Array = jax.Array


class BFieldMLP(eqx.Module):
    mlp: eqx.nn.MLP
    center: float

    def __init__(self, key, center=0.0):
        """Initializes the BFieldMLP model.

        Args:
            key (jax.random.PRNGKey): The PRNG key for initialization.
            center (float, optional): The center of the symmetry.
                Defaults to 0.0.
        """
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=32,
            depth=3,
            activation=jax.nn.gelu,
            key=key,
        )
        self.center = center

    def _raw_eval(self, z: float) -> Array:
        """Evaluates the raw MLP output with symmetry.

        Args:
            z (float): The spatial coordinate.

        Returns:
            Array: The raw MLP output.
        """
        z_sym = (z - self.center) ** 2
        return self.mlp(jnp.array([z_sym]))[0]

    def get_norm_bounds(self, grid: Array):
        """Gets the minimum and maximum of the raw MLP output over a grid.

        Args:
            grid (Array): The spatial grid.

        Returns:
            tuple: A tuple containing the minimum and maximum values.
        """
        raw_grid = jax.vmap(self._raw_eval)(grid)
        return jnp.min(raw_grid), jnp.max(raw_grid)

    def eval_point(self, z: float, out_min: float, out_max: float) -> Array:
        """Evaluates and normalizes the point.

        Args:
            z (float): The spatial coordinate.
            out_min (float): The minimum raw output value.
            out_max (float): The maximum raw output value.

        Returns:
            Array: The normalized evaluated point.
        """
        raw_val = self._raw_eval(z)
        diff = jnp.maximum(out_max - out_min, 1e-12)
        normalized = (raw_val - out_min) / diff
        return 1.0 + 9.0 * normalized


def load_b_trained(seed: int, zs: Array, load_path: str = None):
    """Loads a trained BFieldMLP model and returns relevant functions and
    evaluations.

    Args:
        seed (int): The random seed.
        zs (Array): The spatial grid.
        load_path (str, optional): The path to load the model from.
            Defaults to None.

    Returns:
        tuple: A tuple containing B_fn_grid, dB_fn_grid, B_eval,
            dB_eval, g_eval.
    """
    key = jax.random.PRNGKey(seed)
    loaded_b_model = BFieldMLP(key, center=0.0)
    if load_path:
        loaded_b_model = eqx.tree_deserialise_leaves(load_path, loaded_b_model)
    out_min, out_max = loaded_b_model.get_norm_bounds(zs)

    def B_fn(z):
        return loaded_b_model.eval_point(z, out_min, out_max)

    B_fn_grid = jnp.vectorize(B_fn)
    dB_fn = jax.grad(B_fn)
    dB_fn_grid = jnp.vectorize(dB_fn)
    B_eval = B_fn_grid(zs)
    dB_eval = dB_fn_grid(zs)
    g_eval = dB_eval / B_eval
    return B_fn_grid, dB_fn_grid, B_eval, dB_eval, g_eval


@jax.jit
def cost_rho(rho_final: Array, zs: Array) -> Array:
    """Computes the cost function.

    Args:
        rho_final (Array): The final charge density.
        zs (Array): The spatial grid.

    Returns:
        Array: The computed cost.
    """
    return jnp.trapezoid(rho_final, zs)


def get_initial_distribution_single(
    B_fn_grid: Callable, sigma_z: float, mesh: Mesh
) -> Array:
    """Computes the initial distribution for a single species.

    Args:
        B_fn_grid (Callable): Function to evaluate the magnetic
            field on a grid.
        sigma_z (float): The spatial spread.
        mesh (Mesh): The mesh object.

    Returns:
        Array: The initial distribution function.
    """
    f_eq_Z = jnp.exp(-mesh.Z**2 / (2 * sigma_z**2))

    B_norm = B_fn_grid(mesh.Z)

    f_eq = jnp.exp(-0.5 * ((mesh.V) ** 2 + 2 * B_norm * mesh.MU))

    C_z = 1 / (
        jnp.trapezoid(
            jnp.trapezoid(f_eq, x=mesh.mus, axis=2), x=mesh.vs, axis=1
        )
    )

    f_iv = C_z[:, None, None] * f_eq_Z * f_eq

    rho = jnp.trapezoid(
        jnp.trapezoid(f_iv, x=mesh.mus, axis=2), x=mesh.vs, axis=1
    )

    D = 1 / (
        jnp.trapezoid(
            jnp.trapezoid(
                jnp.trapezoid(f_iv, x=mesh.mus, axis=2), x=mesh.vs, axis=1
            ),
            x=mesh.zs,
            axis=0,
        )
    )

    f_iv = D * f_iv

    return f_iv


def get_initial_distribution_multi(
    B_fn_grid: Callable,
    sigma_z: float,
    mesh: MeshFull,
    m_i: float,
) -> tuple[Array, Array]:
    """Computes the initial distributions for electrons and ions.

    Args:
        B_fn_grid (Callable): Function to evaluate the magnetic
            field on a grid.
        sigma_z (float): The spatial spread.
        mesh (MeshFull): The full mesh object.
        m_i (float): The ion mass.

    Returns:
        tuple[Array, Array]: Initial distribution functions for
            electrons and ions.
    """
    f_eq_Z = jnp.exp(-mesh.Z**2 / (2 * sigma_z**2))

    B_norm = B_fn_grid(mesh.Z)

    # Calculate Electron Maxwellian
    f_eq_e = jnp.exp(-0.5 * (mesh.V_e**2 + 2 * B_norm * mesh.MU))
    C_z_e = 1.0 / (
        jnp.trapezoid(
            jnp.trapezoid(f_eq_e, x=mesh.mus, axis=2), x=mesh.vs_e, axis=1
        )
    )
    f_iv_e = C_z_e[:, None, None] * f_eq_Z * f_eq_e

    # Normalizing factor so the spatial integral fits exactly
    D_e = 1.0 / (
        jnp.trapezoid(
            jnp.trapezoid(
                jnp.trapezoid(f_iv_e, x=mesh.mus, axis=2), x=mesh.vs_e, axis=1
            ),
            x=mesh.zs,
            axis=0,
        )
    )
    f_iv_e = D_e * f_iv_e

    # Calculate Ion Maxwellian
    f_eq_i = jnp.exp(-0.5 * (m_i * mesh.V_i**2 + 2 * B_norm * mesh.MU))
    C_z_i = 1.0 / (
        jnp.trapezoid(
            jnp.trapezoid(f_eq_i, x=mesh.mus, axis=2), x=mesh.vs_i, axis=1
        )
    )
    f_iv_i = C_z_i[:, None, None] * f_eq_Z * f_eq_i

    D_i = 1.0 / (
        jnp.trapezoid(
            jnp.trapezoid(
                jnp.trapezoid(f_iv_i, x=mesh.mus, axis=2), x=mesh.vs_i, axis=1
            ),
            x=mesh.zs,
            axis=0,
        )
    )
    f_iv_i = D_i * f_iv_i

    return f_iv_e, f_iv_i


@eqx.filter_value_and_grad
def compute_loss_single(
    model: BFieldMLP,
    sigma_z: float,
    solver: VlasovPoissonSolver,
    t_final: float,
    mesh: Mesh,
) -> float:
    """Computes the loss for a single species simulation.

    Args:
        model (BFieldMLP): The magnetic field model.
        sigma_z (float): The spatial spread.
        solver (VlasovPoissonSolver): The Vlasov-Poisson solver.
        t_final (float): The final simulation time.
        mesh (Mesh): The mesh object.

    Returns:
        float: The computed loss (negative for maximization).
    """
    # Dynamically find the bounds for the CURRENT weights
    out_min, out_max = model.get_norm_bounds(mesh.zs)

    # Create the bounded functions
    def B_fn(z):
        return model.eval_point(z, out_min, out_max)

    dB_fn = jax.grad(B_fn)
    B_fn_grid = jnp.vectorize(B_fn)
    dB_fn_grid = jnp.vectorize(dB_fn)

    # Evaluate both functions smoothly over the entire space!
    B_eval = B_fn_grid(mesh.zs)
    dB_eval = dB_fn_grid(mesh.zs)
    g_eval = dB_eval / B_eval

    f_iv = get_initial_distribution_single(B_fn_grid, sigma_z, mesh)

    f_arr, rho_last = solver.run_forward_jax_scan_efficient(
        f_iv, B_eval, dB_eval, g_eval, t_final
    )

    loss = jnp.trapezoid(rho_last, mesh.zs)

    return -1.0 * loss  # Since we are maximizing


# Optimization Step


@eqx.filter_jit
def step_single(
    model: BFieldMLP,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    sigma_z: float,
    solver: VlasovPoissonSolver,
    t_final: float,
    mesh: Mesh,
):
    """Performs a single optimization step for the single species model.

    Args:
        model (BFieldMLP): The magnetic field model.
        opt_state (optax.OptState): The optimizer state.
        optimizer (optax.GradientTransformation): The optimizer.
        sigma_z (float): The spatial spread.
        solver (VlasovPoissonSolver): The Vlasov-Poisson solver.
        t_final (float): The final simulation time.
        mesh (Mesh): The mesh object.

    Returns:
        tuple: Updated model, opt_state, and the loss.
    """
    loss, grads = compute_loss_single(model, sigma_z, solver, t_final, mesh)

    # Update weights
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


@eqx.filter_value_and_grad
def compute_loss_multi(
    model: BFieldMLP,
    sigma_z: float,
    solver: VlasovPoissonSolverFull,
    t_final: float,
    m_i: float,
    mesh: MeshFull,
) -> float:
    """Computes the loss for a multi-species simulation.

    Args:
        model (BFieldMLP): The magnetic field model.
        sigma_z (float): The spatial spread.
        solver (VlasovPoissonSolverFull): The full Vlasov-Poisson solver.
        t_final (float): The final simulation time.
        m_i (float): The ion mass.
        mesh (MeshFull): The full mesh object.

    Returns:
        float: The computed loss (negative for maximization).
    """
    out_min, out_max = model.get_norm_bounds(mesh.zs)

    def B_fn(z):
        return model.eval_point(z, out_min, out_max)

    dB_fn = jax.grad(B_fn)
    B_fn_grid = jnp.vectorize(B_fn)
    dB_fn_grid = jnp.vectorize(dB_fn)

    B_eval = B_fn_grid(mesh.zs)
    dB_eval = dB_fn_grid(mesh.zs)
    g_eval = dB_eval / B_eval

    f_iv_e, f_iv_i = get_initial_distribution_multi(
        B_fn_grid, sigma_z, mesh, m_i
    )

    f_e_last, f_i_last = solver.run_forward_jax_scan_efficient(
        f_iv_e, f_iv_i, B_eval, dB_eval, g_eval, t_final, chunk_size=10
    )

    rho_e_final = solver.compute_rho_1d_e(f_e_last)
    rho_i_final = solver.compute_rho_1d_i(f_i_last)

    loss = cost_rho(rho_e_final, mesh.zs) + cost_rho(rho_i_final, mesh.zs)

    return -1.0 * loss  # Since we are maximizing


# Optimization Step


@eqx.filter_jit
def step_multi(
    model: BFieldMLP,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    sigma_z: float,
    solver: VlasovPoissonSolverFull,
    t_final: float,
    m_i: float,
    mesh: MeshFull,
):
    """Performs a single optimization step for the multi-species model.

    Args:
        model (BFieldMLP): The magnetic field model.
        opt_state (optax.OptState): The optimizer state.
        optimizer (optax.GradientTransformation): The optimizer.
        sigma_z (float): The spatial spread.
        solver (VlasovPoissonSolverFull): The full Vlasov-Poisson solver.
        t_final (float): The final simulation time.
        m_i (float): The ion mass.
        mesh (MeshFull): The full mesh object.

    Returns:
        tuple: Updated model, opt_state, and the loss.
    """
    loss, grads = compute_loss_multi(
        model, sigma_z, solver, t_final, m_i, mesh
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
