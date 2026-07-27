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
        self.mlp = eqx.nn.MLP(
            in_size=1, 
            out_size=1, 
            width_size=32, 
            depth=3, 
            activation=jax.nn.gelu, 
            key=key
        )
        self.center = center
        
    def _raw_eval(self, z: float) -> jax.Array:
        z_sym = (z - self.center)**2 
        return self.mlp(jnp.array([z_sym]))[0]
        
    def get_norm_bounds(self, grid: jax.Array):
        raw_grid = jax.vmap(self._raw_eval)(grid)
        return jnp.min(raw_grid), jnp.max(raw_grid)
        
    def eval_point(self, z: float, out_min: float, out_max: float) -> jax.Array:
        raw_val = self._raw_eval(z)
        diff = jnp.maximum(out_max - out_min, 1e-12)
        normalized = (raw_val - out_min) / diff
        return 1.0 + 9.0 * normalized 



def load_b_trained(seed: int, zs: jnp.ndarray, load_path: str=None):
    key = jax.random.PRNGKey(seed)
    loaded_b_model = BFieldMLP(key, center=0.0)
    if load_path:
        loaded_b_model = eqx.tree_deserialise_leaves(load_path, loaded_b_model)
    out_min, out_max = loaded_b_model.get_norm_bounds(zs)
    B_fn = lambda z: loaded_b_model.eval_point(z, out_min, out_max)
    B_fn_grid = jnp.vectorize(B_fn)
    dB_fn = jax.grad(B_fn)
    dB_fn_grid = jnp.vectorize(dB_fn)
    B_eval = B_fn_grid(zs)
    dB_eval = dB_fn_grid(zs)
    g_eval = dB_eval / B_eval
    return B_fn_grid, dB_fn_grid, B_eval, dB_eval, g_eval


@jax.jit
def cost_rho(rho_final: jnp.ndarray, zs: jnp.ndarray) -> jax.Array:
    """Computes the cost function."""
    return jnp.trapezoid(rho_final, zs)


def get_initial_distribution_single(B_fn_grid: Callable, sigma_z:float,
                                  mesh: Mesh) -> Array:

    f_eq_Z = jnp.exp(-mesh.Z**2/(2*sigma_z**2))

    B_norm = B_fn_grid(mesh.Z)

    f_eq = jnp.exp(-0.5*((mesh.V)**2 + 2*B_norm*mesh.MU))

    C_z = 1 / (jnp.trapezoid(jnp.trapezoid(f_eq, x=mesh.mus, axis=2), x=mesh.vs, axis=1))

    f_iv = C_z[:,None,None] * f_eq_Z * f_eq

    rho = jnp.trapezoid(jnp.trapezoid(f_iv, x=mesh.mus, axis=2), x=mesh.vs, axis=1)

    D = 1 / (jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(f_iv, x=mesh.mus, axis=2), x=mesh.vs, axis=1), x=mesh.zs, axis=0))

    f_iv = D * f_iv

    return f_iv


def get_initial_distribution_multi(B_fn_grid: Callable, sigma_z: float,
                                   mesh: MeshFull, m_i: float,
                                   ) -> tuple[Array, Array]:

    f_eq_Z = jnp.exp(-mesh.Z**2/(2*sigma_z**2))

    B_norm = B_fn_grid(mesh.Z)

    # Calculate Electron Maxwellian
    f_eq_e = jnp.exp(-0.5 * (mesh.V_e**2 + 2 * B_norm * mesh.MU))
    C_z_e = 1.0 / (jnp.trapezoid(jnp.trapezoid(f_eq_e, x=mesh.mus, axis=2), x=mesh.vs_e, axis=1))
    f_iv_e = C_z_e[:, None, None] * f_eq_Z * f_eq_e

    # Normalizing factor so the spatial integral fits exactly
    D_e = 1.0 / (jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(f_iv_e, x=mesh.mus, axis=2), x=mesh.vs_e, axis=1), x=mesh.zs, axis=0))
    f_iv_e = D_e * f_iv_e

    # Calculate Ion Maxwellian
    f_eq_i = jnp.exp(-0.5 * (m_i * mesh.V_i**2 + 2 * B_norm * mesh.MU))
    C_z_i = 1.0 / (jnp.trapezoid(jnp.trapezoid(f_eq_i, x=mesh.mus, axis=2),
                                    x=mesh.vs_i, axis=1))
    f_iv_i = C_z_i[:, None, None] * f_eq_Z * f_eq_i

    D_i = 1.0 / (jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(f_iv_i, x=mesh.mus,
                                                            axis=2),
                                             x=mesh.vs_i, axis=1),
                                   x=mesh.zs, axis=0))
    f_iv_i = D_i * f_iv_i

    return f_iv_e, f_iv_i


@eqx.filter_value_and_grad
def compute_loss_single(model: BFieldMLP, sigma_z:float,
                        solver: VlasovPoissonSolver, t_final: float,
                        mesh: Mesh) -> float:
    # Dynamically find the bounds for the CURRENT weights
    out_min, out_max = model.get_norm_bounds(mesh.zs)
    
    # Create the bounded functions
    B_fn = lambda z: model.eval_point(z, out_min, out_max)
    dB_fn = jax.grad(B_fn)
    B_fn_grid = jnp.vectorize(B_fn)
    dB_fn_grid = jnp.vectorize(dB_fn)

    # Evaluate both functions smoothly over the entire space!
    B_eval = B_fn_grid(mesh.zs)
    dB_eval = dB_fn_grid(mesh.zs)
    g_eval = dB_eval / B_eval

    f_iv = get_initial_distribution_single(B_fn_grid, sigma_z, mesh)

    f_arr, rho_last = solver.run_forward_jax_scan_efficient(f_iv, B_eval,
                                                            dB_eval, g_eval,
                                                            t_final)

    loss = jnp.trapezoid(rho_last, mesh.zs)

    return -1.0 * loss #Since we are maximizing

# Optimization Step
@eqx.filter_jit
def step_single(model: BFieldMLP, opt_state: optax.OptState, 
                        optimizer: optax.GradientTransformation,
                        sigma_z:float,
                        solver: VlasovPoissonSolver, t_final: float,
                        mesh: Mesh):
    loss, grads = compute_loss_single(model, sigma_z, solver, t_final, mesh)
    
    # Update weights
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss



@eqx.filter_value_and_grad
def compute_loss_multi(model: BFieldMLP, sigma_z:float,
                       solver: VlasovPoissonSolverFull, t_final: float,
                       m_i: float, mesh: MeshFull) -> float:
    out_min, out_max = model.get_norm_bounds(mesh.zs)
    
    B_fn = lambda z: model.eval_point(z, out_min, out_max)
    dB_fn = jax.grad(B_fn)
    B_fn_grid = jnp.vectorize(B_fn)
    dB_fn_grid = jnp.vectorize(dB_fn)

    B_eval = B_fn_grid(mesh.zs)
    dB_eval = dB_fn_grid(mesh.zs)
    g_eval = dB_eval / B_eval

    f_iv_e, f_iv_i = get_initial_distribution_multi(B_fn_grid, sigma_z, mesh, m_i)

    (f_e_last, f_i_last) = solver.run_forward_jax_scan_efficient(
                f_iv_e, f_iv_i, B_eval, dB_eval, g_eval, t_final, chunk_size=10
            )

    rho_e_final = solver.compute_rho_1d_e(f_e_last)
    rho_i_final = solver.compute_rho_1d_i(f_i_last)

    loss = cost_rho(rho_e_final, mesh.zs) + cost_rho(rho_i_final, mesh.zs)

    return -1.0 * loss #Since we are maximizing

# Optimization Step
@eqx.filter_jit
def step_multi(model: BFieldMLP, opt_state: optax.OptState, 
               optimizer: optax.GradientTransformation,
               sigma_z:float, solver: VlasovPoissonSolverFull, 
               t_final: float, m_i: float, mesh: MeshFull):
    loss, grads = compute_loss_multi(model, sigma_z, solver, t_final, m_i, mesh)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss