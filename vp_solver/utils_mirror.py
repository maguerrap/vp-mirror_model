import jax
import jax.numpy as jnp
import equinox as eqx


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

