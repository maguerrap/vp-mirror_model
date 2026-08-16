import pytest
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from vp_solver.utils_mirror import (
    BFieldMLP,
    cost_rho,
    get_initial_distribution_single,
    get_initial_distribution_multi,
    compute_loss_single,
    step_single,
    compute_loss_multi,
    step_multi,
    load_b_trained
)
from vp_solver.jax_mirror_solver import make_mesh, VlasovPoissonSolver
from vp_solver.jax_mirror_solver_full import make_mesh_full, VlasovPoissonSolverFull

@pytest.fixture
def mesh_single():
    return make_mesh(length_z=1.0, length_v=2.0, length_mu=1.0, nz=10, nv=5, nmu=3)


@pytest.fixture
def mesh_multi():
    return make_mesh_full(length_z=1.0, length_v_e=2.0, length_v_i=1.0, length_mu=1.0, nz=10, nv=5, nmu=3)


@pytest.fixture
def solver_single(mesh_single):
    return VlasovPoissonSolver(mesh_single, dt=0.01)


@pytest.fixture
def solver_multi(mesh_multi):
    return VlasovPoissonSolverFull(mesh_multi, dt=0.01)


@pytest.fixture
def b_model():
    key = jax.random.PRNGKey(0)
    return BFieldMLP(key, center=0.0)


def test_bfield_mlp_basic(b_model):
    zs = jnp.linspace(-1.0, 1.0, 10)
    out_min, out_max = b_model.get_norm_bounds(zs)
    assert out_min <= out_max
    
    val = b_model.eval_point(0.0, out_min, out_max)
    assert val.shape == ()


def test_b_fn_and_grad(b_model):
    zs = jnp.linspace(-1.0, 1.0, 10)
    out_min, out_max = b_model.get_norm_bounds(zs)
    
    def B_fn(z):
        return b_model.eval_point(z, out_min, out_max)
        
    dB_fn = jax.grad(B_fn)
    
    # Test scalar evaluation
    b_val = B_fn(0.0)
    db_val = dB_fn(0.0)
    
    assert b_val.shape == ()
    assert db_val.shape == ()


def test_cost_rho():
    zs = jnp.linspace(-1.0, 1.0, 10)
    rho = jnp.ones_like(zs)
    cost = cost_rho(rho, zs)
    assert cost.shape == ()
    assert jnp.isclose(cost, 2.0)


def test_load_b_trained():
    zs = jnp.linspace(-1.0, 1.0, 10)
    B_fn_grid, dB_fn_grid, B_eval, dB_eval, g_eval = load_b_trained(seed=42, zs=zs, load_path=None)
    
    assert B_eval.shape == (10,)
    assert dB_eval.shape == (10,)
    assert g_eval.shape == (10,)


def test_get_initial_distribution_single(mesh_single):
    def B_fn_grid(z):
        return jnp.ones_like(z)
    
    f_iv = get_initial_distribution_single(B_fn_grid, sigma_z=0.5, mesh=mesh_single)
    assert f_iv.shape == (mesh_single.nz, mesh_single.nv, mesh_single.nmu)


def test_get_initial_distribution_multi(mesh_multi):
    def B_fn_grid(z):
        return jnp.ones_like(z)
    
    f_iv_e, f_iv_i = get_initial_distribution_multi(B_fn_grid, sigma_z=0.5, mesh=mesh_multi, m_i=25.0)
    assert f_iv_e.shape == (mesh_multi.nz, mesh_multi.nv, mesh_multi.nmu)
    assert f_iv_i.shape == (mesh_multi.nz, mesh_multi.nv, mesh_multi.nmu)


def test_single_species_optimization(b_model, solver_single, mesh_single):
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(b_model, eqx.is_array))
    
    loss, grads = compute_loss_single(
        b_model, sigma_z=0.5, solver=solver_single, t_final=0.01, mesh=mesh_single
    )
    assert loss.shape == ()
    
    model_new, opt_state_new, loss_step = step_single(
        b_model, opt_state, optimizer, sigma_z=0.5, solver=solver_single, t_final=0.01, mesh=mesh_single
    )
    assert jnp.isclose(loss_step, loss)


def test_multi_species_optimization(b_model, solver_multi, mesh_multi):
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(b_model, eqx.is_array))
    
    loss, grads = compute_loss_multi(
        b_model, sigma_z=0.5, solver=solver_multi, t_final=0.01, m_i=25.0, mesh=mesh_multi
    )
    assert loss.shape == ()
    
    model_new, opt_state_new, loss_step = step_multi(
        b_model, opt_state, optimizer, sigma_z=0.5, solver=solver_multi, t_final=0.01, m_i=25.0, mesh=mesh_multi
    )
    assert jnp.isclose(loss_step, loss)
