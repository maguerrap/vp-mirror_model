import pytest
import jax
import jax.numpy as jnp

from vp_solver.jax_mirror_solver_full import VlasovPoissonSolverFull, make_mesh_full

# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def solver():
    mesh = make_mesh_full(length_z=1.0, length_v_e=2.0, length_v_i=1.0, length_mu=1.0, nz=10, nv=5, nmu=3)
    dt = 0.01
    return VlasovPoissonSolverFull(mesh, dt)


@pytest.fixture
def f0_e(solver):
    """Initial distribution f_e(z,v_e,mu)."""
    return jnp.exp(-solver.mesh.Z**2 - solver.mesh.V_e**2)


@pytest.fixture
def f0_i(solver):
    """Initial distribution f_i(z,v_i,mu)."""
    return jnp.exp(-solver.mesh.Z**2 - solver.mesh.V_i**2)


# ----------------------------
# Basic tests
# ----------------------------

def test_semilag_z_e_runs(solver, f0_e):
    semilag = solver.build_semilag_z_e()
    f1 = semilag(f0_e)
    assert f1.shape == f0_e.shape


def test_semilag_z_i_runs(solver, f0_i):
    semilag = solver.build_semilag_z_i()
    f1 = semilag(f0_i)
    assert f1.shape == f0_i.shape


def test_semilag_v_e_runs(solver, f0_e):
    semilag = solver.build_semilag_v_e()
    E = jnp.zeros(solver.mesh.nz)
    dB = jnp.zeros(solver.mesh.nz)
    f1 = semilag(f0_e, E, dB)
    assert f1.shape == f0_e.shape


def test_semilag_v_i_runs(solver, f0_i):
    semilag = solver.build_semilag_v_i()
    E = jnp.zeros(solver.mesh.nz)
    dB = jnp.zeros(solver.mesh.nz)
    f1 = semilag(f0_i, E, dB)
    assert f1.shape == f0_i.shape


def test_assemble_A_phi_shape(solver):
    g = jnp.ones(solver.mesh.nz)
    A = solver.assemble_A_for_phi(g)
    assert A.shape == (solver.mesh.nz, solver.mesh.nz)


def test_assemble_A_E_shape(solver):
    A = solver.assemble_A_for_E()
    assert A.shape == (solver.mesh.nz, solver.mesh.nz)


def test_rho_computation(solver, f0_e, f0_i):
    B = jnp.ones(solver.mesh.nz)
    rho_e = solver.compute_rho_e(B, f0_e)
    rho_i = solver.compute_rho_i(B, f0_i)
    assert rho_e.shape == (solver.mesh.nz,)
    assert rho_i.shape == (solver.mesh.nz,)


def test_phi_and_E(solver, f0_e, f0_i):
    B = jnp.ones(solver.mesh.nz)
    g = jnp.ones(solver.mesh.nz)

    A_phi = solver.assemble_A_for_phi(g)
    A_E   = solver.assemble_A_for_E()

    rho_e = solver.compute_rho_e(B, f0_e)
    rho_i = solver.compute_rho_i(B, f0_i)
    rho_net_0 = rho_i - rho_e

    phi = solver.compute_phi_from_rho(A_phi, B, rho_e, rho_i, rho_net_0)
    E   = solver.compute_E_from_phi(A_E, phi)

    assert phi.shape == (solver.mesh.nz,)
    assert E.shape == (solver.mesh.nz,)


# ----------------------------
# Integration test
# ----------------------------
def test_one_step_forward(solver, f0_e, f0_i):
    mesh = solver.mesh

    B = jnp.ones(mesh.nz)
    dB = jnp.zeros(mesh.nz)
    g  = jnp.ones(mesh.nz)

    t_final = solver.dt
    num_steps = int(t_final/solver.dt)

    (f_e_final, f_i_final), E_total_array, ee_array, rho_e_array, rho_i_array = solver.run_forward_jax_scan(
        f0_e, f0_i, B, dB, g, t_final
    )

    assert f_e_final.shape == f0_e.shape
    assert f_i_final.shape == f0_i.shape
    assert E_total_array.shape == (num_steps, mesh.nz)
    assert ee_array.shape == (num_steps,)
    assert rho_e_array.shape == (num_steps + 1, mesh.nz)
    assert rho_i_array.shape == (num_steps + 1, mesh.nz)
