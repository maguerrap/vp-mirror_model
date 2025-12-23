import pytest
import jax
import jax.numpy as jnp

from vp_solver.jax_mirror_solver import VlasovPoissonSolver


# ----------------------------
# Minimal Mesh stub for testing (need at least nz > 5)
# ----------------------------
class Mesh:
    def __init__(self, nz=10, nv=5, nmu=3):
        self.nz  = nz
        self.nv  = nv
        self.nmu = nmu

        # coordinates
        self.zs  = jnp.linspace(-1.0, 1.0, nz)
        self.vs  = jnp.linspace(-2.0, 2.0, nv)
        self.mus = jnp.linspace(0.0, 1.0, nmu)

        # 3D meshgrids
        Z, V, MU = jnp.meshgrid(self.zs, self.vs, self.mus, indexing="ij")
        self.Z  = Z
        self.V  = V
        self.MU = MU

        # spacings
        self.dz  = float(self.zs[1] - self.zs[0])
        self.dv  = float(self.vs[1] - self.vs[0])
        self.dmu = float(self.mus[1] - self.mus[0])


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def solver():
    mesh = Mesh(nz=10, nv=5, nmu=3)
    dt = 0.01
    return VlasovPoissonSolver(mesh, dt)


@pytest.fixture
def f0(solver):
    """Initial distribution f(z,v,mu)."""
    return jnp.exp(-solver.mesh.Z**2 - solver.mesh.V**2)


# ----------------------------
# Basic tests
# ----------------------------

def test_semilag_z_runs(solver, f0):
    semilag = solver.build_semilag_z()
    f1 = semilag(f0)
    assert f1.shape == f0.shape


def test_semilag_v_runs(solver, f0):
    semilag = solver.build_semilag_v()
    E = jnp.zeros(solver.mesh.nz)
    dB = jnp.zeros(solver.mesh.nz)
    f1 = semilag(f0, E, dB)
    assert f1.shape == f0.shape


def test_assemble_A_phi_shape(solver):
    g = jnp.ones(solver.mesh.nz)
    A = solver.assemble_A_for_phi(g)
    assert A.shape == (solver.mesh.nz, solver.mesh.nz)


def test_assemble_A_E_shape(solver):
    A = solver.assemble_A_for_E()
    assert A.shape == (solver.mesh.nz, solver.mesh.nz)


def test_rho_computation(solver, f0):
    B = jnp.ones(solver.mesh.nz)
    rho = solver.compute_rho(B, f0)
    assert rho.shape == (solver.mesh.nz,)


def test_phi_and_E(solver, f0):
    B = jnp.ones(solver.mesh.nz)
    g = jnp.ones(solver.mesh.nz)

    A_phi = solver.assemble_A_for_phi(g)
    A_E   = solver.assemble_A_for_E()

    rho = solver.compute_rho(B, f0)
    rho0 = rho.copy()

    phi = solver.compute_phi_from_rho(A_phi, B, rho, rho0)
    E   = solver.compute_E_from_phi(A_E, phi)

    assert phi.shape == (solver.mesh.nz,)
    assert E.shape == (solver.mesh.nz,)


# ----------------------------
# Integration test
# ----------------------------
def test_one_step_forward(solver, f0):
    mesh = solver.mesh

    B = jnp.ones(mesh.nz)
    dB = jnp.zeros(mesh.nz)
    g  = jnp.ones(mesh.nz)

    t_final = solver.dt
    num_steps = int(t_final/solver.dt)

    f_array, f_total, E_total, ee, rho_arr = solver.run_forward_jax_scan(
        f0, B, dB, g, t_final
    )

    assert f_array.shape == f0.shape   # scan output
    assert f_total.shape == (num_steps, mesh.nz, mesh.nv, mesh.nmu)
    assert E_total.shape == (num_steps, mesh.nz)
    assert ee.shape == (num_steps,)
    assert rho_arr.shape == (num_steps, mesh.nz)
