import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
#from jax.scipy.interpolate import RegularGridInterpolator
import interpax

Array = jax.Array


@dataclasses.dataclass
class Mesh:
    """Mesh object."""
    zs: Array
    dz: float
    vs: Array
    dv: float
    mus: Array
    dmu: float
    #Zz: Array
    #Vz: Array
    #Vv: Array
    #MUv: Array
    V: Array
    Z: Array
    MU: Array
    period_z: float
    period_v: float
    period_mu: float
    nz: int
    nv: int
    nmu: int


def make_mesh(length_z: float, length_v: float, length_mu: float,
              nz: int, nv: int, nmu: int) -> Mesh:
    """Generates mesh object."""
    zs = jnp.linspace(-length_z, length_z, nz, endpoint=True)
    dz = float(zs[1] - zs[0])
    vs = jnp.linspace(-length_v, length_v, nv, endpoint=True)
    mus = jnp.linspace(0, length_mu, nv, endpoint=True)
    dv = float(vs[1] - vs[0])
    dmu = float(mus[1] - mus[0])
    #Zz, Vz = jnp.meshgrid(zs, vs, indexing="ij")
    #Vv, MUv = jnp.meshgrid(vs, mus, indexing="ij")
    Z, V, MU = jnp.meshgrid(zs, vs, mus, indexing="ij")
    return Mesh(
        zs=zs,
        dz=dz,
        vs=vs,
        dv=dv,
        mus = mus,
        dmu = dmu,
        #Zz=Zz,
        #Vz=Vz,
        #Vv=Vv,
        #MUv=MUv,
        Z=Z,
        V=V,
        MU=MU,
        period_z=length_z,
        period_v=length_v,
        period_mu=length_mu,
        nz=nz,
        nv=nv,
        nmu=nmu
    )


@dataclasses.dataclass(frozen=True)
class VlasovPoissonSolver:
    """
    Vlasov–Poisson semi-Lagrangian solver with operator splitting.

    The Vlasov–Poisson system is given by

        ∂_t f(t, z, v, μ)
        + v ∂_x f(t, z, v, μ)
        - (E(t, z) + μ∂_z|B|) ∂_v f(t, z, v, μ) = 0,

    where f(t, z, v, μ) is the particle distribution function.

    The electric field E(t, z) is determined by Poisson's equation:

        E(t, z) = - ∂_z φ(t, z)

        - ∂_zz φ(t, z) + ∂_zφ(∂_z|B|/|B|) = 1 - 2π|B|ρ(t, z),

    with charge density

        ρ(t, z) = ∫ f(t, z, v, μ) dv dμ.

    We split our operator into
        ∂_t f(t, z, v, μ) + v ∂_z f(t, x, v, μ) = 0  (1),
    and
        ∂_t f(t, z, v, μ) - (E(t, z) + μ∂_z|B|) ∂_v f(t, x, v) = 0    (2).
    """

    mesh: Mesh
    dt: float


    def build_semilag_z(self) -> Callable[[Array], Array]:
        """
        Builds semi-Lagrangian to solve (1) via linear interpolation.
        """
        def interp_jax_z(f: Array) -> Array:
            Z_src = self.mesh.Z - 0.5 * self.mesh.V * self.dt
            f_interp = interpax.interp3d(Z_src.flatten(), self.mesh.V.flatten(), self.mesh.MU.flatten(),
                                         self.mesh.zs, self.mesh.vs, self.mesh.mus,
                                         f, method='linear', extrap=[[0.0,0.0], [0.0,0.0], [0.0,0.0]],
                                         period=(None, None, None)
            ).reshape(self.mesh.nz, self.mesh.nv, self.mesh.nmu)

            # mask for BCs
            #inflow_from_left  = (Z_src < self.mesh.zs[0]) & (self.mesh.V > 0)
            #inflow_from_right = (Z_src > self.mesh.zs[-1]) & (self.mesh.V < 0)
            
            #f_bc = jnp.where(inflow_from_left | inflow_from_right, 0.0, f_interp)
            return f_interp
        return interp_jax_z
        #return jax.vmap(interp_jax_z, in_axes=2, out_axes=2)

    def build_semilag_v(self) -> Callable[[Array, Array, Array, Array], Array]:
        """
        Builds semi-Lagrangian to solve (2) via linear interpolation.
        """
        def interp_jax_v(f: Array, E: Array, partial_B: Array) -> Array:
            V_src = self.mesh.V + (E[:,None,None] + self.mesh.MU * partial_B[:,None,None]) * self.dt
            f_interp = interpax.interp3d(self.mesh.Z.flatten(), V_src.flatten(), self.mesh.MU.flatten(),
                                         self.mesh.zs, self.mesh.vs, self.mesh.mus,
                                          f, method='linear', extrap=[[0.0,0.0], [0.0,0.0], [0.0,0.0]],
                                          period=(None, None, None)
            ).reshape(self.mesh.nz, self.mesh.nv, self.mesh.nmu)
            # mask for BCs
            #inflow_from_vmin = (V_src < self.mesh.vs[0])   # coming from below v-boundary
            #inflow_from_vmax = (V_src > self.mesh.vs[-1])  # coming from above v-boundary
            
            #f_bc = jnp.where(inflow_from_vmin | inflow_from_vmax, 0.0, f_interp)
            return f_interp
        return interp_jax_v
        #return jax.vmap(interp_jax_v, in_axes=(0,0,0), out_axes=0)

    def assemble_A_for_phi(self, g: Array) -> Array:

        A_zz = -jnp.diag(jnp.ones(self.mesh.nz-2), k=-2) \
            + 16*jnp.diag(jnp.ones(self.mesh.nz-1), k=-1) \
            - 30*jnp.diag(jnp.ones(self.mesh.nz)) \
            + 16*jnp.diag(jnp.ones(self.mesh.nz-1), k=1) \
            - jnp.diag(jnp.ones(self.mesh.nz-2), k=2)
        A_z = g*jnp.diag(jnp.ones(self.mesh.nz-2), k=-2) \
            - 8*g*jnp.diag(jnp.ones(self.mesh.nz-1), k=-1) \
            + 8*g*jnp.diag(jnp.ones(self.mesh.nz-1), k=1) \
            - g*jnp.diag(jnp.ones(self.mesh.nz-2), k=2)

        A = A_z / (12*self.mesh.dz) - A_zz / (12*self.mesh.dz**2)

        A = A.at[1,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[1,0:5].set(g[1]*jnp.array([-3, -10, 18, -6, 1])/(12*self.mesh.dz)
                            - jnp.array([11, -20, 6, 4, -1])/(12*self.mesh.dz**2))
        A = A.at[-1,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[-2,-5:].set(g[-2]*jnp.array([-1, 6, -18, 10, 3])/(12*self.mesh.dz) 
                             - jnp.array([-1, 4, 6, -20, 11])/(12*self.mesh.dz**2))

        A = A.at[0,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[0,0].set(1.0)
        A = A.at[-1,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[-1,-1].set(1.0)

        return A

    def assemble_A_for_E(self) -> Array:
        A = 8*jnp.diag(jnp.ones(self.mesh.nz-1), k=1) \
            - 8*jnp.diag(jnp.ones(self.mesh.nz-1), k=-1) \
            +jnp.diag(jnp.ones(self.mesh.nz-2), k=-2) \
            - jnp.diag(jnp.ones(self.mesh.nz-2), k=2)
        A = A.at[0,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[0, 0:5].set(jnp.array([-25, 48, -36, 16, -3]))
        A = A.at[1,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[1, 0:5].set(jnp.array([-3, -10, 18, -6, 1]))
        A = A.at[-2,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[-2, -5:].set(jnp.array([-1, 6, -18, 10, 3]))
        A = A.at[-1,:].set(jnp.zeros(self.mesh.nz))
        A = A.at[-1, -5:].set(jnp.array([3, -16, 36, -48, 25]))
        A = -A / (12 * self.mesh.dz)
        return A

    def compute_rho(self, B: Array, f: Array) -> Array:
        """Compute value of ρ(t, z)."""
        return 2*jnp.pi*B*jnp.sum(f, axis=(1,2)) * self.mesh.dv * self.mesh.dmu

    def compute_rho_1d(self, f: Array) -> Array:
        """Compute vale of ρ^{1D}(t,z)."""
        return jnp.sum(f, axis=(1,2)) * self.mesh.dv * self.mesh.dmu

    def compute_phi_from_rho(self, A: Array, B: Array, rho: Array, rho_0: Array) -> Array:
        """Solver for phi using finite differences."""
        b = rho_0 - rho
        b = b.at[0].set(0.0)
        b = b.at[-1].set(0.0)
        phi = jnp.linalg.solve(A, b)
        return phi

    def compute_E_from_phi(self, A: Array, phi: Array) -> Array:
        """Compute value of E(t, z) from φ(t, z)."""
        E = A @ phi
        return E

    def compute_E(self, f: Array, A_phi: Array, A_E: Array, B: Array, rho_0: Array) -> Array:
        """Compute E(t, z)."""
        return self.compute_E_from_phi(A_E,
                                       self.compute_phi_from_rho(A_phi, B,
                                                                 self.compute_rho(B,
                                                                                  f), rho_0))

    def compute_electric_energy(self, E: Array) -> Array:
        """Compute electric energy from E(t, z)."""
        return 0.5 * jnp.sum(jnp.square(E)) * self.mesh.dz

    def run_forward_jax_scan(
        self, f_iv: Array, B: Array, partial_B: Array, g: Array,
        t_final: float) -> tuple:
        
        """
        Compute time integration for the time derivative.
        """

        num_steps = int(t_final / self.dt)
        f = f_iv.copy()
        rho_0 = self.compute_rho(B, f)
        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z = self.build_semilag_z()
        semilag_v = self.build_semilag_v()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        #compute_E_jax = jax.jit(self.compute_E)
        #compute_energy_jax = jax.jit(self.compute_electric_energy)

        @jax.jit
        def time_step_jax(f, t):
            f_half = semilag_z(f)
            E = self.compute_E(f_half, A_phi, A_E, B, rho_0)
            E_total = E
            ee = self.compute_electric_energy(E)
            f = semilag_v(f_half, E, partial_B)
            f = semilag_z(f)
            rho = self.compute_rho_1d(f)
            #rho_B = 2 * jnp.pi * B * rho
            return f, (f, E_total, ee, rho)

        f_array, (f_total, E_total, ee, rho_array) = jax.lax.scan(
            time_step_jax, f, tspan
        )

        return f_array, f_total, E_total, ee, rho_array