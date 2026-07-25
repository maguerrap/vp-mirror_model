import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import interpax

Array = jax.Array

import numpy as np


@dataclasses.dataclass
class MeshFull:
    """Mesh object updated to include cell edges for conservative finite volume advection."""
    zs: Array
    z_edges: Array
    dz: float
    vs_e: Array
    vs_e_edges: Array
    dv_e: float
    vs_i: Array
    vs_i_edges: Array
    dv_i: float
    mus: Array
    dmu: float
    V_e: Array
    V_i: Array
    Z: Array
    MU: Array
    period_z: float
    period_v_e: float
    period_v_i: float
    period_mu: float
    nz: int
    nv: int
    nmu: int


def make_mesh_full(length_z: float, length_v_e: float, length_v_i: float, length_mu: float,
              nz: int, nv: int, nmu: int) -> MeshFull:
    """Generates a combined mesh object for both electrons and ions."""
    zs = jnp.linspace(-length_z, length_z, nz, endpoint=True)
    dz = float(zs[1] - zs[0])
    z_edges = jnp.linspace(-length_z - dz/2, length_z + dz/2, nz + 1, endpoint=True)
    
    vs_e = jnp.linspace(-length_v_e, length_v_e, nv, endpoint=True)
    dv_e = float(vs_e[1] - vs_e[0])
    vs_e_edges = jnp.linspace(-length_v_e - dv_e/2, length_v_e + dv_e/2, nv + 1, endpoint=True)
    
    vs_i = jnp.linspace(-length_v_i, length_v_i, nv, endpoint=True)
    dv_i = float(vs_i[1] - vs_i[0])
    vs_i_edges = jnp.linspace(-length_v_i - dv_i/2, length_v_i + dv_i/2, nv + 1, endpoint=True)
    
    mus = jnp.linspace(0, length_mu, nmu, endpoint=True)
    dmu = float(mus[1] - mus[0])
    
    # Kept for compatibility with other methods (like compute_rho), 
    # but no longer needed for advection!
    Z, V_e, MU = jnp.meshgrid(zs, vs_e, mus, indexing="ij")
    _, V_i, _ = jnp.meshgrid(zs, vs_i, mus, indexing="ij")
    
    return MeshFull(
        zs=zs, z_edges=z_edges, dz=dz,
        vs_e=vs_e, vs_e_edges=vs_e_edges, dv_e=dv_e,
        vs_i=vs_i, vs_i_edges=vs_i_edges, dv_i=dv_i,
        mus=mus, dmu=dmu,
        Z=Z, V_e=V_e, V_i=V_i, MU=MU,
        period_z=length_z, period_v_e=length_v_e,
        period_v_i=length_v_i, period_mu=length_mu,
        nz=nz, nv=nv, nmu=nmu
    )


@dataclasses.dataclass(frozen=True)
class VlasovPoissonSolverFull:
    """
    Vlasov–Poisson Conservative semi-Lagrangian solver with operator splitting.
    """

    mesh: MeshFull
    dt: float
    m_i: float = 25.0

    def build_semilag_z_e(self) -> Callable[[Array], Array]:
        """Builds Conservative semi-Lagrangian to solve z-advection for electrons."""
        def interp_jax_z_e(f: Array) -> Array:
            def advect_z_1d(v_val, f_1d):
                z_edges_src = self.mesh.z_edges - 0.5 * v_val * self.dt
                safe_f = jnp.maximum(f_1d, 1e-8)
                mass = safe_f * self.mesh.dz  # <--- Prevents flat CDFs and NaN gradients
                F_edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
                
                # PCHIP on the primitive function guarantees strictly monotonic mass (no wiggles)
                interpolator = interpax.PchipInterpolator(self.mesh.z_edges, F_edges, check=False)
                F_src = interpolator(z_edges_src)
                
                # Clamping enforces zero-inflow / natural outflow
                F_src = jnp.clip(F_src, 0.0, F_edges[-1])
                return (F_src[1:] - F_src[:-1]) / self.mesh.dz

            advect_z_v = jax.vmap(advect_z_1d, in_axes=(0, 1), out_axes=1)
            advect_z_v_mu = jax.vmap(advect_z_v, in_axes=(None, 2), out_axes=2)
            return advect_z_v_mu(self.mesh.vs_e, f)
        return interp_jax_z_e

    def build_semilag_z_i(self) -> Callable[[Array], Array]:
        """Builds Conservative semi-Lagrangian to solve z-advection for ions."""
        def interp_jax_z_i(f: Array) -> Array:
            def advect_z_1d(v_val, f_1d):
                z_edges_src = self.mesh.z_edges - 0.5 * v_val * self.dt
                safe_f = jnp.maximum(f_1d, 1e-8)
                mass = safe_f * self.mesh.dz  # <--- Prevents flat CDFs and NaN gradients
                F_edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
                
                interpolator = interpax.PchipInterpolator(self.mesh.z_edges, F_edges, check=False)
                F_src = interpolator(z_edges_src)
                
                F_src = jnp.clip(F_src, 0.0, F_edges[-1])
                return (F_src[1:] - F_src[:-1]) / self.mesh.dz

            advect_z_v = jax.vmap(advect_z_1d, in_axes=(0, 1), out_axes=1)
            advect_z_v_mu = jax.vmap(advect_z_v, in_axes=(None, 2), out_axes=2)
            return advect_z_v_mu(self.mesh.vs_i, f)
        return interp_jax_z_i

    def build_semilag_v_e(self) -> Callable[[Array, Array, Array], Array]:
        """Builds Conservative semi-Lagrangian to solve v-advection for electrons."""
        def interp_jax_v_e(f: Array, E: Array, partial_B: Array) -> Array:
            def advect_v_1d(e_val, pb_val, mu_val, f_1d):
                a_val = e_val + mu_val * pb_val
                # Original sign tracking retained
                v_edges_src = self.mesh.vs_e_edges + a_val * self.dt 
                safe_f = jnp.maximum(f_1d, 1e-8)
                mass = safe_f * self.mesh.dv_e  # <--- Prevents flat CDFs and NaN gradients
                F_edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
                
                interpolator = interpax.PchipInterpolator(self.mesh.vs_e_edges, F_edges, check=False)
                F_src = interpolator(v_edges_src)
                
                F_src = jnp.clip(F_src, 0.0, F_edges[-1])
                return (F_src[1:] - F_src[:-1]) / self.mesh.dv_e

            advect_v_z = jax.vmap(advect_v_1d, in_axes=(0, 0, None, 0), out_axes=0)
            advect_v_z_mu = jax.vmap(advect_v_z, in_axes=(None, None, 0, 2), out_axes=2)
            return advect_v_z_mu(E, partial_B, self.mesh.mus, f)
        return interp_jax_v_e

    def build_semilag_v_i(self) -> Callable[[Array, Array, Array], Array]:
        """Builds Conservative semi-Lagrangian to solve v-advection for ions."""
        def interp_jax_v_i(f: Array, E: Array, partial_B: Array) -> Array:
            def advect_v_1d(e_val, pb_val, mu_val, f_1d):
                a_val = -(e_val - mu_val * pb_val) / self.m_i
                v_edges_src = self.mesh.vs_i_edges + a_val * self.dt
                safe_f = jnp.maximum(f_1d, 1e-8)
                mass = safe_f * self.mesh.dv_i  # <--- Prevents flat CDFs and NaN gradients
                F_edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
                
                interpolator = interpax.PchipInterpolator(self.mesh.vs_i_edges, F_edges, check=False)
                F_src = interpolator(v_edges_src)
                
                F_src = jnp.clip(F_src, 0.0, F_edges[-1])
                return (F_src[1:] - F_src[:-1]) / self.mesh.dv_i

            advect_v_z = jax.vmap(advect_v_1d, in_axes=(0, 0, None, 0), out_axes=0)
            advect_v_z_mu = jax.vmap(advect_v_z, in_axes=(None, None, 0, 2), out_axes=2)
            return advect_v_z_mu(E, partial_B, self.mesh.mus, f)
        return interp_jax_v_i

    def assemble_A_for_phi(self, g: Array) -> Array:
        nz = self.mesh.nz
        dz = self.mesh.dz

        A_zz = -jnp.diag(jnp.ones(nz-2), k=-2) \
            + 16*jnp.diag(jnp.ones(nz-1), k=-1) \
            - 30*jnp.diag(jnp.ones(nz)) \
            + 16*jnp.diag(jnp.ones(nz-1), k=1) \
            - jnp.diag(jnp.ones(nz-2), k=2)
        A_z = g*jnp.diag(jnp.ones(nz-2), k=-2) \
            - 8*g*jnp.diag(jnp.ones(nz-1), k=-1) \
            + 8*g*jnp.diag(jnp.ones(nz-1), k=1) \
            - g*jnp.diag(jnp.ones(nz-2), k=2)

        A = A_z / (12*dz) - A_zz / (12*dz**2)

        A = A.at[1,:].set(jnp.zeros(nz))
        A = A.at[1,0:5].set(g[1]*jnp.array([-3, -10, 18, -6, 1])/(12*dz)
                            - jnp.array([11, -20, 6, 4, -1])/(12*dz**2))
        A = A.at[-1,:].set(jnp.zeros(nz))
        A = A.at[-2,-5:].set(g[-2]*jnp.array([-1, 6, -18, 10, 3])/(12*dz) 
                             - jnp.array([-1, 4, 6, -20, 11])/(12*dz**2))

        A = A.at[0,:].set(jnp.zeros(nz))
        A = A.at[0,0].set(1.0)
        A = A.at[-1,:].set(jnp.zeros(nz))
        A = A.at[-1,-1].set(1.0)

        return A

    def assemble_A_for_E(self) -> Array:
        nz = self.mesh.nz
        dz = self.mesh.dz

        A = 8*jnp.diag(jnp.ones(nz-1), k=1) \
            - 8*jnp.diag(jnp.ones(nz-1), k=-1) \
            +jnp.diag(jnp.ones(nz-2), k=-2) \
            - jnp.diag(jnp.ones(nz-2), k=2)
        A = A.at[0,:].set(jnp.zeros(nz))
        A = A.at[0, 0:5].set(jnp.array([-25, 48, -36, 16, -3]))
        A = A.at[1,:].set(jnp.zeros(nz))
        A = A.at[1, 0:5].set(jnp.array([-3, -10, 18, -6, 1]))
        A = A.at[-2,:].set(jnp.zeros(nz))
        A = A.at[-2, -5:].set(jnp.array([-1, 6, -18, 10, 3]))
        A = A.at[-1,:].set(jnp.zeros(nz))
        A = A.at[-1, -5:].set(jnp.array([3, -16, 36, -48, 25]))
        A = -A / (12 * dz)
        return A

    def compute_rho_e(self, B: Array, f: Array) -> Array:
        return 2*jnp.pi*B*jnp.trapezoid(jnp.trapezoid(f, self.mesh.vs_e, axis=1), self.mesh.mus, axis=1)

    def compute_rho_i(self, B: Array, f: Array) -> Array:
        return 2*jnp.pi*B*jnp.trapezoid(jnp.trapezoid(f, self.mesh.vs_i, axis=1), self.mesh.mus, axis=1)

    def compute_rho_1d_e(self, f: Array) -> Array:
        return jnp.trapezoid(jnp.trapezoid(f, self.mesh.vs_e, axis=1), self.mesh.mus, axis=1)

    def compute_rho_1d_i(self, f: Array) -> Array:
        return jnp.trapezoid(jnp.trapezoid(f, self.mesh.vs_i, axis=1), self.mesh.mus, axis=1)

    def compute_phi_from_rho(self, A: Array, B: Array, rho_e: Array, rho_i: Array, rho_net_0: Array) -> Array:
        rho_net = rho_i - rho_e
        b = rho_net
        b = b.at[0].set(0.0)
        b = b.at[-1].set(0.0)
        phi = jnp.linalg.solve(A, b)
        return phi

    def compute_E_from_phi(self, A: Array, phi: Array) -> Array:
        E = A @ phi
        return E

    def compute_E(self, f_e: Array, f_i: Array, A_phi: Array, A_E: Array, B: Array, rho_net_0: Array) -> Array:
        rho_e = self.compute_rho_e(B, f_e)
        rho_i = self.compute_rho_i(B, f_i)
        phi = self.compute_phi_from_rho(A_phi, B, rho_e, rho_i, rho_net_0)
        return self.compute_E_from_phi(A_E, phi)

    def compute_electric_energy(self, E: Array) -> Array:
        return 0.5 * jnp.trapezoid(jnp.square(E), self.mesh.zs)

    def run_forward_jax_scan(
        self, f_e_iv: jax.Array, f_i_iv: jax.Array, B: jax.Array, partial_B: jax.Array, g: jax.Array,
        t_final: float, chunk_size: int = 2) -> tuple:
        
        num_steps = int(t_final / self.dt)

        f_e = f_e_iv
        f_i = f_i_iv
        
        rho_e_0 = self.compute_rho_e(B, f_e)
        rho_i_0 = self.compute_rho_i(B, f_i)
        rho_net_0 = rho_i_0 - rho_e_0
        rho_e_0_1d = self.compute_rho_1d_e(f_e)
        rho_i_0_1d = self.compute_rho_1d_i(f_i)
        
        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z_e = self.build_semilag_z_e()
        semilag_z_i = self.build_semilag_z_i()
        semilag_v_e = self.build_semilag_v_e()
        semilag_v_i = self.build_semilag_v_i()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        def time_step_jax(carry, t):
            f_e, f_i = carry
            
            f_e_half = semilag_z_e(f_e)
            f_i_half = semilag_z_i(f_i)
            
            E = self.compute_E(f_e_half, f_i_half, A_phi, A_E, B, rho_net_0)
            ee = self.compute_electric_energy(E)
            
            f_e_new = semilag_v_e(f_e_half, E, partial_B)
            f_i_new = semilag_v_i(f_i_half, E, partial_B)
            
            f_e_new = semilag_z_e(f_e_new)
            f_i_new = semilag_z_i(f_i_new)
            
            rho_e = self.compute_rho_1d_e(f_e_new)
            rho_i = self.compute_rho_1d_i(f_i_new)
            
            return (f_e_new, f_i_new), (E, ee, rho_e, rho_i)

        num_chunks = num_steps // chunk_size
        remainder = num_steps % chunk_size

        @jax.checkpoint
        def chunk_step_jax(carry, t_chunk):
            return jax.lax.scan(time_step_jax, carry, t_chunk)

        # 1. Process main chunks
        tspan_chunks = tspan[:num_chunks * chunk_size].reshape((num_chunks, chunk_size))

        (f_e_final, f_i_final), chunked_arrays = jax.lax.scan(
            chunk_step_jax, (f_e, f_i), tspan_chunks
        )
        
        # Unpack and flatten the chunked dimensions: (num_chunks, chunk_size, ...) -> (num_chunks * chunk_size, ...)
        E_total_array, ee_array, rho_e_array, rho_i_array = [
            arr.reshape((-1,) + arr.shape[2:]) for arr in chunked_arrays
        ]

        # 2. Process remainder block
        if remainder > 0:
            tspan_remainder = tspan[-remainder:]
            # Use the unpacked carry (f_e_final, f_i_final) and capture remainder history
            (f_e_final, f_i_final), rem_arrays = jax.lax.scan(
                time_step_jax, (f_e_final, f_i_final), tspan_remainder
            )
            
            # Append remainder arrays to the flattened chunk arrays
        #    f_e_array = jnp.concatenate([f_e_array, rem_arrays[0]], axis=0)
        #    f_i_array = jnp.concatenate([f_i_array, rem_arrays[1]], axis=0)
            E_total_array = jnp.concatenate([E_total_array, rem_arrays[0]], axis=0)
            ee_array = jnp.concatenate([ee_array, rem_arrays[1]], axis=0)
            rho_e_array = jnp.concatenate([rho_e_array, rem_arrays[2]], axis=0)
            rho_i_array = jnp.concatenate([rho_i_array, rem_arrays[3]], axis=0)

        # 3. Prepend the initial conditions
        # Note: using `[None, ...]` is cleaner than `[None, :, :, :]` as it adapts to any spatial/velocity grid shape
        #f_e_array = jnp.concatenate([f_e_iv[None, ...], f_e_array], axis=0)
        #f_i_array = jnp.concatenate([f_i_iv[None, ...], f_i_array], axis=0)
        rho_e_array = jnp.concatenate([rho_e_0_1d[None, :], rho_e_array], axis=0)
        rho_i_array = jnp.concatenate([rho_i_0_1d[None, :], rho_i_array], axis=0)

        return (f_e_final, f_i_final), E_total_array, ee_array, rho_e_array, rho_i_array
    
    def run_forward_jax_scan_no_E(
        self, f_e_iv: Array, f_i_iv: Array, B: Array, partial_B: Array, g: Array,
        t_final: float) -> tuple:
        
        num_steps = int(t_final / self.dt)
        f_e = f_e_iv.copy()
        f_i = f_i_iv.copy()
        
        rho_e_0 = self.compute_rho_e(B, f_e)
        rho_i_0 = self.compute_rho_i(B, f_i)
        rho_net_0 = rho_i_0 - rho_e_0
        rho_e_0_1d = self.compute_rho_1d_e(f_e)
        rho_i_0_1d = self.compute_rho_1d_i(f_i)
        
        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z_e = self.build_semilag_z_e()
        semilag_z_i = self.build_semilag_z_i()
        semilag_v_e = self.build_semilag_v_e()
        semilag_v_i = self.build_semilag_v_i()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        @jax.checkpoint
        def time_step_jax(carry, t):
            f_e, f_i = carry
            
            f_e_half = semilag_z_e(f_e)
            f_i_half = semilag_z_i(f_i)
            
            E = self.compute_E(f_e_half, f_i_half, A_phi, A_E, B, rho_net_0)
            
            f_e_new = semilag_v_e(f_e_half, jnp.zeros_like(E), partial_B)
            f_i_new = semilag_v_i(f_i_half, jnp.zeros_like(E), partial_B)
            
            f_e_new = semilag_z_e(f_e_new)
            f_i_new = semilag_z_i(f_i_new)
            
            rho_e = self.compute_rho_1d_e(f_e_new)
            rho_i = self.compute_rho_1d_i(f_i_new)
            
            return (f_e_new, f_i_new), (rho_e, rho_i)

        (f_e_final, f_i_final), (rho_e_array, rho_i_array) = jax.lax.scan(
            time_step_jax, (f_e, f_i), tspan
        )

        rho_e_array = jnp.concatenate([rho_e_0_1d[None, :], rho_e_array], axis=0)
        rho_i_array = jnp.concatenate([rho_i_0_1d[None, :], rho_i_array], axis=0)

        return (f_e_final, f_i_final), rho_e_array, rho_i_array

    def run_forward_jax_scan_efficient(
        self, f_e_iv: Array, f_i_iv: Array, B: Array, partial_B: Array, g: Array,
        t_final: float, chunk_size: int = 10) -> tuple:
        
        num_steps = int(t_final / self.dt)
        f_e = f_e_iv.copy()
        f_i = f_i_iv.copy()
        
        rho_e_0 = self.compute_rho_e(B, f_e)
        rho_i_0 = self.compute_rho_i(B, f_i)
        rho_net_0 = rho_i_0 - rho_e_0
        
        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z_e = self.build_semilag_z_e()
        semilag_z_i = self.build_semilag_z_i()
        semilag_v_e = self.build_semilag_v_e()
        semilag_v_i = self.build_semilag_v_i()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        def time_step_jax(carry, t):
            f_e_curr, f_i_curr = carry
            
            f_e_half = semilag_z_e(f_e_curr)
            f_i_half = semilag_z_i(f_i_curr)
            
            E = self.compute_E(f_e_half, f_i_half, A_phi, A_E, B, rho_net_0)
            
            f_e_new = semilag_v_e(f_e_half, E, partial_B)
            f_i_new = semilag_v_i(f_i_half, E, partial_B)
            
            f_e_new = semilag_z_e(f_e_new)
            f_i_new = semilag_z_i(f_i_new)
            
            return (f_e_new, f_i_new), None

        num_chunks = num_steps // chunk_size
        remainder = num_steps % chunk_size

        @jax.checkpoint
        def chunk_step_jax(carry, t_chunk):
            return jax.lax.scan(time_step_jax, carry, t_chunk)

        tspan_chunks = tspan[:num_chunks * chunk_size].reshape((num_chunks, chunk_size))

        carry_final, _ = jax.lax.scan(
            chunk_step_jax, (f_e, f_i), tspan_chunks
        )

        if remainder > 0:
            tspan_remainder = tspan[-remainder:]
            carry_final, _ = jax.lax.scan(
                time_step_jax, carry_final, tspan_remainder
            )

        f_e_final, f_i_final = carry_final
        
        #rho_e_final = self.compute_rho_1d_e(f_e_final)
        #rho_i_final = self.compute_rho_1d_i(f_i_final)

        return (f_e_final, f_i_final)

    def run_forward_hybrid(
        self, f_e_iv: jax.Array, f_i_iv: jax.Array, B: jax.Array, partial_B: jax.Array, g: jax.Array,
        t_final: float, chunk_size: int = 10) -> tuple:
        
        num_steps = int(t_final / self.dt)
        f_e = f_e_iv
        f_i = f_i_iv
        
        # 1. Initial calculations (Executed on GPU)
        rho_e_0 = self.compute_rho_e(B, f_e)
        rho_i_0 = self.compute_rho_i(B, f_i)
        rho_net_0 = rho_i_0 - rho_e_0
        rho_e_0_1d = self.compute_rho_1d_e(f_e)
        rho_i_0_1d = self.compute_rho_1d_i(f_i)
        
        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z_e = self.build_semilag_z_e()
        semilag_z_i = self.build_semilag_z_i()
        semilag_v_e = self.build_semilag_v_e()
        semilag_v_i = self.build_semilag_v_i()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        def time_step_jax(carry, t):
            f_e, f_i = carry
            
            f_e_half = semilag_z_e(f_e)
            f_i_half = semilag_z_i(f_i)
            
            E = self.compute_E(f_e_half, f_i_half, A_phi, A_E, B, rho_net_0)
            ee = self.compute_electric_energy(E)
            
            f_e_new = semilag_v_e(f_e_half, E, partial_B)
            f_i_new = semilag_v_i(f_i_half, E, partial_B)
            
            f_e_new = semilag_z_e(f_e_new)
            f_i_new = semilag_z_i(f_i_new)
            
            rho_e = self.compute_rho_1d_e(f_e_new)
            rho_i = self.compute_rho_1d_i(f_i_new)
            
            return (f_e_new, f_i_new), (f_e_new, f_i_new, E, ee, rho_e, rho_i)

        # 2. JIT compile the execution of a single chunk of timesteps
        @jax.jit
        def run_chunk(carry, tspan_chunk):
            return jax.lax.scan(time_step_jax, carry, tspan_chunk)

        num_chunks = num_steps // chunk_size
        remainder = num_steps % chunk_size

        carry = (f_e, f_i)
        
        # Pre-allocate standard Python lists to store the CPU-bound arrays
        cpu_f_e, cpu_f_i = [], []
        cpu_E, cpu_ee = [], []
        cpu_rho_e, cpu_rho_i = [], []

        # Helper function to pull a chunk's results off the GPU and into system RAM
        def append_to_cpu(chunk_arrays_gpu):
            # jax.device_get is the critical step: it pulls data from GPU VRAM -> CPU RAM
            chunk_cpu = jax.device_get(chunk_arrays_gpu)
            cpu_f_e.append(chunk_cpu[0])
            cpu_f_i.append(chunk_cpu[1])
            cpu_E.append(chunk_cpu[2])
            cpu_ee.append(chunk_cpu[3])
            cpu_rho_e.append(chunk_cpu[4])
            cpu_rho_i.append(chunk_cpu[5])

        # 3. Process Main Chunks via a standard Python loop
        for i in range(num_chunks):
            start_idx = i * chunk_size
            t_chunk = tspan[start_idx : start_idx + chunk_size]
            
            carry, chunk_arrays = run_chunk(carry, t_chunk)
            append_to_cpu(chunk_arrays)

        # 4. Process Remainder Time Steps
        if remainder > 0:
            t_rem = tspan[-remainder:]
            carry, rem_arrays = run_chunk(carry, t_rem)
            append_to_cpu(rem_arrays)

        # 5. Concatenate everything together on the CPU side using standard NumPy
        
        # First, pull the initial conditions from GPU to CPU
        f_e_iv_cpu = jax.device_get(f_e_iv)
        f_i_iv_cpu = jax.device_get(f_i_iv)
        rho_e_0_1d_cpu = jax.device_get(rho_e_0_1d)
        rho_i_0_1d_cpu = jax.device_get(rho_i_0_1d)

        # Now concatenate the lists of CPU arrays into continuous multi-dimensional arrays
        f_e_array = np.concatenate([f_e_iv_cpu[None, ...]] + cpu_f_e, axis=0)
        f_i_array = np.concatenate([f_i_iv_cpu[None, ...]] + cpu_f_i, axis=0)
        E_total_array = np.concatenate(cpu_E, axis=0)
        ee_array = np.concatenate(cpu_ee, axis=0)
        rho_e_array = np.concatenate([rho_e_0_1d_cpu[None, ...]] + cpu_rho_e, axis=0)
        rho_i_array = np.concatenate([rho_i_0_1d_cpu[None, ...]] + cpu_rho_i, axis=0)

        return carry, f_e_array, f_i_array, E_total_array, ee_array, rho_e_array, rho_i_array