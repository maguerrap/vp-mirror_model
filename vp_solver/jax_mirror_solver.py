import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import interpax

Array = jax.Array


class Mesh(eqx.Module):
    """Mesh object updated to include cell edges for conservative finite volume
    advection."""

    zs: Array
    z_edges: Array
    dz: float
    vs: Array
    vs_edges: Array
    dv: float
    mus: Array
    dmu: float
    V: Array
    Z: Array
    MU: Array
    period_z: float
    period_v: float
    period_mu: float
    nz: int
    nv: int
    nmu: int


def make_mesh(
    length_z: float,
    length_v: float,
    length_mu: float,
    nz: int,
    nv: int,
    nmu: int,
) -> Mesh:
    """Generates a combined mesh object for both electrons and ions.

    Args:
        length_z (float): Half-length of the spatial domain.
        length_v (float): Half-length of the velocity domain.
        length_mu (float): Length of the magnetic moment domain.
        nz (int): Number of grid points in the z direction.
        nv (int): Number of grid points in the v direction.
        nmu (int): Number of grid points in the mu direction.

    Returns:
        Mesh: A mesh object containing grid points and cell edges.
    """
    zs = jnp.linspace(-length_z, length_z, nz, endpoint=True)
    dz = float(zs[1] - zs[0])
    z_edges = jnp.linspace(
        -length_z - dz / 2, length_z + dz / 2, nz + 1, endpoint=True
    )

    vs = jnp.linspace(-length_v, length_v, nv, endpoint=True)
    dv = float(vs[1] - vs[0])
    vs_edges = jnp.linspace(
        -length_v - dv / 2, length_v + dv / 2, nv + 1, endpoint=True
    )

    mus = jnp.linspace(0, length_mu, nmu, endpoint=True)
    dmu = float(mus[1] - mus[0])

    # Kept for compatibility with other methods (like compute_rho),
    # but no longer needed for advection!
    Z, V, MU = jnp.meshgrid(zs, vs, mus, indexing="ij")

    return Mesh(
        zs=zs,
        z_edges=z_edges,
        dz=dz,
        vs=vs,
        vs_edges=vs_edges,
        dv=dv,
        mus=mus,
        dmu=dmu,
        Z=Z,
        V=V,
        MU=MU,
        period_z=length_z,
        period_v=length_v,
        period_mu=length_mu,
        nz=nz,
        nv=nv,
        nmu=nmu,
    )


class VlasovPoissonSolver(eqx.Module):
    """Vlasov–Poisson Conservative semi-Lagrangian solver with operator
    splitting."""

    mesh: Mesh
    dt: float
    m_i: float = 25.0

    def build_semilag_z(self) -> Callable[[Array], Array]:
        """Builds Conservative semi-Lagrangian to solve z-advection for
        electrons.

        Returns:
            Callable[[Array], Array]: A function that takes the distribution
            function f and returns the updated f after z-advection.
        """

        def interp_jax_z(f: Array) -> Array:
            def advect_z_1d(v_val, f_1d):
                z_edges_src = (
                    self.mesh.z_edges - 0.5 * v_val * self.dt
                )  # + 1e-10
                safe_f = jnp.maximum(f_1d, 1e-8)
                # <--- Prevents flat CDFs and NaN gradients
                mass = (safe_f) * self.mesh.dz
                F_edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
                interpolator = interpax.PchipInterpolator(
                    self.mesh.z_edges, F_edges, check=False
                )
                F_src = interpolator(z_edges_src)
                F_src = jnp.clip(F_src, 0.0, F_edges[-1])
                return (F_src[1:] - F_src[:-1]) / self.mesh.dz

            advect_z_v = jax.vmap(advect_z_1d, in_axes=(0, 1), out_axes=1)
            advect_z_v_mu = jax.vmap(advect_z_v, in_axes=(None, 2), out_axes=2)
            return advect_z_v_mu(self.mesh.vs, f)

        return interp_jax_z

    def build_semilag_v(self) -> Callable[[Array, Array, Array], Array]:
        """Builds Conservative semi-Lagrangian to solve v-advection for
        electrons.

        Returns:
            Callable[[Array, Array, Array], Array]: A function that takes the
            distribution function f, electric field E, and partial B field,
            and returns the updated f after v-advection.
        """

        def interp_jax_v(f: Array, E: Array, partial_B: Array) -> Array:
            def advect_v_1d(e_val, pb_val, mu_val, f_1d):
                a_val = e_val + mu_val * pb_val
                v_edges_src = self.mesh.vs_edges + a_val * self.dt  # + 1e-10
                safe_f = jnp.maximum(f_1d, 1e-8)
                mass = (
                    safe_f * self.mesh.dv
                )  # <--- Prevents flat CDFs and NaN gradients
                F_edges = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
                interpolator = interpax.PchipInterpolator(
                    self.mesh.vs_edges, F_edges, check=False
                )
                F_src = interpolator(v_edges_src)
                F_src = jnp.clip(F_src, 0.0, F_edges[-1])
                return (F_src[1:] - F_src[:-1]) / self.mesh.dv

            advect_v_z = jax.vmap(
                advect_v_1d, in_axes=(0, 0, None, 0), out_axes=0
            )
            advect_v_z_mu = jax.vmap(
                advect_v_z, in_axes=(None, None, 0, 2), out_axes=2
            )
            return advect_v_z_mu(E, partial_B, self.mesh.mus, f)

        return interp_jax_v

    def assemble_A_for_phi(self, g: Array) -> Array:
        """Assembles the matrix A for solving Poisson's equation for potential
        phi.

        Args:
            g (Array): The gradient of the magnetic field
                over the magnetic field.

        Returns:
            Array: The assembled matrix A.
        """
        nz = self.mesh.nz
        dz = self.mesh.dz

        A_zz = (
            -jnp.diag(jnp.ones(nz - 2), k=-2)
            + 16 * jnp.diag(jnp.ones(nz - 1), k=-1)
            - 30 * jnp.diag(jnp.ones(nz))
            + 16 * jnp.diag(jnp.ones(nz - 1), k=1)
            - jnp.diag(jnp.ones(nz - 2), k=2)
        )
        A_z = (
            g * jnp.diag(jnp.ones(nz - 2), k=-2)
            - 8 * g * jnp.diag(jnp.ones(nz - 1), k=-1)
            + 8 * g * jnp.diag(jnp.ones(nz - 1), k=1)
            - g * jnp.diag(jnp.ones(nz - 2), k=2)
        )

        A = A_z / (12 * dz) - A_zz / (12 * dz**2)

        A = A.at[1, :].set(jnp.zeros(nz))
        A = A.at[1, 0:5].set(
            g[1] * jnp.array([-3, -10, 18, -6, 1]) / (12 * dz)
            - jnp.array([11, -20, 6, 4, -1]) / (12 * dz**2)
        )
        A = A.at[-1, :].set(jnp.zeros(nz))
        A = A.at[-2, -5:].set(
            g[-2] * jnp.array([-1, 6, -18, 10, 3]) / (12 * dz)
            - jnp.array([-1, 4, 6, -20, 11]) / (12 * dz**2)
        )

        A = A.at[0, :].set(jnp.zeros(nz))
        A = A.at[0, 0].set(1.0)
        A = A.at[-1, :].set(jnp.zeros(nz))
        A = A.at[-1, -1].set(1.0)

        return A

    def assemble_A_for_E(self) -> Array:
        """Assembles the matrix A for computing the electric field E from
        potential phi.

        Returns:
            Array: The assembled matrix A.
        """
        nz = self.mesh.nz
        dz = self.mesh.dz

        A = (
            8 * jnp.diag(jnp.ones(nz - 1), k=1)
            - 8 * jnp.diag(jnp.ones(nz - 1), k=-1)
            + jnp.diag(jnp.ones(nz - 2), k=-2)
            - jnp.diag(jnp.ones(nz - 2), k=2)
        )
        A = A.at[0, :].set(jnp.zeros(nz))
        A = A.at[0, 0:5].set(jnp.array([-25, 48, -36, 16, -3]))
        A = A.at[1, :].set(jnp.zeros(nz))
        A = A.at[1, 0:5].set(jnp.array([-3, -10, 18, -6, 1]))
        A = A.at[-2, :].set(jnp.zeros(nz))
        A = A.at[-2, -5:].set(jnp.array([-1, 6, -18, 10, 3]))
        A = A.at[-1, :].set(jnp.zeros(nz))
        A = A.at[-1, -5:].set(jnp.array([3, -16, 36, -48, 25]))
        A = -A / (12 * dz)
        return A

    def compute_rho(self, B: Array, f: Array) -> Array:
        """Computes the full charge density.

        Args:
            B (Array): The magnetic field.
            f (Array): The distribution function.

        Returns:
            Array: The computed charge density.
        """
        return (
            2
            * jnp.pi
            * B
            * jnp.trapezoid(
                jnp.trapezoid(f, self.mesh.vs, axis=1), self.mesh.mus, axis=1
            )
        )

    def compute_rho_1d(self, f: Array) -> Array:
        """Computes the 1D charge density along z.

        Args:
            f (Array): The distribution function.

        Returns:
            Array: The computed 1D charge density.
        """
        return jnp.trapezoid(
            jnp.trapezoid(f, self.mesh.vs, axis=1), self.mesh.mus, axis=1
        )

    def compute_phi_from_rho(
        self, A: Array, B: Array, rho: Array, rho_0: Array
    ) -> Array:
        """Computes the electric potential phi from the charge density.

        Args:
            A (Array): The matrix A for solving Poisson's equation.
            B (Array): The magnetic field.
            rho (Array): The charge density.
            rho_0 (Array): The initial background charge density.

        Returns:
            Array: The electric potential phi.
        """
        b = rho_0 - rho
        b = b.at[0].set(0.0)
        b = b.at[-1].set(0.0)
        phi = jnp.linalg.solve(A, b)
        return phi

    def compute_E_from_phi(self, A: Array, phi: Array) -> Array:
        """Computes the electric field E from the electric potential phi.

        Args:
            A (Array): The matrix A for computing E from phi.
            phi (Array): The electric potential phi.

        Returns:
            Array: The electric field E.
        """
        E = A @ phi
        return E

    def compute_E(
        self, f: Array, A_phi: Array, A_E: Array, B: Array, rho_0: Array
    ) -> Array:
        """Computes the electric field directly from the distribution function.

        Args:
            f (Array): The distribution function.
            A_phi (Array): The matrix A for solving Poisson's equation.
            A_E (Array): The matrix A for computing E from phi.
            B (Array): The magnetic field.
            rho_0 (Array): The initial background charge density.

        Returns:
            Array: The electric field E.
        """
        rho = self.compute_rho(B, f)
        phi = self.compute_phi_from_rho(A_phi, B, rho, rho_0)
        return self.compute_E_from_phi(A_E, phi)

    def compute_electric_energy(self, E: Array) -> Array:
        """Computes the total electric energy.

        Args:
            E (Array): The electric field E.

        Returns:
            Array: The total electric energy.
        """
        return 0.5 * jnp.trapezoid(jnp.square(E), self.mesh.zs)

    def run_forward_jax_scan(
        self, f_iv: Array, B: Array, partial_B: Array, g: Array, t_final: float
    ) -> tuple:
        """Runs the forward simulation using JAX's lax.scan.

        Args:
            f_iv (Array): The initial distribution function.
            B (Array): The magnetic field.
            partial_B (Array): The spatial derivative of the magnetic field.
            g (Array): The gradient of the magnetic field
                over the magnetic field.
            t_final (float): The final simulation time.

        Returns:
            tuple: A tuple containing the final distribution function f,
            array of f over time, total E over time, electric energy over time,
            and rho over time.
        """
        num_steps = int(t_final / self.dt)
        f = f_iv.copy()

        rho_0 = self.compute_rho(B, f)
        rho_0_1d = self.compute_rho_1d(f)

        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z = self.build_semilag_z()
        semilag_v = self.build_semilag_v()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        @jax.checkpoint
        def time_step_jax(f, t):
            f_half = semilag_z(f)
            E = self.compute_E(f_half, A_phi, A_E, B, rho_0)
            ee = self.compute_electric_energy(E)
            f = semilag_v(f_half, E, partial_B)
            f = semilag_z(f)
            rho = self.compute_rho_1d(f)
            return f, (f, E, ee, rho)

        f_final, (f_array, E_total, ee, rho) = jax.lax.scan(
            time_step_jax, f, tspan
        )
        f_array = jnp.concatenate([f_iv[None, :, :, :], f_array], axis=0)
        rho = jnp.concatenate([rho_0_1d[None, :], rho], axis=0)

        return f_final, f_array, E_total, ee, rho

    def run_forward_jax_scan_no_E(
        self, f_iv: Array, B: Array, partial_B: Array, g: Array, t_final: float
    ) -> tuple:
        """Runs the forward simulation without electric field updates.

        Args:
            f_iv (Array): The initial distribution function.
            B (Array): The magnetic field.
            partial_B (Array): The spatial derivative of the magnetic field.
            g (Array): The gradient of the magnetic field
                over the magnetic field.
            t_final (float): The final simulation time.

        Returns:
            tuple: A tuple containing the final distribution function f
            and rho over time.
        """
        num_steps = int(t_final / self.dt)
        f = f_iv.copy()

        rho_0 = self.compute_rho(B, f)
        rho_0_1d = self.compute_rho_1d(f)

        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z = self.build_semilag_z()
        semilag_v = self.build_semilag_v()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        @jax.checkpoint
        def time_step_jax(f, t):
            f_half = semilag_z(f)
            E = self.compute_E(f_half, A_phi, A_E, B, rho_0)
            f = semilag_v(f_half, jnp.zeros_like(E), partial_B)
            f = semilag_z(f)
            rho = self.compute_rho_1d(f)
            return f, rho

        f_final, (rho) = jax.lax.scan(time_step_jax, f, tspan)

        rho = jnp.concatenate([rho_0_1d[None, :], rho], axis=0)

        return f_final, rho

    def run_forward_jax_scan_efficient(
        self,
        f_iv: Array,
        B: Array,
        partial_B: Array,
        g: Array,
        t_final: float,
        chunk_size: int = 10,
    ) -> tuple:
        """Runs the forward simulation efficiently without saving full
        trajectory.

        Args:
            f_iv (Array): The initial distribution function.
            B (Array): The magnetic field.
            partial_B (Array): The spatial derivative of the magnetic field.
            g (Array): The gradient of the magnetic field
                over the magnetic field.
            t_final (float): The final simulation time.
            chunk_size (int, optional): The chunk size for scanning. 
                Defaults to 10.

        Returns:
            tuple: A tuple containing the final distribution function f
            and the final rho array.
        """
        num_steps = int(t_final / self.dt)
        f = f_iv.copy()

        rho_0 = self.compute_rho(B, f)

        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_z = self.build_semilag_z()
        semilag_v = self.build_semilag_v()

        A_phi = self.assemble_A_for_phi(g)
        A_E = self.assemble_A_for_E()

        @jax.checkpoint
        def time_step_jax(f, t):
            f_half = semilag_z(f)
            E = self.compute_E(f_half, A_phi, A_E, B, rho_0)
            f = semilag_v(f_half, E, partial_B)
            f = semilag_z(f)
            rho = self.compute_rho_1d(f)
            return f, rho

        f_final, rho_array = jax.lax.scan(time_step_jax, f, tspan)

        return f_final, rho_array[-1]
