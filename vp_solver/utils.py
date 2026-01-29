from typing import Callable

import jax
import jax.numpy as jnp

import matplotlib
from matplotlib import figure, axes
import matplotlib.pyplot as plt

from .jax_vp_solver import Mesh, VlasovPoissonSolver


matplotlib.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})


Array = jax.Array


# ===== Auxiliary Functions =====

def external_electric_field(
    ak: Array,
    mesh: Mesh,
    k_0: float
) -> Array:
    """
    Compute the external electric field H(x):

        H(x; a_k, b_k) = ∑_k a_k cos(k₀ k x) + b_k sin(k₀ k x).
    """
    N = ak.shape[1]
    k = jnp.arange(1, N + 1)
    cos_term = ak[0, :] @ jnp.cos(k_0 * k[:, None] * mesh.xs)
    sin_term = ak[1, :] @ jnp.sin(k_0 * k[:, None] * mesh.xs)
    return cos_term + sin_term


# ===== Cost Functions =====

# --- KL Divergence ---

def kl_divergence(
    f_T: Array,
    solver: VlasovPoissonSolver,
    eps: float = 1e-12
) -> Array:
    """
    Compute the KL divergence:

        KL(f_T || f_eq) = ∫∫ f_T log(f_T / f_eq) dx dv.
    """
    dx_dv = solver.mesh.dx * solver.mesh.dv
    norm_final = jnp.sum(f_T) * dx_dv + eps
    norm_eq = jnp.sum(solver.f_eq) * dx_dv + eps

    f_final = f_T / norm_final
    f_eq = solver.f_eq / norm_eq

    kl_div = jnp.sum(
        jax.scipy.special.rel_entr(f_final, f_eq + eps) * dx_dv
    )
    return kl_div


def make_cost_function_kl(
    solver: VlasovPoissonSolver,
    solver_jit: Callable[
        [Array, Array, float], tuple[Array, Array, Array]
    ],
    f_iv: Array,
    k_0: float,
    t_final: float
) -> Callable[[Array], Array]:
    """Generate KL cost function."""
    @jax.jit
    def cost_function_kl(a_k: Array) -> Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, _, _ = solver_jit(f_iv, H, t_final)
        return kl_divergence(f_array, solver)

    return cost_function_kl


# --- Final Electric Energy ---

def make_cost_function_ee(
    solver: VlasovPoissonSolver,
    solver_jit: Callable[
        [Array, Array, float], tuple[Array, Array, Array]
    ],
    f_iv: Array,
    k_0: float,
    t_final: float
) -> Callable[[Array], Array]:
    """
    Generate electric energy cost function.

        EE = ∫ [E(T, x)]² dx
    """
    @jax.jit
    def cost_function_ee(a_k: Array) -> Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        _, _, ee_array = solver_jit(f_iv, H, t_final)
        return ee_array[-1]

    return cost_function_ee


# --- Total Electric Energy Over Time ---

def electric_energy_in_time(
    ee_array: Array,
    solver: VlasovPoissonSolver
) -> Array:
    """
    Compute the total electric energy over time:

        EE = ∫∫ [E(t, x)]² dx dt
    """
    return jnp.sum(ee_array) * solver.dt


def make_cost_function_eet(
    solver: VlasovPoissonSolver,
    solver_jit: Callable[
        [Array, Array, float], tuple[Array, Array, Array]
    ],
    f_iv: Array,
    k_0: float,
    t_final: float
) -> Callable[[Array], Array]:
    """Generate total-time electric energy cost function."""
    @jax.jit
    def cost_function_eet(a_k: Array) -> Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        _, _, ee_array = solver_jit(f_iv, H, t_final)
        return electric_energy_in_time(ee_array, solver)

    return cost_function_eet



# ===== Plotting Functions =====

def plot_feq_distribution(
    fig: figure.Figure,
    ax: axes.Axes,
    f_eq: Array,
    title: str,
    mesh: Mesh,
    sci: bool = False
) -> None:
    """
    Plot the equilibrium distribution f_eq(x, v).
    """
    im = ax.imshow(
        f_eq.T,
        extent=(float(mesh.xs[0]), float(mesh.xs[-1]),
                float(mesh.vs[0]), float(mesh.vs[-1])),
        aspect="auto",
        cmap="plasma",
    )
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale("log")


def plot_distribution(
    fig: figure.Figure,
    ax: axes.Axes,
    data: Array,
    title: str,
    time: float,
    mesh: Mesh,
    sci: bool = False
) -> None:
    """
    Plot the plasma distribution f(x, v) at a given time.
    """
    im = ax.imshow(
        data.T,
        extent=(float(mesh.xs[0]),float(mesh.xs[-1]),
                float(mesh.vs[0]), float(mesh.vs[-1])),
        aspect="auto",
        cmap="plasma",
    )
    ax.set_title(f"{title} (T={time:.0f})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale("log")


def plot_inital_solve(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_eq: Array,
    f_array_1: Array,
    ee_array_1: Array,
    f_array_2: Array,
    ee_array_2: Array,
    mesh: Mesh,
    t_values: Array,
    sci: bool = False,
) -> None:
    """
    Plot solution comparison between H ≡ 0 and a good initial guess.
    """
    plot_feq_distribution(
        fig, axs[0], f_eq, "Distribution of $f_{eq}$", mesh, sci
    )
    plot_distribution(
        fig, axs[1], f_array_1,
        "Distribution of $f[H\\equiv 0]$",
        float(t_values[-1]), mesh, sci
    )
    plot_distribution(
        fig, axs[2], f_array_2,
        "Distribution of $f[H]$",
        float(t_values[-1]), mesh, sci
    )

    axs[3].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[3].plot(t_values, ee_array_1, label="No $H$")
    axs[3].plot(t_values, ee_array_2, label="Good initial $H$")
    axs[3].set_xlabel("$t$")
    axs[3].set_title("$\\mathcal{E}_{f}(t)$")
    axs[3].legend()


def plot_results_TS(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_final: Array,
    E_array: Array,
    H: Array,
    ee_array: Array,
    objective_values: Array,
    t_values: Array,
    mesh: Mesh,
) -> None:
    """
    Plot results of optimization for Two-Stream equilibrium.
    """
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, "Distribution of $f[H]$",
        float(t_values[-1]), mesh
    )

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label="$H(x)$")
    axs[1].plot(mesh.xs, E_array[0] - H, label=f"$E(t={0*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[99] - H, label=f"$E(t={100*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[199] - H, label=f"$E(t={200*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[299] - H, label=f"$E(t={300*dt:.0f},x)$")
    axs[1].set_xlabel("$x$")
    axs[1].set_title("Electric fields")
    axs[1].legend(loc="upper right")

    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[2].plot(t_values, ee_array)
    axs[2].set_xlabel("$t$")
    axs[2].set_title("$\\mathcal{E}_{f}(t)$")

    axs[3].plot(objective_values)
    axs[3].set_yscale("log")
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")


def plot_results_BoT(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_final: Array,
    E_array: Array,
    H: Array,
    ee_array: Array,
    objective_values: Array,
    t_values: Array,
    mesh: Mesh,
) -> None:
    """
    Plot results of optimization for Bump-on-Tail equilibrium.
    """
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, "Distribution of $f[H]$",
        float(t_values[-1]), mesh
    )

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label="$H(x)$")
    axs[1].plot(mesh.xs, E_array[0] - H, label=f"$E(t={0*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[199] - H, label=f"$E(t={200*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[299] - H, label=f"$E(t={300*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[399] - H, label=f"$E(t={400*dt:.0f},x)$")
    axs[1].set_xlabel("$x$")
    axs[1].set_title("Electric fields")
    axs[1].legend(loc="upper right")

    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[2].plot(t_values, ee_array)
    axs[2].set_xlabel("$t$")
    axs[2].set_title("$\\mathcal{E}_{f}(t)$")

    axs[3].plot(objective_values)
    axs[3].set_yscale("log")
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")


# Functional interface for hermite splines (recommended for use in JIT)
def cubic_hermite_spline_eval(xi, x_knots, y_knots, dydx_knots):
    """
    Evaluate cubic Hermite spline (fully JIT-compatible).
    
    Parameters:
    -----------
    xi : array
        Evaluation points
    x_knots : array
        Knot x-coordinates (must be sorted)
    y_knots : array
        Function values at knots
    dydx_knots : array
        Derivative values at knots
    derivative_order : int
        0 for function, 1 for first derivative, 2 for second derivative
    
    Returns:
    --------
    yi : array
        Evaluated values
    """
    #xi = jnp.atleast_1d(xi)
    n = len(x_knots)
    
    # Find segments
    i = jnp.searchsorted(x_knots, xi, side='right') - 1
    i = jnp.clip(i, 0, n - 2)
    
    # Get segment data
    x0 = x_knots[i]
    x1 = x_knots[i + 1]
    y0 = y_knots[i]
    y1 = y_knots[i + 1]
    m0 = dydx_knots[i]
    m1 = dydx_knots[i + 1]
    
    # Normalized position
    dx = x1 - x0
    t = (xi - x0) / dx
    
    # Hermite basis functions
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)
    
    yi = h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1
    
    return yi

def cubic_hermite_spline_eval_1(xi, x_knots, y_knots, dydx_knots):
    """
    Evaluate first derivative of cubic Hermite spline (fully JIT-compatible).
    
    Parameters:
    -----------
    xi : array
        Evaluation points
    x_knots : array
        Knot x-coordinates (must be sorted)
    y_knots : array
        Function values at knots
    dydx_knots : array
        Derivative values at knots
    derivative_order : int
        0 for function, 1 for first derivative, 2 for second derivative
    
    Returns:
    --------
    yi : array
        Evaluated values
    """
    #xi = jnp.atleast_1d(xi)
    n = len(x_knots)
    
    # Find segments
    i = jnp.searchsorted(x_knots, xi, side='right') - 1
    i = jnp.clip(i, 0, n - 2)
    
    # Get segment data
    x0 = x_knots[i]
    x1 = x_knots[i + 1]
    y0 = y_knots[i]
    y1 = y_knots[i + 1]
    m0 = dydx_knots[i]
    m1 = dydx_knots[i + 1]
    
    # Normalized position
    dx = x1 - x0
    t = (xi - x0) / dx
    
    # Hermite basis functions
    # First derivatives
    dh00_dt = 6*t**2 - 6*t
    dh10_dt = 3*t**2 - 4*t + 1
    dh01_dt = -6*t**2 + 6*t
    dh11_dt = 3*t**2 - 2*t
        
    yi = (dh00_dt * y0 + dh10_dt * dx * m0 + 
          dh01_dt * y1 + dh11_dt * dx * m1) / dx
    
    return yi

def cubic_hermite_spline_eval_2(xi, x_knots, y_knots, dydx_knots):
    """
    Evaluate second derivative of cubic Hermite spline (fully JIT-compatible).
    
    Parameters:
    -----------
    xi : array
        Evaluation points
    x_knots : array
        Knot x-coordinates (must be sorted)
    y_knots : array
        Function values at knots
    dydx_knots : array
        Derivative values at knots
    derivative_order : int
        0 for function, 1 for first derivative, 2 for second derivative
    
    Returns:
    --------
    yi : array
        Evaluated values
    """
    #xi = jnp.atleast_1d(xi)
    n = len(x_knots)
    
    # Find segments
    i = jnp.searchsorted(x_knots, xi, side='right') - 1
    i = jnp.clip(i, 0, n - 2)
    
    # Get segment data
    x0 = x_knots[i]
    x1 = x_knots[i + 1]
    y0 = y_knots[i]
    y1 = y_knots[i + 1]
    m0 = dydx_knots[i]
    m1 = dydx_knots[i + 1]
    
    # Normalized position
    dx = x1 - x0
    t = (xi - x0) / dx
    
    # Hermite basis functions
    # Second derivatives
    d2h00_dt2 = 12*t - 6
    d2h10_dt2 = 6*t - 4
    d2h01_dt2 = -12*t + 6
    d2h11_dt2 = 6*t - 2
        
    yi = (d2h00_dt2 * y0 + d2h10_dt2 * dx * m0 + 
            d2h01_dt2 * y1 + d2h11_dt2 * dx * m1) / dx**2
    
    return yi



def build_B(B_min, B_max, z_min, z_max, z_b,
            B_b, z_0):
    """
    Build magnetic field B(z) and its partial derivative with respect to parameters
    using cubic Hermite splines with zero derivatives at critical points.
    """
    
    def B_1(params, z):
        """
        Build magnetic field B(z) from parameters
        when z_0 == z_min.
        """
        n_params = params.shape[0]
        
        # Generate interior z locations (between z_0 and z_max, excluding z_max)
        z_interior = jnp.linspace(z_0, z_max, num=n_params, endpoint=False)
        
        # Build z array with all knots (may contain duplicates)
        z_sorted = jnp.concatenate([
            z_interior,
            jnp.array([z_max, z_b])
        ])
        
        # Build corresponding B array
        B_sorted = jnp.concatenate([
            params,
            jnp.array([B_max, B_b])
        ])

        # Find indices of critical points in final array
        idx_min = jnp.argmin(jnp.abs(z_sorted - z_min))
        idx_max = jnp.argmin(jnp.abs(z_sorted - z_max))

        # Initialize derivative array
        n = len(B_sorted)
        indices = jnp.arange(n)
        
        # Compute all possible derivatives
        dz_forward = z_sorted[1:] - z_sorted[:-1]
        dB_forward = B_sorted[1:] - B_sorted[:-1]
        
        # Central differences
        dB_central = jnp.zeros(n)
        dB_central = dB_central.at[1:-1].set(
            (B_sorted[2:] - B_sorted[:-2]) / (z_sorted[2:] - z_sorted[:-2])
        )
        
        # Forward difference for first point
        dB_first = dB_forward[0] / dz_forward[0]
        
        # Backward difference for last point
        dB_last = dB_forward[-1] / dz_forward[-1]
        
        # Assemble derivatives
        dB_knots = jnp.where(
            indices == 0,
            dB_first,
            jnp.where(
                indices == n - 1,
                dB_last,
                dB_central
            )
        )
        
        # Enforce zero derivatives at critical points and neighbors
        is_critical = (indices == idx_min) | (indices == idx_max)
        is_before_min = (indices == idx_min - 1) & (idx_min > 0)
        is_after_max = (indices == idx_max + 1) & (idx_max < n - 1)
        
        dB_knots = jnp.where(is_critical | is_before_min | is_after_max, 0.0, dB_knots)

        z_eval = jnp.clip(jnp.abs(z), 0, z_b)
        return cubic_hermite_spline_eval(z_eval, z_sorted, B_sorted, dB_knots)
    
    def partial_B_1(params, z):
        """
        Build partial derivative of B(z) with respect to z
        when z_0 == z_min.
        """
        n_params = params.shape[0]
        
        # Generate interior z locations (between z_0 and z_max, excluding z_max)
        z_interior= jnp.linspace(z_0, z_max, num=n_params, endpoint=False)
        
        # Build z array with all knots (may contain duplicates)
        z_sorted = jnp.concatenate([
            z_interior,
            jnp.array([z_max, z_b])
        ])
        
        # Build corresponding B array
        B_sorted = jnp.concatenate([
            params,
            jnp.array([B_max, B_b])
        ])

        # Find indices of critical points in final array
        idx_min = jnp.argmin(jnp.abs(z_sorted - z_min))
        idx_max = jnp.argmin(jnp.abs(z_sorted - z_max))

        # Initialize derivative array
        n = len(B_sorted)
        indices = jnp.arange(n)
        
        # Compute all possible derivatives
        dz_forward = z_sorted[1:] - z_sorted[:-1]
        dB_forward = B_sorted[1:] - B_sorted[:-1]
        
        # Central differences
        dB_central = jnp.zeros(n)
        dB_central = dB_central.at[1:-1].set(
            (B_sorted[2:] - B_sorted[:-2]) / (z_sorted[2:] - z_sorted[:-2])
        )
        
        # Forward difference for first point
        dB_first = dB_forward[0] / dz_forward[0]
        
        # Backward difference for last point
        dB_last = dB_forward[-1] / dz_forward[-1]
        
        # Assemble derivatives
        dB_knots = jnp.where(
            indices == 0,
            dB_first,
            jnp.where(
                indices == n - 1,
                dB_last,
                dB_central
            )
        )
        
        # Enforce zero derivatives at critical points and neighbors
        is_critical = (indices == idx_min) | (indices == idx_max)
        is_before_min = (indices == idx_min - 1) & (idx_min > 0)
        is_after_max = (indices == idx_max + 1) & (idx_max < n - 1)
        
        dB_knots = jnp.where(is_critical | is_before_min | is_after_max, 0.0, dB_knots)

        sign = jnp.sign(z)
        sign = jnp.where(z == 0, 0, sign)
        z_eval = jnp.clip(jnp.abs(z), 0, z_b)
        return sign * cubic_hermite_spline_eval_1(z_eval, z_sorted, B_sorted, dB_knots)

    def B_2(params, z):
        """
        Build magnetic field B(z) from parameters
        when z_0 != z_min.
        """
        n_params = params.shape[0]
        
        # Generate interior z locations (between z_0 and z_max, excluding z_max)
        z_interior = jnp.linspace(z_0, z_max, num=n_params, endpoint=False)
        
        # Build z array with all knots (may contain duplicates)
        z_unsorted = jnp.concatenate([
            jnp.array([z_min]),
            z_interior,
            jnp.array([z_max, z_b])
        ])
        
        # Build corresponding B array
        B_unsorted = jnp.concatenate([
            jnp.array([B_min]),
            params,
            jnp.array([B_max, B_b])
        ])
        
        # Sort both arrays by z values
        sort_indices = jnp.argsort(z_unsorted)
        z_sorted = z_unsorted[sort_indices]
        B_sorted = B_unsorted[sort_indices]

        # Find indices of critical points in final array
        idx_min = jnp.argmin(jnp.abs(z_sorted - z_min))
        idx_max = jnp.argmin(jnp.abs(z_sorted - z_max))

        # Initialize derivative array
        n = len(B_sorted)
        indices = jnp.arange(n)
        
        # Compute all possible derivatives
        dz_forward = z_sorted[1:] - z_sorted[:-1]
        dB_forward = B_sorted[1:] - B_sorted[:-1]
        
        # Central differences
        dB_central = jnp.zeros(n)
        dB_central = dB_central.at[1:-1].set(
            (B_sorted[2:] - B_sorted[:-2]) / (z_sorted[2:] - z_sorted[:-2])
        )
        
        # Forward difference for first point
        dB_first = dB_forward[0] / dz_forward[0]
        
        # Backward difference for last point
        dB_last = dB_forward[-1] / dz_forward[-1]
        
        # Assemble derivatives
        dB_knots = jnp.where(
            indices == 0,
            dB_first,
            jnp.where(
                indices == n - 1,
                dB_last,
                dB_central
            )
        )
        
        # Enforce zero derivatives at critical points and neighbors
        is_critical = (indices == idx_min) | (indices == idx_max)
        is_before_min = (indices == idx_min - 1) & (idx_min > 0)
        is_after_max = (indices == idx_max + 1) & (idx_max < n - 1)
        
        dB_knots = jnp.where(is_critical | is_before_min | is_after_max, 0.0, dB_knots)

        z_eval = jnp.clip(jnp.abs(z), 0, z_b)
        return cubic_hermite_spline_eval(z_eval, z_sorted, B_sorted, dB_knots)
    
    def partial_B_2(params, z):
        """
        Build partial derivative of B(z) with respect to z
        when z_0 != z_min.
        """
        n_params = params.shape[0]
        
        # Generate interior z locations (between z_0 and z_max, excluding z_max)
        z_interior = jnp.linspace(z_0, z_max, num=n_params, endpoint=False)
        
        # Build z array with all knots (may contain duplicates)
        z_unsorted = jnp.concatenate([
            jnp.array([z_min]),
            z_interior,
            jnp.array([z_max, z_b])
        ])
        
        # Build corresponding B array
        B_unsorted = jnp.concatenate([
            jnp.array([B_min]),
            params,
            jnp.array([B_max, B_b])
        ])
        
        # Sort both arrays by z values
        sort_indices = jnp.argsort(z_unsorted)
        z_sorted = z_unsorted[sort_indices]
        B_sorted = B_unsorted[sort_indices]

        # Find indices of critical points in final array
        idx_min = jnp.argmin(jnp.abs(z_sorted - z_min))
        idx_max = jnp.argmin(jnp.abs(z_sorted - z_max))

        # Initialize derivative array
        n = len(B_sorted)
        indices = jnp.arange(n)
        
        # Compute all possible derivatives
        dz_forward = z_sorted[1:] - z_sorted[:-1]
        dB_forward = B_sorted[1:] - B_sorted[:-1]
        
        # Central differences
        dB_central = jnp.zeros(n)
        dB_central = dB_central.at[1:-1].set(
            (B_sorted[2:] - B_sorted[:-2]) / (z_sorted[2:] - z_sorted[:-2])
        )
        
        # Forward difference for first point
        dB_first = dB_forward[0] / dz_forward[0]
        
        # Backward difference for last point
        dB_last = dB_forward[-1] / dz_forward[-1]
        
        # Assemble derivatives
        dB_knots = jnp.where(
            indices == 0,
            dB_first,
            jnp.where(
                indices == n - 1,
                dB_last,
                dB_central
            )
        )
        
        # Enforce zero derivatives at critical points and neighbors
        is_critical = (indices == idx_min) | (indices == idx_max)
        is_before_min = (indices == idx_min - 1) & (idx_min > 0)
        is_after_max = (indices == idx_max + 1) & (idx_max < n - 1)
        
        dB_knots = jnp.where(is_critical | is_before_min | is_after_max, 0.0, dB_knots)

        sign = jnp.sign(z)
        sign = jnp.where(z == 0, 0, sign)
        z_eval = jnp.clip(jnp.abs(z), 0, z_b)
        return sign * cubic_hermite_spline_eval_1(z_eval, z_sorted, B_sorted, dB_knots)


    if z_0 == z_min:
        return B_1, partial_B_1
    else:
        return B_2, partial_B_2