import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

from .jax_mirror_solver import Mesh
from .jax_mirror_solver_full import MeshFull
from .utils_mirror import cost_rho

Array = jax.Array

vcost_rho = jax.vmap(cost_rho, in_axes=(0,None), out_axes=0)

#################### Single species plotting #######################

def plot_magnetic_fields(B_eval: Array, dB_eval: Array,
                         g_eval: Array, zs: Array) -> None:
    fig, axs = plt.subplots(1,3, figsize=(25,7))

    axs[0].plot(zs, B_eval)
    axs[0].set_xlabel('$z$', fontsize=20)
    axs[0].set_title('$|\\mathbf{B}(z)|$')


    axs[1].plot(zs, dB_eval)
    axs[1].set_xlabel('$z$', fontsize=20)
    axs[1].set_title('$\\partial_z |\\mathbf{B}(z)|$')


    axs[2].plot(zs, g_eval)
    axs[2].set_xlabel('$z$', fontsize=20)
    axs[2].set_title('$\\partial_z |\\mathbf{B}(z)| / |\\mathbf{B}(z)|$')

    plt.tight_layout()
    plt.show()


def plot_initial_distribution_single(f_iv: Array, mesh: Mesh) -> None:
    fig, axs = plt.subplots(1,3 ,figsize=(28, 7))

    im = axs[0].imshow(f_iv[:,:,0].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    axs[0].set_xlabel("$z$", fontsize=20)
    axs[0].set_ylabel("$v_e$", fontsize=20)
    axs[0].set_title("$f_e(0,z,v,0)$")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    im = axs[1].imshow(f_iv[:,:,int(mesh.nmu/4)].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    axs[1].set_xlabel("$z$", fontsize=20)
    #axs[0,1].set_ylabel("$v$")
    axs[1].set_title("$f_e(0,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/4)]))
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    im = axs[2].imshow(f_iv[:,:,-1].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    axs[2].set_xlabel("$z$", fontsize=20)
    #axs[2].set_ylabel("$v$")
    axs[2].set_title("$f_e(0,z,v,{:.3f})$".format(mesh.mus[-1]))
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    plt.show()


def plot_final_distribution_single_E_effect(f_array: Array,
                                            f_array_no_E: Array,
                                            mesh: Mesh) -> None:
    fig, axs = plt.subplots(2,3 ,figsize=(28, 15))

    im = axs[0,0].imshow(f_array_no_E[:,:,0].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    #axs[0,0].set_xlabel("$z$", fontsize=20)
    axs[0,0].set_ylabel("$v_e$", fontsize=20)
    axs[0,0].set_title("$f_e(T,z,v,0)$")
    fig.colorbar(im, ax=axs[0,0], fraction=0.046, pad=0.04)

    im = axs[0,1].imshow(f_array_no_E[:,:,int(mesh.nmu/16)].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    #axs[0,1].set_xlabel("$z$", fontsize=20)
    #axs[1,1].set_ylabel("$v$")
    axs[0,1].set_title("$f_e(T,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/16)]))
    fig.colorbar(im, ax=axs[0,1], fraction=0.046, pad=0.04)

    im = axs[0,2].imshow(f_array_no_E[:,:,-1].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    #axs[0,2].set_xlabel("$z$", fontsize=20)
    #axs[2].set_ylabel("$v$")
    axs[0,2].set_title("$f_e(T,z,v,{:.3f})$".format(mesh.mus[-1]))
    fig.colorbar(im, ax=axs[0,2], fraction=0.046, pad=0.04)

    im = axs[1,0].imshow(f_array[:,:,0].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    axs[1,0].set_xlabel("$z$", fontsize=20)
    axs[1,0].set_ylabel("$v_e$", fontsize=20)
    #axs[1,0].set_title("$f_e(T,z,v,0)$")
    fig.colorbar(im, ax=axs[1,0], fraction=0.046, pad=0.04)

    im = axs[1,1].imshow(f_array[:,:,int(mesh.nmu/16)].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    axs[1,1].set_xlabel("$z$", fontsize=20)
    #axs[1,1].set_ylabel("$v$")
    #axs[1,1].set_title("$f_e(T,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/2)]))
    fig.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)

    im = axs[1,2].imshow(f_array[:,:,-1].T, origin="lower",
                        extent=[mesh.zs[0], mesh.zs[-1],
                                mesh.vs[0], mesh.vs[-1]],
                        aspect='auto', cmap='plasma')
    axs[1,2].set_xlabel("$z$", fontsize=20)
    #axs[1,2].set_ylabel("$v$")
    #axs[1,2].set_title("$f_e(T,z,v,{:.3f})$".format(mesh.mus[-1]))
    fig.colorbar(im, ax=axs[1,2], fraction=0.046, pad=0.04)

    plt.show()


def plot_energies(t_values: jnp.ndarray, ee_array: jnp.ndarray,
                   E_array: jnp.ndarray, dB_eval: jnp.ndarray,
                    mesh: Mesh) -> None:
    
    fig, axs = plt.subplots(1,3 ,figsize=(28, 7))

    axs[0].plot(t_values[1:], ee_array)
    axs[0].set_title("$\\mathcal{E}(t)$")
    axs[0].set_xlabel("$t$", fontsize=20)


    axs[1].plot(mesh.zs, E_array[0,:],
                label='$E({:.1f},z)$'.format(t_values[0]))
    axs[1].plot(mesh.zs, E_array[5,:],
                label='$E({:.1f},z)$'.format(t_values[5]))
    axs[1].plot(mesh.zs, E_array[10,:],
                label='$E({:.1f},z)$'.format(t_values[10]))
    axs[1].plot(mesh.zs, E_array[50,:],
                label='$E({:.1f},z)$'.format(t_values[50]))
    axs[1].plot(mesh.zs, E_array[-1,:],
                label='$E({:.1f},z)$'.format(t_values[-1]))
    axs[1].set_title("$E(t,z)$")
    axs[1].set_xlabel("$z$", fontsize=20)
    axs[1].legend()


    axs[2].plot(mesh.zs, mesh.mus[10]*dB_eval,
                label='$\\mu = {:.3f}$'.format(mesh.mus[10]))
    axs[2].plot(mesh.zs, mesh.mus[25]*dB_eval,
                label='$\\mu = {:.3f}$'.format(mesh.mus[25]))
    axs[2].plot(mesh.zs, mesh.mus[50]*dB_eval,
                label='$\\mu = {:.3f}$'.format(mesh.mus[50]))
    axs[2].plot(mesh.zs, mesh.mus[-1]*dB_eval,
                label='$\\mu = {:.3f}$'.format(mesh.mus[-1]))
    axs[2].plot(mesh.zs, E_array[50,:], label='$E({:.1f},z)$'.format(t_values[50]),
                linestyle='--', color='black')
    axs[2].set_title("$\\mu \\partial_z|\\mathbf{B}(z)|$")
    axs[2].set_xlabel("$z$", fontsize=20)
    axs[2].legend()


    plt.show()


def plot_rhos_single(rho_array: Array, rho_array_no_E: Array,
                     B_eval: Array, t_values: Array, mesh: Mesh) -> None:
    fig, axs = plt.subplots(1,2 ,figsize=(22, 7))

    cmap = plt.get_cmap("tab10")
    line = matplotlib.lines.Line2D([0], [0], color='k', linestyle='--',  
                                   label='No $E$')

    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array[0,:], color=cmap(0),
                label='$\\rho({:.2f},z)$'.format(t_values[0]))
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array_no_E[0,:], color=cmap(0),
                linestyle='--')
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array[10,:], color=cmap(1),
                label='$\\rho({:.2f},z)$'.format(t_values[10]))
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array_no_E[10,:], color=cmap(1),
                linestyle='--')
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array[100,:], color=cmap(2),
                label='$\\rho({:.2f},z)$'.format(t_values[100]))
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array_no_E[100,:], color=cmap(2),
                linestyle='--')
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array[-1,:], color=cmap(3),
                label='$\\rho({:.2f},z)$'.format(t_values[-1]))
    axs[0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_array_no_E[-1,:], color=cmap(3),
                linestyle='--')
    axs[0].set_title("$\\rho_e(t,z)$")
    axs[0].set_xlabel("$z$", fontsize=20)
    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(line)
    axs[0].legend(handles=handles)


    axs[1].plot(mesh.zs, rho_array[0,:], color=cmap(0),
                label='$\\rho({:.2f},z)$'.format(t_values[0]))
    axs[1].plot(mesh.zs, rho_array_no_E[0,:], color=cmap(0), linestyle='--')
    axs[1].plot(mesh.zs, rho_array[10,:], color=cmap(1),
                label='$\\rho({:.2f},z)$'.format(t_values[10]))
    axs[1].plot(mesh.zs, rho_array_no_E[10,:], color=cmap(1), linestyle='--')
    axs[1].plot(mesh.zs, rho_array[100,:], color=cmap(2),
                label='$\\rho({:.2f},z)$'.format(t_values[100]))
    axs[1].plot(mesh.zs, rho_array_no_E[100,:], color=cmap(2), linestyle='--')
    axs[1].plot(mesh.zs, rho_array[-1,:], color=cmap(3),
                label='$\\rho({:.2f},z)$'.format(t_values[-1]))
    axs[1].plot(mesh.zs, rho_array_no_E[-1,:], color=cmap(3), linestyle='--')
    axs[1].set_title("$\\rho_e^{1D}(t,z)$")
    axs[1].set_xlabel("$z$", fontsize=20)
    handles, labels = axs[1].get_legend_handles_labels()
    handles.append(line)
    axs[1].legend(handles=handles)

    plt.show()
    

def plot_int_rho_single(rho_array: Array, rho_array_no_E:Array,
                        t_values:Array, LZ:float, LV: float,
                        prop_trapped: float, mesh: Mesh) -> None:
    fig, ax = plt.subplots(1,1 ,figsize=(9, 7))

    ax.plot(t_values, vcost_rho(rho_array, mesh.zs), label='With $E$')
    ax.plot(t_values, vcost_rho(rho_array_no_E, mesh.zs), label='Without $E$')
    ax.set_title("$\\int \\rho_e^{1D}(t,z)\\mathrm{d}z$")
    ax.set_xlabel("$t$", fontsize=20)
    ax.hlines(y=prop_trapped, xmin=0, xmax=t_values[-1], colors='k',
              linestyles='--', label='Trapped fraction')
    ax.vlines(x=LZ/LV, ymin=0.94, ymax=1.0, colors='r', linestyles='--',
              label='Minimum time before leak')
    ax.legend()

    plt.show()