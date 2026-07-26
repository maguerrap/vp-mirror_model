import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

from .jax_mirror_solver import Mesh
from .jax_mirror_solver_full import MeshFull
from .utils_mirror import cost_rho

Array = jax.Array

vcost_rho = jax.vmap(cost_rho, in_axes=(0,None), out_axes=0)

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


#################### Single species plotting #######################


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


def plot_energies_single(t_values: jnp.ndarray, ee_array: jnp.ndarray,
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


#################### Multi species plotting #######################

def plot_initial_distribution_multi(f_iv_e: Array, f_iv_i: Array, 
                                  mesh: MeshFull) -> None:
    fig, axs = plt.subplots(2,3 ,figsize=(28, 15))
    
    im = axs[0,0].imshow(f_iv_e[:,:,0].T, origin="lower", 
                        extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_e[0], mesh.vs_e[-1]],
                        aspect='auto', cmap='plasma')
    #axs[0,0].set_xlabel("$z$", fontsize=20)
    axs[0,0].set_ylabel("$v_e$", fontsize=20)
    axs[0,0].set_title("$f_e(0,z,v,0)$")
    fig.colorbar(im, ax=axs[0,0], fraction=0.046, pad=0.04)

    im = axs[0,1].imshow(f_iv_e[:,:,int(mesh.nmu/2)].T, origin="lower", extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_e[0], mesh.vs_e[-1]],
                        aspect='auto', cmap='plasma')
    #axs[0,1].set_xlabel("$z$", fontsize=20)
    #axs[0,1].set_ylabel("$v$")
    axs[0,1].set_title("$f_e(0,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/2)]))
    fig.colorbar(im, ax=axs[0,1], fraction=0.046, pad=0.04)

    im = axs[0,2].imshow(f_iv_e[:,:,-1].T, origin="lower", extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_e[0], mesh.vs_e[-1]],
                        aspect='auto', cmap='plasma')
    #axs[0,2].set_xlabel("$z$", fontsize=20)
    #axs[0,2].set_ylabel("$v$")
    axs[0,2].set_title("$f_e(0,z,v,{:.3f})$".format(mesh.mus[-1]))
    fig.colorbar(im, ax=axs[0,2], fraction=0.046, pad=0.04)


    im = axs[1,0].imshow(f_iv_i[:,:,0].T, origin="lower", extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_i[0], mesh.vs_i[-1]],
                        aspect='auto', cmap='plasma')
    axs[1,0].set_xlabel("$z$", fontsize=20)
    axs[1,0].set_ylabel("$v_i$", fontsize=20)
    axs[1,0].set_title("$f_i(0,z,v,0)$")
    fig.colorbar(im, ax=axs[1,0], fraction=0.046, pad=0.04)

    im = axs[1,1].imshow(f_iv_i[:,:,int(mesh.nmu/2)].T, origin="lower", extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_i[0], mesh.vs_i[-1]],
                        aspect='auto', cmap='plasma')
    axs[1,1].set_xlabel("$z$", fontsize=20)
    #axs[1,1].set_ylabel("$v_i$")
    axs[1,1].set_title("$f_i(0,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/2)]))
    fig.colorbar(im, ax=axs[1,1], fraction=0.046, pad=0.04)

    im = axs[1,2].imshow(f_iv_i[:,:,-1].T, origin="lower", extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_i[0], mesh.vs_i[-1]],
                        aspect='auto', cmap='plasma')
    axs[1,2].set_xlabel("$z$", fontsize=20)
    #axs[1,2].set_ylabel("$\\mu$")
    axs[1,2].set_title("$f_i(0,z,v,{:.3f})$".format(mesh.mus[-1]))
    fig.colorbar(im, ax=axs[1,2], fraction=0.046, pad=0.04)

    plt.show()


def normalize_final_E_effect_multi(f_e_array: Array, f_e_array_no_E: Array,
                                   f_i_array: Array, f_i_array_no_E: Array,
                                   mesh: MeshFull) -> tuple:

	all_data_il = jnp.concatenate([
		f_i_array_no_E[:,:,0].ravel(),
		f_i_array[:,:,0].ravel(),
	])

	vmin_il = all_data_il.min()
	vmax_il = all_data_il.max()

	norm_il = matplotlib.colors.Normalize(vmin=vmin_il, vmax=vmax_il)

	all_data_ic = jnp.concatenate([
		f_i_array_no_E[:,:,int(mesh.nmu/16)].ravel(),
		f_i_array[:,:,int(mesh.nmu/16)].ravel(),
	])

	vmin_ic = all_data_ic.min()
	vmax_ic = all_data_ic.max()

	norm_ic = matplotlib.colors.Normalize(vmin=vmin_ic, vmax=vmax_ic)

	all_data_ir = jnp.concatenate([
		f_i_array_no_E[:,:,-1].ravel(),
		f_i_array[:,:,-1].ravel(),
	])

	vmin_ir = all_data_ir.min()
	vmax_ir = all_data_ir.max()

	norm_ir = matplotlib.colors.Normalize(vmin=vmin_ir, vmax=vmax_ir)

	all_data_el = jnp.concatenate([
		f_e_array_no_E[:,:,0].ravel(),
		f_e_array[:,:,0].ravel(),
	])

	vmin_el = all_data_el.min()
	vmax_el = all_data_el.max()

	norm_el = matplotlib.colors.Normalize(vmin=vmin_el, vmax=vmax_el)

	all_data_ec = jnp.concatenate([
		f_e_array_no_E[:,:,int(mesh.nmu/16)].ravel(),
		f_e_array[:,:,int(mesh.nmu/16)].ravel(),
	])

	vmin_ec = all_data_ec.min()
	vmax_ec = all_data_ec.max()

	norm_ec = matplotlib.colors.Normalize(vmin=vmin_ec, vmax=vmax_ec)

	all_data_er = jnp.concatenate([
		f_e_array_no_E[:,:,-1].ravel(),
		f_e_array[:,:,-1].ravel(),
	])

	vmin_er = all_data_er.min()
	vmax_er = all_data_er.max()

	norm_er = matplotlib.colors.Normalize(vmin=vmin_er, vmax=vmax_er)

	return norm_il, norm_ic, norm_ir, norm_el, norm_ec, norm_er 


def plot_final_distribution_multi(f_e_array: Array, f_i_array: Array,
									   norm_il, norm_ic, norm_ir, norm_el,
									   norm_ec, norm_er,
									   mesh: MeshFull) -> None:

	fig, axs = plt.subplots(2,3 ,figsize=(28, 15))

	im_el = axs[0,0].imshow(f_e_array[:,:,0].T, origin="lower",
						extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_e[0], mesh.vs_e[-1]],
					aspect='auto', cmap='plasma', norm=norm_el)
	#axs[0,0].set_xlabel("$z$", fontsize=20)
	axs[0,0].set_ylabel("$v_e$", fontsize=20)
	axs[0,0].set_title("$f_e(T,z,v,0)$")

	im_ec = axs[0,1].imshow(f_e_array[:,:,int(mesh.nmu/16)].T, origin="lower", 
					extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_e[0], mesh.vs_e[-1]],
					aspect='auto', cmap='plasma', norm=norm_ec)
	#axs[0,1].set_xlabel("$z$", fontsize=20)
	#axs[0,1].set_ylabel("$v$")
	axs[0,1].set_title("$f_e(T,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/16)]))

	im_er = axs[0,2].imshow(f_e_array[:,:,-1].T, origin="lower",
					extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_e[0], mesh.vs_e[-1]],
					aspect='auto', cmap='plasma', norm=norm_er)
	#axs[0,2].set_xlabel("$z$", fontsize=20)
	#axs[0,2].set_ylabel("$v$")
	axs[0,2].set_title("$f_e(T,z,v,{:.3f})$".format(mesh.mus[-1]))


	im_il = axs[1,0].imshow(f_i_array[:,:,0].T, origin="lower",
					extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_i[0], mesh.vs_i[-1]],
					aspect='auto', cmap='plasma', norm=norm_il)
	axs[1,0].set_xlabel("$z$", fontsize=20)
	axs[1,0].set_ylabel("$v_i$", fontsize=20)
	axs[1,0].set_title("$f_i(T,z,v,0)$")

	im_ic = axs[1,1].imshow(f_i_array[:,:,int(mesh.nmu/16)].T, origin="lower",
					extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_i[0], mesh.vs_i[-1]],
					aspect='auto', cmap='plasma', norm=norm_ic)
	axs[1,1].set_xlabel("$z$", fontsize=20)
	#axs[1,1].set_ylabel("$v$")
	axs[1,1].set_title("$f_i(T,z,v,{:.3f})$".format(mesh.mus[int(mesh.nmu/16)]))

	im_ir = axs[1,2].imshow(f_i_array[:,:,-1].T, origin="lower",
					extent=[mesh.zs[0], mesh.zs[-1], mesh.vs_i[0], mesh.vs_i[-1]],
					aspect='auto', cmap='plasma', norm=norm_ir)
	axs[1,2].set_xlabel("$z$", fontsize=20)
	#axs[1,2].set_ylabel("$v$")
	axs[1,2].set_title("$f_i(T,z,v,{:.3f})$".format(mesh.mus[-1]))


	fig.colorbar(im_el, ax=axs[0,0], shrink=1.0, fraction=0.046, pad=0.04)
	fig.colorbar(im_ec, ax=axs[0,1], shrink=1.0, fraction=0.046, pad=0.04)
	fig.colorbar(im_er, ax=axs[0,2], shrink=1.0, fraction=0.046, pad=0.04)

	fig.colorbar(im_il, ax=axs[1,0], shrink=1.0, fraction=0.046, pad=0.04)
	fig.colorbar(im_ic, ax=axs[1,1], shrink=1.0, fraction=0.046, pad=0.04)
	fig.colorbar(im_ir, ax=axs[1,2], shrink=1.0, fraction=0.046, pad=0.04)

	plt.show()


def plot_energies_multi(t_values: Array, ee_array: Array, E_total_array: Array,
                        dB_eval: Array, mesh: MeshFull) -> None:

	fig, axs = plt.subplots(1,3 ,figsize=(28, 7))

	axs[0].plot(t_values[1:], ee_array)
	axs[0].set_title("$\\mathcal{E}(t)$")
	axs[0].set_xlabel("$t$", fontsize=20)


	axs[1].plot(mesh.zs, E_total_array[0,:], label='$E({:.1f},z)$'.format(t_values[0]))
	axs[1].plot(mesh.zs, E_total_array[40,:], label='$E({:.1f},z)$'.format(t_values[40]))
	axs[1].plot(mesh.zs, E_total_array[80,:], label='$E({:.1f},z)$'.format(t_values[80]))
	axs[1].plot(mesh.zs, E_total_array[150,:], label='$E({:.1f},z)$'.format(t_values[150]))
	axs[1].plot(mesh.zs, E_total_array[500,:], label='$E({:.1f},z)$'.format(t_values[500]))
	axs[1].plot(mesh.zs, E_total_array[-1,:], label='$E({:.1f},z)$'.format(t_values[-1]))
	axs[1].set_title("$E(t,z)$")
	axs[1].set_xlabel("$z$", fontsize=20)
	axs[1].legend()


	axs[2].plot(mesh.zs, mesh.mus[10]*dB_eval, label='$\\mu = {:.3f}$'.format(mesh.mus[10]))
	axs[2].plot(mesh.zs, mesh.mus[25]*dB_eval, label='$\\mu = {:.3f}$'.format(mesh.mus[25]))
	axs[2].plot(mesh.zs, mesh.mus[50]*dB_eval, label='$\\mu = {:.3f}$'.format(mesh.mus[50]))
	axs[2].plot(mesh.zs, mesh.mus[-1]*dB_eval, label='$\\mu = {:.3f}$'.format(mesh.mus[-1]))
	axs[2].plot(mesh.zs, E_total_array[40,:], label='$E({:.1f},z)$'.format(t_values[50]), linestyle='--', color='black')
	axs[2].set_title("$\\mu \\partial_z|\\mathbf{B}(z)|$")
	axs[2].set_xlabel("$z$", fontsize=20)
	axs[2].legend()

	plt.show()


def plot_rhos_multi(rho_e_array: Array,
					rho_e_array_no_E: Array,
					rho_i_array: Array,
					rho_i_array_no_E: Array,
					B_eval: Array,
					t_values: Array,
					mesh: MeshFull) -> None:

	fig, axs = plt.subplots(2,2 ,figsize=(22, 15))

	cmap = plt.get_cmap("tab10")
	line = matplotlib.lines.Line2D([0], [0], color='k', linestyle='--', label='No $E$')

	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array[0,:], color=cmap(0), label='$\\rho({:.2f},z)$'.format(t_values[0]))
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array_no_E[0,:], color=cmap(0), linestyle='--')
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array[10,:], color=cmap(1), label='$\\rho({:.2f},z)$'.format(t_values[10]))
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array_no_E[10,:], color=cmap(1), linestyle='--')
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array[100,:], color=cmap(2), label='$\\rho({:.2f},z)$'.format(t_values[100]))
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array_no_E[100,:], color=cmap(2), linestyle='--')
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array[250,:], color=cmap(3), label='$\\rho({:.2f},z)$'.format(t_values[250]))
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array_no_E[250,:], color=cmap(3), linestyle='--')
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array[-1,:], color=cmap(4), label='$\\rho({:.2f},z)$'.format(t_values[-1]))
	axs[0,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_e_array_no_E[-1,:], color=cmap(4), linestyle='--')
	axs[0,0].set_title("$\\rho_e(t,z)$", fontsize=20)
	axs[0,0].set_xlabel("$z$", fontsize=20)
	handles, labels = axs[0,0].get_legend_handles_labels()
	handles.append(line)
	#axs[0,0].set_yscale('log')
	axs[0,0].legend(handles=handles)

	axs[0,1].plot(mesh.zs, rho_e_array[0,:], color=cmap(0), label='$\\rho({:.2f},z)$'.format(t_values[0]))
	axs[0,1].plot(mesh.zs, rho_e_array_no_E[0,:], color=cmap(0), linestyle='--')
	axs[0,1].plot(mesh.zs, rho_e_array[10,:], color=cmap(1), label='$\\rho({:.2f},z)$'.format(t_values[10]))
	axs[0,1].plot(mesh.zs, rho_e_array_no_E[10,:], color=cmap(1), linestyle='--')
	axs[0,1].plot(mesh.zs, rho_e_array[100,:], color=cmap(2), label='$\\rho({:.2f},z)$'.format(t_values[100]))
	axs[0,1].plot(mesh.zs, rho_e_array_no_E[100,:], color=cmap(2), linestyle='--')
	axs[0,1].plot(mesh.zs, rho_e_array[250,:], color=cmap(3), label='$\\rho({:.2f},z)$'.format(t_values[250]))
	axs[0,1].plot(mesh.zs, rho_e_array_no_E[250,:], color=cmap(3), linestyle='--')
	axs[0,1].plot(mesh.zs, rho_e_array[-1,:], color=cmap(4), label='$\\rho({:.2f},z)$'.format(t_values[-1]))
	axs[0,1].plot(mesh.zs, rho_e_array_no_E[-1,:], color=cmap(4), linestyle='--')
	axs[0,1].set_title("$\\rho_e^{1D}(t,z)$", fontsize=20)
	axs[0,1].set_xlabel("$z$", fontsize=20)
	handles, labels = axs[0,1].get_legend_handles_labels()
	handles.append(line)
	#axs[0,1].set_yscale('log')
	axs[0,1].legend(handles=handles)

	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array[0,:], color=cmap(0), label='$\\rho({:.2f},z)$'.format(t_values[0]))
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array_no_E[0,:], color=cmap(0), linestyle='--')
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array[10,:], color=cmap(1), label='$\\rho({:.2f},z)$'.format(t_values[10]))
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array_no_E[10,:], color=cmap(1), linestyle='--')
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array[100,:], color=cmap(2), label='$\\rho({:.2f},z)$'.format(t_values[100]))
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array_no_E[100,:], color=cmap(2), linestyle='--')
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array[250,:], color=cmap(3), label='$\\rho({:.2f},z)$'.format(t_values[250]))
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array_no_E[250,:], color=cmap(3), linestyle='--')
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array[-1,:], color=cmap(4), label='$\\rho({:.2f},z)$'.format(t_values[-1]))
	axs[1,0].plot(mesh.zs, 2*jnp.pi*B_eval*rho_i_array_no_E[-1,:], color=cmap(4), linestyle='--')
	axs[1,0].set_title("$\\rho_i(t,z)$", fontsize=20)
	axs[1,0].set_xlabel("$z$", fontsize=20)
	handles, labels = axs[1,0].get_legend_handles_labels()
	handles.append(line)
	#axs[1,0].set_yscale('log')
	axs[1,0].legend(handles=handles)

	axs[1,1].plot(mesh.zs, rho_i_array[0,:], color=cmap(0), label='$\\rho({:.2f},z)$'.format(t_values[0]))
	axs[1,1].plot(mesh.zs, rho_i_array_no_E[0,:], color=cmap(0), linestyle='--')
	axs[1,1].plot(mesh.zs, rho_i_array[10,:], color=cmap(1), label='$\\rho({:.2f},z)$'.format(t_values[10]))
	axs[1,1].plot(mesh.zs, rho_i_array_no_E[10,:], color=cmap(1), linestyle='--')
	axs[1,1].plot(mesh.zs, rho_i_array[100,:], color=cmap(2), label='$\\rho({:.2f},z)$'.format(t_values[100]))
	axs[1,1].plot(mesh.zs, rho_i_array_no_E[100,:], color=cmap(2), linestyle='--')
	axs[1,1].plot(mesh.zs, rho_i_array[250,:], color=cmap(3), label='$\\rho({:.2f},z)$'.format(t_values[250]))
	axs[1,1].plot(mesh.zs, rho_i_array_no_E[250,:], color=cmap(3), linestyle='--')
	axs[1,1].plot(mesh.zs, rho_i_array[-1,:], color=cmap(4), label='$\\rho({:.2f},z)$'.format(t_values[-1]))
	axs[1,1].plot(mesh.zs, rho_i_array_no_E[-1,:], color=cmap(4), linestyle='--')
	axs[1,1].set_title("$\\rho_i^{1D}(t,z)$", fontsize=20)
	axs[1,1].set_xlabel("$z$", fontsize=20)
	handles, labels = axs[1,1].get_legend_handles_labels()
	handles.append(line)
	#axs[1,1].set_yscale('log')
	axs[1,1].legend(handles=handles)

	plt.show()


def plot_int_rho_multi(rho_e_array: Array,
					rho_e_array_no_E: Array,
					rho_i_array: Array,
					rho_i_array_no_E: Array,
					prop_trapped: float,
					LZ: float,
					LV: float,
					m_i: float,
					ts: Array,
					mesh: MeshFull) -> None:

	fig, axs = plt.subplots(1,2 ,figsize=(22, 7))

	axs[0].plot(ts, vcost_rho(rho_e_array, mesh.zs), label='With $E$')
	axs[0].plot(ts, vcost_rho(rho_e_array_no_E, mesh.zs), label='Without $E$')
	axs[0].set_title("$\\int \\rho_e^{1D}(t,z)\\mathrm{d}z$")
	axs[0].set_xlabel("$t$", fontsize=20)
	axs[0].hlines(y=prop_trapped, xmin=0, xmax=ts[-1], colors='k',
				  linestyles='--', label='Trapped fraction')
	axs[0].vlines(x=LZ/LV, ymin=0.94, ymax=1.0, colors='r',
				  linestyles='--', label='Minimum time before leak')
	axs[0].legend()


	axs[1].plot(ts, vcost_rho(rho_i_array, mesh.zs), label='With $E$')
	axs[1].plot(ts, vcost_rho(rho_i_array_no_E, mesh.zs), label='Without $E$')
	axs[1].set_title("$\\int \\rho_i^{1D}(t,z)\\mathrm{d}z$")
	axs[1].set_xlabel("$t$", fontsize=20)
	axs[1].hlines(y=prop_trapped, xmin=0, xmax=ts[-1], colors='k',
				  linestyles='--', label='Trapped fraction')
	axs[1].vlines(x=(LZ/LV)*jnp.sqrt(m_i), ymin=0.94, ymax=1.0, colors='r',
				  linestyles='--', label='Minimum time before leak')
	axs[1].legend()

	plt.show()