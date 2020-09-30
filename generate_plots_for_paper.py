import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits
import generate_animations

num_cores_max = 30

abm_files = file_loaders.ABM_simulations()
N_files = len(abm_files)
Path("Figures/Paper").mkdir(parents=True, exist_ok=True)

ABM_parameter_vanilla = "v__1.0__N_tot__5800000__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__algo__2__N_init__100__lambda_E__1.0__lambda_I__1.0__make_random_initial_infections__1__N_connect_retries__0"

ABM_parameter_spatial = "v__1.0__N_tot__5800000__rho__0.1__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__algo__2__N_init__100__lambda_E__1.0__lambda_I__1.0__make_random_initial_infections__1__N_connect_retries__0"

filename_hdf5_vanilla = "Data/network/v__1.0__N_tot__5800000__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__algo__2__N_init__100__lambda_E__1.0__lambda_I__1.0__make_random_initial_infections__1__N_connect_retries__0__ID__0.hdf5"

filename_hdf5_spatial = "Data/network/v__1.0__N_tot__5800000__rho__0.1__epsilon_rho__0.0__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__algo__2__N_init__100__lambda_E__1.0__lambda_I__1.0__make_random_initial_infections__0__N_connect_retries__0__ID__0.hdf5"

# x=x

#%%

# ███████ ██  ██████  ██    ██ ██████  ███████     ██████
# ██      ██ ██       ██    ██ ██   ██ ██               ██
# █████   ██ ██   ███ ██    ██ ██████  █████        █████
# ██      ██ ██    ██ ██    ██ ██   ██ ██          ██
# ██      ██  ██████   ██████  ██   ██ ███████     ███████
#
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Regular&t=Figure%202


# reload(generate_animations)
fig_coordinates_vanilla, _ = generate_animations.make_paper_screenshot(
    filename_hdf5_vanilla, title="5.8M, vanilla", i_day=80
)
fig_coordinates_vanilla
fig_coordinates_vanilla.savefig("Figures/Paper/Figure_2a_coordinates_vanilla.pdf", dpi=100)

# reload(plot)
fig_ABM_vanilla, _ = plot.plot_single_ABM_simulation(
    ABM_parameter_vanilla, abm_files, add_top_text=False, xlim=(0, 300)
)
fig_ABM_vanilla.savefig("Figures/Paper/Figure_2b_ABM_vanilla.pdf", dpi=100)

plot.plot_1D_scan(scan_parameter="mu", figname_pdf="Figures/Paper/Figure_2c_1D_scan_mu.pdf")
plot.plot_1D_scan(
    scan_parameter="beta", figname_pdf="Figures/Paper/Figure_2d_1D_scan_beta.pdf", labelpad=0
)


#%%

#%%


# ███████ ██  ██████  ██    ██ ██████  ███████     ██████
# ██      ██ ██       ██    ██ ██   ██ ██               ██
# █████   ██ ██   ███ ██    ██ ██████  █████        █████
# ██      ██ ██    ██ ██    ██ ██   ██ ██               ██
# ██      ██  ██████   ██████  ██   ██ ███████     ██████


reload(plot)

fig_contacts_vanilla, _ = plot._plot_single_number_of_contacts(
    filename_hdf5_vanilla,
    make_fraction_subplot=False,
    xlim=(18, 62),
    figsize=(5 * 0.9, 8 * 0.9),
    title="5.8M, vanilla",
    add_legend=False,
    xlabel="",
    ylabel="",
    fontsize=40,
    labelsize=48,
)

fig_contacts_vanilla.savefig("Figures/Paper/Figure_3a_contacts_vanilla.pdf", dpi=100)

reload(plot)
fig_contacts_spatial, ax_contacts_spatial = plot._plot_single_number_of_contacts(
    filename_hdf5_spatial,
    make_fraction_subplot=False,
    figsize=(8 * 1.2, 4 * 1.2),
    title="5.8M, spatial",
    add_legend=True,
    loc=(0.15, 0.5),
    fontsize=30,
    labelsize=30,
    add_average_arrows=True,
)
fig_contacts_spatial.savefig("Figures/Paper/Figure_3a_contacts_spatial.pdf", dpi=100)

# reload(generate_animations)
fig_coordinates_spatial, _ = generate_animations.make_paper_screenshot(
    filename_hdf5_spatial,
    title="5.8M, spatial",
    i_day=200,
    R_eff_max=7,
)
fig_coordinates_spatial
fig_coordinates_spatial.savefig("Figures/Paper/Figure_3b_coordinates_spatial.pdf", dpi=100)


fig_ABM_spatial, _ = plot.plot_single_ABM_simulation(
    ABM_parameter_spatial, abm_files, add_top_text=False, xlim=(0, 250)
)
fig_ABM_spatial.savefig("Figures/Paper/Figure_3c_ABM_spatial.pdf", dpi=100)


# plot.plot_1D_scan(scan_parameter="rho", figname_pdf="Figures/Paper/Figure_3de_1D_scan_rho.pdf")
res_rho = plot.get_1D_scan_results(scan_parameter="rho", non_default_parameters={})
res_rho_beta_007 = plot.get_1D_scan_results(
    scan_parameter="rho", non_default_parameters={"beta": 0.007}
)

res_rho_sigmabeta_1 = plot.get_1D_scan_results(
    scan_parameter="rho", non_default_parameters={"sigma_beta": 1}
)

res_rho_sigmamu_1 = plot.get_1D_scan_results(
    scan_parameter="rho", non_default_parameters={"sigma_mu": 1}
)

res_rho_sigmabeta_1_sigmamu_1 = plot.get_1D_scan_results(
    scan_parameter="rho", non_default_parameters={"sigma_beta": 1, "sigma_mu": 1}
)
#%%
reload(plot)

ylim = np.array([(0.75, 2.1), (0.2, 2.8)])

fig, (ax0, ax1) = plot._plot_1D_scan_res(
    res_rho,
    scan_parameter="rho",
    ylim=ylim,
    label=r"$\beta = 0.01$",
    fmt=".",
    wspace=0.5,
)

plot._plot_1D_scan_res(
    res_rho_sigmabeta_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["red"],
    label=r"$\sigma_\beta = 1$",
    fmt="v",
    # labelpad=-20,
)

plot._plot_1D_scan_res(
    res_rho_sigmamu_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["orange"],
    label=r"$\sigma_\mu = 1$",
    fmt="^",
    # labelpad=-20,
)

plot._plot_1D_scan_res(
    res_rho_sigmabeta_1_sigmamu_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["green"],
    label=r"$\sigma_\mu=\sigma_\beta=1$",
    labelpad=-20,
    fmt="x",
)


ax0.axhline(1, color="grey", ls="--")
ax1.axhline(1, color="grey", ls="--")


ax0.set_ylabel(r"$I_\mathrm{max}^\mathrm{ABM} \, / \,\, I_\mathrm{max}^\mathrm{SEIR}$", labelpad=10)
ax1.set_ylabel(r"$R_\infty^\mathrm{ABM} \, / \,\, R_\infty^\mathrm{SEIR}$", labelpad=15)


ax0b = ax0.twinx()  # instantiate a second axes that shares the same x-axis
# ax1b = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = plot.d_colors["blue"]
ax0b.tick_params(axis="y", labelcolor=color)
# ax1b.tick_params(axis="y", labelcolor=color)

errorbar_kwargs = dict(
    fmt="*",
    elinewidth=1,
    capsize=10,
)

ax0b.errorbar(
    res_rho_beta_007[0],
    res_rho_beta_007[1],
    res_rho_beta_007[3],
    **errorbar_kwargs,
    color=color,
    ecolor=color,
)


ax1.errorbar(
    res_rho_beta_007[0],
    res_rho_beta_007[2],
    res_rho_beta_007[4],
    **errorbar_kwargs,
    color=color,
    ecolor=color,
    label=r"$\beta = 0.007$",
)

lines, labels = ax1.get_legend_handles_labels()
lines = [lines[-1]] + lines[:-1]
labels = [labels[-1]] + labels[:-1]
# lines2, labels2 = ax1b.get_legend_handles_labels()
# lines = lines2 + lines
# labels = labels2 + labels
ax1.legend(
    lines,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(-0.3, 1.2),
    columnspacing=0.8,
    handletextpad=-0,
)

# ax0.set(xlim=(-0.02, 0.52))
ax0.set(xticks=[0, 0.5])
ax1.set(xticks=[0, 0.5])

scale_I = 10
scale_R = 4
ax0b.set(ylim=ylim[0] * scale_I)
# ax1b.set(ylim=ylim[1] * scale_R)
# ax1b.set_yticks(ax1.get_yticks()[1:-1] * scale_R)
# ax1b.axhline(1, color=color, ls="--")

#%%

fig.savefig(
    "Figures/Paper/Figure_3de_1D_scan_rho.pdf", dpi=100
)  # bbox_inches='tight', pad_inches=0.3


plot.plot_1D_scan(
    scan_parameter="epsilon_rho",
    non_default_parameters=dict(rho=0.1),
    figname_pdf="Figures/Paper/Figure_3fg_1D_scan_epsilon_rho.pdf",
)


#%%


# ███████ ██  ██████  ██    ██ ██████  ███████     ██   ██
# ██      ██ ██       ██    ██ ██   ██ ██          ██   ██
# █████   ██ ██   ███ ██    ██ ██████  █████       ███████
# ██      ██ ██    ██ ██    ██ ██   ██ ██               ██
# ██      ██  ██████   ██████  ██   ██ ███████          ██


num_cores = utils.get_num_cores(num_cores_max)
all_fits = fits.get_fit_results(abm_files, force_rerun=False, num_cores=num_cores)

fig_plots_vanilla, _ = plot.plot_single_fit(
    ABM_parameter_vanilla, all_fits, add_top_text=False, xlim=(0, 350)
)
fig_plots_vanilla.savefig("Figures/Paper/Figure_4a_fits_vanilla.pdf", dpi=100)

fig_plots_spatial, _ = plot.plot_single_fit(
    ABM_parameter_spatial, all_fits, add_top_text=False, xlim=(0, 210)
)
fig_plots_spatial.savefig("Figures/Paper/Figure_4b_fits_spatial.pdf", dpi=100)


# plot.plot_1D_scan_fit_results(
# all_fits, scan_parameter="rho", figname_pdf="Figures/Paper/Figure_4cd_1D_scan_fit_rho.pdf"
# )

#%%

res_rho = plot.get_1D_scan_fit_results(
    all_fits,
    scan_parameter="rho",
    non_default_parameters={},
)

res_rho_beta_007 = plot.get_1D_scan_fit_results(
    all_fits,
    scan_parameter="rho",
    non_default_parameters={"beta": 0.007},
)

res_rho_sigmabeta_1 = plot.get_1D_scan_fit_results(
    all_fits,
    scan_parameter="rho",
    non_default_parameters={"sigma_beta": 1},
)

res_rho_sigmamu_1 = plot.get_1D_scan_fit_results(
    all_fits,
    scan_parameter="rho",
    non_default_parameters={"sigma_mu": 1},
)

res_rho_sigmabeta_1_sigmamu_1 = plot.get_1D_scan_fit_results(
    all_fits,
    scan_parameter="rho",
    non_default_parameters={"sigma_beta": 1, "sigma_mu": 1},
)

#%%

reload(plot)

ylim = np.array([(0.8, 8.5), (0.8, 4.3)])

fig, (ax0, ax1) = plot._plot_1D_scan_res(
    res_rho_beta_007,  # res_rho,
    scan_parameter="rho",
    ylim=ylim,
    label=r"$\beta = 0.007$",
    color=plot.d_colors["blue"],
    wspace=0.45,
    fmt="*",
)

plot._plot_1D_scan_res(
    res_rho,
    scan_parameter="rho",
    axes=(ax0, ax1),
    # color="black",
    label=r"$\beta = 0.01$",
    fmt=".",
)

plot._plot_1D_scan_res(
    res_rho_sigmabeta_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["red"],
    label=r"$\sigma_\beta = 1$",
    fmt="v",
)


plot._plot_1D_scan_res(
    res_rho_sigmamu_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["orange"],
    label=r"$\sigma_\mu = 1$",
    fmt="^",
)

plot._plot_1D_scan_res(
    res_rho_sigmabeta_1_sigmamu_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["green"],
    label=r"$\sigma_\mu = \sigma_\beta = 1$",
    labelpad=-20,
    fmt="x",
)

ax0.axhline(1, color="grey", ls="--")
ax1.axhline(1, color="grey", ls="--")

ax0.set_ylabel(r"$I_\mathrm{max}^\mathrm{fit} \, / \,\, I_\mathrm{max}^\mathrm{ABM}$", labelpad=10)
ax1.set_ylabel(r"$R_\infty^\mathrm{fit} \, / \,\, R_\infty^\mathrm{ABM}$", labelpad=15)


lines, labels = ax1.get_legend_handles_labels()
ax1.legend(
    lines,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(-0.3, 1.2),
    columnspacing=0.8,
    handletextpad=-0,
)

ax0.set(xticks=[0, 0.5])
ax1.set(xticks=[0, 0.5])


#%%
fig.savefig(
    "Figures/Paper/Figure_4cd_1D_scan_fit_rho.pdf", dpi=100
)  # bbox_inches='tight', pad_inches=0.3

#%%

plot.plot_1D_scan_fit_results(
    all_fits,
    scan_parameter="epsilon_rho",
    non_default_parameters=dict(rho=0.1),
    figname_pdf="Figures/Paper/Figure_4ef_1D_scan_fit_epsilon_rho.pdf",
)

# %%

# import h5py
# f = h5py.File(filename_hdf5_spatial, "r")
# d = f.attrs
# with h5py.File(filename_hdf5_spatial, "r") as f:
# d = f.attrs

# with h5py.File(self.filenames["network_network"], "w") as f:  #
#     f.create_dataset("time_elapsed", data=time_elapsed)

#     for key, val in self.cfg.items():
#         f.attrs[key] = val

# # %%
