import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits
from tqdm import tqdm

num_cores_max = 30

abm_files = file_loaders.ABM_simulations()
N_files = len(abm_files)
Path("Figures/Paper").mkdir(parents=True, exist_ok=True)

#%%

ID = 0

cfg_vanilla = utils.DotDict(
    {
        "version": 1.0,
        "N_tot": 5_800_000,
        "rho": 0.0,
        "epsilon_rho": 0.04,
        "mu": 40.0,
        "sigma_mu": 0.0,
        "beta": 0.01,
        "sigma_beta": 0.0,
        "algo": 2,
        "N_init": 100,
        "lambda_E": 1.0,
        "lambda_I": 1.0,
        "make_random_initial_infections": True,
        "clustering_connection_retries": 0,
        "N_events": 0,
    }
)
hash_vanilla = utils.query_cfg(cfg_vanilla)[0].hash
filename_hdf5_vanilla = str(
    list(Path(f"./Data/network/{hash_vanilla}").rglob(f"*ID__{ID}.hdf5"))[0]
)


cfg_vanilla_007 = utils.DotDict({**cfg_vanilla, "beta": 0.007, "N_tot": 580_000})
hash_vanilla_007 = utils.query_cfg(cfg_vanilla_007)[0].hash
filename_hdf5_vanilla_007 = str(
    list(Path(f"./Data/network/{hash_vanilla_007}").rglob(f"*ID__{ID}.hdf5"))[0]
)

cfg_spatial = utils.DotDict({**cfg_vanilla, "rho": 0.1})
hash_spatial = utils.query_cfg(cfg_spatial)[0].hash
filename_hdf5_spatial = str(
    list(Path(f"./Data/network/{hash_spatial}").rglob(f"*ID__{ID}.hdf5"))[0]
)

cfg_spatial_local_outbreak = utils.DotDict(
    {
        "epsilon_rho": 0.04,
        "rho": 0.1,
        "N_tot": 5_800_000,
        "beta": 0.007,
        "make_random_initial_infections": False,
    }
)
hash_spatial_local_outbreak = utils.query_cfg(cfg_spatial_local_outbreak)[0].hash
filename_hdf5_spatial_local_outbreak = str(
    list(Path(f"./Data/network/{hash_spatial_local_outbreak}").rglob(f"*ID__{ID}.hdf5"))[0]
)


#%%

# ███████ ██  ██████  ██    ██ ██████  ███████     ██████
# ██      ██ ██       ██    ██ ██   ██ ██               ██
# █████   ██ ██   ███ ██    ██ ██████  █████        █████
# ██      ██ ██    ██ ██    ██ ██   ██ ██          ██
# ██      ██  ██████   ██████  ██   ██ ███████     ███████

# reload(plot)
fig_contacts_vanilla, _ = plot.plot_single_number_of_contacts(
    filename_hdf5_vanilla,
    make_fraction_subplot=False,
    xlim=(18, 62),
    figsize=(5 * 0.9, 6.5 * 0.9),
    title="5.8M, vanilla",
    add_legend=False,
    xlabel="",
    ylabel="",
    fontsize=40,
    labelsize=48,
)

fig_contacts_vanilla.savefig("Figures/Paper/Figure_2a_contacts_vanilla.pdf", dpi=100)

# reload(plot)
fig_contacts_spatial, ax_contacts_spatial = plot.plot_single_number_of_contacts(
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
fig_contacts_spatial.savefig("Figures/Paper/Figure_2a_contacts_spatial.pdf", dpi=100)

reload(plot)
fig_coordinates_spatial, _ = plot.make_paper_screenshot(
    filename_hdf5_spatial_local_outbreak,
    title="5.8M, spatial",
    i_day=80,
    R_eff_max=7,
    dpi=40,
)
fig_coordinates_spatial
fig_coordinates_spatial.savefig("Figures/Paper/Figure_2b_coordinates_spatial.pdf", dpi=75)


plot.make_paper_screenshot(
    filename_hdf5_vanilla,
    title="5.8M, vanilla",
    i_day=100,
    R_eff_max=7,
    dpi=40,
)[0].savefig("Figures/Paper/Figure_X_coordinates_vanilla.pdf", dpi=75)
plot.make_paper_screenshot(
    filename_hdf5_vanilla_007,
    title="580k, vanilla, beta=0.007",
    i_day=100,
    R_eff_max=7,
    dpi=40,
)[0].savefig("Figures/Paper/Figure_X_coordinates_vanilla_beta_007.pdf", dpi=75)


#%%
# reload(plot)
fig_ABM_spatial, axes_ABM_spatial = plot.plot_single_ABM_simulation(
    cfg_spatial,
    abm_files,
    add_top_text=False,
    xlim=(0, 230),
    ylim_scale=1.25,
    legend_fontsize=28,
    d_label_loc={"I": "upper right", "R": "upper left"},
)
axes_ABM_spatial[0].text(
    35,
    7.7 / 100,
    r"$I_\mathrm{peak}^\mathrm{ABM}$",
    fontsize=38,
)
axes_ABM_spatial[0].text(
    115,
    5.6 / 100,
    r"$I_\mathrm{peak}^\mathrm{SEIR}$",
    fontsize=38,
    color=plot.d_colors["red"],
)
axes_ABM_spatial[1].text(
    172,
    41 / 100,
    r"$R_\infty^\mathrm{ABM}$",
    fontsize=38,
)
axes_ABM_spatial[1].text(
    172,
    69 / 100,
    r"$R_\infty^\mathrm{SEIR}$",
    fontsize=38,
    color=plot.d_colors["red"],
)
fig_ABM_spatial.savefig("Figures/Paper/Figure_2cd_ABM_spatial.pdf", dpi=100)


# reload(plot)
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
# reload(plot)

ylim = np.array([(0.75, 2.1), (0.2, 1.8)])

fig, (ax0, ax1) = plot._plot_1D_scan_res(
    res_rho,
    scan_parameter="rho",
    ylim=ylim,
    label=r"$\beta = 0.01$",
    fmt=".",
    wspace=0.5,
    add_title=False,
    add_horizontal_line=True,
)

plot._plot_1D_scan_res(
    res_rho_sigmabeta_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["red"],
    label=r"$\sigma_\beta = 1$",
    fmt="v",
    add_title=False,
    # labelpad=-20,
)

plot._plot_1D_scan_res(
    res_rho_sigmamu_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["orange"],
    label=r"$\sigma_\mu = 1$",
    fmt="^",
    add_title=False,
    # labelpad=-20,
)

plot._plot_1D_scan_res(
    res_rho_sigmabeta_1_sigmamu_1,
    scan_parameter="rho",
    axes=(ax0, ax1),
    color=plot.d_colors["green"],
    label=r"$\sigma_\mu=\sigma_\beta=1$",
    labelpad=-20,
    add_title=False,
    fmt="x",
)


# ax0.axhline(1, color="grey", ls="--")
# ax1.axhline(1, color="grey", ls="--")


ax0.set_ylabel(
    r"$I_\mathrm{peak}^\mathrm{ABM} \, / \,\, I_\mathrm{peak}^\mathrm{SEIR}$", labelpad=10
)
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
# lines.insert()
labels.insert(1, labels.pop(-1))
lines.insert(1, lines.pop(-1))
# lines2, labels2 = ax1b.get_legend_handles_labels()
# lines = lines2 + lines
# labels = labels2 + labels
ax1.legend(
    lines,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(-0.28, 1.22),
    columnspacing=0.6,
    handletextpad=-0.5,
    fontsize=27,
)

# ax0.set(xlim=(-0.02, 0.52))
ax0.set(xticks=[0, 0.5])
ax1.set(xticks=[0, 0.5])

scale_I = 10
scale_R = 4
ax0b.set(ylim=ylim[0] * scale_I)

#%%

fig.savefig(
    "Figures/Paper/Figure_2ef_1D_scan_rho.pdf", dpi=100
)  # bbox_inches='tight', pad_inches=0.3


# reload(plot)
plot.plot_1D_scan(
    scan_parameter="epsilon_rho",
    non_default_parameters=dict(rho=0.1),
    figname_pdf="Figures/Paper/Figure_2gh_1D_scan_epsilon_rho.pdf",
    add_horizontal_line=True,
    ylim=[(None, None), (0.5, 1.01)],
)

#%%


# ███████ ██  ██████  ██    ██ ██████  ███████     ██████
# ██      ██ ██       ██    ██ ██   ██ ██               ██
# █████   ██ ██   ███ ██    ██ ██████  █████        █████
# ██      ██ ██    ██ ██    ██ ██   ██ ██               ██
# ██      ██  ██████   ██████  ██   ██ ███████     ██████
#
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Regular&t=Figure%203

num_cores = utils.get_num_cores(num_cores_max)
all_fits = fits.get_fit_results(abm_files, force_rerun=False, num_cores=num_cores)

all_fits_y_max_1_percent = all_fits

all_fits_y_max_2_percent = fits.get_fit_results(
    abm_files,
    force_rerun=False,
    num_cores=num_cores,
    y_max=0.02,
)

all_fits_y_max_05_percent = fits.get_fit_results(
    abm_files,
    force_rerun=False,
    num_cores=num_cores,
    y_max=0.005,
)

fit_objects_vanilla = all_fits[hash_vanilla]
fit_objects_spatial = all_fits[hash_spatial]

# reload(plot)
fig_plots_vanilla, _ = plot.plot_single_fit(
    cfg_vanilla,
    fit_objects_vanilla,
    add_top_text=False,
    xlim=(0, 350),
    legend_fontsize=30,
)
fig_plots_vanilla.savefig("Figures/Paper/Figure_3a_fits_vanilla.pdf", dpi=100)

plot.plot_single_fit(
    cfg_vanilla,
    all_fits_y_max_2_percent[hash_vanilla],
    add_top_text=False,
    xlim=(0, 350),
    legend_fontsize=30,
)[0].savefig("Figures/Paper/Figure_3a_fits_vanilla_y_max_2_percent.pdf", dpi=100)
plot.plot_single_fit(
    cfg_vanilla,
    all_fits_y_max_05_percent[hash_vanilla],
    add_top_text=False,
    xlim=(0, 350),
    legend_fontsize=30,
)[0].savefig("Figures/Paper/Figure_3a_fits_vanilla_y_max_0.5_percent.pdf", dpi=100)


fig_plots_spatial, _ = plot.plot_single_fit(
    cfg_spatial,
    fit_objects_spatial,
    add_top_text=False,
    xlim=(0, 210),
    legend_fontsize=30,
)
fig_plots_spatial.savefig("Figures/Paper/Figure_3b_fits_spatial.pdf", dpi=100)

plot.plot_single_fit(
    cfg_spatial,
    all_fits_y_max_2_percent[hash_spatial],
    add_top_text=False,
    xlim=(0, 210),
    legend_fontsize=30,
)[0].savefig("Figures/Paper/Figure_3b_fits_spatial_y_max_2_percent.pdf", dpi=100)
plot.plot_single_fit(
    cfg_spatial,
    all_fits_y_max_05_percent[hash_spatial],
    add_top_text=False,
    xlim=(0, 210),
    legend_fontsize=30,
)[0].savefig("Figures/Paper/Figure_3b_fits_spatial_y_max_05_percent.pdf", dpi=100)


#%%

for percent, all_fits_percent in zip(
    [0.5, 1, 2],
    [all_fits_y_max_05_percent, all_fits_y_max_1_percent, all_fits_y_max_2_percent],
):

    res_rho = plot.get_1D_scan_fit_results(
        all_fits_percent,
        scan_parameter="rho",
        non_default_parameters={},
    )

    res_rho_beta_007 = plot.get_1D_scan_fit_results(
        all_fits_percent,
        scan_parameter="rho",
        non_default_parameters={"beta": 0.007},
    )

    res_rho_sigmabeta_1 = plot.get_1D_scan_fit_results(
        all_fits_percent,
        scan_parameter="rho",
        non_default_parameters={"sigma_beta": 1},
    )

    res_rho_sigmamu_1 = plot.get_1D_scan_fit_results(
        all_fits_percent,
        scan_parameter="rho",
        non_default_parameters={"sigma_mu": 1},
    )

    res_rho_sigmabeta_1_sigmamu_1 = plot.get_1D_scan_fit_results(
        all_fits_percent,
        scan_parameter="rho",
        non_default_parameters={"sigma_beta": 1, "sigma_mu": 1},
    )

    #%%

    # reload(plot)

    ylim = np.array([(0.8, 8.5), (0.8, 4.3)])

    fig, (ax0, ax1) = plot._plot_1D_scan_res(
        res_rho_beta_007,  # res_rho,
        scan_parameter="rho",
        ylim=ylim,
        label=r"$\beta = 0.007$",
        color=plot.d_colors["blue"],
        wspace=0.45,
        fmt="*",
        add_title=False,
        add_horizontal_line=True,
    )

    plot._plot_1D_scan_res(
        res_rho,
        scan_parameter="rho",
        axes=(ax0, ax1),
        # color="black",
        label=r"$\beta = 0.01$",
        fmt=".",
        add_title=False,
    )

    plot._plot_1D_scan_res(
        res_rho_sigmabeta_1,
        scan_parameter="rho",
        axes=(ax0, ax1),
        color=plot.d_colors["red"],
        label=r"$\sigma_\beta = 1$",
        fmt="v",
        add_title=False,
    )

    plot._plot_1D_scan_res(
        res_rho_sigmamu_1,
        scan_parameter="rho",
        axes=(ax0, ax1),
        color=plot.d_colors["orange"],
        label=r"$\sigma_\mu = 1$",
        fmt="^",
        add_title=False,
    )

    plot._plot_1D_scan_res(
        res_rho_sigmabeta_1_sigmamu_1,
        scan_parameter="rho",
        axes=(ax0, ax1),
        color=plot.d_colors["green"],
        label=r"$\sigma_\mu = \sigma_\beta = 1$",
        labelpad=-20,
        fmt="x",
        add_title=False,
    )

    # ax0.axhline(1, color="grey", ls="--")
    # ax1.axhline(1, color="grey", ls="--")

    ax0.set_ylabel(
        r"$I_\mathrm{peak}^\mathrm{fit} \, / \,\, I_\mathrm{peak}^\mathrm{ABM}$", labelpad=10
    )
    ax1.set_ylabel(r"$R_\infty^\mathrm{fit} \, / \,\, R_\infty^\mathrm{ABM}$", labelpad=15)

    lines, labels = ax1.get_legend_handles_labels()
    labels.insert(1, labels.pop(0))
    lines.insert(1, lines.pop(0))

    ax1.legend(
        lines,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(-0.28, 1.22),
        columnspacing=0.6,
        handletextpad=-0.5,
        fontsize=27,
    )

    ax0.set(xticks=[0, 0.5])
    ax1.set(xticks=[0, 0.5])

    #%%

    if percent == 1:
        fig.savefig(
            "Figures/Paper/Figure_3cd_1D_scan_fit_rho.pdf", dpi=100
        )  # bbox_inches='tight', pad_inches=0.3
    else:
        fig.savefig(
            f"Figures/Paper/Figure_3cd_1D_scan_fit_rho_y_max_{percent}_percent.pdf", dpi=100
        )  # bbox_inches='tight', pad_inches=0.3

    #%%

    plot.plot_1D_scan_fit_results(
        all_fits_percent,
        scan_parameter="epsilon_rho",
        non_default_parameters=dict(rho=0.1),
        figname_pdf="Figures/Paper/Figure_3ef_1D_scan_fit_epsilon_rho.pdf"
        if percent == 1
        else f"Figures/Paper/Figure_3ef_1D_scan_fit_epsilon_rho_y_max_{percent}_percent.pdf",
        add_horizontal_line=True,
    )

# %%
