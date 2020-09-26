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
plot.plot_1D_scan(scan_parameter="beta", figname_pdf="Figures/Paper/Figure_2d_1D_scan_beta.pdf")

#%%

#%%


# ███████ ██  ██████  ██    ██ ██████  ███████     ██████
# ██      ██ ██       ██    ██ ██   ██ ██               ██
# █████   ██ ██   ███ ██    ██ ██████  █████        █████
# ██      ██ ██    ██ ██    ██ ██   ██ ██               ██
# ██      ██  ██████   ██████  ██   ██ ███████     ██████


# reload(plot)

fig_contacts_vanilla, _ = plot._plot_single_number_of_contacts(
    filename_hdf5_vanilla,
    make_fraction_subplot=False,
    xlim=(18, 62),
    figsize=(5, 8),
    title="5.8M, vanilla",
    add_legend=False,
)
fig_contacts_vanilla.savefig("Figures/Paper/Figure_3a_contacts_vanilla.pdf", dpi=100)

fig_contacts_spatial, _ = plot._plot_single_number_of_contacts(
    filename_hdf5_spatial,
    make_fraction_subplot=False,
    # xlim=(18, 62),
    figsize=(8 * 1.2, 4 * 1.2),
    title="5.8M, spatial",
    add_legend=True,
    loc=(0.15, 0.5),
)
fig_contacts_spatial.savefig("Figures/Paper/Figure_3a_contacts_spatial.pdf", dpi=100)

# reload(generate_animations)
fig_coordinates_spatial, _ = generate_animations.make_paper_screenshot(
    filename_hdf5_spatial, title="5.8M, spatial", i_day=200
)
fig_coordinates_spatial
fig_coordinates_spatial.savefig("Figures/Paper/Figure_3b_coordinates_spatial.pdf", dpi=100)

#%%

fig_ABM_spatial, _ = plot.plot_single_ABM_simulation(
    ABM_parameter_spatial, abm_files, add_top_text=False, xlim=(0, 250)
)
fig_ABM_spatial.savefig("Figures/Paper/Figure_3c_ABM_spatial.pdf", dpi=100)

plot.plot_1D_scan(scan_parameter="rho", figname_pdf="Figures/Paper/Figure_3de_1D_scan_rho.pdf")
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


plot.plot_1D_scan_fit_results(
    all_fits, scan_parameter="rho", figname_pdf="Figures/Paper/Figure_4cd_1D_scan_fit_rho.pdf"
)

plot.plot_1D_scan_fit_results(
    all_fits,
    scan_parameter="epsilon_rho",
    non_default_parameters=dict(rho=0.1),
    figname_pdf="Figures/Paper/Figure_4ef_1D_scan_fit_epsilon_rho.pdf",
)

# %%
