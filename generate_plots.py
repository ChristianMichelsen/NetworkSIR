import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# from iminuit import Minuit
from collections import defaultdict
import joblib
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits
from src import database

rc_params.set_rc_params(dpi=100)
num_cores_max = 30

make_1D_scan = True
force_rerun = True
verbose = True
make_fits = False


#%%

reload(plot)
reload(file_loaders)

abm_files = file_loaders.ABM_simulations(verbose=True)
N_files = len(abm_files)

#%%

reload(plot)


network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
plot.plot_corona_type(
    network_files,
    force_rerun=force_rerun,
    xlim=(10, 100),
    N_max_runs=3,
    reposition_x_axis=True,
    normalize=False,
)

x = x


#%%

reload(plot)
# plot.plot_ABM_simulations(abm_files, force_rerun=force_rerun)
plot.plot_ABM_simulations(abm_files, force_rerun=force_rerun, xlim=(0, 150))

x = x

plot.plot_corona_type_ratio_plot(network_files, force_rerun=force_rerun, xlim=(10, 100))

# for cfg in network_files.iter_cfgs():
#     break

#%%

reload(plot)

parameters_1D_scan = [
    # dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=1)),
    # dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=10)),
    # dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=100)),
    # dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=1_000)),
    # dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=10_000)),
    # dict(scan_parameter="mu"),
    # dict(scan_parameter="beta", non_default_parameters=dict(rho=0.1)),
    # dict(scan_parameter="beta"),
    # dict(scan_parameter="beta", non_default_parameters=dict(sigma_beta=1)),
    # dict(scan_parameter="beta", non_default_parameters=dict(sigma_beta=1, rho=0.1)),
    # dict(scan_parameter="N_tot", do_log=True),
    # dict(scan_parameter="N_tot", do_log=True, non_default_parameters=dict(rho=0.1)),
    # dict(scan_parameter="N_init", do_log=True),
    # dict(scan_parameter="N_init", do_log=True, non_default_parameters=dict(rho=0.1)),
    # dict(scan_parameter="rho"),
    # dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0)),
    # dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0.02)),
    # dict(scan_parameter="rho", non_default_parameters=dict(beta=0.007)),
    # dict(scan_parameter="rho", non_default_parameters=dict(sigma_beta=1)),
    # dict(scan_parameter="rho", non_default_parameters=dict(sigma_mu=1)),
    # dict(scan_parameter="rho", non_default_parameters=dict(sigma_mu=1, sigma_beta=1)),
    # dict(scan_parameter="rho", non_default_parameters=dict(algo=1)),
    # dict(scan_parameter="rho", non_default_parameters=dict(N_tot=5_800_000)),
    # dict(scan_parameter="epsilon_rho"),
    # dict(scan_parameter="epsilon_rho", non_default_parameters=dict(rho=0.1)),
    # dict(scan_parameter="epsilon_rho", non_default_parameters=dict(rho=0.1, algo=1)),
    # dict(scan_parameter="sigma_beta"),
    # dict(scan_parameter="sigma_beta", non_default_parameters=dict(rho=0.1)),
    # dict(scan_parameter="sigma_beta", non_default_parameters=dict(sigma_mu=1)),
    # dict(scan_parameter="sigma_beta", non_default_parameters=dict(rho=0.1, sigma_mu=1)),
    # dict(scan_parameter="sigma_mu"),
    # dict(scan_parameter="sigma_mu", non_default_parameters=dict(rho=0.1)),
    # dict(scan_parameter="sigma_mu", non_default_parameters=dict(sigma_beta=1)),
    # dict(scan_parameter="sigma_mu", non_default_parameters=dict(rho=0.1, sigma_beta=1)),
    # dict(scan_parameter="lambda_E"),
    # dict(scan_parameter="lambda_I"),
    dict(
        scan_parameter="beta_UK_multiplier",
        non_default_parameters=dict(N_init=2_000, N_init_UK=200, beta=0.004),
    ),
]

# reload(plot)
if make_1D_scan:
    for parameter_1D_scan in parameters_1D_scan:
        plot.plot_1D_scan(**parameter_1D_scan)

#%%

if make_fits:

    reload(fits)
    num_cores = utils.get_num_cores(num_cores_max)
    all_fits = fits.get_fit_results(abm_files, force_rerun=False, num_cores=num_cores, y_max=0.01)

    reload(plot)
    plot.plot_fits(all_fits, force_rerun=force_rerun, verbose=verbose)

#%%

reload(plot)
if make_1D_scan:
    for parameter_1D_scan in parameters_1D_scan:
        plot.plot_1D_scan_fit_results(all_fits, **parameter_1D_scan)

#%%
reload(plot)
# force_rerun=True
network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
plot.plot_number_of_contacts(network_files, force_rerun=force_rerun)

# %%

reload(plot)

from matplotlib.backends.backend_pdf import PdfPages

d_query = utils.DotDict(
    {
        "beta": 0.004,
    },
)

cfgs = utils.query_cfg(d_query)
cfgs.sort(key=lambda cfg: cfg["beta_UK_multiplier"])

pdf_name = Path(f"Figures/ABM_simulations_UK.pdf")
utils.make_sure_folder_exist(pdf_name)
with PdfPages(pdf_name) as pdf:
    for cfg in tqdm(cfgs, desc="Plotting only events"):
        filenames = utils.hash_to_filenames(cfg.hash)
        fig, ax = plot.plot_single_ABM_simulation(cfg, abm_files)
        pdf.savefig(fig, dpi=100)
        plt.close("all")

#%%

if False:

    d_query = utils.DotDict(
        {
            # "epsilon_rho": 0.02,
            # "N_tot": 580_000,
            # "rho": 0.0,
            # "beta": 0.0108,
            # "weighted_random_initial_infections": True,
            # "results_delay_in_clicks": [30, 30, 30],
            # "event_size_mean": 7.9997,
            "hash": "1e04392284"
        },
    )

    cfgs = utils.query_cfg(d_query)
    for cfg in cfgs:
        print(cfg)

# x = x

# cfgs.sort(key=lambda cfg: cfg["N_tot"])
# [cfg.hash for cfg in cfgs]

# plot.plot_single_ABM_simulation(cfgs[0], abm_files)

#%%

# R_eff for beta 1D-scan
if False:
    cfgs, _ = utils.get_1D_scan_cfgs_all_filenames(
        scan_parameter="beta",
        non_default_parameters={},
        # non_default_parameters=dict(weighted_random_initial_infections=True),
    )
    cfgs.sort(key=lambda cfg: cfg["beta"])

    plot.plot_R_eff_beta_1D_scan(cfgs, abm_files)


# %%

reload(plot)
reload(database)


# x = x

# plot MCMC results
variable = "event_size_max"
variable = "results_delay_in_clicks"
reverse_order = True
extra_selections = {"tracking_rates": [1.0, 0.8, 0.0]}

# variable_subset = [
#     [20, 20, 20],
#     [30, 30, 30],
# ]

N_max_figures = 2
N_max_figures = None


plot.make_MCMC_plots(
    variable="results_delay_in_clicks",
    abm_files=abm_files,
    extra_selections={"tracking_rates": [1.0, 0.8, 0.0]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=0,
    reverse_order=True,  # True since a higher value of results_delay_in_clicks is less intervention
    # variable_subset=variable_subset,
)


plot.make_MCMC_plots(
    variable="results_delay_in_clicks",
    abm_files=abm_files,
    extra_selections={"tracking_rates": [1.0, 0.8, 0.25]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=0,
    reverse_order=True,  # True since a higher value of results_delay_in_clicks is less intervention
    # variable_subset=variable_subset,
)


plot.make_MCMC_plots(
    variable="results_delay_in_clicks",
    abm_files=abm_files,
    extra_selections={"tracking_rates": [1.0, 0.8, 0.5]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=0,
    reverse_order=True,  # True since a higher value of results_delay_in_clicks is less intervention
    # variable_subset=variable_subset,
)


plot.make_MCMC_plots(
    variable="results_delay_in_clicks",
    abm_files=abm_files,
    extra_selections={"tracking_rates": [1.0, 0.8, 0.75]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=0,
    reverse_order=True,  # True since a higher value of results_delay_in_clicks is less intervention
    # variable_subset=variable_subset,
)


# plot MCMC results
plot.make_MCMC_plots(
    variable="tracking_rates",
    abm_files=abm_files,
    extra_selections={"results_delay_in_clicks": [30, 30, 30]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=-1,
    reverse_order=False,
)

# plot MCMC results
plot.make_MCMC_plots(
    variable="tracking_rates",
    abm_files=abm_files,
    extra_selections={"results_delay_in_clicks": [20, 20, 20]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=-1,
    reverse_order=False,
)


# plot MCMC results
plot.make_MCMC_plots(
    variable="tracking_rates",
    abm_files=abm_files,
    extra_selections={"results_delay_in_clicks": [10, 10, 10]},
    N_max_figures=N_max_figures,
    index_in_list_to_sortby=-1,
    reverse_order=False,
)
# %%

reload(plot)

network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
# for cfg in network_files.iter_cfgs():
#     fig, axes = plot.plot_corona_type_single_plot(cfg, network_files)


plot.plot_corona_type_single_plot(network_files, force_rerun=False)

# %%
