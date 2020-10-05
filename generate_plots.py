import numpy as np
import matplotlib.pyplot as plt
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

rc_params.set_rc_params()
num_cores_max = 30

do_make_1D_scan = True
force_rerun = True
verbose = False

#%%

reload(plot)
reload(file_loaders)

abm_files = file_loaders.ABM_simulations()
N_files = len(abm_files)


#%%
plot.plot_ABM_simulations(abm_files, force_rerun=force_rerun)

#%%

reload(plot)

parameters_1D_scan = [
    dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=1)),
    dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=10)),
    dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=100)),
    dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=1_000)),
    dict(scan_parameter="event_size_max", non_default_parameters=dict(N_events=10_000)),
    dict(scan_parameter="mu"),
    dict(scan_parameter="beta", non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="beta"),
    dict(scan_parameter="beta", non_default_parameters=dict(sigma_beta=1)),
    dict(scan_parameter="beta", non_default_parameters=dict(sigma_beta=1, rho=0.1)),
    dict(scan_parameter="N_tot", do_log=True),
    dict(scan_parameter="N_tot", do_log=True, non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="N_init", do_log=True),
    dict(scan_parameter="N_init", do_log=True, non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="rho"),
    dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0)),
    dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0.02)),
    dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0.04)),
    dict(scan_parameter="rho", non_default_parameters=dict(beta=0.007)),
    dict(scan_parameter="rho", non_default_parameters=dict(sigma_beta=1)),
    dict(scan_parameter="rho", non_default_parameters=dict(sigma_mu=1)),
    dict(scan_parameter="rho", non_default_parameters=dict(sigma_mu=1, sigma_beta=1)),
    dict(scan_parameter="rho", non_default_parameters=dict(algo=1)),
    dict(scan_parameter="rho", non_default_parameters=dict(N_tot=5_800_000)),
    dict(scan_parameter="epsilon_rho"),
    dict(scan_parameter="epsilon_rho", non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="epsilon_rho", non_default_parameters=dict(rho=0.1, algo=1)),
    dict(scan_parameter="sigma_beta"),
    dict(scan_parameter="sigma_beta", non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="sigma_beta", non_default_parameters=dict(sigma_mu=1)),
    dict(scan_parameter="sigma_beta", non_default_parameters=dict(rho=0.1, sigma_mu=1)),
    dict(scan_parameter="sigma_mu"),
    dict(scan_parameter="sigma_mu", non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="sigma_mu", non_default_parameters=dict(sigma_beta=1)),
    dict(scan_parameter="sigma_mu", non_default_parameters=dict(rho=0.1, sigma_beta=1)),
    dict(scan_parameter="lambda_E"),
    dict(scan_parameter="lambda_I"),
]

# reload(plot)
if do_make_1D_scan:
    for parameter_1D_scan in parameters_1D_scan:
        plot.plot_1D_scan(**parameter_1D_scan)

#%%

reload(fits)
num_cores = utils.get_num_cores(num_cores_max)
all_fits = fits.get_fit_results(abm_files, force_rerun=False, num_cores=num_cores)


#%%

reload(plot)
plot.plot_fits(all_fits, force_rerun=force_rerun, verbose=verbose)

#%%

reload(plot)
if do_make_1D_scan:
    for parameter_1D_scan in parameters_1D_scan:
        plot.plot_1D_scan_fit_results(all_fits, **parameter_1D_scan)

#%%
# reload(plot)
# force_rerun=True
network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
plot.plot_number_of_contacts(network_files, force_rerun=force_rerun)

# %%
