import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# from iminuit import Minuit
from collections import defaultdict
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits

rc_params.set_rc_params()
num_cores_max = 30

do_make_1D_scan = True
force_rerun = False
verbose = False

#%%

reload(plot)

abm_files = file_loaders.ABM_simulations()
N_files = len(abm_files)


#%%
plot.plot_ABM_simulations(abm_files, force_rerun=force_rerun)

#%%

parameters_1D_scan = [
    dict(scan_parameter="mu"),
    dict(scan_parameter="beta", non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="beta"),
    dict(scan_parameter="N_tot", do_log=True),
    dict(scan_parameter="N_tot", do_log=True, non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="N_init", do_log=True),
    dict(scan_parameter="N_init", do_log=True, non_default_parameters=dict(rho=0.1)),
    dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0)),
    dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0.02)),
    dict(scan_parameter="rho", non_default_parameters=dict(epsilon_rho=0.04)),
    dict(scan_parameter="rho", non_default_parameters=dict(beta=0.005)),
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


#%%

num_cores = utils.get_num_cores(num_cores_max)

all_fits = fits.get_fit_results(abm_files, force_rerun=False, num_cores=num_cores)

#%%

#%%

plot.plot_fits(all_fits, force_rerun=force_rerun, verbose=verbose)


#%%

# reload(plot)
if do_make_1D_scan:
    for parameter_1D_scan in parameters_1D_scan:
        plot.plot_1D_scan_fit_results(all_fits, **parameter_1D_scan)


#%%

# reload(plot)
# force_rerun = True

network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
plot.plot_number_of_contacts(network_files, force_rerun=force_rerun)

filename = network_files.all_files[0]


#%%

# reload(plot)

# # Custom Plots
# ABM_parameter = "N_tot__5800000__N_init__100__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__algo__2"
# fig, ax = plot.plot_single_fit(
#     ABM_parameter, all_fits, add_text=False, xlim=(0, 240), legend_loc={"I": "upper left", "R": "upper left"}
# )
# fig.savefig("Figures/paper/fits_vanilla.pdf")


# #%%

# ABM_parameter = "N_tot__5800000__N_init__100__rho__0.1__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__algo__2"
# fig, ax = plot.plot_single_fit(
#     ABM_parameter, all_fits, add_text=False, xlim=(0, 220), legend_loc={"I": "upper right", "R": "lower right"}
# )
# fig.savefig("Figures/paper/fits_rho.pdf")

# # %%
