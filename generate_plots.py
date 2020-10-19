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

x=x

#%%
plot.plot_ABM_simulations(abm_files, force_rerun=force_rerun)

#%%

# x=x

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
all_fits = fits.get_fit_results(abm_files, force_rerun=False, num_cores=num_cores, y_max=0.01)

#%%

reload(plot)
plot.plot_fits(all_fits, force_rerun=force_rerun, verbose=verbose)

#%%

reload(plot)
if do_make_1D_scan:
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
        "mu": 20,
        "beta": 0.012,
    },
)

cfgs = utils.query_cfg(d_query)
cfgs.sort(key=lambda cfg: cfg["N_events"])

pdf_name = Path(f"Figures/ABM_simulations_events.pdf")
utils.make_sure_folder_exist(pdf_name)
with PdfPages(pdf_name) as pdf:
    for cfg in tqdm(cfgs, desc="Plotting only events"):
        filenames = utils.hash_to_filenames(cfg.hash)
        fig, ax = plot.plot_single_ABM_simulation(cfg, abm_files)
        pdf.savefig(fig, dpi=100)
        plt.close("all")

#%%

d_query = utils.DotDict(
    {
        # "epsilon_rho": 0.02,
        "N_tot": 580_000,
        "rho": 0.0,
        "beta": 0.007,
    },
)

cfgs = utils.query_cfg(d_query)
for cfg in cfgs:
    print(cfg)
# cfgs.sort(key=lambda cfg: cfg["N_tot"])
# [cfg.hash for cfg in cfgs]
# %%


from src import SIR

from collections import deque
from bisect import insort, bisect_left
from itertools import islice


def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d) // 2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result


def plot_R_eff(cfg):
    filenames = abm_files.cfg_to_filenames(cfg)

    T_peaks = []
    for filename in filenames:
        df = file_loaders.pandas_load_file(filename)
        T_peak = df["time"].iloc[df["I"].argmax()]
        T_peaks.append(T_peak)

    fig, ax = plt.subplots()
    for i, filename in enumerate(filenames):
        df = file_loaders.pandas_load_file(filename)
        df_interpolated = SIR.interpolate_df(df)
        T_peak = df["time"].iloc[df["I"].argmax()]
        time = df_interpolated["time"].values[:-1] - T_peak + np.mean(T_peaks)
        S = (cfg.N_tot - df_interpolated[["E", "I", "R"]].sum(axis=1)).values
        # I = df_interpolated["I"].values
        R = df_interpolated["R"].values
        R_eff = -(S[1:] - S[:-1]) / (R[1:] - R[:-1])
        R_eff_running_median = np.array(running_median_insort(R_eff, 7))
        ax.plot(
            time,
            R_eff,
            "-k",
            alpha=0.1,
            label="$\mathcal{R}_\mathrm{eff}$" if i == 0 else None,
        )
        ax.plot(
            time,
            R_eff_running_median,
            "-k",
            label="Running median $(\mathcal{R}_\mathrm{eff}, 7)$" if i == 0 else None,
        )
    ax.legend()
    ax.set(ylim=(-0.01, np.percentile(R_eff_running_median, 95)))

    title = utils.dict_to_title(cfg, len(filenames))
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=0.82)

    return fig, ax


cfgs, _ = utils.get_1D_scan_cfgs_all_filenames(
    scan_parameter="beta",
    non_default_parameters={},
)
cfgs.sort(key=lambda cfg: cfg["beta"])


#%%

pdf_name = "Figures/R_eff.pdf"
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(pdf_name) as pdf:
    for cfg in tqdm(
        cfgs,
        desc="Plotting R_eff for beta 1D-scan",
    ):
        fig, ax = plot_R_eff(cfg)
        pdf.savefig(fig, dpi=100)
        plt.close("all")

# %%
