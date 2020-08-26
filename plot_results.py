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
from src import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits
rc_params.set_rc_params()
num_cores_max = 30

savefig = False
do_animate = False
save_and_show_all_plots = True
plot_SIR_comparison = True if utils.is_local_computer() else False

#%%


reload(plot)

abn_files = file_loaders.ABNFiles()
N_files = len(abn_files)


if plot_SIR_comparison:

    plot.make_SIR_curves(abn_files, 'I', force_overwrite=False)
    plot.make_SIR_curves(abn_files, 'R', force_overwrite=False)



#%%

if False:

    # reload(plot)

    plot.plot_1D_scan('mu')
    plot.plot_1D_scan('beta')

    plot.plot_1D_scan('N_tot', do_log=True)
    plot.plot_1D_scan('N_init', do_log=True, ylim=(0.9, 1.))
    plot.plot_1D_scan('N_init', do_log=True, ylim=(0.9, 1.), non_default_parameters=dict(algo= 1))

    plot.plot_1D_scan('beta_scaling',  non_default_parameters=dict(N_tot=5_800_000, epsilon_rho=0, rho=100, N_init=100))
    plot.plot_1D_scan('beta_scaling', non_default_parameters=dict(N_tot=5_800_000, epsilon_rho=0, rho=100, N_init=1000))
    plot.plot_1D_scan('beta_scaling',  non_default_parameters=dict(N_tot=580_000,   epsilon_rho=0, rho=100, N_init=100))
    plot.plot_1D_scan('beta_scaling',  non_default_parameters=dict(N_tot=580_000,   epsilon_rho=0, rho=100, N_init=1000))

    plot.plot_1D_scan('rho')
    plot.plot_1D_scan('rho', non_default_parameters=dict(algo=1))
    plot.plot_1D_scan('rho', non_default_parameters=dict(N_tot=5_800_000))
    plot.plot_1D_scan('epsilon_rho', non_default_parameters=dict(rho=100))
    plot.plot_1D_scan('epsilon_rho', non_default_parameters=dict(rho=100, algo=1))

    plot.plot_1D_scan('sigma_beta')
    plot.plot_1D_scan('sigma_beta', non_default_parameters=dict(algo=1))
    plot.plot_1D_scan('sigma_beta', non_default_parameters=dict(rho=100))
    plot.plot_1D_scan('sigma_beta', non_default_parameters=dict(rho=100, algo=1))
    plot.plot_1D_scan('sigma_beta', non_default_parameters=dict(sigma_mu=1))
    plot.plot_1D_scan('sigma_beta', non_default_parameters=dict(sigma_mu=1, rho=100))
    plot.plot_1D_scan('sigma_beta', non_default_parameters=dict(rho=100))
    plot.plot_1D_scan('sigma_mu')
    plot.plot_1D_scan('sigma_mu', non_default_parameters=dict(algo=1))
    plot.plot_1D_scan('sigma_mu', non_default_parameters=dict(rho=100))
    plot.plot_1D_scan('sigma_mu', non_default_parameters=dict(rho=100, algo=1))
    plot.plot_1D_scan('lambda_E')
    plot.plot_1D_scan('lambda_I')


#%%


#%%

num_cores = utils.get_num_cores(num_cores_max)

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="covariance is not positive-semidefinite.")
    all_fits = fits.get_fit_results(abn_files, force_rerun=False, num_cores=num_cores)


#%%


from scipy.special import erf
def get_confidence_intervals_median(x, N_sigma=1):
    median = np.median(x)
    sigma = 100*erf(N_sigma/np.sqrt(2))
    p_lower = 50 - sigma/2
    p_upper = 50 + sigma/2
    lower_bound = np.percentile(x, p_lower)
    upper_bound = np.percentile(x, p_upper)
    errors = median-lower_bound, upper_bound-median
    return median, errors


#%%

from matplotlib.ticker import EngFormatter

do_log = False
def plot_fits(all_fits, force_overwrite=False, verbose=False, do_log=False):

    pdf_name = f"Figures/Fits.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_overwrite:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf:

        leg_loc = {'I': 'upper right', 'R': 'lower right'}
        d_ylabel = {'I': 'Infected', 'R': 'Recovered'}

        for ABN_parameter, fit_objects in tqdm(all_fits.items()):
            # break

            if len(fit_objects) == 0:
                print(f"Skipping {ABN_parameter}")
                continue

            cfg = utils.string_to_dict(ABN_parameter)

            fit_values_deterministic = {'lambda_E': cfg.lambda_E,
                          'lambda_I': cfg.lambda_I,
                          'beta': cfg.beta,
                          'tau': 0}


            fig, axes = plt.subplots(ncols=2, figsize=(18, 7), constrained_layout=True)
            fig.subplots_adjust(top=0.8)


            for i, fit_object in enumerate(fit_objects.values()):
                # break

                df = file_loaders.pandas_load_file(fit_object.filename)
                t = df['time'].values
                T_max = max(t)*1.1
                df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)

                # if df_fit['I'].max() > 150_000:
                #     assert False

                lw = 0.8
                for I_or_R, ax in zip(['I', 'R'], axes):

                    label = 'Simulations' if i == 0 else None
                    ax.plot(t, df[I_or_R], 'k-', lw=lw, label=label)

                    label_min = 'Fit Range' if i == 0 else None
                    ax.axvline(fit_object.t.min(), lw=lw, alpha=0.8, label=label_min)
                    ax.axvline(fit_object.t.max(), lw=lw, alpha=0.8)

                    label = 'Fits' if i == 0 else None
                    ax.plot(df_fit['time'], df_fit[I_or_R], lw=lw, color='green', label=label)


            human = utils.human_format


            all_I_max_MC = []
            all_R_inf_MC = []
            for i, fit_object in enumerate(fit_objects.values()):
                all_I_max_MC.extend(fit_object.I_max_MC)
                all_R_inf_MC.extend(fit_object.R_inf_MC)

            I_median, I_errors = get_confidence_intervals_median(all_I_max_MC)
            s = utils.format_uncertanties(I_median, I_errors, 'I')
            axes[0].text(-0.15, -0.25, s, horizontalalignment='left',
                    transform=axes[0].transAxes, fontsize=24)


            R_median, R_errors = get_confidence_intervals_median(all_R_inf_MC)
            s = utils.format_uncertanties(R_median, R_errors, 'R')
            axes[1].text(-0.15, -0.25, s, horizontalalignment='left',
                    transform=axes[1].transAxes, fontsize=24)


            df_SIR = fit_object.calc_df_fit(fit_values=fit_values_deterministic, ts=0.1, T_max=T_max)

            for I_or_R, ax in zip(['I', 'R'], axes):

                ax.plot(df_SIR['time'], df_SIR[I_or_R], lw=lw*5, color='red', label='SIR', zorder=0)

                if do_log:
                    ax.set_yscale('log', nonposy='clip')

                ax.set(xlim=(0, None), ylim=(0, None))

                ax.set(xlabel='Time', ylabel=d_ylabel[I_or_R])
                ax.set_rasterized(True)
                ax.set_rasterization_zorder(0)
                ax.yaxis.set_major_formatter(EngFormatter())

                leg = ax.legend(loc=leg_loc[I_or_R])
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
                    legobj.set_alpha(1.0)

            title = utils.dict_to_title(cfg, len(fit_objects))
            fig.suptitle(title, fontsize=24)
            plt.subplots_adjust(wspace=0.3)


            pdf.savefig(fig, dpi=100)
            plt.close('all')


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")
    plot_fits(all_fits, force_overwrite=False, verbose=False, do_log=False)


#%%






