import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src import rc_params
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd
from matplotlib.ticker import EngFormatter
from collections import defaultdict
import warnings

try:
    from src.utils import utils

    # from src import simulation_utils
    from src import file_loaders
    from src import SIR
except ImportError:
    import utils

    # import simulation_utils
    import file_loaders
    import SIR

from numba import njit
from functools import lru_cache

rc_params.set_rc_params()


def compute_df_deterministic(cfg, variable, T_max=100):
    # checks that the curve has flattened out
    while True:
        df_deterministic = SIR.integrate(cfg, T_max, dt=0.01, ts=1)
        delta_1_day = df_deterministic[variable].iloc[-2] - df_deterministic[variable].iloc[-1]
        if variable == "R":
            delta_1_day *= -1
        delta_rel = delta_1_day / cfg.N_tot
        if 0 <= delta_rel < 1e-5:
            break
        T_max *= 1.1
    return df_deterministic


def plot_ABM_simulations(abm_files, force_rerun=False):

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/ABM_simulations.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")

        d_ylabel = {"I": "Infected", "R": "Recovered"}
        d_label_loc = {"I": "upper right", "R": "lower right"}

        for ABM_parameter in tqdm(abm_files.keys, desc="Plotting individual ABM parameters"):
            # break

            cfg = utils.string_to_dict(ABM_parameter)

            fig, axes = plt.subplots(ncols=2, figsize=(18, 7), constrained_layout=True)
            fig.subplots_adjust(top=0.8)

            T_max = 0
            lw = 0.1 * 10 / np.sqrt(len(abm_files[ABM_parameter]))

            stochastic_noise_I = []
            stochastic_noise_R = []

            # file, i = abm_files[ABM_parameter][0], 0
            for i, file in enumerate(abm_files[ABM_parameter]):
                df = file_loaders.pandas_load_file(file)
                t = df["time"].values
                label = "ABM" if i == 0 else None

                axes[0].plot(t, df["I"], lw=lw, c="k", label=label)
                axes[1].plot(t, df["R"], lw=lw, c="k", label=label)

                if t.max() > T_max:
                    T_max = t.max()

                stochastic_noise_I.append(df["I"].max())
                stochastic_noise_R.append(df["R"].iloc[-1])

            for variable, ax in zip(["I", "R"], axes):

                df_deterministic = compute_df_deterministic(cfg, variable, T_max=T_max)

                ax.plot(
                    df_deterministic["time"],
                    df_deterministic[variable],
                    lw=2.5,
                    color="red",
                    label="SEIR",
                )
                leg = ax.legend(loc=d_label_loc[variable])
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)

                ax.set(xlabel="Time", ylim=(0, None), ylabel=d_ylabel[variable])
                # ax.set_xlabel('Time', ha='right')
                ax.xaxis.set_label_coords(0.91, -0.14)
                ax.yaxis.set_major_formatter(EngFormatter())

            names = [r"I_\mathrm{max}^\mathrm{ABM}", r"R_\infty^\mathrm{ABM}"]
            for name, x, ax in zip(names, [stochastic_noise_I, stochastic_noise_R], axes):

                mu, std = np.mean(x), utils.SDOM(x)

                n_digits = int(np.log10(utils.round_to_uncertainty(mu, std)[0])) + 1
                rel_uncertainty = std / mu
                s_mu = utils.human_format_scientific(mu, digits=n_digits)

                s = (
                    r"$ "
                    + f"{name} = ({s_mu[0]}"
                    + r"\pm "
                    + f"{rel_uncertainty*100:.2}"
                    + r"\% )"
                    + r"\cdot "
                    + f"{s_mu[1]}"
                    + r"$"
                )
                ax.text(
                    -0.1,
                    -0.2,
                    s,
                    horizontalalignment="left",
                    transform=ax.transAxes,
                    fontsize=24,
                )

            title = utils.dict_to_title(cfg, len(abm_files[ABM_parameter]))
            fig.suptitle(title, fontsize=24)
            plt.subplots_adjust(wspace=0.3)

            pdf.savefig(fig, dpi=100)
            plt.close("all")


# %%


def compute_ABM_SEIR_proportions(filenames):
    "Compute the fraction (z) between ABM and SEIR for I_max and R_inf "

    I_max_ABM = []
    R_inf_ABM = []
    for filename in filenames:
        try:
            df = file_loaders.pandas_load_file(filename)
        except EmptyDataError:
            print(f"Empty file error at {filename}")
            continue
        I_max_ABM.append(df["I"].max())
        R_inf_ABM.append(df["R"].iloc[-1])
    I_max_ABM = np.array(I_max_ABM)
    R_inf_ABM = np.array(R_inf_ABM)

    T_max = max(df["time"].max() * 1.2, 300)
    cfg = utils.string_to_dict(filename)
    df_SIR = SIR.integrate(cfg, T_max, dt=0.01, ts=0.1)

    # break out if the SIR model dies out
    if df_SIR["I"].max() < cfg.N_init:
        N = len(I_max_ABM)
        return np.full(N, np.nan), np.full(N, np.nan), cfg

    z_rel_I = I_max_ABM / df_SIR["I"].max()
    z_rel_R = R_inf_ABM / df_SIR["R"].iloc[-1]

    return z_rel_I, z_rel_R, cfg


def get_1D_scan_results(scan_parameter, non_default_parameters):
    "Compute the fraction between ABM and SEIR for all simulations related to the scan_parameter"

    simulation_parameters_1D_scan = utils.get_simulation_parameters_1D_scan(scan_parameter, non_default_parameters)
    N_simulation_parameters = len(simulation_parameters_1D_scan)
    if N_simulation_parameters <= 1:
        return None

    base_dir = Path("Data") / "ABM"

    x = np.zeros(N_simulation_parameters)
    y_I = np.zeros(N_simulation_parameters)
    y_R = np.zeros(N_simulation_parameters)
    sy_I = np.zeros(N_simulation_parameters)
    sy_R = np.zeros(N_simulation_parameters)
    n = np.zeros(N_simulation_parameters)

    # ABM_parameter = simulation_parameters_1D_scan[0]
    for i, ABM_parameter in enumerate(tqdm(simulation_parameters_1D_scan, desc=scan_parameter)):
        filenames = [str(filename) for filename in base_dir.rglob("*.csv") if f"{ABM_parameter}/" in str(filename)]

        z_rel_I, z_rel_R, cfg = compute_ABM_SEIR_proportions(filenames)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_I[i] = utils.SDOM(z_rel_I)
        sy_R[i] = utils.SDOM(z_rel_R)
        n[i] = len(z_rel_I)

    if np.isfinite(y_I).sum() <= 1 and np.isfinite(y_R).sum() <= 1:
        return None

    return x, y_I, y_R, sy_I, sy_R, n, cfg


def extract_limits(lim):
    """ deals with both limits of the form (0, 1) and [(0, 1), (0.5, 1.5)] """
    if isinstance(lim, (tuple, list)):
        if isinstance(lim[0], (float, int)):
            lim0 = lim1 = lim
        elif isinstance(lim[0], (tuple, list)):
            lim0, lim1 = lim
    else:
        lim0 = lim1 = (None, None)

    return lim0, lim1


def _plot_1D_scan_res(res, scan_parameter, ylim, do_log):

    x, y_I, y_R, sy_I, sy_R, n, cfg = res

    d_par_pretty = utils.get_parameter_to_latex()
    title = utils.dict_to_title(cfg, exclude=[scan_parameter, "ID"])
    xlabel = r"$" + d_par_pretty[scan_parameter] + r"$"

    ylim0, ylim1 = extract_limits(ylim)

    # n>1 datapoints
    mask = n > 1

    factor = 0.8
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16 * factor, 9 * factor))  #
    fig.suptitle(title, fontsize=28 * factor)

    ax0.errorbar(
        x[mask],
        y_I[mask],
        sy_I[mask],
        fmt=".",
        color="black",
        ecolor="black",
        elinewidth=1,
        capsize=10,
    )
    ax0.errorbar(
        x[~mask],
        y_I[~mask],
        sy_I[~mask],
        fmt=".",
        color="grey",
        ecolor="grey",
        elinewidth=1,
        capsize=10,
    )
    ax0.set(xlabel=xlabel, ylim=ylim0)

    ax1.errorbar(
        x[mask],
        y_R[mask],
        sy_R[mask],
        fmt=".",
        color="black",
        ecolor="black",
        elinewidth=1,
        capsize=10,
    )
    ax1.errorbar(
        x[~mask],
        y_R[~mask],
        sy_R[~mask],
        fmt=".",
        color="grey",
        ecolor="grey",
        elinewidth=1,
        capsize=10,
    )
    ax1.set(xlabel=xlabel, ylim=ylim1)

    if do_log:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    fig.tight_layout()
    fig.subplots_adjust(top=0.8, wspace=0.45)

    return fig, (ax0, ax1)


from pandas.errors import EmptyDataError


def plot_1D_scan(scan_parameter, do_log=False, ylim=None, non_default_parameters=None):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_results(scan_parameter, non_default_parameters)
    if not res:
        return None

    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log)

    ax0.set(ylabel=r"$I_\mathrm{max}^\mathrm{ABM} \, / \,\, I_\mathrm{max}^\mathrm{SEIR}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{ABM} \, / \,\, R_\infty^\mathrm{SEIR}$")

    figname_pdf = f"Figures/1D_scan/1D_scan_{scan_parameter}"
    for key, val in non_default_parameters.items():
        figname_pdf += f"_{key}_{val}"
    figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%


def rug_plot(xs, ax, ymax=0.1, **kwargs):
    for x in xs:
        ax.axvline(x, ymax=ymax, **kwargs)


def plot_single_fit(ABM_parameter, all_fits, add_text=True, xlim=(0, None), ylim=(0, None), legend_loc=None):

    fit_objects = all_fits[ABM_parameter]

    d_ylabel = {"I": "Infected", "R": "Recovered"}

    if legend_loc is None:
        legend_loc = {"I": "upper right", "R": "lower right"}

    relative_names = [
        r"\frac{I_\mathrm{max}^\mathrm{fit}} {I_\mathrm{max}^\mathrm{ABM}}",
        r"\frac{R_\inf^\mathrm{fit}}{R_\inf^\mathrm{fit}}",
    ]

    cfg = utils.string_to_dict(ABM_parameter)
    axhline_ymin = 0.1 / 100 * cfg.N_tot
    axhline_ymax = 1 / 100 * cfg.N_tot

    fig, axes = plt.subplots(ncols=2, figsize=(18, 7), constrained_layout=True)
    fig.subplots_adjust(top=0.8)

    for i, fit_object in enumerate(fit_objects.values()):
        # break

        df = file_loaders.pandas_load_file(fit_object.filename)
        t = df["time"].values
        T_max = max(t) * 1.1
        df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)

        lw = 0.8
        for I_or_R, ax in zip(["I", "R"], axes):

            label_min = "Fit Range" if i == 0 else None
            axvline_kwargs = dict(lw=lw, alpha=0.9, color="lightgrey")
            tmp = df.query("@fit_object.t.min() <= time <= @fit_object.t.max()")
            ax.fill_between(tmp["time"], tmp[I_or_R], label=label_min, **axvline_kwargs)

            label = "ABM" if i == 0 else None
            ax.plot(t, df[I_or_R], "k-", lw=lw, label=label)

            label = "Fits" if i == 0 else None
            ax.plot(
                df_fit["time"],
                df_fit[I_or_R],
                lw=lw,
                color="green",
                label=label,
            )

            # x_rug = [fit_object.t.min(), fit_object.t.max()]
            # rug_plot(x_rug, ax, ymin=-0.01, ymax=0.03, color="k", lw=lw)

    if add_text:

        # calculate monte carlo simulated fit results
        all_I_max_MC = []
        all_R_inf_MC = []
        for i, fit_object in enumerate(fit_objects.values()):
            all_I_max_MC.extend(fit_object.I_max_MC)
            all_R_inf_MC.extend(fit_object.R_inf_MC)
        d_fits = {"I": all_I_max_MC, "R": all_R_inf_MC}

        # and plot them
        for I_or_R, ax in zip(["I", "R"], axes):
            median, errors = utils.get_central_confidence_intervals(d_fits[I_or_R])
            s = utils.format_asymmetric_uncertanties(median, errors, I_or_R)
            axes[0].text(-0.15, -0.25, s, transform=ax.transAxes, fontsize=22)

        # calculate fraction between fit and ABM simulation
        z = defaultdict(list)
        for fit_object in fit_objects.values():
            z["I"].append(fit_object.I_max_fit / fit_object.I_max_ABM)
            z["R"].append(fit_object.R_inf_fit / fit_object.R_inf_ABM)

        for I_or_R, name, ax in zip(["I", "R"], relative_names, axes):
            mu, std = np.mean(z[I_or_R]), utils.SDOM(z[I_or_R])
            n_digits = int(np.log10(utils.round_to_uncertainty(mu, std)[0])) + 1
            s_mu = utils.human_format_scientific(mu, digits=n_digits)
            s = r"$ " + f"{name} = {s_mu[0]}" + r"\pm " + f"{std:.{n_digits}f}" + r"$"
            ax.text(0.3, -0.25, s, transform=ax.transAxes, fontsize=22)

    fit_values_deterministic = {
        "lambda_E": cfg.lambda_E,
        "lambda_I": cfg.lambda_I,
        "beta": cfg.beta,
        "tau": 0,
    }

    df_SIR = fit_object.calc_df_fit(fit_values=fit_values_deterministic, ts=0.1, T_max=T_max)

    for I_or_R, ax in zip(["I", "R"], axes):

        ax.plot(
            df_SIR["time"],
            df_SIR[I_or_R],
            lw=lw * 8,
            color="red",
            label="SEIR",
            zorder=0,
        )

        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel="Time", ylabel=d_ylabel[I_or_R])
        if add_text:
            ax.xaxis.set_label_coords(0.91, -0.14)
        ax.yaxis.set_major_formatter(EngFormatter())

        leg = ax.legend(loc=legend_loc[I_or_R])
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_alpha(1.0)

    title = utils.dict_to_title(cfg, len(fit_objects))
    fig.suptitle(title, fontsize=24)
    plt.subplots_adjust(wspace=0.3)

    return fig, ax


def plot_fits(all_fits, force_rerun=False, verbose=False):

    pdf_name = f"Figures/Fits.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")

        for ABM_parameter, fit_objects in tqdm(all_fits.items(), desc="Plotting all fits"):
            # break
            # cfg = utils.string_to_dict(ABM_parameter)
            # if cfg.N_tot == 5_800_000 and cfg.rho == 0.1:
            # break

            # skip if no fits
            if len(fit_objects) == 0:
                if verbose:
                    print(f"Skipping {ABM_parameter}")
                continue

            fig, ax = plot_single_fit(ABM_parameter, all_fits)

            pdf.savefig(fig, dpi=100)
            plt.close("all")


#%%


def compute_fit_ABM_proportions(fit_objects):
    "Compute the fraction (z) between the fits and the ABM simulations for I_max and R_inf "

    N = len(fit_objects)

    I_max_fit = np.zeros(N)
    R_inf_fit = np.zeros(N)
    I_max_ABM = np.zeros(N)
    R_inf_ABM = np.zeros(N)

    for i, fit_object in enumerate(fit_objects.values()):
        # break

        df = file_loaders.pandas_load_file(fit_object.filename)
        I_max_ABM[i] = df["I"].max()
        R_inf_ABM[i] = df["R"].iloc[-1]

        t = df["time"].values
        T_max = max(t) * 1.1
        df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)
        I_max_fit[i] = df_fit["I"].max()
        R_inf_fit[i] = df_fit["R"].iloc[-1]

    z_rel_I = I_max_fit / I_max_ABM
    z_rel_R = R_inf_fit / R_inf_ABM

    return z_rel_I, z_rel_R


def get_1D_scan_fit_results(all_fits, scan_parameter, non_default_parameters):
    "Compute the fraction between ABM and SEIR for all simulations related to the scan_parameter"

    simulation_parameters_1D_scan = utils.get_simulation_parameters_1D_scan(scan_parameter, non_default_parameters)

    selected_fits = {key: val for key, val in all_fits.items() if key in simulation_parameters_1D_scan}

    N_simulation_parameters = len(selected_fits)
    if N_simulation_parameters <= 1:
        return None

    N = len(selected_fits)

    x = np.zeros(N)
    y_I = np.zeros(N)
    y_R = np.zeros(N)
    sy_I = np.zeros(N)
    sy_R = np.zeros(N)
    n = np.zeros(N)

    it = tqdm(enumerate(selected_fits.items()), desc=scan_parameter, total=N)
    for i, (ABM_parameter, fit_objects) in it:
        # break

        cfg = utils.string_to_dict(ABM_parameter)
        z_rel_I, z_rel_R = compute_fit_ABM_proportions(fit_objects)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_I[i] = utils.SDOM(z_rel_I)
        sy_R[i] = utils.SDOM(z_rel_R)
        n[i] = len(z_rel_I)

    return x, y_I, y_R, sy_I, sy_R, n, cfg


from pandas.errors import EmptyDataError


def plot_1D_scan_fit_results(all_fits, scan_parameter, do_log=False, ylim=None, non_default_parameters=None):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_fit_results(all_fits, scan_parameter, non_default_parameters)
    if not res:
        return None

    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log)

    ax0.set(ylabel=r"$I_\mathrm{max}^\mathrm{fit} \, / \,\, I_\mathrm{max}^\mathrm{ABM}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{fit} \, / \,\, R_\infty^\mathrm{ABM}$")

    figname_pdf = f"Figures/1D_scan_fits/1D_scan_fit_{scan_parameter}"
    for key, val in non_default_parameters.items():
        figname_pdf += f"_{key}_{val}"
    figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%

import h5py
from matplotlib.ticker import EngFormatter


def _load_my_state_and_my_number_of_contacts(filename):
    with h5py.File(filename, "r") as f:
        my_state = f["my_state"][()]
        my_number_of_contacts = f["my_number_of_contacts"][()]
    return my_state, my_number_of_contacts


def _plot_number_of_contacts(filename):

    my_state, my_number_of_contacts = _load_my_state_and_my_number_of_contacts(filename)

    mask_S = my_state[-1] == -1
    mask_R = my_state[-1] == 8

    x_min = np.percentile(my_number_of_contacts, 0.01)
    x_max = np.percentile(my_number_of_contacts, 99)
    x_range = (x_min, x_max)
    N_bins = int(x_max - x_min)

    d_colors = {"Total": "C0", "Immune": "C2", "Risk": "C1"}
    kwargs = {"bins": N_bins, "range": x_range, "histtype": "step"}

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    H_all = ax1.hist(my_number_of_contacts, label="Total", color=d_colors["Total"], **kwargs)
    H_S = ax1.hist(my_number_of_contacts[mask_S], label="At Risk", color=d_colors["Risk"], **kwargs)
    H_R = ax1.hist(my_number_of_contacts[mask_R], label="Immune", color=d_colors["Immune"], **kwargs)

    x = 0.5 * (H_all[1][:-1] + H_all[1][1:])
    frac_S = H_S[0] / H_all[0]
    s_frac_S = np.sqrt(frac_S * (1 - frac_S) / H_all[0])
    frac_R = H_R[0] / H_all[0]
    s_frac_R = np.sqrt(frac_R * (1 - frac_R) / H_all[0])

    kwargs_errorbar = dict(fmt=".", elinewidth=1.5, capsize=4, capthick=1.5)
    ax2.errorbar(x, frac_S, s_frac_S, color=d_colors["Risk"], **kwargs_errorbar)
    ax2.errorbar(x, frac_R, s_frac_R, color=d_colors["Immune"], **kwargs_errorbar)

    ax1.legend()
    # ax1.legend()
    title = utils.string_to_title(filename)
    ax1.yaxis.set_major_formatter(EngFormatter())
    ax1.set(ylabel="Counts", xlim=x_range)
    ax1.set_title(title, pad=40, fontsize=30)
    ax2.set(xlabel="Total number of contacts", ylim=(0, 1), ylabel=r"Fraction")

    return fig, ax1


def plot_number_of_contacts(network_files, force_rerun=False):

    if len(network_files) == 0:
        return None

    pdf_name = f"Figures/Number_of_contacts.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")
        for network_filename in tqdm(network_files, desc="Number of contacts"):
            cfg = utils.string_to_dict(str(network_filename))
            if cfg.ID != 0:
                continue
            else:
                fig, ax = _plot_number_of_contacts(network_filename)
                pdf.savefig(fig, dpi=100)
                plt.close("all")


#%%


# def _load_my_state_and_coordinates(filename):
#     with h5py.File(filename, "r") as f:
#         my_state = f["my_state"][()]
#         coordinates = f["coordinates"][()]
#     return my_state, coordinates


# @njit
# def hist2d_numba(data_2D, bins, ranges):
#     H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
#     delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)
#     for t in range(data_2D.shape[0]):
#         i = (data_2D[t, 0] - ranges[0, 0]) * delta[0]
#         j = (data_2D[t, 1] - ranges[1, 0]) * delta[1]
#         if 0 <= i < bins[0] and 0 <= j < bins[1]:
#             H[int(i), int(j)] += 1
#     return H


# @njit
# def get_ranges(x):
#     return np.array(([x[:, 0].min(), x[:, 0].max()], [x[:, 1].min(), x[:, 1].max()]))


# def histogram2d(data_2D, bins=None, ranges=None):
#     if bins is None:
#         print("No binning provided, using (100, 100) as default")
#         bins = np.array((100, 100))
#     if isinstance(bins, int):
#         bins = np.array([bins, bins])
#     elif isinstance(bins, list) or isinstance(bins, tuple):
#         bins = np.array(bins, dtype=int)
#     if ranges is None:
#         ranges = get_ranges(data_2D)
#         ranges[:, 0] *= 0.99
#         ranges[:, 1] *= 1.01
#     return hist2d_numba(data_2D, bins=bins, ranges=ranges)


# def _get_N_bins_xy(coordinates):

#     lon_min = coordinates[:, 0].min()
#     lon_max = coordinates[:, 0].max()
#     lon_mid = np.mean([lon_min, lon_max])

#     lat_min = coordinates[:, 1].min()
#     lat_max = coordinates[:, 1].max()
#     lat_mid = np.mean([lat_min, lat_max])

#     N_bins_x = int(utils.haversine(lon_min, lat_mid, lon_max, lat_mid)) + 1
#     N_bins_y = int(utils.haversine(lon_mid, lat_min, lon_mid, lat_max)) + 1

#     return N_bins_x, N_bins_y


# if False:

#     filename = "Data/network/N_tot__5800000__N_init__100__rho__0.01__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__algo__2__ID__0.hdf5"

#     day = -1
#     grid_km = 2

#     my_state, coordinates = _load_my_state_and_coordinates(filename)

#     N_bins_x, N_bins_y = _get_N_bins_xy(coordinates)
#     ranges = get_ranges(coordinates)

#     my_state_day = my_state[day]

#     kwargs = {"bins": (N_bins_x / grid_km, N_bins_y / grid_km), "ranges": ranges}
#     counts_1d_all = histogram2d(coordinates, **kwargs)
#     counts_1d_R = histogram2d(coordinates[my_state_day == 8], **kwargs)

#     H = counts_1d_R / counts_1d_all
#     H_1d = H[counts_1d_all > 0]

#     from matplotlib.colors import LogNorm

#     fig, ax = plt.subplots()
#     norm = LogNorm(vmin=0.01, vmax=1)
#     im = ax.imshow(H.T, interpolation="none", origin="lower", cmap="viridis")  # norm=norm
#     fig.colorbar(im, ax=ax, extend="max")

#     fig2, ax2 = plt.subplots()
#     ax2.hist(H[counts_1d_all > 0], 100)
#     # ax2.set_yscale("log")

#     fig3, ax3 = plt.subplots()
#     ax3.hist(counts_1d_all.flatten(), 100)
#     # ax3.set_yscale("log")


#     # %%
