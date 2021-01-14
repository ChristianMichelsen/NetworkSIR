import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src import rc_params
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd
from matplotlib.ticker import PercentFormatter, EngFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
from collections import defaultdict
import warnings
from importlib import reload

try:
    from src.utils import utils

    # from src import simulation_utils
    from src import file_loaders
    from src import SIR
    from src import database
except ImportError:
    import utils

    # import simulation_utils
    import file_loaders
    import SIR
    import database
import generate_R_eff_fits


from numba import njit
from functools import lru_cache

rc_params.set_rc_params()

# https://carto.com/carto-colors/
d_colors = {
    "blue": "#1D6996",
    "red": "#E41A1C",
    "green": "#0F8554",
    "orange": "#E17C05",
    "purple": "#94346E",
    "light_blue": "#38A6A5",
    "light_green": "#73AF48",
    "yellow": "#EDAD08",
    "grey": "#666666",
}


def compute_df_deterministic(cfg, variable, T_max=100):
    # checks that the curve has flattened out
    while True:
        df_deterministic = SIR.integrate(cfg, T_max, dt=0.01, ts=0.1)
        delta_1_day = df_deterministic[variable].iloc[-2] - df_deterministic[variable].iloc[-1]
        if variable == "R":
            delta_1_day *= -1
        delta_rel = delta_1_day / cfg.N_tot
        if 0 <= delta_rel < 1e-5:
            break
        T_max *= 1.1
    return df_deterministic


# class LatexEngFormatter(EngFormatter):
#     def __init__(self, unit="", places=None, sep=" "):
#         self.has_printed = False
#         super().__init__(unit=unit, places=places, sep=sep, usetex=False, useMathText=False)

#     def __call__(self, x, pos=None):
#         s = super().__call__(x, pos)
#         s = s.split(self.sep)
#         s[-1] = r"\, \mathrm{" + s[-1] + r"}"
#         s = r"$" + self.sep.join(s) + r"$"
#         return s


def plot_single_ABM_simulation(
    cfg,
    abm_files,
    add_top_text=True,
    xlim=(0, None),
    ylim_scale=1.0,
    legend_fontsize=30,
    d_label_loc=None,
):

    filenames = abm_files.cfg_to_filenames(cfg)

    if not isinstance(cfg, utils.DotDict):
        cfg = utils.DotDict(cfg)

    d_ylabel = {"I": "Fraction Infected", "R": "Fraction Recovered"}
    if d_label_loc is None:
        d_label_loc = {"I": "upper right", "R": "lower right"}

    N_tot = cfg.N_tot

    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.75)

    T_max = 0
    lw = 0.3 * 10 / np.sqrt(len(filenames))
    lw_SEIR = 4

    stochastic_noise_I = []
    stochastic_noise_R = []

    # file, i = abm_files[ABM_parameter][0], 0
    for i, filename in enumerate(filenames):
        # break
        df = file_loaders.pandas_load_file(filename)
        t = df["time"].values
        label = r"ABM" if i == 0 else None

        axes[0].plot(t, df["I"] / N_tot, lw=lw, c="k", label=label)
        axes[1].plot(t, df["R"] / N_tot, lw=lw, c="k", label=label)

        if t.max() > T_max:
            T_max = t.max()

        stochastic_noise_I.append(df["I"].max())
        stochastic_noise_R.append(df["R"].iloc[-1])

    for variable, ax in zip(["I", "R"], axes):

        df_deterministic = compute_df_deterministic(cfg, variable, T_max=T_max)

        ax.plot(
            df_deterministic["time"],
            df_deterministic[variable] / N_tot,
            lw=lw_SEIR,
            color=d_colors["red"],
            label="SEIR",
        )
        leg = ax.legend(loc=d_label_loc[variable], fontsize=legend_fontsize)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(lw_SEIR)

        ax.set(
            xlabel="Time [days]",
            ylim=(0, None),
            ylabel=d_ylabel[variable],
            xlim=xlim,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * ylim_scale)
        # ax.set_xlabel('Time', ha='right')
        # ax.xaxis.set_label_coords(0.91, -0.14)
        # ax.yaxis.set_major_formatter(EngFormatter())
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    if add_top_text:
        names = [r"I_\mathrm{peak}^\mathrm{ABM}", r"R_\infty^\mathrm{ABM}"]
        for name, x, ax in zip(names, [stochastic_noise_I, stochastic_noise_R], axes):
            s = utils.format_relative_uncertainties(x, name)
            ax.text(
                0.5,
                1.05,
                s,
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )

    title = utils.dict_to_title(cfg, len(filenames))
    fig.suptitle(title, fontsize=15)
    plt.subplots_adjust(wspace=0.4)

    return fig, axes


def plot_single_ABM_simulation_test_focus(
    cfg, abm_files, add_top_text=True, xlim=(0, None), legend_fontsize=30
):

    filenames = abm_files.cfg_to_filenames(cfg)
    if filenames is None:
        return None

    if not isinstance(cfg, utils.DotDict):
        cfg = utils.DotDict(cfg)

    d_ylabel = {"I": "Infected", "R": "Recovered"}
    d_label_loc = {"I": "upper right", "R": "lower right"}

    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)

    T_max = 0
    lw = 0.3 * 10 / np.sqrt(len(filenames))

    stochastic_noise_I = []
    stochastic_noise_R = []

    # file, i = abm_files[ABM_parameter][0], 0
    for i, filename in enumerate(filenames):
        df = file_loaders.pandas_load_file(filename)
        t = df["time"].values
        label = r"ABM" if i == 0 else None

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
            lw=lw * 4,
            color=d_colors["red"],
            label="SEIR",
        )
        leg = ax.legend(loc=d_label_loc[variable], fontsize=legend_fontsize)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(lw * 4)

        ax.set(
            xlabel="Time [days]",
            ylim=(0, None),
            ylabel=d_ylabel[variable],
            xlim=xlim,
        )
        # ax.set_xlabel('Time', ha='right')
        # ax.xaxis.set_label_coords(0.91, -0.14)
        ax.yaxis.set_major_formatter(EngFormatter())

    if add_top_text:
        names = [r"I_\mathrm{max}^\mathrm{ABM}", r"R_\infty^\mathrm{ABM}"]
        for name, x, ax in zip(names, [stochastic_noise_I, stochastic_noise_R], axes):
            s = utils.format_relative_uncertainties(x, name)
            ax.text(
                0.5,
                1.05,
                s,
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )

    title = utils.dict_to_title(cfg, len(filenames))
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    return fig, ax


def plot_ABM_simulations(abm_files, force_rerun=False, plot_found_vs_real_inf=False, **kwargs):

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/ABM_simulations.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_rerun:
        print(f"{pdf_name} already exists\n", flush=True)
        return None

    # def iter_folders(self):
    #     for cfg in self.cfgs:
    #         filenames = self.d[cfg.hash]
    #         yield cfg, filenames

    with PdfPages(pdf_name) as pdf:

        # for ABM_parameter in tqdm(abm_files.keys, desc="Plotting individual ABM parameters"):
        for cfg in tqdm(
            abm_files.iter_cfgs(),
            desc="Plotting individual ABM parameters",
            total=len(abm_files.cfgs),
        ):

            #     break
            if plot_found_vs_real_inf:
                fig_ax = plot_single_ABM_simulation_test_focus(cfg, abm_files)
            else:
                fig_ax = plot_single_ABM_simulation(cfg, abm_files, **kwargs)

            if fig_ax is not None:
                fig, ax = fig_ax
                pdf.savefig(fig, dpi=100)
            plt.close("all")


# %%


def compute_ABM_SEIR_proportions(cfg, filenames):
    "Compute the fraction (z) between ABM and SEIR for I_max and R_inf "

    I_max_ABM = []
    R_inf_ABM = []
    for filename in filenames:
        try:
            df = file_loaders.pandas_load_file(filename)
        except EmptyDataError as e:
            print(f"Empty file error at {filename}")
            raise e
            # continue
        I_max_ABM.append(df["I"].max())
        R_inf_ABM.append(df["R"].iloc[-1])
    I_max_ABM = np.array(I_max_ABM)
    R_inf_ABM = np.array(R_inf_ABM)

    T_max = max(df["time"].max() * 10, 300)
    df_SIR = SIR.integrate(cfg, T_max, dt=0.01, ts=0.1)

    # break out if the SIR model dies out
    if df_SIR["I"].max() < cfg.N_init:
        N = len(I_max_ABM)
        return np.full(N, np.nan), np.full(N, np.nan)

    z_rel_I = I_max_ABM / df_SIR["I"].max()
    z_rel_R = R_inf_ABM / df_SIR["R"].iloc[-1]

    return z_rel_I, z_rel_R


# def get_1D_scan_cfgs_all_filenames(scan_parameter, non_default_parameters):
#     return cfgs, all_filenames


def get_1D_scan_results(scan_parameter, non_default_parameters):
    "Compute the fraction between ABM and SEIR for all simulations related to the scan_parameter"

    cfgs, all_filenames = utils.get_1D_scan_cfgs_all_filenames(
        scan_parameter, non_default_parameters
    )
    N_cfgs = len(cfgs)
    if N_cfgs <= 1:
        return None

    x = np.zeros(N_cfgs)
    y_I = np.zeros(N_cfgs)
    y_R = np.zeros(N_cfgs)
    sy_I = np.zeros(N_cfgs)
    sy_R = np.zeros(N_cfgs)
    n = np.zeros(N_cfgs)

    # ABM_parameter = simulation_parameters_1D_scan[0]
    it = zip(cfgs, all_filenames)
    for i, (cfg, filenames) in enumerate(tqdm(it, desc=scan_parameter, total=N_cfgs)):

        z_rel_I, z_rel_R = compute_ABM_SEIR_proportions(cfg, filenames)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        sy_I[i] = utils.SDOM(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_R[i] = utils.SDOM(z_rel_R)

        n[i] = len(z_rel_I)

    if np.isfinite(y_I).sum() <= 1 and np.isfinite(y_R).sum() <= 1:
        return None

    return x, y_I, y_R, sy_I, sy_R, n, cfg


def extract_limits(lim):
    """ deals with both limits of the form (0, 1) and [(0, 1), (0.5, 1.5)] """
    if isinstance(lim, (tuple, list, np.ndarray)):
        if isinstance(lim[0], (float, int)):
            lim0 = lim1 = lim
        elif isinstance(lim[0], (tuple, list, np.ndarray)):
            lim0, lim1 = lim
    else:
        lim0 = lim1 = (None, None)

    return lim0, lim1


def _plot_1D_scan_res(
    res,
    scan_parameter,
    ylim=None,
    do_log=False,
    add_title=True,
    add_horizontal_line=False,
    **kwargs,
):

    # kwargs = {}

    x, y_I, y_R, sy_I, sy_R, n, cfg = res

    d_par_pretty = utils.get_parameter_to_latex()
    title = utils.dict_to_title(cfg, exclude=[scan_parameter, "ID"])

    label_pretty = d_par_pretty[scan_parameter]
    xlabel = r"$" + label_pretty + r"$"
    if scan_parameter == "rho":
        xlabel += r"\, \Huge [km$^{-1}$]"

    ylim0, ylim1 = extract_limits(ylim)

    # n>1 datapoints
    mask = n > 1
    mask_n1 = n == 1
    if scan_parameter == "event_size_max":
        mask_event_size_max = x != 0
        mask = mask & mask_event_size_max
        mask_n1 = mask_n1 & mask_event_size_max

    factor = 0.7

    if "axes" not in kwargs:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16 * factor, 9 * factor))  #
        if add_title:
            fig.suptitle(title, fontsize=16 * factor)
    else:
        ax0, ax1 = kwargs["axes"]

    errorbar_kwargs = dict(
        fmt=kwargs.get("fmt", "."),
        elinewidth=1,
        capsize=10,
    )

    ax0.errorbar(
        x[mask],
        y_I[mask],
        sy_I[mask],
        **errorbar_kwargs,
        color=kwargs.get("color", "black"),
        ecolor=kwargs.get("color", "black"),
        # label=kwargs.get("label", None),
    )
    ax0.errorbar(
        x[mask_n1],
        y_I[mask_n1],
        sy_I[mask_n1],
        **errorbar_kwargs,
        color="grey",
        ecolor="grey",
    )
    ax0.set(ylim=ylim0)

    ax1.errorbar(
        x[mask],
        y_R[mask],
        sy_R[mask],
        **errorbar_kwargs,
        color=kwargs.get("color", "black"),
        ecolor=kwargs.get("color", "black"),
        label=kwargs.get("label", None),
    )
    ax1.errorbar(
        x[mask_n1],
        y_R[mask_n1],
        sy_R[mask_n1],
        **errorbar_kwargs,
        color="grey",
        ecolor="grey",
    )
    ax1.set(ylim=ylim1)

    if add_horizontal_line:
        ax0.axhline(1, color="grey", ls="--", lw=2)
        ax1.axhline(1, color="grey", ls="--", lw=2)

    if scan_parameter == "event_size_max":
        mask_limit = ~mask_event_size_max
        if any(mask_limit):
            ax0.axhline(y_I[mask_limit], ls="--", color=kwargs.get("color", "black"))
            ax1.axhline(y_R[mask_limit], ls="--", color=kwargs.get("color", "black"))

    ax0.set_xlabel(xlabel, labelpad=kwargs.get("labelpad", -5))
    ax1.set_xlabel(xlabel, labelpad=kwargs.get("labelpad", -5))

    if "label" in kwargs:
        ax1.legend()

    if do_log:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    if "axes" not in kwargs:
        fig.tight_layout()
        fig.subplots_adjust(top=0.8, wspace=kwargs.get("wspace", 0.55))
        return fig, (ax0, ax1)


from pandas.errors import EmptyDataError


def plot_1D_scan(
    scan_parameter,
    do_log=False,
    ylim=None,
    non_default_parameters=None,
    figname_pdf=None,
    **kwargs,
):

    if non_default_parameters is None:
        non_default_parameters = {}

    res = get_1D_scan_results(scan_parameter, non_default_parameters)
    if res is None:
        return None

    # kwargs = {}
    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log, **kwargs)

    ax0.set(ylabel=r"$I_\mathrm{peak}^\mathrm{ABM} \, / \,\, I_\mathrm{peak}^\mathrm{SEIR}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{ABM} \, / \,\, R_\infty^\mathrm{SEIR}$")

    if figname_pdf is None:
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


def compute_reduced_chi2s(fit_objects):
    red_chi2 = []
    for fit_object in fit_objects.values():
        red_chi2.append(fit_object.chi2 / fit_object.N)
    return np.array(red_chi2)


def plot_single_fit(
    cfg,
    fit_objects,
    add_top_text=True,
    xlim=(0, None),
    ylim=(0, None),
    legend_loc=None,
    legend_fontsize=28,
    add_chi2=True,
):

    relative_names = [
        r"\frac{I_\mathrm{peak}^\mathrm{fit}} {I_\mathrm{peak}^\mathrm{ABM}}",
        r"\frac{R_\infty^\mathrm{fit}} {R_\infty^\mathrm{fit}}",
    ]

    # fit_objects = all_fits[ABM_parameter]

    N_tot = cfg.N_tot
    d_ylabel = {"I": r"Fraction Infected", "R": r"Fraction Recovered"}

    if legend_loc is None:
        legend_loc = {"I": "upper right", "R": "lower right"}

    # cfg = utils.string_to_dict(ABM_parameter)

    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)

    for i, fit_object in enumerate(fit_objects.values()):
        # break

        df = file_loaders.pandas_load_file(fit_object.filename)
        t = df["time"].values
        T_max = max(t) * 1.1
        df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)

        lw = 0.9
        for I_or_R, ax in zip(["I", "R"], axes):

            label = "ABM" if i == 0 else None
            ax.plot(t, df[I_or_R] / N_tot, "k-", lw=lw, label=label)

            label = "Fits" if i == 0 else None
            ax.plot(
                df_fit["time"],
                df_fit[I_or_R] / N_tot,
                lw=lw,
                color=d_colors["green"],
                label=label,
            )

            if i == 0:
                axvline_kwargs = dict(lw=lw, color=d_colors["blue"], alpha=0.4)
                tmp = df.query("@fit_object.t.min() <= time <= @fit_object.t.max()")
                ax.fill_between(tmp["time"], tmp[I_or_R] / N_tot, **axvline_kwargs)

                vertical_lines = tmp["time"].iloc[0], tmp["time"].iloc[-1]
                line_kwargs = dict(ymax=0.6, color=d_colors["blue"], lw=2 * lw)
                ax.axvline(vertical_lines[0], **line_kwargs)
                ax.axvline(vertical_lines[1], **line_kwargs)

                ax.text(
                    vertical_lines[0] * 0.65,
                    0.3 * ax.get_ylim()[1],
                    "Fit Range",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=34,
                    rotation=90,
                    color=d_colors["blue"],
                )

            # x_rug = [fit_object.t.min(), fit_object.t.max()]
            # rug_plot(x_rug, ax, ymin=-0.01, ymax=0.03, color="k", lw=lw)

    if add_top_text:

        # calculate monte carlo simulated fit results
        # all_I_max_MC = []
        # all_R_inf_MC = []
        all_I_max_fit = []
        all_R_inf_fit = []
        for i, fit_object in enumerate(fit_objects.values()):
            # all_I_max_MC.extend(fit_object.I_max_MC)
            # all_R_inf_MC.extend(fit_object.R_inf_MC)
            all_I_max_fit.append(fit_object.I_max_fit)
            all_R_inf_fit.append(fit_object.R_inf_fit)
        d_fits = {"I": all_I_max_fit, "R": all_R_inf_fit}

        names_fit = {}
        names_fit["I"] = r"I_\mathrm{peak}^\mathrm{fit}"
        names_fit["R"] = r"R_\infty^\mathrm{fit}"

        for I_or_R, ax in zip(["I", "R"], axes):

            s = utils.format_relative_uncertainties(d_fits[I_or_R], names_fit[I_or_R])

            ax.text(
                0.1,
                1.05,
                s,
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )

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
            ax.text(0.55, 1.05, s, transform=ax.transAxes, fontsize=12)

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
            df_SIR[I_or_R] / N_tot,
            lw=lw * 4,
            color=d_colors["red"],
            label="SEIR",
            zorder=0,
        )

        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel="Time [days]", ylabel=d_ylabel[I_or_R])
        # if add_top_text:
        #     ax.xaxis.set_label_coords(0.91, -0.14)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    if add_chi2:
        reduced_chi2s = compute_reduced_chi2s(fit_objects)
        mean_chi2 = np.mean(reduced_chi2s)
        std_chi2 = utils.SDOM(reduced_chi2s)
        s = r"${\tilde{\chi}}^2 = " + f"{mean_chi2:.2f} \pm {std_chi2:.2f}$"
        axes[1].text(
            0.53,
            0.05,
            s,
            transform=axes[1].transAxes,
            fontsize=24,
        )

    leg = axes[0].legend(loc=legend_loc["I"], fontsize=legend_fontsize)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
        legobj.set_alpha(1.0)

    title = utils.dict_to_title(cfg, len(fit_objects))
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    return fig, ax


def plot_fits(all_fits, force_rerun=False, verbose=False):

    pdf_name = f"Figures/Fits.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf:

        for hash_, fit_objects in tqdm(all_fits.items(), desc="Plotting all fits"):
            # break

            # skip if no fits
            if len(fit_objects) == 0:
                if verbose:
                    print(f"Skipping {hash_}")
                continue

            cfg = utils.hash_to_cfg(hash_)
            fig_ax = plot_single_fit(cfg, fit_objects)
            if fig_ax is not None:
                fig, ax = fig_ax
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

    cfgs, _ = utils.get_1D_scan_cfgs_all_filenames(scan_parameter, non_default_parameters)

    cfg_hashes = set([cfg.hash for cfg in cfgs])
    selected_fits = {hash_: val for hash_, val in all_fits.items() if hash_ in cfg_hashes}

    N_cfgs = len(selected_fits)
    if N_cfgs <= 1:
        return None

    N = len(selected_fits)

    x = np.zeros(N)
    y_I = np.zeros(N)
    y_R = np.zeros(N)
    sy_I = np.zeros(N)
    sy_R = np.zeros(N)
    n = np.zeros(N)

    it = tqdm(enumerate(selected_fits.items()), desc=scan_parameter, total=N)
    for i, (hash_, fit_objects) in it:
        # break

        cfg = utils.hash_to_cfg(hash_)
        # cfg = utils.string_to_dict(ABM_parameter)
        z_rel_I, z_rel_R = compute_fit_ABM_proportions(fit_objects)

        x[i] = cfg[scan_parameter]

        if len(z_rel_I) > 0:
            y_I[i] = np.mean(z_rel_I)
            sy_I[i] = utils.SDOM(z_rel_I)
        else:
            y_I[i] = sy_I[i] = np.nan

        if len(z_rel_R) > 0:
            y_R[i] = np.mean(z_rel_R)
            sy_R[i] = utils.SDOM(z_rel_R)
        else:
            y_R[i] = sy_R[i] = np.nan

        n[i] = len(z_rel_I)

    return x[n > 0], y_I[n > 0], y_R[n > 0], sy_I[n > 0], sy_R[n > 0], n[n > 0], cfg


from pandas.errors import EmptyDataError


def plot_1D_scan_fit_results(
    all_fits,
    scan_parameter,
    do_log=False,
    ylim=None,
    non_default_parameters=None,
    figname_pdf=None,
    **kwargs,
):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_fit_results(all_fits, scan_parameter, non_default_parameters)
    if not res:
        return None

    fig, (ax0, ax1) = _plot_1D_scan_res(
        res,
        scan_parameter,
        ylim,
        do_log,
        **kwargs,
    )

    ax0.set(ylabel=r"$I_\mathrm{peak}^\mathrm{fit} \, / \,\, I_\mathrm{peak}^\mathrm{ABM}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{fit} \, / \,\, R_\infty^\mathrm{ABM}$")

    if figname_pdf is None:
        figname_pdf = f"Figures/1D_scan_fits/1D_scan_fit_{scan_parameter}"
        for key, val in non_default_parameters.items():
            figname_pdf += f"_{key}_{val}"
        figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%

import h5py


def _load_my_state_and_my_number_of_contacts(filename):
    with h5py.File(filename, "r") as f:
        my_state = f["my_state"][()]
        my_number_of_contacts = f["my_number_of_contacts"][()]
    return my_state, my_number_of_contacts


def plot_single_number_of_contacts(
    filename,
    make_fraction_subplot=True,
    figsize=None,
    xlim=None,
    title=None,
    add_legend=True,
    loc="best",
    xlabel=None,
    ylabel=None,
    fontsize=30,
    labelsize=None,
    add_average_arrows=False,
    title_pad=20,
):

    if xlabel is None:
        xlabel = "Number of contacts"

    if ylabel is None:
        ylabel = "Counts"

    my_state, my_number_of_contacts = _load_my_state_and_my_number_of_contacts(filename)

    cfg = file_loaders.filename_to_cfg(filename)
    N_tot = cfg.N_tot
    factor = 1 / N_tot

    mask_S = my_state[-1] == -1
    mask_R = my_state[-1] == 8

    if xlim is None:
        x_min = np.percentile(my_number_of_contacts, 0.01)
        x_max = np.percentile(my_number_of_contacts, 99)
    else:
        x_min, x_max = xlim
    x_range = (x_min, x_max)
    N_bins = int(x_max - x_min)

    kwargs = {"bins": N_bins, "range": x_range, "histtype": "step"}

    if make_fraction_subplot:
        fig, (ax1, ax2) = plt.subplots(
            figsize=figsize, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    H_all = ax1.hist(
        my_number_of_contacts,
        weights=factor * np.ones_like(my_number_of_contacts),
        label="All",
        color=d_colors["blue"],
        **kwargs,
    )
    H_S = ax1.hist(
        my_number_of_contacts[mask_S],
        weights=factor * np.ones_like(my_number_of_contacts[mask_S]),
        label="Susceptable",
        color=d_colors["red"],
        **kwargs,
    )
    H_R = ax1.hist(
        my_number_of_contacts[mask_R],
        weights=factor * np.ones_like(my_number_of_contacts[mask_R]),
        label="Recovered",
        color=d_colors["green"],
        **kwargs,
    )

    x = 0.5 * (H_all[1][:-1] + H_all[1][1:])
    frac_S = H_S[0] / H_all[0]
    s_frac_S = np.sqrt(frac_S * (1 - frac_S) / (H_all[0] / factor))
    frac_R = H_R[0] / H_all[0]
    s_frac_R = np.sqrt(frac_R * (1 - frac_R) / (H_all[0] / factor))

    if make_fraction_subplot:
        kwargs_errorbar = dict(fmt=".", elinewidth=1.5, capsize=4, capthick=1.5)
        ax2.errorbar(x, frac_S, s_frac_S, color=d_colors["red"], **kwargs_errorbar)
        ax2.errorbar(x, frac_R, s_frac_R, color=d_colors["green"], **kwargs_errorbar)

    if add_legend:
        ax1.legend(loc=loc)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax1.set(xlim=x_range)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_title(title, pad=title_pad, fontsize=fontsize)

    if make_fraction_subplot:
        ax2.set(ylim=(0, 1), ylabel=r"Fraction")
        ax2.set(xlabel=xlabel)
    else:
        ax1.set_xlabel(xlabel, fontsize=fontsize)

    if labelsize:
        ax1.tick_params(axis="both", labelsize=labelsize)

    if add_average_arrows:
        ymax = ax1.get_ylim()[1]
        mean_all = np.mean(my_number_of_contacts)
        mean_S = np.mean(my_number_of_contacts[mask_S])
        mean_R = np.mean(my_number_of_contacts[mask_R])

        arrowprops = dict(ec="white", width=6, headwidth=20, headlength=15)
        ax1.annotate(
            "",
            xy=(mean_all, ymax * 0.01),
            xytext=(mean_all, ymax * 0.2),
            arrowprops=dict(**arrowprops, fc=d_colors["blue"]),
        )
        ax1.annotate(
            "",
            xy=(mean_S, ymax * 0.01),
            xytext=(mean_S, ymax * 0.2),
            arrowprops=dict(**arrowprops, fc=d_colors["red"]),
        )
        ax1.annotate(
            "",
            xy=(mean_R, ymax * 0.01),
            xytext=(mean_R, ymax * 0.2),
            arrowprops=dict(**arrowprops, fc=d_colors["green"]),
        )

    return fig, ax1


def plot_number_of_contacts(network_files, force_rerun=False):

    if len(network_files) == 0:
        return None

    pdf_name = f"Figures/Number_of_contacts.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf:
        for network_filename in tqdm(
            network_files.iter_all_files(),
            desc="Number of contacts",
            total=len(network_files),
        ):
            # cfg = utils.string_to_dict(str(network_filename))
            if "_ID__0" in network_filename:
                fig_ax = plot_single_number_of_contacts(network_filename)
                if fig_ax is not None:
                    fig, ax = fig_ax
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


#%%

import generate_animations
from matplotlib.lines import Line2D


def make_paper_screenshot(
    filename,
    title="",
    i_day=1,
    dpi=50,
    R_eff_max=4,
    do_tqdm=False,
    verbose=False,
):

    animation = generate_animations.AnimateSIR(
        filename, do_tqdm=do_tqdm, verbose=verbose, N_max=i_day + 1
    )
    if animation.df_counts is None:
        animation._initialize_data()

    geo_plot_kwargs = {}
    geo_plot_kwargs["S"] = dict(alpha=0.9, norm=animation.norm_100)
    geo_plot_kwargs["I"] = dict(alpha=1.0, norm=animation.norm_10)
    geo_plot_kwargs["R"] = dict(alpha=1.0, norm=animation.norm_1000)

    fig = plt.figure(figsize=(10 * 1.3, 12 * 1.3))
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    for state in animation.states:
        if animation.df_counts.loc[i_day, state] > 0:
            ax.scatter_density(
                *animation.coordinates[animation._get_mask(i_day, state)].T,
                color=animation.d_colors[state],
                dpi=dpi,
                **geo_plot_kwargs[state],
            )

    ax.set(xlim=(7.9, 13.3), ylim=(54.5, 58.2), xlabel="Longitude")
    ax.set_ylabel("Latitude", rotation=90)  # fontsize=20, labelpad=20
    ax.set_title(title, pad=40, fontsize=32)

    kw_args_circle = dict(xdata=[0], ydata=[0], marker="o", color="w", markersize=16)
    circles = [
        Line2D(
            label=animation.state_names[state],
            markerfacecolor=animation.d_colors[state],
            **kw_args_circle,
        )
        for state in animation.states
    ]
    ax.legend(handles=circles, fontsize=30, frameon=False, loc=(0, 0.82))

    # secondary plots:

    # These are in unitless percentage of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.63, 0.75, 0.39 * 0.6, 0.08]

    i_day_max = i_day + max(3, i_day * 0.1)

    # delta_width = 0 * width / 100
    ax2 = fig.add_axes([left, bottom, width, height])
    I_up_to_today = animation.df_counts["I"].iloc[: i_day + 1] / animation.cfg["N_tot"]
    ax2.plot(
        I_up_to_today.index,
        I_up_to_today,
        "-",
        color=animation.d_colors["I"],
        lw=3,
    )
    ax2.plot(
        I_up_to_today.index[-1],
        I_up_to_today.iloc[-1],
        "o",
        color=animation.d_colors["I"],
    )
    I_max = np.max(I_up_to_today)
    ax2.set(
        ylim=(0, I_max * 1.3),
        xlim=(0, i_day_max),
    )
    decimals = max(int(-np.log10(I_max)) - 1, 0)  # max important, otherwise decimals=-1
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))
    ax2.text(
        0.01,
        1.18,
        "Fraction Infected",
        fontsize=30,
        transform=ax2.transAxes,
        rotation=0,
        ha="left",
    )
    ax2.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    # add_spines(ax2, exclude=["upper", "right"])
    generate_animations.remove_spines(ax2)

    ax3 = fig.add_axes([left, bottom - height * 2, width, height])

    if i_day > 0:
        R_eff_up_to_today = animation._interpolate_R_eff(animation.R_eff_smooth[: i_day + 1])
        z = (R_eff_up_to_today["R_eff"] > 1) / 1
        ax3.scatter(
            R_eff_up_to_today["t"],
            R_eff_up_to_today["R_eff"],
            s=10,
            c=z,
            **animation._scatter_kwargs,
        )
        R_eff_today = R_eff_up_to_today.iloc[-1]
        z_today = R_eff_today["R_eff"] > 1
        ax3.scatter(
            R_eff_today["t"],
            R_eff_today["R_eff"],
            s=100,
            c=z_today,
            **animation._scatter_kwargs,
        )

    ax3.axhline(1, ls="--", color="k", lw=1)  # x = 0
    ax3.set(
        ylim=(0, R_eff_max * 1.1),
        xlim=(0, i_day_max),
    )
    ax3.set_xlabel(r"Time [days]", fontsize=30)
    ax3.text(
        0.01,
        1.18,
        r"$\mathcal{R}_\mathrm{eff}$",
        fontsize=30,
        transform=ax3.transAxes,
        rotation=0,
        ha="left",
        # va="center",
    )
    ax3.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    generate_animations.remove_spines(ax3)

    scalebar = AnchoredSizeBar(
        ax.transData,
        generate_animations.longitudes_per_50km,
        "50 km",
        loc="lower left",
        sep=10,
        color="black",
        frameon=False,
        size_vertical=0.003,
        fontproperties=generate_animations.fontprops,
        bbox_to_anchor=Bbox.from_bounds(8, 54.52, 0, 0),
        bbox_transform=ax.transData,
    )

    ax.add_artist(scalebar)

    plt.close("all")
    return fig, (ax, ax2, ax3)


#%%


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


def plot_R_eff(cfg, abm_files):
    filenames = abm_files.cfg_to_filenames(cfg)
    if filenames is None:
        return None

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


from matplotlib.backends.backend_pdf import PdfPages


def plot_R_eff_beta_1D_scan(cfgs, abm_files):
    pdf_name = "Figures/R_eff.pdf"

    with PdfPages(pdf_name) as pdf:
        for cfg in tqdm(cfgs, desc="Plotting R_eff for beta 1D-scan"):
            fig_ax = plot_R_eff(cfg, abm_files)
            if fig_ax is not None:
                fig, ax = fig_ax
                pdf.savefig(fig, dpi=100)
            plt.close("all")


#%%


# reload(generate_R_eff_fits)


def plot_multiple_ABM_simulations(
    cfgs,
    abm_files,
    variable,
    R_effs,
    reverse_order=False,
    days=None,
    xlim=(0, None),
    legend_fontsize=20,
    d_label_loc=None,
    ylim_scale=1,
    figsize=(16, 7),
    dpi=100,
):

    d_ylabel = {"I": "Fraction Infected", "R": "Fraction Recovered"}
    if d_label_loc is None:
        d_label_loc = {"I": "upper left", "R": "upper left"}

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(top=0.75)

    for i_cfg, cfg in enumerate(cfgs):

        filenames = abm_files.cfg_to_filenames(cfg)
        if filenames is None:
            return None

        if not isinstance(cfg, utils.DotDict):
            cfg = utils.DotDict(cfg)

        N_tot = cfg.N_tot

        T_max = 0
        lw = 0.3 * 10 / np.sqrt(len(filenames))

        for i_filename, filename in enumerate(filenames):
            df = file_loaders.pandas_load_file(filename)
            t = df["time"].values
            label = f"{cfg[variable]}" if i_filename == 0 else None

            if i_cfg > len(colors):
                ls = "--"
            else:
                ls = "-"

            plot_kwargs = dict(lw=lw, ls=ls, c=colors[i_cfg % len(colors)], label=label)
            axes[0].plot(t, df["I"] / N_tot, **plot_kwargs)
            # axes[1].plot(t, df["R"] / N_tot, **plot_kwargs)

            if t.max() > T_max:
                T_max = t.max()

    # for state, ax in zip(["I", "R"], axes):
    # break

    df_deterministic = compute_df_deterministic(cfg, "I", T_max=T_max)

    axes[0].plot(
        df_deterministic["time"],
        df_deterministic["I"] / N_tot,
        lw=lw * 4,
        color="k",
        label="SEIR",
    )

    legend_title = r"$" + utils.get_parameter_to_latex()[variable] + r"$"
    leg = axes[0].legend(
        loc=d_label_loc["I"],
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=30,
        labelspacing=0.1,
    )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(lw * 4)

    axes[0].set(
        xlabel="Time [days]",
        ylim=(0, None),
        ylabel=d_ylabel["I"],
        xlim=xlim,
    )
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * ylim_scale)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    xlabel = r"$" + utils.get_parameter_to_latex()[variable] + r"$"
    generate_R_eff_fits.plot_R_effs_single_comparison(
        R_effs,
        variable,
        reverse_order,
        days=days,
        ax=axes[1],
        xlabel=xlabel,
    )

    # cfg.pop(variable, None)
    title = utils.dict_to_title(cfg, exclude=["hash", variable])
    fig.suptitle(title, fontsize=15)
    plt.subplots_adjust(wspace=0.4)

    return fig, axes


#%%


def make_MCMC_plots(
    variable,
    abm_files,
    reverse_order=False,
    days=None,
    N_max_figures=None,
    variable_subset=None,
    extra_selections=None,
    index_in_list_to_sortby=0,
    force_rerun=False,
):

    s_extra = ""
    if extra_selections:
        s_extra += "__"
        for key, val in extra_selections.items():
            s_extra += f"__{key}__{val}"

    pdf_name = f"Figures/MCMC_{variable}{s_extra}.pdf"

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists\n", flush=True)
        return None

    cfgs_to_plot = database.get_MCMC_data(
        variable,
        variable_subset,
        N_max=N_max_figures,
        extra_selections=extra_selections,
    )

    if len(cfgs_to_plot) == 0:
        print(f"No runs to plot for {variable}")
        return

    if N_max_figures is not None:
        print(f"Only plotting the first {N_max_figures} MCMC figures", flush=True)
        cfgs_to_plot = cfgs_to_plot[:N_max_figures]

    with PdfPages(pdf_name) as pdf:
        for cfgs in tqdm(cfgs_to_plot, desc=f"Plotting MCMC runs for {variable}"):
            # break

            R_effs = generate_R_eff_fits.compute_R_eff_fits_from_cfgs(
                cfgs,
                abm_files,
                variable,
                index_in_list_to_sortby=index_in_list_to_sortby,
                do_tqdm=False,
            )

            fig_ax = plot_multiple_ABM_simulations(
                cfgs, abm_files, variable, R_effs, reverse_order, days
            )

            if fig_ax is not None:
                fig, ax = fig_ax
                pdf.savefig(fig, dpi=100)
            plt.close("all")


#%%


def _load_corona_type_data(filename, start_day=0, end_day=-1):
    with h5py.File(filename, "r") as f:
        my_corona_type = f["my_corona_type"][()]
        if end_day == -1:
            end_day = len(f["my_state"])
        my_state = f["my_state"][start_day:end_day]
    return my_corona_type, my_state


@njit
def get_I_corona_types(my_state, my_corona_type):
    # getting I states that also has specified corona_type
    I_states = (my_state >= 4) & (my_state < 8)
    type_0 = I_states & (my_corona_type == 0)
    type_1 = I_states & (my_corona_type == 1)

    type_1_sum = type_1.sum(axis=1)
    type_0_sum = I_states.sum(axis=1) - type_1_sum
    return type_0_sum, type_1_sum
    # return (I_states & (my_corona_type[start_day:end_day, :] == corona_type)).sum(axis=1)


xlim = (0, None)
ylim_scale = 1.0
legend_fontsize = 30
d_label_loc = None


def plot_corona_type_single_plot(
    cfg,
    network_files,
    xlim=(0, None),
    ylim_scale=1.0,
    legend_fontsize=30,
    d_label_loc=None,
    N_max_runs=None,
    reposition_x_axis=False,
    normalize=True,
):

    filenames = network_files.cfg_to_filenames(cfg)
    # filenames = [filename for filename in network_files.iter_all_files()]
    if N_max_runs:
        filenames = filenames[:N_max_runs]

    if not isinstance(cfg, utils.DotDict):
        cfg = utils.DotDict(cfg)

    # d_ylabel = {"I": "Fraction Infected", "R": "Fraction Infected"}
    # if d_label_loc is None:
    # d_label_loc = {"I": "upper right", "R": "upper right"}

    N_tot = cfg.N_tot if normalize else 1
    start_day = xlim[0]
    end_day = xlim[1] if xlim[1] is not None else -1
    delta_t = xlim[0] if reposition_x_axis else 0

    lw = 0.3 * 10 / np.sqrt(len(filenames))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.75)

    # file, i = abm_files[ABM_parameter][0], 0
    for i, filename in enumerate(filenames):
        # break
        df = file_loaders.pandas_load_file(filename)
        my_corona_type, my_state = _load_corona_type_data(filename, start_day, end_day)
        c = f"C{i}"

        t = df["time"].values - delta_t
        label = r"Total" if i == 0 else None

        ax.plot(t, df["I"] / N_tot, lw=lw * 1.5, c=c, label=label)

        t_days = np.arange(len(my_state)) + start_day - delta_t
        I_normal, I_UK = get_I_corona_types(my_state, my_corona_type)

        ax.plot(
            t_days,
            I_normal / N_tot,
            lw=lw / 1.5,
            ls="dotted",
            c=c,
            label=r"DK" if i == 0 else None,
        )
        ax.plot(
            t_days,
            I_UK / N_tot,
            lw=lw / 1.5,
            ls="dashed",
            c=c,
            label=r"UK" if i == 0 else None,
        )

    ax.legend(fontsize=legend_fontsize)

    if reposition_x_axis:
        xlim = (xlim[0] - delta_t, xlim[1] - delta_t)

    ax.set(
        xlabel="Tid [dage]",
        ylim=(0, None),
        ylabel="Inficerede",
        xlim=xlim,
    )
    ax.set_ylim(0, ax.get_ylim()[1] * ylim_scale)
    if normalize:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    else:
        ax.yaxis.set_major_formatter(EngFormatter())

    title = utils.dict_to_title(cfg, len(filenames))
    fig.suptitle(title, fontsize=15)
    return fig, ax


def plot_corona_type(network_files, force_rerun=False, **kwargs):

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/corona_type_infection_curves_simulations.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_rerun:
        print(f"{pdf_name} already exists\n", flush=True)
        return None

    with PdfPages(pdf_name) as pdf:

        for cfg in tqdm(
            network_files.iter_cfgs(),
            desc="Plotting corona type infections",
            total=len(network_files.cfgs),
        ):
            # break

            #     break
            fig_ax = plot_corona_type_single_plot(cfg, network_files, **kwargs)

            if fig_ax is not None:
                fig, ax = fig_ax
                pdf.savefig(fig, dpi=100)
            plt.close("all")


#%%


# xlim = (10, 100)


def plot_corona_type_ratio_plot_single_plot(
    cfg,
    network_files,
    xlim=(0, None),
    # ylim_scale=1.0,
):

    filenames = network_files.cfg_to_filenames(cfg)
    # filenames = filenames[0:1]

    if not isinstance(cfg, utils.DotDict):
        cfg = utils.DotDict(cfg)

    d_ylabel = {"I": "UK / DK", "R": "log10 (UK / DK)"}

    N_tot = cfg.N_tot

    start_day = xlim[0]
    end_day = xlim[1] if xlim[1] is not None else -1

    lw = 0.3 * 10 / np.sqrt(len(filenames))

    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.75)

    # file, i = abm_files[ABM_parameter][0], 0
    for i, filename in enumerate(filenames):
        # break
        df = file_loaders.pandas_load_file(filename)
        my_corona_type, my_state = _load_corona_type_data(filename, start_day, end_day)

        t_days = np.arange(len(my_state)) + start_day

        I_normal, I_UK = get_I_corona_types(my_state, my_corona_type)

        axes[0].plot(
            t_days,
            I_UK / I_normal,
            lw=lw,
            c="k",
            label=r"UK / DK" if i == 0 else None,
        )

        axes[1].plot(
            t_days,
            I_UK / I_normal,
            lw=lw,
            c="k",
            label=r"log10 (UK / DK)" if i == 0 else None,
        )
        axes[1].set_yscale("log", base=10, nonpositive="mask")

    for variable, ax in zip(["I", "R"], axes):

        # leg = ax.legend(loc=d_label_loc[variable], fontsize=legend_fontsize)
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(lw * 4)

        ax.set(
            xlabel="Time [days]",
            ylim=(0, None) if variable == "I" else None,
            ylabel=d_ylabel[variable],
            xlim=xlim,
        )
        # ax.set_ylim(0, ax.get_ylim()[1] * ylim_scale)
        # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    title = utils.dict_to_title(cfg, len(filenames))
    fig.suptitle(title, fontsize=15)
    plt.subplots_adjust(wspace=0.4)

    return fig, axes


def plot_corona_type_ratio_plot(network_files, force_rerun=False, **kwargs):

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/corona_type_infection_ratio_curves_simulations.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_rerun:
        print(f"{pdf_name} already exists\n", flush=True)
        return None

    with PdfPages(pdf_name) as pdf:

        for cfg in tqdm(
            network_files.iter_cfgs(),
            desc="Plotting corona type infection ratios",
            total=len(network_files.cfgs),
        ):

            #     break
            fig_ax = plot_corona_type_ratio_plot_single_plot(cfg, network_files, **kwargs)

            if fig_ax is not None:
                fig, ax = fig_ax
                pdf.savefig(fig, dpi=100)
            plt.close("all")