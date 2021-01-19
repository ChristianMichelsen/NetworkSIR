import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from iminuit import Minuit
from collections import defaultdict
import joblib
from importlib import reload
from iminuit import Minuit, describe
from iminuit.util import make_func_code
from IPython.display import display
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc_context
import sigfig
import matplotlib


from src.utils import utils
from src import file_loaders
from src import fits
from src import plot
from src import database
import generate_R_eff_fits

num_cores_max = 30
# delta_time = 8
# delta_time = 4.7
delta_time = 7


#%%


def pandas_load_file(filename):
    df_raw = pd.read_csv(filename)  # .convert_dtypes()

    for state in ["E", "I"]:
        df_raw[state] = sum(
            (df_raw[col] for col in df_raw.columns if state in col and len(col) == 2)
        )

    # only keep relevant columns
    df = df_raw[["Time", "E", "I", "R"]].copy()
    df.rename(columns={"Time": "time"}, inplace=True)

    # remove duplicate timings
    df = df.loc[df["time"].drop_duplicates().index]

    # keep only every 10th data point and reset index
    df = df.iloc[::10].reset_index(drop=True)

    return df


def load_df(filenames):

    frames = [pandas_load_file(filename.replace(".hdf5", ".csv"))["I"] for filename in filenames]
    df_I = pd.concat(frames, axis=1)

    mean = np.mean(df_I, axis=1)
    std = np.std(df_I, axis=1)
    sdom = std / np.sqrt(df_I.shape[1])

    df = pd.concat([mean, sdom], axis=1).rename(columns={0: "mean", 1: "sdom"})
    return df


def exponential(t, I_0, R_eff, T):
    return I_0 * R_eff ** (t / T)


def linear(x, a, b):
    return a * x + b


class FittingClassChi2:
    def __init__(self, x, y, sy, days=None, verbose=True, model="exponential"):

        if days is None:
            self.x = x
            self.y = y
            self.sy = sy
        else:
            self.x = x[days]
            self.y = y[days]
            self.sy = sy[days]
        self.verbose = verbose

        self.model = self.init_model(model)
        self.func_code = make_func_code(describe(self.model)[1:])
        self.N_fit_parameters = len(describe(self.model)[1:])
        self.N = len(x)
        self.df = self.N - self.N_fit_parameters
        self.is_fitted = True

    def init_model(self, model):
        if model == "exponential":
            self.fit_kwargs = {
                "I_0": self.y[0],
                "R_eff": 1,
                "limit_R_eff": (0, None),
                # "T": 8,
                "T": 4.7,
                "fix_T": True,
            }
            self.fit_kwargs_retry = {
                "I_0": self.y[0],
                "R_eff": 0.5,
                "limit_R_eff": (0, None),
                # "T": 8,
                "T": 4.7,
                "fix_T": True,
            }
            return exponential
        elif model == "linear":
            self.fit_kwargs = {
                "a": 1,
                "b": self.y[0],
            }
            self.fit_kwargs_retry = {
                "a": 0,
                "b": 1,
            }
            return linear
        else:
            raise AssertionError(f"Model: {model} could not be recognized.")

    def __call__(self, *par):
        yhat = self.model(self.x, *par)
        chi2 = np.sum((yhat - self.y) ** 2 / self.sy ** 2)
        return chi2

    def fit(self):

        m = Minuit(self, errordef=1, pedantic=False, **self.fit_kwargs)
        m.migrad()

        self.bad_fit = False
        if not m.fmin.is_valid:

            m = Minuit(self, errordef=1, pedantic=False, **self.fit_kwargs_retry)
            m.migrad()
            if not m.fmin.is_valid:
                self.bad_fit = True

        if self.verbose:
            display(m.fmin)
            display(m.params)

        # self.m = m # do not save m object to be dill'able
        self.chi2 = m.fval
        self.is_fitted = True
        self._set_values_and_errors(m)
        return self

    def _set_values_and_errors(self, m):
        self.values = dict(m.values)
        self.errors = dict(m.errors)


def fit_df(df):

    x = df.index.values
    y = df["mean"].values
    sy = df["sdom"].values

    N = len(x)
    R_eff = {}
    for day in range(delta_time, N):
        days = np.arange(day - delta_time, day)
        fit = FittingClassChi2(x, y, sy, days, verbose=False).fit()
        if not fit.bad_fit:
            R_eff[day] = {"mean": fit.values["R_eff"], "std": fit.errors["R_eff"]}

    R_eff = pd.DataFrame(R_eff).T
    return R_eff


#%%

def compute_R_eff(*abm_files=None, cfg=None, df=None):
    if abm_files is None and cfg is None:
        filenames = abm_files.cfg_to_filenames(cfg)
        df = load_df(filenames)
    elif df is None:
        pass
    else:
        raise AssertionError("Must specify either abm-files and cfg, or df")

    R_eff = fit_df(df)
    return R_eff


def compute_R_eff_fits_from_cfgs(
    cfgs, abm_files, variable=None, index_in_list_to_sortby=0, do_tqdm=True
):

    R_effs = {}

    if do_tqdm:
        cfgs = tqdm(cfgs, desc="Creating R_eff (fits)")

    counter = 0
    for cfg in cfgs:
        # break
        R_eff = compute_R_eff(abm_files, cfg)
        if variable is None or variable == "all":
            key = counter
        else:
            key = cfg[variable]

        counter += 1
        # if variable is a list, use first value
        if isinstance(key, list):
            key = key[index_in_list_to_sortby]
        R_effs[key] = R_eff
    return R_effs


def plot_R_effs_single_comparison(
    R_effs, variable, reverse_order=False, days=None, title=None, xlabel=None, ax=None
):

    if days is None:
        days = [20, 25, 30, 35, 40]

    if xlabel is None:
        xlabel = variable
    if reverse_order:
        xlabel += " (reversed)"
    # title = utils.dict_to_title(cfgs[0], exclude=["hash", variable, "version"])

    xmin = np.min(list(R_effs.keys()))
    xmax = np.max(list(R_effs.keys()))
    delta = xmax - xmin
    xmin -= delta / 10
    xmax += delta / 10
    xlim = [xmin, xmax]

    no_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        no_ax = True

    for i, day in enumerate(days):
        # break
        df = pd.DataFrame.from_dict(
            {key: R_effs[key].loc[day] for key in R_effs.keys()},
            orient="index",
        )

        fit = FittingClassChi2(df.index, df["mean"], df["std"], verbose=False, model="linear").fit()
        xx = np.linspace(*xlim, num=2)
        yy = fit.model(xx, **fit.values)
        ax.plot(xx, yy, "-", alpha=0.5, color=f"C{i}")

        a_fit = sigfig.round(str(fit.values["a"]), uncertainty=fit.errors["a"])

        ax.errorbar(
            df.index,
            df["mean"],
            df["std"],
            fmt=".",
            color=f"C{i}",
            label=f"Day: {day}, a={a_fit}",
            elinewidth=1,
            capsize=4,
            capthick=1,
        )

    ax.set(xlabel=xlabel, ylabel="$\mathcal{R}_\mathrm{eff}$", xlim=xlim)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.4)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if reverse_order:
        ax.invert_xaxis()
        # ax.set_xlim(ax.get_xlim()[::-1])

    if no_ax:
        return fig


def plot_R_eff(cfgs, abm_files):

    for cfg in cfgs:
        # break

        R_eff = compute_R_eff(abm_files, cfg)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(
            R_eff.index,
            R_eff["mean"],
            R_eff["std"],
            fmt=".",
            color=f"k",
            elinewidth=1,
            capsize=4,
            capthick=1,
        )


# reload(file_loaders)
# reload(database)

if __name__ == "__main__" and False:

    matplotlib.style.use("default")
    sns.set_style("white")
    sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2})

    abm_files = file_loaders.ABM_simulations(verbose=True)
    N_files = len(abm_files)

    # plot MCMC results
    # variable = "all"
    variable = "N_daily_vaccinations"
    # extra_selections = {"tracking_rates": [1.0, 0.8, 0.0]}

    # N_max_figures = 10
    # N_max_figures = None

    cfgs_to_plot = database.get_MCMC_data(
        variable,
        # variable_subset=None,
        # N_max=N_max_figures,
        # extra_selections=extra_selections,
    )

    days = [20, 25, 30, 35, 40]

    cfgs = cfgs_to_plot[0]

    s_extra = ""
    if extra_selections:
        s_extra += "__"
        for key, val in extra_selections.items():
            s_extra += f"__{key}__{val}"

    pdf_name = f"Figures/R_eff_MCMC_{variable}{s_extra}.pdf"
    desc = f"Plotting R_eff fits for {len(cfgs_to_plot)} MCMC cfgs"
    with PdfPages(pdf_name) as pdf:
        for cfgs in tqdm(cfgs_to_plot):

            R_effs = compute_R_eff_fits_from_cfgs(
                cfgs,
                abm_files,
                variable=variable,
                index_in_list_to_sortby=0,
                do_tqdm=False,
            )
            fig = plot_R_effs_single_comparison(R_effs, variable, days=days)

            pdf.savefig(fig, dpi=100, bbox_inches="tight")
            plt.close("all")
