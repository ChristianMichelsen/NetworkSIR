from collections import defaultdict
from sklearn.model_selection import ParameterSampler
import joblib
import numpy as np
from pathlib import Path
from iminuit import Minuit
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats import uniform as sp_uniform
from copy import copy, deepcopy
from importlib import reload
import warnings
from p_tqdm import p_umap
from functools import partial

try:
    from src.utils import utils
    from src import file_loaders
    from src import SIR
except ImportError:
    import utils
    import file_loaders
    import SIR

# reload(SIR)


def uniform(a, b):
    loc = a
    scale = b - a
    return sp_uniform(loc, scale)


def extract_data(t, y, T_max, N_tot, y_max=0.01):
    """Extract data where:
    1) y is larger than 1‰ (permille) of N_tot
    1) y is smaller than y_max of N_tot (default 1%)
    1) t is less than T_max"""
    mask_min_1_permille = y > N_tot * 1 / 1000
    mask_max_1_percent = y < N_tot * y_max
    mask_T_max = t < T_max
    mask = mask_min_1_permille & mask_max_1_percent & mask_T_max
    return t[mask], y[mask]


def add_fit_results_to_fit_object(fit_object, filename, cfg, T_max, df, make_MC_fits=False):

    fit_object.filename = filename

    I_max_SIR, R_inf_SIR = SIR.calc_deterministic_results(cfg, T_max * 1.2, dt=0.01, ts=0.1)
    I_max_fit, R_inf_fit = fit_object.compute_I_max_R_inf(T_max=T_max * 1.5)

    fit_object.I_max_ABM = np.max(df["I"])
    fit_object.I_max_fit = I_max_fit
    fit_object.I_max_SIR = I_max_SIR

    fit_object.R_inf_ABM = df["R"].iloc[-1]
    fit_object.R_inf_fit = R_inf_fit
    fit_object.R_inf_SIR = R_inf_SIR

    if make_MC_fits:
        SIR_results, I_max_MC, R_inf_MC = fit_object.make_monte_carlo_fits(
            N_samples=100, T_max=T_max * 1.5, ts=0.1
        )
        # fit_object.SIR_results = SIR_results
        fit_object.I_max_MC = I_max_MC
        fit_object.R_inf_MC = R_inf_MC


def draw_random_p0(cfg, N_max_fits):
    param_grid = {
        #   'lambda_E': uniform(cfg.lambda_E/10, cfg.lambda_E*5),
        #   'lambda_I': uniform(cfg.lambda_I/10, cfg.lambda_I*5),
        "beta": uniform(cfg.beta / 10, cfg.beta * 5),
        "tau": uniform(-10, 10),
    }
    i = 0
    while i < N_max_fits:
        random_p0 = list(ParameterSampler(param_grid, n_iter=1))[0]
        yield i, random_p0
        i += 1


def refit_if_needed(fit_object, cfg, bounds, fix, minuit, N_max_fits=10, debug=False):

    fit_failed = True
    i_refits = 0

    if fit_object.valid_fit:
        fit_failed = False

    else:

        minuit_dict = dict(
            pedantic=False,
            print_level=0,
            **bounds,
            errordef=Minuit.LEAST_SQUARES,
            **fix,
        )

        # max_reduced_chi2 = np.linspace(3, 3, N_max_fits)

        best_fit_red_chi2 = 1e10
        best_fit = None

        for i_refits, random_p0 in draw_random_p0(cfg, N_max_fits):

            minuit = Minuit(fit_object, **random_p0, **minuit_dict)
            minuit.migrad()
            # fit_object.set_minuit(minuit, max_reduced_chi2[i_refits])
            fit_object.set_minuit(minuit, max_reduced_chi2=100)

            better_chi2 = fit_object.reduced_chi2 < best_fit_red_chi2
            semi_valid = fit_object._valid_fit(minuit, max_reduced_chi2=None)
            if better_chi2 and semi_valid:
                best_fit = copy(fit_object)
                best_fit_red_chi2 = fit_object.reduced_chi2

            if debug:
                print(i_refits, fit_object.reduced_chi2)

            if fit_object.valid_fit:
                fit_failed = False
                fit_object = best_fit
                break

    # if unable to fit the data, stop the fit
    if fit_failed:
        return fit_object, fit_failed

    # compute better errors (slow!)
    # minuit.minos()
    # fit_object.set_minuit(minuit)

    fit_object.N_refits = i_refits
    return fit_object, fit_failed


def run_actual_fit(t, y, sy, cfg, dt, ts):

    debug = False
    # debug = True

    # np.random.seed(cfg.ID)
    np.random.seed(42)

    if debug:
        reload(SIR)
        print("delete this")

    # reload(SIR)
    normal_priors = dict(
        # multiplier=0,
        # lambda_E={'mean': cfg.lambda_E, 'std': cfg.lambda_E/10},
        # lambda_I={'mean': cfg.lambda_I, 'std': cfg.lambda_I/10},
        # beta=    {'mean': 0.01, 'std': 0.05},
    )

    p0 = dict(
        lambda_E=cfg.lambda_E,
        lambda_I=cfg.lambda_I,
        beta=cfg.beta,
        tau=0,
    )

    bounds = dict(
        limit_lambda_E=(1e-6, None),
        limit_lambda_I=(1e-6, None),
        limit_beta=(1e-6, None),
    )

    fix = dict(
        fix_lambda_E=True,
        fix_lambda_I=True,
    )

    fit_object = SIR.FitSIR(t, y, sy, normal_priors, cfg, dt=dt, ts=ts)
    minuit = Minuit(
        fit_object,
        pedantic=False,
        print_level=0,
        **p0,
        **bounds,
        **fix,
        errordef=Minuit.LEAST_SQUARES,
    )

    minuit.migrad()

    fit_object.set_minuit(minuit)

    fit_object, fit_failed = refit_if_needed(fit_object, cfg, bounds, fix, minuit, debug=debug)

    return fit_object, fit_failed


def fit_single_file(filename, cfg, ts=0.1, dt=0.01, y_max=0.01):

    df = file_loaders.pandas_load_file(filename)

    df_interpolated = SIR.interpolate_df(df)

    # Time at end of simulation
    T_max = df["time"].max()

    # time at peak I (peak infection)
    T_peak = df["time"].iloc[df["I"].argmax()]

    # extract data between 1 permille and 1 percent I of N_tot and lower than T_max
    t, y = extract_data(
        t=df_interpolated["time"].values,
        y=df_interpolated["I"].values,
        T_max=T_peak,
        N_tot=cfg.N_tot,
        y_max=y_max,
    )
    sy = np.sqrt(y)

    if len(t) < 5:
        return filename, f"Too few datapoints (N = {len(t)})"

    fit_object, fit_failed = run_actual_fit(t, y, sy, cfg, dt, ts)
    if fit_failed:
        return filename, "Fit failed"

    try:
        add_fit_results_to_fit_object(
            fit_object,
            filename,
            cfg,
            T_max,
            df,
            make_MC_fits=False,
        )
    except AttributeError as e:
        print(filename)
        print("\n\n")
        raise e

    return filename, fit_object


#%%

from collections import Counter


def fit_multiple_files(cfg, filenames, num_cores=1, do_tqdm=True, y_max=0.01, verbose=False):

    func = partial(fit_single_file, cfg=cfg, y_max=y_max)

    if num_cores == 1:
        if do_tqdm:
            filenames = tqdm(filenames)
        results = [func(filename) for filename in filenames]

    else:
        results = p_umap(func, filenames, num_cpus=num_cores, disable=True)

    reject_counter = Counter()

    # postprocess results from multiprocessing:
    fit_objects = {}
    for filename, fit_result in results:

        if isinstance(fit_result, str):
            if verbose:
                print(f"\n\n{filename} was rejected due to {fit_result.lower()}")
            reject_counter[fit_result.lower()] += 1

        else:
            fit_object = fit_result
            fit_objects[filename] = fit_object
            reject_counter["no rejection"] += 1

    return fit_objects, reject_counter


def get_fit_results(abm_files, force_rerun=False, num_cores=1, y_max=0.01):

    all_fits_file = f"Data/fits_ymax_{y_max}.joblib"

    if Path(all_fits_file).exists() and not force_rerun:
        print("Loading all Imax fits", flush=True)
        return joblib.load(all_fits_file)

    else:

        all_fits = {}
        print(
            f"Fitting {len(abm_files.all_filenames)} files with {len(abm_files.cfgs)} different simulation parameters, please wait.",
            flush=True,
        )

        reject_counter = Counter()

        desc = "Fitting ABM simulations"
        for cfg, filenames in tqdm(abm_files.iter_folders(), total=len(abm_files.cfgs), desc=desc):
            # break
            output_filename = Path("Data/fits") / f"fits_{cfg.hash}_ymax_{y_max}.joblib"
            utils.make_sure_folder_exist(output_filename)

            if output_filename.exists():
                all_fits[cfg.hash] = joblib.load(output_filename)

            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="covariance is not positive-semidefinite."
                    )
                    fit_results, reject_counter_tmp = fit_multiple_files(
                        cfg,
                        filenames,
                        num_cores=num_cores,
                        y_max=y_max,
                    )

                joblib.dump(fit_results, output_filename)
                all_fits[cfg.hash] = fit_results
                reject_counter += reject_counter_tmp

        print(reject_counter)

        joblib.dump(all_fits, all_fits_file)
        return all_fits


if False:

    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter

    filename = "Data/ABM/e24e6303fc/ABM_2020-10-12_e24e6303fc_ID__0.hdf5"
    cfg = file_loaders.filename_to_cfg(filename)
    filename, fit_result = fit_single_file(filename, cfg, ts=0.1, dt=0.01, y_max=0.01)

    t = fit_result.t
    T_max = max(t) * 1.1
    df_fit = fit_result.calc_df_fit(ts=0.1, T_max=T_max)

    xlim = (63, 97)
    ylim = (0, 75_000)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(t, fit_result.y, fit_result.sy, fmt=".", label="ABM")
    ax.plot(df_fit["time"], df_fit["I"], label="Fit")
    ax.set(xlim=xlim, title="Fit", ylim=ylim)
    ax.text(
        0.1,
        0.8,
        f"$\chi^2 = {fit_result.chi2:.1f}, N = {fit_result.N}$",
        transform=ax.transAxes,
        fontsize=24,
    )
    ax.yaxis.set_major_formatter(EngFormatter())


#%%

import pandas as pd

from iminuit import Minuit, describe
from iminuit.util import make_func_code
from IPython.display import display


def exponential(t, I_0, R_eff, T):
    return I_0 * R_eff ** (t / T)


class FitSingleInfection_R_eff:
    def __init__(self, I, x=None, verbose=True):

        self.I = I.copy()
        if x is not None:
            self.x = x.copy()
        else:
            self.x = np.arange(len(I))
        self.sy = np.sqrt(I)

        self.verbose = verbose

        self.model = exponential
        self.fit_kwargs = {
            "I_0": self.I[0],
            "R_eff": 1,
            "limit_R_eff": (0, None),
            "T": 4.7,
            "fix_T": True,
        }
        self.func_code = make_func_code(describe(self.model)[1:])
        self.N_fit_parameters = len(describe(self.model)[1:])
        self.N = len(I)
        self.df = self.N - self.N_fit_parameters

    def __call__(self, *par):
        yhat = self.model(self.x, *par)
        chi2 = np.sum((yhat - self.I) ** 2 / self.sy ** 2)
        return chi2

    def fit_single_week(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        m = Minuit(self, errordef=1, pedantic=False, **self.fit_kwargs)
        m.migrad()
        if not m.fmin.is_valid:
            print("Not valid fit")
        if verbose:
            display(m.fmin)
            display(m.params)
        return dict(m.values), dict(m.errors)

    def fit_daily_R_eff(self, time_shift=0, keep_all_times=True):
        self.I_org = self.I.copy()
        self.x_org = self.x.copy()
        self.sy_org = self.sy.copy()
        R_eff = {}
        for day in range(7, self.N):
            days = np.arange(day - 7, day)
            self.I = self.I_org[days]
            self.x = self.x_org[days]
            self.sy = self.sy_org[days]
            values, errors = self.fit_single_week(verbose=False)

            if keep_all_times or day - time_shift >= 0:
                R_eff[day - time_shift] = {
                    "mean": values["R_eff"],
                    "std": errors["R_eff"],
                    "day_start": days[0] - time_shift,
                    "day_end": days[-1] - time_shift,
                }
        R_eff = pd.DataFrame(R_eff).T
        self.I = self.I_org
        self.x = self.x_org
        self.sy = self.sy_org
        del self.I_org
        del self.x_org
        del self.sy_org
        return R_eff.convert_dtypes().copy()
