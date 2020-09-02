
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

try:
    from src import utils
    from src import file_loaders
    from src import SIR
    from src import parallel
except ImportError:
    import utils
    import file_loaders
    import SIR
    import parallel

reload(SIR)


def uniform(a, b):
    loc = a
    scale = b-a
    return sp_uniform(loc, scale)


def extract_data(t, y, T_max, N_tot):
    """ Extract data where:
        1) y is larger than 1‰ (permille) of N_tot
        1) y is smaller than 1% (percent) of N_tot
        1) t is less than T_max"""
    mask_min_1_permille = (y > N_tot*1/1000)
    mask_max_1_percent = (y < N_tot*1/100)
    mask_T_max = t < T_max
    mask = mask_min_1_permille & mask_max_1_percent & mask_T_max
    return t[mask], y[mask]


def add_fit_results_to_fit_object(fit_object, filename, cfg, T_max, df):

    fit_object.filename = filename

    I_max_SIR, R_inf_SIR = SIR.calc_deterministic_results(cfg, T_max*1.2, dt=0.01, ts=0.1)
    I_max_fit, R_inf_fit = fit_object.compute_I_max_R_inf(T_max=T_max*1.5)


    fit_object.I_max_ABN = np.max(df['I'])
    fit_object.I_max_fit = I_max_fit
    fit_object.I_max_SIR = I_max_SIR

    fit_object.R_inf_ABN = df['R'].iloc[-1]
    fit_object.R_inf_fit = R_inf_fit
    fit_object.R_inf_SIR = R_inf_SIR

    SIR_results, I_max_MC, R_inf_MC = fit_object.make_monte_carlo_fits(N_samples=100, T_max=T_max*1.5, ts=0.1)
    # fit_object.SIR_results = SIR_results
    fit_object.I_max_MC = I_max_MC
    fit_object.R_inf_MC = R_inf_MC


def draw_random_p0(cfg, N_max_fits):
    param_grid = {
                #   'lambda_E': uniform(cfg.lambda_E/10, cfg.lambda_E*5),
                #   'lambda_I': uniform(cfg.lambda_I/10, cfg.lambda_I*5),
                  'beta':     uniform(cfg.beta/10,     cfg.beta*5),
                  'tau':      uniform(-10, 10),
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

        minuit_dict = dict(pedantic=False, print_level=0, **bounds, errordef=Minuit.LEAST_SQUARES, **fix)

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

    np.random.seed(cfg.ID)

    if debug:
        reload(SIR)
        print('delete this')

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
    minuit = Minuit(fit_object, pedantic=False, print_level=0, **p0, **bounds, **fix, errordef=Minuit.LEAST_SQUARES)

    minuit.migrad()

    fit_object.set_minuit(minuit)

    fit_object, fit_failed = refit_if_needed(fit_object, cfg, bounds, fix, minuit, debug=debug)

    if debug:
        print(f"chi2 = {fit_object.chi2:.3f}")
        print("")
        print(fit_object.get_fit_parameters())
        print("")
        print(normal_priors)
        print("")
        print(fit_object.correlations)
        print("")

        I_max_SIR, R_inf_SIR = SIR.calc_deterministic_results(cfg, 500, dt=0.01, ts=0.1)
        print(f"I_max_SIR = {I_max_SIR/1e6:.2f} * 10^6")

        I_max_fit, R_inf_fit = fit_object.compute_I_max_R_inf(T_max=500)
        print(f"I_max_fit = {I_max_fit/1e6:.2f} * 10^6")

        SIR_results, I_max_MC, R_inf_MC = fit_object.make_monte_carlo_fits(N_samples=100, T_max=500, ts=0.1)
        fig, ax = plt.subplots()
        ax.hist(I_max_MC, 50);
        ax.xaxis.set_major_formatter(EngFormatter())

    return fit_object, fit_failed


def fit_single_file(filename, ts=0.1, dt=0.01):

    cfg = utils.string_to_dict(filename)
    df = file_loaders.pandas_load_file(filename)

    df_interpolated = SIR.interpolate_df(df)

    # Time at end of simulation
    T_max = df['time'].max()

    # time at peak I (peak infection)
    T_peak = df['time'].iloc[df['I'].argmax()]

    # extract data between 1 permille and 1 percent I of N_tot and lower than T_max
    t, y = extract_data(t=df_interpolated['time'].values,
                        y=df_interpolated['I'].values,
                        T_max=T_peak,
                        N_tot=cfg.N_tot)
    sy = np.sqrt(y)

    if len(t) < 5:
        return filename, f'Too few datapoints (N = {len(t)})'

    fit_object, fit_failed = run_actual_fit(t, y, sy, cfg, dt, ts)
    if fit_failed:
        return filename, 'Fit failed'

    try:
        add_fit_results_to_fit_object(fit_object, filename, cfg, T_max, df)
    except AttributeError as e:
        print(filename)
        print("\n\n")
        raise e

    return filename, fit_object



#%%

import warnings

def fit_multiple_files(filenames, num_cores=1, do_tqdm=True):

    if num_cores == 1:
        if do_tqdm:
            filenames = tqdm(filenames)
        results = [fit_single_file(filename) for filename in filenames]

    else:
        results = parallel.p_umap(fit_single_file, filenames, num_cpus=num_cores, do_tqdm=False)
        # with mp.Pool(num_cores) as p:
            # results = list(p.imap_unordered(fit_single_file, filenames))

    # postprocess results from multiprocessing:
    fit_objects = {}
    for filename, fit_result in results:

        if isinstance(fit_result, str):
            print(f"\n\n{filename} was rejected due to {fit_result.lower()}")

        else:
            fit_object = fit_result
            fit_objects[filename] = fit_object
    return fit_objects



def get_fit_results(abn_files, force_rerun=False, num_cores=1):

    all_fits_file = 'Data/fits.joblib'

    if Path(all_fits_file).exists() and not force_rerun:
        print("Loading all Imax fits", flush=True)
        return joblib.load(all_fits_file)

    else:

        all_fits = {}
        print(f"Fitting {len(abn_files.all_files)} files with {len(abn_files.ABN_parameters)} different simulation parameters, please wait.", flush=True)

        for ABN_parameter in tqdm(abn_files.keys):
            # break
            cfg = utils.string_to_dict(ABN_parameter)
            files = abn_files[ABN_parameter]
            output_filename = Path('Data/fits') / f'fits_{ABN_parameter}.joblib'
            utils.make_sure_folder_exist(output_filename)

            if output_filename.exists() and not force_rerun:
                all_fits[ABN_parameter] = joblib.load(output_filename)

            else:
                fit_results = fit_multiple_files(files, num_cores=num_cores)
                joblib.dump(fit_results, output_filename)
                all_fits[ABN_parameter] = fit_results

        joblib.dump(all_fits, all_fits_file)
        return all_fits




filename = '../Data/ABN/N_tot__580000__N_init__100__N_ages__1__mu__40.0__sigma_mu__1.0__beta__0.04__sigma_beta__1.0__rho__0.0__lambda_E__1.0__lambda_I__2.0__epsilon_rho__0.01__beta_scaling__1.0__age_mixing__1.0__algo__2/N_tot__580000__N_init__100__N_ages__1__mu__40.0__sigma_mu__1.0__beta__0.04__sigma_beta__1.0__rho__0.0__lambda_E__1.0__lambda_I__2.0__epsilon_rho__0.01__beta_scaling__1.0__age_mixing__1.0__algo__2__ID__006.csv'
ts = 0.1
dt = 0.01
# filename = filename.replace('N_tot__580000__', 'N_tot__5800000__')
# filename = filename.replace('N_init__1000__', 'N_init__100__')
# filename = filename.replace('rho__100.0__', 'rho__15.0__')

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


if False:


    cfg = utils.string_to_dict(filename)
    df = file_loaders.pandas_load_file(filename)

    plt.plot(df['time'], df['I'])

    df_interpolated = SIR.interpolate_df(df)

    # Time at end of simulation
    T_max = df['time'].max()

    # time at peak I (peak infection)
    T_peak = df['time'].iloc[df['I'].argmax()]

    # extract data between 1 permille and 1 percent I of N_tot and lower than T_max
    t, y = extract_data(t=df_interpolated['time'].values,
                        y=df_interpolated['I'].values,
                        T_max=T_peak,
                        N_tot=cfg.N_tot)
    sy = np.sqrt(y)


    plt.errorbar(t, y, yerr=sy, fmt='.')

    fig, ax = plt.subplots()
    for i, SIR_result in enumerate(SIR_results):
        t = SIR_result[:, 0]
        I = SIR_result[:, -2]
        R = SIR_result[:, -1]
        ax.plot(t, I)
    # ax.legend()


    fig2, ax2 = plt.subplots()
    ax2.hist(fit_object.I_max_MC, 50)


    df = fit_object.calc_df_fit(T_max=100)
    fig3, ax3 = plt.subplots()
    ax3.plot(df['time'], df['I'])
    ax3.errorbar(fit_object.t, fit_object.y, yerr=fit_object.sy, fmt='.')
    ax3.set(xlim=(fit_object.t.min()*0.9, fit_object.t.max()*1.1),
            ylim=(fit_object.y.min()*0.9, fit_object.y.max()*1.1), )



    fit_object, fit_failed = run_actual_fit(t, y, sy, cfg, dt, ts, filename)
    add_fit_results_to_fit_object(fit_object, filename, cfg, T_max, df)
    print(f"{fit_object.I_max_SIR/1e6=}")
    print(f"{fit_object.I_max_ABN/1e6=}")
    print(f"{fit_object.I_max_fit/1e6=}")
    fig4, ax4 = plt.subplots()
    ax4.hist(fit_object.I_max_MC, 50);


    SIR.calc_deterministic_results(cfg, T_max*1.2, dt=0.01, ts=0.1)