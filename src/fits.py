
from collections import defaultdict
from sklearn.model_selection import ParameterSampler
import joblib
import numpy as np
from pathlib import Path
from iminuit import Minuit
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats import uniform as sp_uniform
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



def uniform(a, b):
    loc = a
    scale = b-a
    return sp_uniform(loc, scale)


def try_refit(fit_object, cfg, FIT_MAX):
    N_refits = 0
    while True:
        N_refits += 1
        param_grid = {'lambda_E': uniform(cfg.lambda_E/10, cfg.lambda_E*5),
                      'lambda_I': uniform(cfg.lambda_I/10, cfg.lambda_I*5),
                       'beta': uniform(cfg.beta/10, cfg.beta*5),
                       'tau': uniform(-10, 10),
                    }
        param_list = list(ParameterSampler(param_grid, n_iter=1))[0]
        minuit = Minuit(fit_object, pedantic=False, print_level=0, **param_list)
        minuit.migrad()
        fit_object.set_minuit(minuit)
        if (0.001 <= fit_object.chi2 / fit_object.N <= 10) or N_refits > FIT_MAX:
            break
    return fit_object, N_refits


def run_actual_fit(t, y, cfg, dt, ts, filename):

    fit_object = SIR.FitSIR(t, y, cfg, dt=dt, ts=ts)

    p0 = dict(lambda_E=cfg.lambda_E, lambda_I=cfg.lambda_I, beta=cfg.beta, tau=0)
    minuit = Minuit(fit_object, pedantic=False, print_level=0, **p0)

    minuit.migrad()
    fit_object.set_chi2(minuit)

    fit_failed = False
    if fit_object.chi2 / fit_object.N > 100:
        FIT_MAX = 100
        fit_object, N_refits = try_refit(fit_object, cfg, FIT_MAX)
        fit_object.N_refits = N_refits
        if N_refits > FIT_MAX:
            # print(f"\n{filename} was rejected after {N_refits} tries\n", flush=True)
            fit_failed = True

    fit_object.set_minuit(minuit)

    return fit_object, fit_failed


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

    fit_object.I_max_ABN = np.max(df['I'])
    fit_object.I_max_fit = fit_object.compute_I_max()
    fit_object.I_max_SIR = I_max_SIR

    fit_object.R_inf_ABN = df['R'].iloc[-1]
    fit_object.R_inf_fit = fit_object.compute_R_inf(T_max=T_max*2)
    fit_object.R_inf_SIR = R_inf_SIR


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

    if len(t) < 5:
        # print(f"\n\n{filename} was rejected since len(t) = {len(t)}\n", flush=True)
        return filename, f'Too few datapoints (N = {len(t)})'

    fit_object, fit_failed = run_actual_fit(t, y, cfg, dt, ts, filename)
    if fit_failed:
        return filename, 'Too many fit retries'

    add_fit_results_to_fit_object(fit_object, filename, cfg, T_max, df)

    return filename, fit_object



#%%


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

