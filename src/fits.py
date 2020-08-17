
from collections import defaultdict
from sklearn.model_selection import ParameterSampler
import joblib
from pathlib import Path
from iminuit import Minuit

try:
    from src import utils
except ImportError:
    import utils


def uniform(a, b):
    loc = a
    scale = b-a
    return sp_uniform(loc, scale)


@lru_cache(maxsize=None)
def calc_Imax_R_inf_SIRerministic(mu, lambda_E, lambda_I, beta, y0, Tmax, dt, ts):
    ODE_result_SIR = ODE_integrate(y0, Tmax, dt, ts, mu, lambda_E, lambda_I, beta)
    I_max = np.max(ODE_result_SIR[:, 2])
    R_inf = ODE_result_SIR[-1, 3]
    return I_max, R_inf

# N_peak_fits = 20
dark_figure = 40
I_lockdown_DK = 350*dark_figure
I_lockdown_rel = I_lockdown_DK / 5_824_857

def try_refit(fit_object, cfg, FIT_MAX):
    N_refits = 0
    # if (not minuit.get_fmin().is_valid) :
    continue_fit = True
    while continue_fit:
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
            continue_fit = False
    return fit_object, N_refits


def fit_single_file_Imax(filename, ts=0.1, dt=0.01):

    cfg = filename_to_dotdict(filename)

    df, df_interpolated, time, t_interpolated = pandas_load_file(filename, return_only_df=False)
    R_inf_ABN = df['R'].iloc[-1]

    Tmax = int(df['Time'].max())+1 # max number of days
    N_tot = cfg.N_tot
    y0 = N_tot-cfg.N_init, N_tot,   cfg.N_init,0,0,0,      0,0,0,0,   0#, cfg.N_init

    I_min = 100
    I_lockdown = I_lockdown_rel * cfg.N_tot # percent
    iloc_start = np.argmax(I_min <= df_interpolated['I'])
    iloc_lockdown = np.argmax(I_lockdown <= df_interpolated['I']) + 1

    #return None if simulation never reaches minimum requirements
    if I_lockdown < I_min:
        print(f"\nI_lockdown < I_min ({I_lockdown:.1f} < {I_min}) for file: \n{filename}\n")
        return filename, None
    if df_interpolated['I'].max() < I_min:
        print(f"Never reached I_min={I_min}, only {df_interpolated['I'].max():.1f} for file: \n{filename}\n", flush=True)
        return filename, None
    if df_interpolated['I'].max() < I_lockdown:
        print(f"Never reached I_lockdown={I_lockdown:.1f}, only {df_interpolated['I'].max():.1f} for file: \n{filename}\n", flush=True)
        return filename, None
    if iloc_lockdown - iloc_start < 10:
        print(f"Fit period less than 10 days, only ={iloc_lockdown - iloc_start:.1f}, for file: \n{filename}\n", flush=True)
        return filename, None

    y_truth_interpolated = df_interpolated['I']
    I_max_SIR, R_inf_SIR = calc_Imax_R_inf_SIRerministic(cfg.mu, cfg.lambda_E, cfg.lambda_I, cfg.beta, y0, Tmax*2, dt, ts)
    Tmax_peak = df_interpolated['I'].argmax()*1.2
    I_max_ABN = np.max(df['I'])

    fit_object = CustomChi2(t_interpolated[iloc_start:iloc_lockdown], y_truth_interpolated.to_numpy(float)[iloc_start:iloc_lockdown], y0, Tmax_peak, dt=dt, ts=ts, mu=cfg.mu, y_min=I_min)

    minuit = Minuit(fit_object, pedantic=False, print_level=0, lambda_E=cfg.lambda_E, lambda_I=cfg.lambda_I, beta=cfg.beta, tau=0)

    minuit.migrad()
    fit_object.set_chi2(minuit)

    if fit_object.chi2 / fit_object.N > 100:
        FIT_MAX = 100
        fit_object, N_refits = try_refit(fit_object, cfg, FIT_MAX)
        fit_object.N_refits = N_refits
        if N_refits > FIT_MAX:
            print(f"\n\n{filename} was discarded after {N_refits} tries\n", flush=True)
            return filename, None

    fit_object.set_minuit(minuit)

    fit_object.filename = filename
    fit_object.I_max_ABN = I_max_ABN
    fit_object.I_max_fit = fit_object.compute_I_max()
    fit_object.I_max_SIR = I_max_SIR

    fit_object.R_inf_ABN = R_inf_ABN
    fit_object.R_inf_fit = fit_object.compute_R_inf(Tmax=Tmax*2)
    fit_object.R_inf_SIR = R_inf_SIR

    return filename, fit_object



#%%



import multiprocessing as mp
from tqdm import tqdm


def calc_fit_Imax_results(filenames, num_cores_max=30):

    N_files = len(filenames)

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max

    with mp.Pool(num_cores) as p:
        # results = list(tqdm(p.imap_unordered(fit_single_file_Imax, filenames), total=N_files))
        results = list(p.imap_unordered(fit_single_file_Imax, filenames))

    # postprocess results from multiprocessing:
    fit_objects = {}
    for filename, fit_object in results:
        if fit_object:
            fit_objects[filename] = fit_object#.fit_objects_Imax
    return fit_objects



def filename_to_ID(filename):
    return int(filename.split('ID_')[1].strip('.csv'))

def filename_to_par_string(filename):
    return dict_to_str(filename_to_dotdict(filename))


def filenames_to_subgroups(filenames):
    cfg_str = defaultdict(list)
    for filename in sorted(filenames):
        s = dict_to_str(filename_to_dotdict(filename))
        cfg_str[s].append(filename)
    return cfg_str


def get_fit_results(abn_files, force_rerun=False, num_cores_max=20):

    all_fits_file = 'fits_Imax.joblib'

    if Path(all_fits_file).exists() and not force_rerun:
        print("Loading all Imax fits", flush=True)
        return joblib.load(all_fits_file)

    else:

        all_Imax_fits = {}
        print(f"Fitting I_max for {len(abn_files.all_files)} files with on {len(abn_files.ABN_parameters)} different simulation parameters, please wait.", flush=True)

        for ABN_parameter in tqdm(abn_files.keys):
            cfg = utils.string_to_dict(ABN_parameter)
            for i, file in enumerate(abn_files[ABN_parameter]):


        # for sim_pars, files in tqdm(cfg_str.items()):
            # break
            # print(sim_pars)
            output_filename = Path('Data/fits') / f'fits_{ABN_parameter}.joblib'
            utils.make_sure_folder_exist(output_filename)

            if output_filename.exists() and not force_rerun:
                all_Imax_fits[sim_pars] = joblib.load(output_filename)

            else:
                fit_results = calc_fit_Imax_results(files, num_cores_max=num_cores_max)
                joblib.dump(fit_results, output_filename)
                all_Imax_fits[sim_pars] = fit_results


        joblib.dump(all_Imax_fits, all_fits_file)
        return all_Imax_fits

