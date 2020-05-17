import numpy as np
from numba import njit
from scipy import interpolate
import pandas as pd
from pathlib import Path
from scipy.stats import uniform as sp_uniform
import plotly.graph_objects as go
import SimulateDenmark_extra_funcs
import matplotlib.pyplot as plt
import pickle

def get_filenames():
    filenames = Path('Data/NetworkSimulation').rglob(f'*.csv')
    return [str(file) for file in sorted(filenames)]


def pandas_load_file(filename, return_only_df=False):
    df_raw = pd.read_csv(filename).convert_dtypes()

    for state in ['E', 'I']:
        df_raw[state] = sum([df_raw[col] for col in df_raw.columns if state in col and len(col) == 2])

    # only keep relevant columns
    df = df_raw[['Time', 'E', 'I', 'R']].copy()
    if return_only_df:
        return df

    # make first value at time 0
    t0 = df['Time'].min()
    df['Time'] -= t0
    time = df['Time']

    # t0 = time.min()
    t_interpolated = np.arange(int(time.max())+1)
    cols_to_interpolate = ['E', 'I', 'R']
    df_interpolated = interpolate_dataframe(df, time, t_interpolated, cols_to_interpolate)

    return df, df_interpolated, time, t_interpolated


# from scipy.signal import savgol_filter
def interpolate_array(y, time, t_interpolated, force_positive=True):
    f = interpolate.interp1d(time, y, kind='cubic', fill_value=0, bounds_error=False)
    with np.errstate(invalid="ignore"):
        y_hat = f(t_interpolated)
        if force_positive:
            y_hat[y_hat<0] = 0
    return y_hat

# from scipy.signal import savgol_filter
def interpolate_dataframe(df, time, t_interpolated, cols_to_interpolate):
    data_interpolated = {}
    for col in cols_to_interpolate:
        y = df[col]
        y_hat = interpolate_array(y, time, t_interpolated)
        data_interpolated[col] = y_hat
    df_interpolated = pd.DataFrame(data_interpolated)
    df_interpolated['Time'] = t_interpolated
    return df_interpolated

@njit
def ODE_integrate(y0, Tmax, dt, ts, mu0, Mrate1, Mrate2, beta): 

    S, N0, E1, E2, E3, E4, I1, I2, I3, I4, R = y0

    click = 0
    ODE_result_SIR = np.zeros((int(Tmax/ts)+1, 5))
    Times = np.linspace(0, Tmax, int(Tmax/dt)+1)

    for Time in Times:

        dS  = -beta*mu0*2/N0*(I1+I2+I3+I4)*S
        dE1 = beta*mu0*2/N0*(I1+I2+I3+I4)*S - Mrate1*E1

        dE2 = Mrate1*E1 - Mrate1*E2
        dE3 = Mrate1*E2 - Mrate1*E3
        dE4 = Mrate1*E3 - Mrate1*E4

        dI1 = Mrate1*E4 - Mrate2*I1
        dI2 = Mrate2*I1 - Mrate2*I2
        dI3 = Mrate2*I2 - Mrate2*I3
        dI4 = Mrate2*I3 - Mrate2*I4

        # R0  += dt*beta*mu0/N0*(I1+I2+I3+I4)*S

        dR  = Mrate2*I4

        S  += dt*dS
        E1 = E1 + dt*dE1
        E2 = E2 + dt*dE2
        E3 = E3 + dt*dE3
        E4 = E4 + dt*dE4
        
        I1 += dt*dI1
        I2 += dt*dI2
        I3 += dt*dI3
        I4 += dt*dI4

        R += dt*dR

        if Time >= ts*click: # - t0:
            ODE_result_SIR[click, :] = [
                            S, 
                            E1+E2+E3+E4, 
                            I1+I2+I3+I4,
                            R,
                            Time, # RT
                            # R0,
                            ]
            click += 1
    return ODE_result_SIR



from functools import lru_cache


from iminuit.util import make_func_code
from iminuit import describe

class CustomChi2:  # override the class with a better one
    
    def __init__(self, t_interpolated, y_truth, y0, Tmax, dt, ts, mu0, y_min=10):
        
        self.t_interpolated = t_interpolated
        self.y_truth = y_truth#.to_numpy(int)
        self.y0 = y0
        self.Tmax = Tmax
        self.dt = dt
        self.ts = ts
        self.mu0 = mu0
        self.sy = np.sqrt(self.y_truth) #if sy is None else sy
        self.y_min = y_min
        self.N = sum(self.y_truth > self.y_min)
        self.N_refits = 0

    def __call__(self, Mrate1, Mrate2, beta, tau):  # par are a variable number of model parameters
        # compute the function value
        y_hat = self._calc_yhat_interpolated(Mrate1, Mrate2, beta, tau)
        mask = (self.y_truth > self.y_min)
        # compute the chi2-value
        chi2 = np.sum((self.y_truth[mask] - y_hat[mask])**2/self.sy[mask]**2)
        if np.isnan(chi2):
            return 1e10
        return chi2

    def __repr__(self):
        return f'CustomChi2(\n\t{self.t_interpolated=}, \n\t{self.y_truth=}, \n\t{self.y0=}, \n\t{self.Tmax=}, \n\t{self.dt=}, \n\t{self.ts=}, \n\t{self.mu0=}, \n\t{self.y_min=},\n\t)'.replace('=', ' = ').replace('array(', '').replace('])', ']')

    @lru_cache(maxsize=None)
    def _calc_ODE_result_SIR(self, Mrate1, Mrate2, beta, ts=None, Tmax=None):
        ts = ts if ts is not None else self.ts
        Tmax = Tmax if Tmax is not None else self.Tmax
        return ODE_integrate(self.y0, Tmax, self.dt, ts, self.mu0, Mrate1, Mrate2, beta)

    def _calc_yhat_interpolated(self, Mrate1, Mrate2, beta, tau):
        ODE_result_SIR = self._calc_ODE_result_SIR(Mrate1, Mrate2, beta)
        I_SIR = ODE_result_SIR[:, 2]
        time = ODE_result_SIR[:, 4]
        y_hat = interpolate_array(I_SIR, time, self.t_interpolated+tau)
        return y_hat

    def set_chi2(self, minuit):
        self.chi2 = self.__call__(**minuit.values)
        return self.chi2

    def set_minuit(self, minuit):
        # self.minuit = minuit
        # self.m = minuit
        self.parameters = minuit.parameters
        self.values = minuit.np_values()
        self.errors = minuit.np_values()

        self.fit_values = dict(minuit.values)
        self.fit_errors = dict(minuit.errors)

        self.chi2 = self.__call__(**self.fit_values)
        self.is_valid = minuit.get_fmin().is_valid

        try:
            self.correlations = minuit.np_matrix(correlation=True)
            self.covariances = minuit.np_matrix(correlation=False)

        except RuntimeError:
            pass

        return None
    
    def get_fit_par(self, parameter):
        return self.fit_values[parameter], self.fit_errors[parameter]

    def get_all_fit_pars(self):
        all_fit_pars = {}
        for parameter in self.parameters:
            all_fit_pars[parameter] = self.get_fit_par(parameter)
        df_fit_parameters = pd.DataFrame(all_fit_pars, index=['mean', 'std'])
        return df_fit_parameters

    def get_correlations(self):
        return pd.DataFrame(self.correlations, 
                            index=self.parameters, 
                            columns=self.parameters)

    def calc_df_fit(self, ts=0.01, values=None, Tmax=None):
        if values is None:
            values = self.values
        Mrate1, Mrate2, beta, tau = values
        ODE_result_SIR = self._calc_ODE_result_SIR(Mrate1, Mrate2, beta, ts=ts, Tmax=Tmax)
        cols = ['S', 'E', 'I', 'R', 'Time']
        df_fit = pd.DataFrame(ODE_result_SIR, columns=cols).convert_dtypes()
        df_fit['Time'] -= tau
        df_fit['N'] = df_fit[['S', 'E', 'I', 'R']].sum(axis=1)
        if df_fit.iloc[-1]['R'] == 0:
            df_fit = df_fit.iloc[:-1]
        return df_fit

    def compute_I_max(self, ts=0.1, values=None):
        if values is None:
            values = self.values
        Mrate1, Mrate2, beta, tau = values
        ODE_result_SIR = self._calc_ODE_result_SIR(Mrate1, Mrate2, beta, ts=ts)
        I_max = np.max(ODE_result_SIR[:, 2])
        return I_max
    
    def compute_R_inf(self, ts=0.1, values=None, Tmax=None):
        if values is None:
            values = self.values
        Mrate1, Mrate2, beta, tau = values
        ODE_result_SIR = self._calc_ODE_result_SIR(Mrate1, Mrate2, beta, ts=ts, Tmax=Tmax)
        R_inf = ODE_result_SIR[-1, 3]
        return R_inf



def dict_to_str(d):
    string = ''
    for key, val in d.items():
        string += f"{key}_{val}_"
    return string[:-1]


def filename_to_dotdict(filename, SK_P1_UK=False):
    return SimulateDenmark_extra_funcs.filename_to_dotdict(filename, SK_P1_UK=SK_P1_UK)

def string_to_dict(string, SK_P1_UK=False):
    return SimulateDenmark_extra_funcs.filename_to_dotdict(string, normal_string=True, SK_P1_UK=SK_P1_UK)

def dict_to_title(d, N=None, exclude=None):

    if type(d) == 'dict':
        cfg = SimulateDenmark_extra_funcs.DotDict(d)
    else:
        cfg = d
    N0_str = human_format(cfg.N0)
    title = f"N={N0_str}, β={cfg.beta:.4f}, γ={cfg.gamma:.1f}, σ={cfg.sigma:.1f},  α={cfg.alpha:.1f}, μ={cfg.mu:.1f}, λ1={cfg.Mrate1:.1f}, λ2={cfg.Mrate2:.1f}, Ninit={cfg.Ninit}, BB={cfg.BB}"
    if N:
        title += f", #{N}"

    if exclude:
        d_translate = {'beta': 'β', 'N0': 'N', 'mu': 'μ', 'alpha': 'α', 'Ninit': 'Ninit', 'sigma': 'σ', 'gamma': 'γ'}
        new_title = ''
        for s in title.split():
            if not d_translate[exclude] in s:
                new_title += f"{s} "
        title = new_title[:-1]
    
    return title

def filename_to_title(filename):
    return dict_to_title(filename_to_dotdict(filename))


def uniform(a, b):
    loc = a
    scale = b-a
    return sp_uniform(loc, scale)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])



# %%%%

from collections import defaultdict
from sklearn.model_selection import ParameterSampler
import joblib
from pathlib import Path
from iminuit import Minuit


# def fit_single_file(filename, ts=0.1, dt=0.01, FIT_MAX=100):

#     # ts = 0.1 # frequency of "observations". Now 1 pr. day
#     # dt = 0.01 # stepsize in integration
#     # FIT_MAX = 100

#     N_refits = 0
#     discarded_files = []

#     cfg = filename_to_dotdict(str(filename))
#     parameters_as_string = dict_to_str(cfg)
#     # d = extra_funcs.string_to_dict(parameters_as_string)

#     df, df_interpolated, time, t_interpolated = pandas_load_file(filename)
#     y_truth = df_interpolated['I'].to_numpy(int)
#     Tmax = int(time.max())+1 # max number of days
#     N0 = cfg.N0
#     # y0 =  S, N0,                E1,E2,E3,E4,  I1,I2,I3,I4,  R
#     y0 = N0-cfg.Ninit,N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0

#     # reload(extra_funcs)
#     fit_object = CustomChi2(t_interpolated, y_truth, y0, Tmax, dt=dt, ts=ts, mu0=cfg.mu, y_min=10)

#     minuit = Minuit(fit_object, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta, tau=0)

#     minuit.migrad()
#     fit_object.set_chi2(minuit)

#     i_fit = 0
#     # if (not minuit.get_fmin().is_valid) :
#     if fit_object.chi2 / fit_object.N > 100:

#         continue_fit = True
#         while continue_fit:
#             i_fit += 1
#             N_refits += 1

#             param_grid = {'Mrate1': uniform(0.1, 10), 
#                         'Mrate2': uniform(0.1, 10), 
#                         'beta': uniform(0.1/20, 20/20), 
#                         'tau': uniform(-10, 10),
#                         }
#             param_list = list(ParameterSampler(param_grid, n_iter=1))[0]
#             minuit = Minuit(fit_object, pedantic=False, print_level=0, **param_list)
#             minuit.migrad()
#             fit_object.set_minuit(minuit)

#             if fit_object.chi2 / fit_object.N <= 10 or i_fit>FIT_MAX:
#                 continue_fit = False
            
#     if i_fit <= FIT_MAX:
#         fit_object.set_minuit(minuit)
#         return filename, fit_object, N_refits

#     else:
#         print(f"\n\n{filename} was discarded\n", flush=True)
#         return filename, None, N_refits



@lru_cache(maxsize=None)
def calc_Imax_R_inf_deterministic(mu, Mrate1, Mrate2, beta, y0, Tmax, dt, ts):
    ODE_result_SIR = ODE_integrate(y0, Tmax, dt, ts, mu, Mrate1, Mrate2, beta)
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
        param_grid = {'Mrate1': uniform(cfg.Mrate1/10, cfg.Mrate1*5), 
                    'Mrate2': uniform(cfg.Mrate2/10, cfg.Mrate2*5), 
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

    # print(filename)

    # ts = 0.1 # frequency of "observations". Now 1 pr. day
    # dt = 0.01 # stepsize in integration

    cfg = filename_to_dotdict(filename)

    df, df_interpolated, time, t_interpolated = pandas_load_file(filename, return_only_df=False)
    R_inf_net = df['R'].iloc[-1]

    Tmax = int(df['Time'].max())+1 # max number of days
    N0 = cfg.N0
    y0 = N0-cfg.Ninit, N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0#, cfg.Ninit

    I_min = 100
    I_lockdown = I_lockdown_rel * cfg.N0 # percent
    iloc_start = np.argmax(I_min <= df_interpolated['I'])
    iloc_lockdown = np.argmax(I_lockdown <= df_interpolated['I']) + 1
    
    #return None if simulation never reaches minimum requirements
    if I_lockdown < I_min:
        print(f"\nI_lockdown < I_min ({I_lockdown:.1f} < {I_min}) for file: \n{cfg}\n")
        return filename, None
    if df_interpolated['I'].max() < I_min:
        print(f"Never reached I_min={I_min}, only {df_interpolated['I'].max():.1f} for file: \n{cfg}\n", flush=True)
        return filename, None
    if df_interpolated['I'].max() < I_lockdown:
        print(f"Never reached I_lockdown={I_lockdown:.1f}, only {df_interpolated['I'].max():.1f} for file: \n{cfg}\n", flush=True)
        return filename, None

    y_truth_interpolated = df_interpolated['I']
    I_max_det, R_inf_det = calc_Imax_R_inf_deterministic(cfg.mu, cfg.Mrate1, cfg.Mrate2, cfg.beta, y0, Tmax*2, dt, ts)
    Tmax_peak = df_interpolated['I'].argmax()*1.2
    I_max_net = np.max(df['I'])

    fit_object = CustomChi2(t_interpolated[iloc_start:iloc_lockdown], y_truth_interpolated.to_numpy(int)[iloc_start:iloc_lockdown], y0, Tmax_peak, dt=dt, ts=ts, mu0=cfg.mu, y_min=10)

    minuit = Minuit(fit_object, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta, tau=0)

    try:
        minuit.migrad()
    except: 
        print(filename)
        raise ValueError()
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
    fit_object.I_max_net = I_max_net
    fit_object.I_max_hat = fit_object.compute_I_max()
    fit_object.I_max_det = I_max_det
    
    fit_object.R_inf_net = R_inf_net
    fit_object.R_inf_hat = fit_object.compute_R_inf(Tmax=Tmax*2)
    fit_object.R_inf_det = R_inf_det

    return filename, fit_object


    # reload(extra_funcs)
    # N_peak_fits = N_peak_fits
    # I_max_net = np.max(I)
    # I_maxs = np.zeros(N_peak_fits)
    # betas = np.zeros(N_peak_fits)
    # betas_std = np.zeros(N_peak_fits)
        # I_max = fit_object.compute_I_max()
        # I_maxs[imax] = I_max
        # betas[imax] = fit_object.fit_values['beta']
        # betas_std[imax] = fit_object.fit_errors['beta']
    # return filename, times_maxs_normalized, I_maxs, I_max_net, betas, betas_std




#%%



import multiprocessing as mp
from tqdm import tqdm


# def calc_fit_results(filenames, num_cores_max=20):

#     N_files = len(filenames)

#     num_cores = mp.cpu_count() - 1
#     if num_cores >= num_cores_max:
#         num_cores = num_cores_max

#     # print(f"Fitting {N_files} network-based simulations with {num_cores} cores, please wait.", flush=True)
#     with mp.Pool(num_cores) as p:
#         # results = list(tqdm(p.imap_unordered(fit_single_file, filenames), total=N_files))
#         results = list(p.imap_unordered(fit_single_file, filenames))

#     # modify results from multiprocessing

#     N_refits_total = 0
#     discarded_files = []
#     all_fit_objects = {}
#     for filename, fit_object, N_refits in results:
        
#         if fit_object is None:
#             discarded_files.append(filename)
#         else:
#             all_fit_objects[filename] = fit_object
        
#         N_refits_total += N_refits

#     return all_fit_objects, discarded_files, N_refits_total



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


# def get_fit_normal_results(filenames, force_rerun=False, num_cores_max=20):


#     all_fits_file = 'fits_normal.joblib'

#     if Path(all_fits_file).exists() and not force_rerun:
#         print("Loading all normal fits", flush=True)
#         return joblib.load(all_fits_file)

#     else:

#         all_normal_fits = {}
#         cfg_str = filenames_to_subgroups(filenames)
#         print(f"Fitting normal for {len(filenames)} simulations spaced on {len(cfg_str.items())} different runs, please wait.", flush=True)

#         for sim_pars, files in tqdm(cfg_str.items()):
#             output_filename = Path('Data/fits_normal') / f'normal_fits_{sim_pars}.joblib'
#             output_filename.parent.mkdir(parents=True, exist_ok=True)

#             if output_filename.exists() and not force_rerun:
#                 all_normal_fits[sim_pars] = joblib.load(output_filename)

#             else:
#                 fit_results = calc_fit_results(files, num_cores_max=num_cores_max)
#                 joblib.dump(fit_results, output_filename)
#                 all_normal_fits[sim_pars] = fit_results

#         joblib.dump(all_normal_fits, all_fits_file)
#         return all_normal_fits



def get_fit_Imax_results(filenames, force_rerun=False, num_cores_max=20):


    # bins = np.linspace(0, 1, N_peak_fits+1)
    # bin_centers = (bins[1:] + bins[:-1])/2

    all_fits_file = 'fits_Imax.joblib'

    if Path(all_fits_file).exists() and not force_rerun:
        print("Loading all Imax fits", flush=True)
        return joblib.load(all_fits_file)

    else:

        all_Imax_fits = {}
        cfg_str = filenames_to_subgroups(filenames)
        print(f"Fitting I_max for {len(filenames)} simulations spaced on {len(cfg_str.items())} different runs, please wait.", flush=True)

        for sim_pars, files in tqdm(cfg_str.items()):
            # print(sim_pars)
            output_filename = Path('Data/fits_Imax') / f'Imax_fits_{sim_pars}.joblib'
            output_filename.parent.mkdir(parents=True, exist_ok=True)

            try:

                if output_filename.exists() and not force_rerun:
                    all_Imax_fits[sim_pars] = joblib.load(output_filename)

                else:
                    fit_results = calc_fit_Imax_results(files, num_cores_max=num_cores_max)
                    joblib.dump(fit_results, output_filename)
                    all_Imax_fits[sim_pars] = fit_results

            except:
                print(f"{sim_pars} has error, skipping for now") # TODO FIXME

        joblib.dump(all_Imax_fits, all_fits_file)
        return all_Imax_fits


def cut_percentiles(x, p1, p2=None):
    if p2 is None:
        p1 = p1/2
        p2 = 100 - p1
    
    x = x[~np.isnan(x)]

    mask = (np.percentile(x, p1) < x) & (x < np.percentile(x, p2))
    return x[mask]


def fix_and_sort_index(df):
    df.index = df.index.map(filename_to_ID)
    return df.sort_index(ascending=True, inplace=False)

def Imax_fits_to_df(Imax_res, filenames_to_use, I_maxs_times):
    res_tmp = {k: Imax_res[k] for k in filenames_to_use}
    df = fix_and_sort_index(pd.DataFrame(res_tmp).T)
    df.columns = I_maxs_times
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df.loc['sdom'] = df.std() / np.sqrt(len(df)-1)
    # I_max_normed_by_pars[par_string] = df
    return df


def get_filenames_to_use_Imax(par_string):
    filenames_to_use = Path(f'Data/NetworkSimulation/{par_string}').glob(f"*.csv")
    filenames_to_use = [str(s) for s in filenames_to_use]
    return sorted(filenames_to_use)



def extract_normalized_Imaxs(d_fit_objects_all_IDs, I_maxs_net, filenames_to_use, bin_centers_Imax):
    I_maxs_normed_res = {}
    for filename, fit_objects in d_fit_objects_all_IDs.items():
        I_maxs = np.zeros(N_peak_fits)
        for i_fit_object, fit_object in enumerate(fit_objects):
            I_maxs[i_fit_object]  = fit_object.compute_I_max()
        ID = filename_to_ID(filename)
        I_maxs_normed_res[filename] = I_maxs / I_maxs_net[ID]
    df_I_maxs_normed = Imax_fits_to_df(I_maxs_normed_res, filenames_to_use, bin_centers_Imax)
    return df_I_maxs_normed


def extract_relative_Imaxs(d_fit_objects_all_IDs, I_maxs_net, filenames_to_use, bin_centers_Imax):
    I_maxs_relative_res = {}
    for filename, fit_objects in d_fit_objects_all_IDs.items():
        I_maxs = np.zeros(N_peak_fits)
        I_current_pos = np.zeros(N_peak_fits)
        for i_fit_object, fit_object in enumerate(fit_objects):
            I_maxs[i_fit_object] = fit_object.compute_I_max()
            I_current_pos[i_fit_object] = fit_object.y_truth[-1]
        ID = filename_to_ID(filename)
        I_maxs_relative_res[filename] = (I_maxs_net[ID]-I_maxs) / (I_maxs_net[ID]-I_current_pos)
    df_I_maxs_relative = Imax_fits_to_df(I_maxs_relative_res, filenames_to_use, bin_centers_Imax)
    return df_I_maxs_relative

def extract_relative_Imaxs_relative_I(d_fit_objects_all_IDs, I_maxs_net, filenames_to_use):
    
    I_maxs_relative_res = {}
    I_relative_res = {}
    for filename, fit_objects in d_fit_objects_all_IDs.items():
        # break
        I_maxs = np.zeros(N_peak_fits)
        I_current_pos = np.zeros(N_peak_fits)
        I_rel = np.zeros(N_peak_fits)

        for i_fit_object, fit_object in enumerate(fit_objects):
            N0 = fit_object.y0[1]
            I_rel[i_fit_object] = fit_object.y_truth[-1] / N0 # percent
            I_maxs[i_fit_object] = fit_object.compute_I_max()
            I_current_pos[i_fit_object] = fit_object.y_truth[-1]

        ID = filename_to_ID(filename)
        I_maxs_relative_res[filename] = (I_maxs_net[ID]-I_maxs) / (I_maxs_net[ID]-I_current_pos)
        I_relative_res[filename] = I_rel

    x = np.stack(I_relative_res.values())
    y = np.stack(I_maxs_relative_res.values())

    x_flat = x.flatten()
    y_flat = y.flatten()

    # df_I_rel = pd.DataFrame.from_dict(I_relative_res).T
    x_min = x.min()*0.99
    x_max = x.max()*1.01
    N_bins = N_peak_fits
    bins = np.linspace(x_min, x_max, N_bins+1)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    indices = np.digitize(x_flat, bins) - 1

    df_xy = pd.DataFrame({'x': x_flat, 'y':y_flat, 'bin': indices, 'bin_center': bin_centers[indices]})
    
    def calc_binned_mean_sdom(df_group):
        mean = np.mean(df_group['y'])
        std = np.std(df_group['y'])
        sdom = std / np.sqrt(len(df_group) -1)
        d = {'x': df_group['bin_center'].iloc[0], 
             'mean': mean,
             'std': std,
             'sdom': sdom}
        return pd.Series(d)
        # return d

    df_binned = df_xy.groupby('bin').apply(calc_binned_mean_sdom)
    return df_binned


def extract_fit_parameter(par, d_fit_objects_all_IDs, filenames_to_use, bin_centers_Imax):
    par_tmp = {}
    par_std_tmp = {}
    for filename, fit_objects in d_fit_objects_all_IDs.items():
        pars = np.zeros(N_peak_fits)
        pars_std = np.zeros(N_peak_fits)
        for i_fit_object, fit_object in enumerate(fit_objects):
            pars[i_fit_object]  = fit_object.fit_values[par]
            pars_std[i_fit_object]  = fit_object.fit_errors[par]
        par_tmp[filename] = pars 
        par_std_tmp[filename] = pars_std 
    df_par = Imax_fits_to_df(par_tmp, filenames_to_use, bin_centers_Imax)
    df_par_std = Imax_fits_to_df(par_std_tmp, filenames_to_use, bin_centers_Imax)
    return df_par, df_par_std


def mask_df(df, cut_val):
    mask = (-cut_val <= df.loc['mean']) & (df.loc['mean'] <= cut_val)
    return df.loc[:, mask]


from pandas.errors import EmptyDataError

def plot_SIR_model_comparison(force_overwrite=False, max_N_plots=100):

    pdf_name = f"Figures/SIR_comparison.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_overwrite:
        print(f"{pdf_name} already exists")
        return None
    
    else:
        from matplotlib.backends.backend_pdf import PdfPages

        base_dir = Path('Data') / 'NetworkSimulation'
        all_sim_pars = sorted([str(x.name) for x in base_dir.glob('*') if '.DS' not in str(x.name)])

        with PdfPages(pdf_name) as pdf:

            # sim_par = all_sim_pars[0]
            for sim_par in tqdm(all_sim_pars):

                ID_files = list((base_dir/sim_par).rglob('*.csv'))

                cfg = string_to_dict(sim_par)

                fig, ax = plt.subplots(figsize=(20, 10))

                Tmax = 0
                lw = 0.1
                
                it = enumerate(ID_files[:max_N_plots]) if max_N_plots < len(ID_files) else enumerate(ID_files[:max_N_plots])

                for i, filename_ID in it:
                    try:
                        df = pandas_load_file(filename_ID, return_only_df=True)
                    except EmptyDataError as e:
                        print(f"Skipping {filename_ID} because empty file")
                        continue
                        # raise e
                    label = 'Simulations' if i == 0 else None
                    # lw = 1 if i == 0 else 0.1
                    ax.plot(df['Time'].values, df['I'].values, lw=lw, c='k', label=label)
                    if df['Time'].max() > Tmax:
                        Tmax = df['Time'].max() 

                Tmax = max(Tmax, 50)

                y0 = cfg.N0-cfg.Ninit, cfg.N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0#, cfg.Ninit
                dt = 0.01
                ts = 0.1

                ODE_result_SIR = ODE_integrate(y0, Tmax, dt, ts, mu0=cfg.mu, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta)
                # print(y0, Tmax, dt, ts, cfg)
                I_SIR = ODE_result_SIR[:, 2]
                time = ODE_result_SIR[:, 4]
                cols = ['S', 'E_sum', 'I_sum', 'R', 'Time']
                df_fit = pd.DataFrame(ODE_result_SIR, columns=cols).convert_dtypes()

                ax.plot(time, I_SIR, lw=15*lw, color='red', label='SIR')
                leg = ax.legend(loc='upper right')
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
                
                title = dict_to_title(cfg, len(ID_files))
                ax.set(title=title, xlabel='Time', ylabel='I')
                
                ax.set_rasterized(True)
                ax.set_rasterization_zorder(0)

                pdf.savefig(fig, dpi=100)
                plt.close('all')





#%%

def get_filenames_different_than_default(find_par):

    base_dir = Path('Data') / 'NetworkSimulation'
    all_sim_pars = sorted([str(x.name) for x in base_dir.glob('*') if str(x.name) != '.DS_Store'])

    all_sim_pars_as_dict = {s: string_to_dict(s) for s in all_sim_pars}
    df_sim_pars = pd.DataFrame.from_dict(all_sim_pars_as_dict, orient='index')

    default_pars = dict(
                    N0 = 500_000,
                    mu = 20.0,  # Average number connections
                    alpha = 0.0, # Spatial parameter
                    beta = 0.01, # Mean rate
                    sigma = 0.0, # Spread in rate
                    Mrate1 = 1.0, # E->I
                    Mrate2 = 1.0, # I->R
                    gamma = 0.0, # Parameter for skewed connection shape
                    nts = 0.1, 
                    Nstates = 9,
                    BB = 1,
                    Ninit = 100, # 100 Initial Infected
                )


    if isinstance(find_par, str):
        find_par = [find_par]

    query = ''
    for key, val in default_pars.items():
        if not key in find_par:
            query += f"{key} == {val} & "
    query = query[:-3]

    df_different_than_default = df_sim_pars.query(query).sort_values(find_par)
    return list(df_different_than_default.index)




#%%

def plot_variable_other_than_default(par):

    filenames_par_rest_default = get_filenames_different_than_default(par)

    base_dir = Path('Data') / 'NetworkSimulation'
    # I_max_rel = {}

    x = np.zeros(len(filenames_par_rest_default))
    y = np.zeros_like(x)
    sy = np.zeros_like(x)
    n = np.zeros_like(x)

    # i_simpar, sim_par = 0, filenames_par_rest_default[0]
    for i_simpar, sim_par in enumerate(tqdm(filenames_par_rest_default)):
        filenames = [str(filename) for filename in base_dir.rglob('*.csv') if sim_par in str(filename)]
        N_files = len(filenames)

        I_max_net = np.zeros(N_files)
        for i_filename, filename in enumerate(filenames):
            cfg = filename_to_dotdict(filename)
            df = pandas_load_file(filename, return_only_df=True)
            I_max_net[i_filename] = df['I'].max()

        Tmax = df['Time'].max()*1.2
        y0 = cfg.N0-cfg.Ninit, cfg.N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0#, cfg.Ninit
        dt = 0.01
        ts = 0.1
        ODE_result_SIR = ODE_integrate(y0, Tmax, dt, ts, mu0=cfg.mu, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta)
        # print(y0, Tmax, dt, ts, cfg)
        I_SIR = ODE_result_SIR[:, 2]
        I_max_SIR = np.max(I_SIR)
        z_rel = I_max_net / I_max_SIR
        # I_max_rel[cfg[par]] = I_max_net / I_max_SIR
        x[i_simpar] = cfg[par]
        y[i_simpar] = np.mean(z_rel)
        sy[i_simpar] = np.std(z_rel) / np.sqrt(len(z_rel))
        n[i_simpar] = len(z_rel)

    title = dict_to_title(cfg, exclude=par)

    n_text = [f"n = {int(ni)}" for ni in n]
    fig = go.Figure(data=go.Scatter(x=x, y=y, text=n_text, mode='markers', error_y=dict(array=sy)))
    fig.update_layout(title=title,
                    xaxis_title=par,
                    yaxis_title='I_max_net / I_max_SIR',
                    height=600, width=800,
                    showlegend=False,
                    )
    # fig.show()

    figname_html = Path(f"Figures/par_SIR_network_relative/html/par_SIR_network_relative_{par}.html")
    figname_png = Path(f"Figures/par_SIR_network_relative/png/par_SIR_network_relative_{par}.png")
    Path(figname_html).parent.mkdir(parents=True, exist_ok=True)
    Path(figname_png).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(figname_html))
    fig.write_image(str(figname_png))


