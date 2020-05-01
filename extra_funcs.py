import numpy as np
from numba import njit
from scipy import interpolate
import pandas as pd
from pathlib import Path
from scipy.stats import uniform as sp_uniform
import plotly.graph_objects as go
import SimulateNetwork_extra_funcs
import matplotlib.pyplot as plt

def get_filenames():
    filenames = Path('Data').rglob(f'*.csv')
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

    # mu0 = 1

    S, N0, E1, E2, E3, E4, I1, I2, I3, I4, R, R0 = y0

    click = 0
    ODE_result_SIR = np.zeros((int(Tmax/ts)+1, 6))
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

        R0  += dt*beta*mu0/N0*(I1+I2+I3+I4)*S

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
                            R0,
                            ]
            click += 1
    return ODE_result_SIR





from iminuit.util import make_func_code
from iminuit import describe

class CustomChi2:  # override the class with a better one
    
    def __init__(self, t_interpolated, y_truth, y0, Tmax, dt, ts, mu0, y_min=0):
        
        # self.f = f  # model predicts y for given x
        # self.time = time
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
        # self.func_code = make_func_code(describe(self._calc_yhat_interpolated))

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

    def _calc_ODE_result_SIR(self, Mrate1, Mrate2, beta, ts=None):
        ts = ts if ts is not None else self.ts
        return ODE_integrate(self.y0, self.Tmax, self.dt, ts, self.mu0, Mrate1, Mrate2, beta)

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

    def calc_df_fit(self, ts=0.01, values=None):
        if values is None:
            values = self.values
        Mrate1, Mrate2, beta, tau = values
        ODE_result_SIR = self._calc_ODE_result_SIR(Mrate1, Mrate2, beta, ts=ts)
        cols = ['S', 'E_sum', 'I_sum', 'R', 'Time', 'R0']
        df_fit = pd.DataFrame(ODE_result_SIR, columns=cols).convert_dtypes()
        df_fit['Time'] -= tau
        df_fit['N'] = df_fit[['S', 'E_sum', 'I_sum', 'R']].sum(axis=1)
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



def dict_to_str(d):
    string = ''
    for key, val in d.items():
        string += f"{key}_{val}_"
    return string[:-1]


def filename_to_dotdict(filename):
    return SimulateNetwork_extra_funcs.filename_to_dotdict(filename)

def string_to_dict(string):
    return SimulateNetwork_extra_funcs.filename_to_dotdict(string, normal_string=True)

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


def fit_single_file(filename, ts=0.1, dt=0.01, FIT_MAX=100):


    # ts = 0.1 # frequency of "observations". Now 1 pr. day
    # dt = 0.01 # stepsize in integration
    # FIT_MAX = 100

    N_refits = 0
    discarded_files = []

    cfg = filename_to_dotdict(str(filename))
    parameters_as_string = dict_to_str(cfg)
    # d = extra_funcs.string_to_dict(parameters_as_string)

    df, df_interpolated, time, t_interpolated = pandas_load_file(filename)
    y_truth = df_interpolated['I'].to_numpy(int)
    Tmax = int(time.max())+1 # max number of days
    N0 = cfg.N0
    # y0 =  S, N0,                E1,E2,E3,E4,  I1,I2,I3,I4,  R, R0
    y0 = N0-cfg.Ninit,N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0, cfg.Ninit

    # reload(extra_funcs)
    fit_object = CustomChi2(t_interpolated, y_truth, y0, Tmax, dt=dt, ts=ts, mu0=cfg.mu, y_min=10)

    minuit = Minuit(fit_object, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta, tau=0)

    minuit.migrad()
    fit_object.set_chi2(minuit)

    i_fit = 0
    # if (not minuit.get_fmin().is_valid) :
    if fit_object.chi2 / fit_object.N > 100:

        continue_fit = True
        while continue_fit:
            i_fit += 1
            N_refits += 1

            param_grid = {'Mrate1': uniform(0.1, 10), 
                        'Mrate2': uniform(0.1, 10), 
                        'beta': uniform(0.1, 20), 
                        'tau': uniform(-10, 10),
                        }
            param_list = list(ParameterSampler(param_grid, n_iter=1))[0]
            minuit = Minuit(fit_object, pedantic=False, print_level=0, **param_list)
            minuit.migrad()
            fit_object.set_minuit(minuit)

            if fit_object.chi2 / fit_object.N <= 10 or i_fit>FIT_MAX:
                continue_fit = False
            
    if i_fit <= FIT_MAX:
        fit_object.set_minuit(minuit)
        return filename, fit_object, N_refits

    else:
        print(f"\n\n{filename} was discarded\n", flush=True)
        return filename, None, N_refits



N_peak_fits = 20
def fit_single_file_Imax(filename, ts=0.1, dt=0.01, for_animation=False):

    # ts = 0.1 # frequency of "observations". Now 1 pr. day
    # dt = 0.01 # stepsize in integration

    cfg = filename_to_dotdict(filename)
    parameters_as_string = dict_to_str(cfg)

    df = pandas_load_file(filename, return_only_df=True)
    # y_truth = df_interpolated['I'].to_numpy(int)
    Tmax = int(df['Time'].max())+1 # max number of days
    N0 = cfg.N0
    y0 = N0-cfg.Ninit, N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0, cfg.Ninit

    I = df['I'].to_numpy(int)
    Time = df['Time'].to_numpy()
    
    I_cut_min = 0.2 / 1000 * cfg.N0 # percent
    # I_cut_min = 0.05 / 100 * I.max() # percent
    iloc_min = np.argmax(I > I_cut_min)
    iloc_max = np.argmax(I) 

    delta_iloc = (iloc_max - iloc_min) // N_peak_fits
    indices = np.linspace(iloc_min, iloc_max, N_peak_fits+1).astype(int) - delta_iloc // 2
    df_prefit = df.iloc[indices]

    # Time from beginning to peak
    I_time_duration = Time[iloc_max] - Time[iloc_min]

    t_interpolated = df_prefit['Time'].to_numpy()
    y_truth = df_prefit['I'].to_numpy(int)

    Tmax = Time[iloc_max]*1.1


    # reload(extra_funcs)
    I_max_truth = np.max(I)
    times_maxs = t_interpolated[1:] - Time[iloc_min]
    times_maxs_normalized = times_maxs / I_time_duration
    fit_objects_Imax = []
    for imax in range(N_peak_fits):
        fit_object = CustomChi2(t_interpolated[:imax+2], y_truth[:imax+2], y0, Tmax, dt=dt, ts=ts, mu0=cfg.mu, y_min=10)
        minuit = Minuit(fit_object, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta, tau=0)
        minuit.migrad()
        fit_object.set_minuit(minuit)
        fit_objects_Imax.append(fit_object)

    if not for_animation:
        return filename, times_maxs_normalized, I_max_truth, fit_objects_Imax
    else:
        return t_interpolated, y_truth, fit_objects_Imax


    # reload(extra_funcs)
    # N_peak_fits = N_peak_fits
    # I_max_truth = np.max(I)
    # I_maxs = np.zeros(N_peak_fits)
    # betas = np.zeros(N_peak_fits)
    # betas_std = np.zeros(N_peak_fits)
        # I_max = fit_object.compute_I_max()
        # I_maxs[imax] = I_max
        # betas[imax] = fit_object.fit_values['beta']
        # betas_std[imax] = fit_object.fit_errors['beta']
    # return filename, times_maxs_normalized, I_maxs, I_max_truth, betas, betas_std



def animate_Imax_fit_filename(filename):

    t_interpolated, y_truth, fit_objects_Imax = fit_single_file_Imax(filename, ts=0.1, dt=0.01, for_animation=True)

    df = pandas_load_file(filename, return_only_df=True)
    fignames = []
    for imax in tqdm(range(N_peak_fits)):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['I'], name=f'Simulation', line=dict(color='black', width=2)))

        df_fit = fit_objects_Imax[imax].calc_df_fit(ts=0.01)
        fig.add_trace(go.Scatter(x=df_fit['Time'], y=df_fit['I_sum'], name=f'Fit'))

        fig.add_trace(go.Scatter(x=t_interpolated[:imax+2], y=y_truth[:imax+2], name=f'Data', mode='markers', marker=dict(size=10, color=1)))

        fig.update_xaxes(range=[df['Time'].min(), df['Time'].max()])
        fig.update_yaxes(range=[0, df['I'].max()*1.1])

        # Edit the layout
        fig.update_layout(title=f'Simulation comparison, {imax=}',
                        xaxis_title='Time',
                        yaxis_title='Count',
                        height=600, width=800,
                        # showlegend=False,
                        )

        fig.update_yaxes(rangemode="tozero")
        # fig.show()
        figname = f"Figures/.tmp_{imax}.png"
        fig.write_image(figname)
        fignames.append(figname)

    import imageio # conda install imageio
    gifname = 'Figures/Imax_animation_N' + filename.strip('Data/NetworkSimulation_').strip('.csv') + '.gif'
    with imageio.get_writer(gifname, mode='I', duration=0.5) as writer:
        for figname in fignames:
            image = imageio.imread(figname)
            writer.append_data(image)
            Path(figname).unlink() # delete file
    

    return None


#%%



import multiprocessing as mp
from tqdm import tqdm


def calc_fit_results(filenames, num_cores_max=20):

    N_files = len(filenames)

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max

    print(f"Fitting {N_files} network-based simulations with {num_cores} cores, please wait.", flush=True)
    with mp.Pool(num_cores) as p:
        results = list(tqdm(p.imap_unordered(fit_single_file, filenames), total=N_files))

    # modify results from multiprocessing

    N_refits_total = 0
    discarded_files = []
    all_fit_objects = {}
    for filename, fit_object, N_refits in results:
        
        if fit_object is None:
            discarded_files.append(filename)
        else:
            all_fit_objects[filename] = fit_object
        
        N_refits_total += N_refits

    return all_fit_objects, discarded_files, N_refits_total



def calc_fit_Imax_results(filenames, num_cores_max=30):

    N_files = len(filenames)

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max

    print(f"Fitting I_max for {N_files} network-based simulations with {num_cores} cores, please wait.", flush=True)
    with mp.Pool(num_cores) as p:
        results = list(tqdm(p.imap_unordered(fit_single_file_Imax, filenames), total=N_files))

    # postprocess results from multiprocessing:
    I_maxs_truth = {}
    fit_objects = {}
    bins = np.linspace(0, 1, N_peak_fits+1)
    bin_centers = (bins[1:] + bins[:-1])/2

    # filename, times_maxs_normalized, I_max_truth, fit_objects_Imax = results[0]
    for filename, times_maxs_normalized, I_max_truth, fit_objects_Imax in results:
        # if one fit in each bin:
        if np.all(1 == np.histogram(times_maxs_normalized, bins)[0]):
            I_maxs_truth[filename] = I_max_truth
            fit_objects[filename] = fit_objects_Imax
            # I_maxs_normed[filename] = I_maxs / I_max_truth
            # betas[filename] = beta
            # betas_std[filename] = beta_std
        
    return I_maxs_truth, fit_objects, bin_centers



def filename_to_ID(filename):
    return int(filename.split('ID_')[1].strip('.csv'))

def filename_to_par_string(filename):
    return dict_to_str(filename_to_dotdict(filename))


def get_fit_results(filenames, force_rerun=False, num_cores_max=20):

    output_filename = 'fit_results.joblib'

    if Path(output_filename).exists() and not force_rerun:
        print("Loading fit results")
        return joblib.load(output_filename)

    else:
        fit_results = calc_fit_results(filenames, num_cores_max=num_cores_max)
        print(f"Finished fitting, saving results to {output_filename}", flush=True)
        joblib.dump(fit_results, output_filename)
        return fit_results


def get_fit_Imax_results(filenames, force_rerun=False, num_cores_max=20):

    output_filename = 'fit_Imax_results.joblib'

    if Path(output_filename).exists() and not force_rerun:
        print("Loading Imax fit results")
        return joblib.load(output_filename)

    else:
        fit_results = calc_fit_Imax_results(filenames, num_cores_max=num_cores_max)
        print(f"Finished Imax fitting, saving results to {output_filename}", flush=True)
        joblib.dump(fit_results, output_filename)
        return fit_results


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
    df.loc['sdom'] = df.std() / np.sqrt(len(df)-2)
    # I_max_normed_by_pars[par_string] = df
    return df


def extract_normalized_Imaxs(d_fit_objects_all_IDs, I_maxs_truth, filenames_to_use, bin_centers_Imax):
    I_maxs_normed_res = {}
    for filename, fit_objects in d_fit_objects_all_IDs.items():
        I_maxs = np.zeros(N_peak_fits)
        for i_fit_object, fit_object in enumerate(fit_objects):
            I_maxs[i_fit_object]  = fit_object.compute_I_max()
        I_maxs_normed_res[filename] = I_maxs / I_maxs_truth[filename]
    df_I_maxs_normed = Imax_fits_to_df(I_maxs_normed_res, filenames_to_use, bin_centers_Imax)
    return df_I_maxs_normed


def extract_relative_Imaxs(d_fit_objects_all_IDs, I_maxs_truth, filenames_to_use, bin_centers_Imax):
    I_maxs_relative_res = {}
    for filename, fit_objects in d_fit_objects_all_IDs.items():
        I_maxs = np.zeros(N_peak_fits)
        I_current_pos = np.zeros(N_peak_fits)
        for i_fit_object, fit_object in enumerate(fit_objects):
            I_maxs[i_fit_object] = fit_object.compute_I_max()
            I_current_pos[i_fit_object] = fit_object.y_truth[-1]
        I_maxs_relative_res[filename] = (I_maxs_truth[filename]-I_maxs) / (I_maxs_truth[filename]-I_current_pos)
    df_I_maxs_relative = Imax_fits_to_df(I_maxs_relative_res, filenames_to_use, bin_centers_Imax)
    return df_I_maxs_relative

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




def plot_SIR_model_comparison(force_overwrite=False):

    pdf_name = f"Figures/SIR_comparison.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_overwrite:
        print(f"{pdf_name} already exists")
        return None
    
    else:

        from matplotlib.backends.backend_pdf import PdfPages

        base_dir = Path('Data') / 'NetworkSimulation'
        all_sim_pars = sorted([str(x.name) for x in base_dir.glob('*') if str(x.name) != '.DS_Store'])

        with PdfPages(pdf_name) as pdf:

            # sim_par = all_sim_pars[0]
            for sim_par in tqdm(all_sim_pars):

                ID_files = list((base_dir/sim_par).rglob('*.csv'))

                cfg = string_to_dict(sim_par)

                # dfs = []
                fig, ax = plt.subplots(figsize=(16, 10))
                # filename_ID = ID_files[0]
                Tmax = 0
                lw = 0.5
                for i, filename_ID in enumerate(ID_files):
                    df = pandas_load_file(filename_ID, return_only_df=True)
                    label = 'Simulations' if i == 0 else None
                    # lw = 1 if i == 0 else 0.1
                    ax.plot(df['Time'].values, df['I'].values, lw=lw, c='k', label=label)
                    if df['Time'].max() > Tmax:
                        Tmax = df['Time'].max() 

                Tmax = max(Tmax, 50)

                y0 = cfg.N0-cfg.Ninit, cfg.N0,   cfg.Ninit,0,0,0,      0,0,0,0,   0, cfg.Ninit
                dt = 0.01
                ts = 0.1

                ODE_result_SIR = ODE_integrate(y0, Tmax, dt, ts, mu0=cfg.mu, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta)
                # print(y0, Tmax, dt, ts, cfg)
                I_SIR = ODE_result_SIR[:, 2]
                time = ODE_result_SIR[:, 4]
                cols = ['S', 'E_sum', 'I_sum', 'R', 'Time', 'R0']
                df_fit = pd.DataFrame(ODE_result_SIR, columns=cols).convert_dtypes()

                ax.plot(time, I_SIR, lw=10*lw, color='red', label='SIR')
                leg = ax.legend(loc='upper right')
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
                
                N0_str = human_format(cfg.N0)
                title = f"N={N0_str}, β={cfg.beta:.4f}, γ={cfg.gamma:.1f}, σ={cfg.sigma:.1f},  α={cfg.alpha:.1f}, ψ={cfg.psi:.1f}, #{len(ID_files)}"

                ax.set(title=title, xlabel='Time', ylabel='I')
                
                pdf.savefig(fig)





#%%


def get_filenames_different_than_default(find_par):

    base_dir = Path('Data') / 'NetworkSimulation'
    all_sim_pars = sorted([str(x.name) for x in base_dir.glob('*') if str(x.name) != '.DS_Store'])

    all_sim_pars_as_dict = {s: string_to_dict(s) for s in all_sim_pars}
    df_sim_pars = pd.DataFrame.from_dict(all_sim_pars_as_dict, orient='index')

    default_pars = dict(
                        N0 = 50_000,
                        mu = 20.0,  # Average number connections
                        alpha = 0.0, # Spatial parameter
                        psi = 0.0, # cluster effect
                        beta = 0.01, # Mean rate
                        sigma = 0.0, # Spread in rate
                        Mrate1 = 1.0, # E->I
                        Mrate2 = 1.0, # I->R
                        gamma = 0.0, # Parameter for skewed connection shape
                        nts = 0.1, 
                        Nstates = 9,
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
