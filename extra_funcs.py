import numpy as np
from numba import njit
from scipy import interpolate
import pandas as pd
from pathlib import Path
from scipy.stats import uniform as sp_uniform



def get_filenames():
    filenames = Path('Data').glob(f'*.csv')
    return sorted(filenames)


def pandas_load_file(filename):
    df_raw = pd.read_csv(filename).convert_dtypes()

    for state in ['E', 'I']:
        df_raw[state] = sum([df_raw[col] for col in df_raw.columns if state in col and len(col) == 2])

    # only keep relevant columns
    df = df_raw[['Time', 'E', 'I', 'R', 'NR0Inf']].copy()

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

    S, S0, E1, E2, E3, E4, I1, I2, I3, I4, R, R0 = y0

    click = 0
    res_sir = np.zeros((int(Tmax/ts)+1, 6))
    Times = np.linspace(0, Tmax, int(Tmax/dt)+1)

    for Time in Times:

        dS  = -beta*mu0/S0*(I1+I2+I3+I4)*S
        dE1 = beta*mu0/S0*(I1+I2+I3+I4)*S - Mrate1*E1
        dE2 = Mrate1*E1 - Mrate1*E2
        dE3 = Mrate1*E2 - Mrate1*E3
        dE4 = Mrate1*E3 - Mrate1*E4

        dI1 = Mrate1*E4 - Mrate2*I1
        dI2 = Mrate2*I1 - Mrate2*I2
        dI3 = Mrate2*I2 - Mrate2*I3
        dI4 = Mrate2*I3 - Mrate2*I4

        R0  += dt*beta*mu0/S0*(I1+I2+I3+I4)*S

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
            res_sir[click, :] = [
                            S, 
                            E1+E2+E3+E4, 
                            I1+I2+I3+I4,
                            R,
                            Time, # RT
                            R0,
                            ]
            click += 1
    return res_sir





from iminuit.util import make_func_code
from iminuit import describe

class CustomChi2:  # override the class with a better one
    
    def __init__(self, time, t_interpolated, y_true, y0, Tmax, dt, ts, mu0, y_min=0):
        
        # self.f = f  # model predicts y for given x
        self.time = time
        self.t_interpolated = t_interpolated
        self.y_true = y_true.values
        self.y0 = y0
        self.Tmax = Tmax
        self.dt = dt
        self.ts = ts
        self.mu0 = mu0
        self.sy = np.sqrt(y_true.values) #if sy is None else sy
        self.y_min = y_min
        self.N = sum(self.y_true > self.y_min)
        # self.func_code = make_func_code(describe(self._calc_yhat_interpolated))

    def __call__(self, Mrate1, Mrate2, beta, tau):  # par are a variable number of model parameters
        # compute the function value
        y_hat = self._calc_yhat_interpolated(Mrate1, Mrate2, beta, tau)
        mask = (self.y_true > self.y_min)
        # compute the chi2-value
        chi2 = np.sum((self.y_true[mask] - y_hat[mask])**2/self.sy[mask]**2)
        return chi2

    def _calc_res_sir(self, Mrate1, Mrate2, beta, ts=None):
        ts = self.ts if ts is None else ts
        return ODE_integrate(self.y0, self.Tmax, self.dt, ts, self.mu0, Mrate1, Mrate2, beta)

    def _calc_yhat_interpolated(self, Mrate1, Mrate2, beta, tau):
        res_sir = self._calc_res_sir(Mrate1, Mrate2, beta)
        I_SIR = res_sir[:, 2]
        time = res_sir[:, 4]
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
        res_sir = self._calc_res_sir(Mrate1, Mrate2, beta, ts=ts)
        cols = ['S', 'E_sum', 'I_sum', 'R', 'Time', 'R0']
        df_fit = pd.DataFrame(res_sir, columns=cols).convert_dtypes()
        df_fit['Time'] -= tau
        df_fit['N'] = df_fit[['S', 'E_sum', 'I_sum', 'R']].sum(axis=1)
        if df_fit.iloc[-1]['R'] == 0:
            df_fit = df_fit.iloc[:-1]
        return df_fit
    



def dict_to_str(d):
    string = ''
    for key, val in d.items():
        string += f"{key}_{val}_"
    return string[:-1]


import NewSpeedImprove_extra_funcs

def filename_to_dotdict(filename):
    return NewSpeedImprove_extra_funcs.filename_to_dotdict(filename)

def string_to_dict(string):
    return NewSpeedImprove_extra_funcs.filename_to_dotdict(string, normal_string=True)

def uniform(a, b):
    loc = a
    scale = b-a
    return sp_uniform(loc, scale)
