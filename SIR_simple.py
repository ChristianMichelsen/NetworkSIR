from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from pathlib import Path
from numba import njit
from scipy import interpolate
import configuration


savefig = False
cfg = configuration.load()

# from scipy.signal import savgol_filter
def interpolate_array(y, time, t_interpolated, force_positive=True):
    f = interpolate.interp1d(time, y, kind='cubic', fill_value=0, bounds_error=False)
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



#%%

testN = 0

N0 = cfg.N0

mu = cfg.mu  # Average number connections
alpha = cfg.alpha # Spatial parameter
beta = cfg.beta # Mean rate
sigma = cfg.sigma # Spread in rate
Ninit = cfg.Ninit # Initial Infected


filename = Path('Data') / f'Run_{int(mu)}_N{int(N0/1000)}_In{Ninit}_alpha{int(alpha)}_beta{int(beta)}_sigmaF{int(sigma*100)}_test{testN}.csv'
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

#%%

@njit
def ODE_integrate(y0, Tmax, dt, ts, Mrate1, Mrate2, mu0, beta): 

    S, E1, E2, E3, E4, I1, I2, I3, I4, R, R0 = y0

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


ts = cfg.ts # frequency of "observations". Now 1 pr. day
Tmax = int(time.max())+1 # max number of days
dt = cfg.dt # stepsize in integration
mu0 = cfg.mu
beta = cfg.beta
Mrate1 = cfg.Mrate1
Mrate2 = cfg.Mrate1
S0 = N0
Ninit = cfg.Ninit


# y0 =  S,     E1,E2,E3,E4, I1,I2,I3,I4,  R, R0
y0 = S0-Ninit, Ninit,0,0,0,   0,0,0,0,   0, Ninit

res_sir = ODE_integrate(y0, Tmax, dt, ts, Mrate1, Mrate2, mu0, beta)

cols = ['S', 'E_sum', 'I_sum', 'R', 'Time', 'R0']
df_sir = pd.DataFrame(res_sir, columns=cols).convert_dtypes()
df_sir['N'] = df_sir[['S', 'E_sum', 'I_sum', 'R']].sum(axis=1)
df_sir

#%%

# interpolate_array(y, time, t_interpolated, force_positive=True):

y_true = df_interpolated['I']

def calc_yhat_interpolated(Mrate1, Mrate2, mu0, beta, tau):
    res_sir = ODE_integrate(y0, Tmax, dt, ts, Mrate1, Mrate2, mu0, beta)
    I_SIR = res_sir[:, 2]
    time = res_sir[:, 4]
    y_hat = interpolate_array(I_SIR, time, t_interpolated+tau)
    return y_hat

def calc_chi2(Mrate1, Mrate2, mu0, beta, tau):
    y_hat = calc_yhat_interpolated(Mrate1, Mrate2, mu0, beta, tau)
    mask = (y_true > 10)
    chi2 = np.sum((y_hat[mask]-y_true[mask])**2 / y_true[mask])
    return chi2

def calc_df_fit(Mrate1, Mrate2, mu0, beta, tau, ts=0.01):
    res_sir = ODE_integrate(y0, Tmax, dt, ts, Mrate1, Mrate2, mu0, beta)
    cols = ['S', 'E_sum', 'I_sum', 'R', 'Time', 'R0']
    df_fit = pd.DataFrame(res_sir, columns=cols).convert_dtypes()
    df_fit['Time'] -= tau
    df_fit['N'] = df_fit[['S', 'E_sum', 'I_sum', 'R']].sum(axis=1)
    return df_fit

#%%

from ExternalFunctions import Chi2Regression
from iminuit import Minuit

# chi2_exp = Chi2Regression(quadratic, N0s[mask], times[mask], np.sqrt(times[mask]))
minuit = Minuit(calc_chi2, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, mu0=cfg.mu, beta=cfg.beta, tau=0)
minuit.migrad()
if (not minuit.get_fmin().is_valid) :
    print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

print(minuit.parameters)
print(minuit.np_values())
print(minuit.np_errors())

y_hat = calc_yhat_interpolated(*minuit.args)
df_y = pd.DataFrame({'y_true': y_true.values, 'y_hat': y_hat})
chi2 = calc_chi2(*minuit.args)

df_fit = calc_df_fit(*minuit.args, ts=0.01)


#%%

cols_to_interpolate = ['E_sum', 'I_sum', 'R']
df_sir_interpolated = interpolate_dataframe(df_sir, df_sir['Time'], t_interpolated, cols_to_interpolate)
df_sir_interpolated


# %%

fig = go.Figure()

for s in ['E', 'I', 'R']:
    fig.add_trace(go.Scatter(x=df['Time'], y=df[s], name=f'{s} raw network'))
    # fig.add_trace(go.Scatter(x=df_interpolated['Time'], y=df_interpolated[s], name=f'{s} interpolated network'))
    ss = f'{s}_sum' if s != 'R' else s
    # fig.add_trace(go.Scatter(x=df_sir['Time'], y=df_sir[ss], name=f'{s} SIR simulation'))
    # fig.add_trace(go.Scatter(x=df_sir_interpolated['Time'], y=df_sir_interpolated[ss], name=f'{s} SIR interpolated simulation'))
    fig.add_trace(go.Scatter(x=df_fit['Time'], y=df_fit[ss], name=f'{s} FIT'))

k_scale = 2/3
k_scale = 1


# Edit the layout
fig.update_layout(title=f'Simulation comparison, {testN=}',
                   xaxis_title='Time',
                   yaxis_title='Count',
                   height=600*k_scale, width=800*k_scale,
                   )

fig.update_yaxes(rangemode="tozero")



fig.show()
if savefig:
    fig.write_html(f"Figures/{filename.stem}.html")




# %%
