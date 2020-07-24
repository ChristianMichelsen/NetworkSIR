import numpy as np
import pandas as pd
from scipy import interpolate
from numba import njit

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
    # df_interpolated['time'] = t_interpolated
    df_interpolated.insert(loc=0, column='time', value=t_interpolated)

    return df_interpolated


def interpolate_df(df, t_interpolated=None):

    # make first value at time 0
    t0 = df['time'].min()
    df['time'] -= t0
    time = df['time']

    if not t_interpolated:
        t_interpolated = np.arange(int(time.max())+1)
    cols_to_interpolate = ['E', 'I', 'R']
    df_interpolated = interpolate_dataframe(df, time, t_interpolated, cols_to_interpolate)
    return df_interpolated



@njit
def _integrate(y0, Tmax, dt, ts, mu, lambda_E, lambda_I, beta):

    S, N_tot, E1, E2, E3, E4, I1, I2, I3, I4, R = y0
    mu /= 2 # to correct for mu scaling

    click = 0
    SIR_result = np.zeros((int(Tmax/ts)+1, 5))
    times = np.linspace(0, Tmax, int(Tmax/dt)+1)

    for time in times:

        dS  = -beta*mu*2/N_tot*(I1+I2+I3+I4)*S

        dE1 = beta*mu*2/N_tot*(I1+I2+I3+I4)*S - lambda_E*E1
        dE2 = lambda_E*E1 - lambda_E*E2
        dE3 = lambda_E*E2 - lambda_E*E3
        dE4 = lambda_E*E3 - lambda_E*E4

        dI1 = lambda_E*E4 - lambda_I*I1
        dI2 = lambda_I*I1 - lambda_I*I2
        dI3 = lambda_I*I2 - lambda_I*I3
        dI4 = lambda_I*I3 - lambda_I*I4

        dR  = lambda_I*I4

        S  += dt*dS
        E1 += dt*dE1
        E2 += dt*dE2
        E3 += dt*dE3
        E4 += dt*dE4

        I1 += dt*dI1
        I2 += dt*dI2
        I3 += dt*dI3
        I4 += dt*dI4

        R += dt*dR

        if time >= ts*click: # - t0:
            SIR_result[click] = [
                            time, # RT
                            S,
                            E1+E2+E3+E4,
                            I1+I2+I3+I4,
                            R,
                            ]
            click += 1
    return SIR_result


def integrate(cfg, Tmax, dt=0.01, ts=0.1):
    y0 = cfg.N_tot-cfg.N_init, cfg.N_tot,   cfg.N_init,0,0,0,      0,0,0,0,   0
    SIR_result = _integrate(y0, Tmax, dt, ts, mu=cfg.mu, lambda_E=cfg.lambda_E, lambda_I=cfg.lambda_I, beta=cfg.beta)
    cols = ['time', 'S', 'E', 'I', 'R']
    df_fit = pd.DataFrame(SIR_result, columns=cols).convert_dtypes()
    return df_fit