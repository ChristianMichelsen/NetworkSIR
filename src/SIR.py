import numpy as np
import pandas as pd
from scipy import interpolate
from numba import njit
from functools import lru_cache
import matplotlib.pyplot as plt

try:
    from src import file_loaders
except ImportError:
    import file_loaders


# from scipy.signal import savgol_filter
def interpolate_array(y, time, t_interpolated, force_positive=True):
    f = interpolate.interp1d(time, y, kind="cubic", fill_value=0, bounds_error=False)
    with np.errstate(invalid="ignore"):
        y_hat = f(t_interpolated)
        if force_positive:
            y_hat[y_hat < 0] = 0
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
    df_interpolated.insert(loc=0, column="time", value=t_interpolated)

    return df_interpolated


def interpolate_df(df, t_interpolated=None, cols_to_interpolate=None):

    """Interpolates a the columns of a dataframe (df) based on t_interpolated.
    By default it downsamples the dataframe to one observation daily (unless t_interpolated is specifically set) for the columns E, I, and R (unless specifically set)."""

    # make first value at time 0
    t0 = df["time"].min()
    df["time"] -= t0
    time = df["time"]

    if not t_interpolated:
        t_interpolated = np.arange(int(time.max()) + 1)

    if not cols_to_interpolate:
        cols_to_interpolate = ["E", "I", "R"]

    df_interpolated = interpolate_dataframe(df, time, t_interpolated, cols_to_interpolate)

    return df_interpolated


@njit
def _numba_SIR_integrate(y0, T_max, dt, ts, mu, lambda_E, lambda_I, beta):

    S, N_tot, E1, E2, E3, E4, I1, I2, I3, I4, R = y0
    mu /= 2  # to correct for mu scaling

    click = 0
    SIR_result = np.zeros((int(T_max / ts) + 1, 5))
    times = np.linspace(0, T_max, int(T_max / dt) + 1)

    for time in times:

        dS = -beta * mu * 2 / N_tot * (I1 + I2 + I3 + I4) * S

        dE1 = beta * mu * 2 / N_tot * (I1 + I2 + I3 + I4) * S - lambda_E * E1
        dE2 = lambda_E * E1 - lambda_E * E2
        dE3 = lambda_E * E2 - lambda_E * E3
        dE4 = lambda_E * E3 - lambda_E * E4

        dI1 = lambda_E * E4 - lambda_I * I1
        dI2 = lambda_I * I1 - lambda_I * I2
        dI3 = lambda_I * I2 - lambda_I * I3
        dI4 = lambda_I * I3 - lambda_I * I4

        dR = lambda_I * I4

        S += dt * dS
        E1 += dt * dE1
        E2 += dt * dE2
        E3 += dt * dE3
        E4 += dt * dE4

        I1 += dt * dI1
        I2 += dt * dI2
        I3 += dt * dI3
        I4 += dt * dI4

        R += dt * dR

        if time >= ts * click:  # - t0:
            SIR_result[click] = [
                time,  # RT
                S,
                E1 + E2 + E3 + E4,
                I1 + I2 + I3 + I4,
                R,
            ]
            click += 1
    return SIR_result


@lru_cache(maxsize=None)
def numba_SIR_integrate(y0, T_max, dt, ts, mu, lambda_E, lambda_I, beta):
    """ Wrapper function for '_numba_SIR_integrate' which is cached """
    SIR_result = _numba_SIR_integrate(y0, T_max, dt, ts, mu, lambda_E, lambda_I, beta)
    if SIR_result_to_R(SIR_result)[-1] == 0:
        SIR_result = SIR_result[:-1]
    return SIR_result


def cfg_to_y0(cfg):
    """ S, N_tot, E1, E2, E3, E4, I1, I2, I3, I4, R = y0 """
    y0 = (
        cfg.N_tot - cfg.N_init,
        cfg.N_tot,
        cfg.N_init / 8,
        cfg.N_init / 8,
        cfg.N_init / 8,
        cfg.N_init / 8,
        cfg.N_init / 8,
        cfg.N_init / 8,
        cfg.N_init / 8,
        cfg.N_init / 8,
        0,
    )
    return y0


def SIR_result_to_dataframe(SIR_result):
    cols = ["time", "S", "E", "I", "R"]
    return pd.DataFrame(SIR_result, columns=cols).convert_dtypes()


def integrate(cfg, T_max, dt=0.01, ts=0.1, return_dataframe=True):
    y0 = cfg_to_y0(cfg)
    SIR_result = numba_SIR_integrate(
        y0,
        T_max,
        dt,
        ts,
        mu=cfg.mu,
        lambda_E=cfg.lambda_E,
        lambda_I=cfg.lambda_I,
        beta=cfg.beta,
    )
    if return_dataframe:
        return SIR_result_to_dataframe(SIR_result)
    else:
        return SIR_result


def SIR_result_to_time(SIR_result):
    return SIR_result[:, 0]


def SIR_result_to_I(SIR_result):
    return SIR_result[:, -2]


def SIR_result_to_R(SIR_result):
    return SIR_result[:, -1]


def get_I_max(I):
    return np.max(I)


def get_R_inf(R):
    return R[-1]


def calc_deterministic_results(cfg, T_max, dt=0.01, ts=0.1):
    SIR_result = integrate(cfg, T_max, dt, ts, return_dataframe=False)
    I_max = get_I_max(SIR_result_to_I(SIR_result))
    R_inf = get_R_inf(SIR_result_to_R(SIR_result))
    return I_max, R_inf


#%%


#%%


from functools import lru_cache
from iminuit.util import make_func_code
from iminuit import describe


class FitSIR:  # override the class with a better one
    def __init__(self, t, y, sy, priors, cfg, dt, ts):

        """ priors: dict(multiplier=0.1, lambda_E={'mean': mean, 'std': std}, lambda_I={'mean': mean, 'std': std} ...) """

        self.t = t
        self.y = y
        self.sy = sy
        self.priors = priors
        self.cfg = cfg
        self.y0 = cfg_to_y0(cfg)
        self.mu = cfg.mu
        self.T_max = np.max(t)
        self.dt = dt
        self.ts = ts
        self.N_refits = 0
        self.minuit_is_set = False
        self.N = len(t)
        self.mask = self.y > 0  # max to exclude 0 divisions

    def __call__(self, lambda_E, lambda_I, beta, tau):
        # compute the function value
        y_hat = self._compute_yhat(lambda_E, lambda_I, beta, tau)
        # compute the chi2-value
        chi2 = np.sum((self.y[self.mask] - y_hat[self.mask]) ** 2 / self.sy[self.mask] ** 2)

        if np.isnan(chi2):
            return 1e10
        return chi2

    def __repr__(self):
        s = (
            f"FitSIR(\n\tself.t=[{self.t[0]}, ..., {self.t[-1]}], \n\tself.y=[{self.y[0]:.1f}, ..., {self.y[-1]:.1f}], \n\t{self.T_max=}, \n\t{self.dt=}, \n\t{self.ts=}, \n\t{self.N=})".replace(
                "=", " = "
            )
            .replace("array(", "")
            .replace("])", "]")
        )
        if self.minuit_is_set:
            s += "\n\n"
            s += f"chi2 = {self.chi2:.1f}, valid_fit = {self.valid_fit} \n\n"
            s += str(self.get_fit_parameters())
        return s

    def _compute_result_SIR(self, lambda_E, lambda_I, beta, ts=None, T_max=None):
        if not ts:
            ts = self.ts
        if not T_max:
            T_max = self.T_max
        return numba_SIR_integrate(self.y0, T_max, self.dt, ts, self.mu, lambda_E, lambda_I, beta)

    def _compute_yhat(self, lambda_E, lambda_I, beta, tau):
        SIR_result = self._compute_result_SIR(lambda_E, lambda_I, beta)
        I = SIR_result_to_I(SIR_result)
        time = SIR_result_to_time(SIR_result)
        y_hat = interpolate_array(I, time, self.t + tau)
        return y_hat

    def set_chi2(self, minuit):
        self.chi2 = self.__call__(**minuit.values)
        self.reduced_chi2 = self.chi2 / self.N
        return self.chi2

    def _valid_fit(self, minuit, max_reduced_chi2=100, verbose=False):
        if max_reduced_chi2:
            good_chi2 = 0.001 <= self.reduced_chi2 <= max_reduced_chi2
        else:
            good_chi2 = 0.001 <= self.reduced_chi2

        has_correlations = self.has_correlations
        valid_hesse = not minuit.get_fmin().hesse_failed
        good_errors = np.all(minuit.np_errors()[:-1] / np.abs(minuit.np_values()[:-1]) < 2)
        valid_fit = good_chi2 and has_correlations and valid_hesse and good_errors

        if verbose:
            print(f"{good_chi2=}, {has_correlations=}, {valid_hesse=}, {good_errors=}")

        return valid_fit

    def set_minuit(self, minuit, max_reduced_chi2=100):
        self.minuit_is_set = True
        # self.minuit = minuit
        # self.m = minuit

        self.fit_values = dict(minuit.values)
        self.fit_errors = dict(minuit.errors)
        self.fit_fixed = dict(minuit.fixed)
        self.fit_names = list(minuit.parameters)
        for parameter in self.fit_names:
            if self.fit_fixed[parameter]:
                self.fit_values[parameter] = "Fixed"
                self.fit_errors[parameter] = "Fixed"

        self.chi2 = minuit.fval
        self.reduced_chi2 = self.chi2 / self.N
        # self.is_valid = minuit.get_fmin().is_valid

        try:
            self.has_correlations = True
            self.correlations = minuit.np_matrix(correlation=True)
            self.covariances = minuit.np_matrix(correlation=False)

        except RuntimeError:
            self.has_correlations = False

        self.max_reduced_chi2 = max_reduced_chi2
        with np.errstate(divide="ignore", invalid="ignore"):
            self.valid_fit = self._valid_fit(minuit, self.max_reduced_chi2)
        return self

    def get_fit_parameter(self, parameter):
        return self.fit_values[parameter], self.fit_errors[parameter]

    def get_fit_parameters(self):

        if not self.minuit_is_set:
            raise AssertionError("Minuit has to be set ('.set_minuit(minuit)')")

        fit_parameters = {}
        for parameter in self.fit_names:
            fit_parameters[parameter] = self.get_fit_parameter(parameter)
        df_fit_parameters = pd.DataFrame(fit_parameters, index=["mean", "std"])
        return df_fit_parameters

    def get_correlations(self):
        return pd.DataFrame(self.correlations, index=self.parameters, columns=self.parameters)

    def _fix_fixed_parameters(self, fit_values, parameter):
        if fit_values[parameter] == "Fixed":
            return self.cfg[parameter]
        else:
            return fit_values[parameter]

    def _get_fit_values(self, fit_values):

        if fit_values is None:
            fit_values = self.fit_values

        if isinstance(fit_values, (list, np.ndarray)):
            # if some values are fixed
            if len(fit_values) != 4:
                tmp = []
                i = 0
                for parameter in self.fit_names:
                    if self.fit_fixed[parameter]:
                        tmp.append(self.cfg[parameter])
                    else:
                        # assume the fit_values arre ordered
                        tmp.append(fit_values[i])
                        i += 1
                fit_values = tmp
            names = ["lambda_E", "lambda_I", "beta", "tau"]
            fit_values = dict(zip(names, fit_values))

        if not isinstance(fit_values, dict):
            raise AssertionError("fit_values has to be a dictionary")

        lambda_E = self._fix_fixed_parameters(fit_values, "lambda_E")
        lambda_I = self._fix_fixed_parameters(fit_values, "lambda_I")
        beta = self._fix_fixed_parameters(fit_values, "beta")
        tau = self._fix_fixed_parameters(fit_values, "tau")
        return lambda_E, lambda_I, beta, tau

    def calc_df_fit(self, ts=0.1, fit_values=None, T_max=None):
        lambda_E, lambda_I, beta, tau = self._get_fit_values(fit_values)
        SIR_result = self._compute_result_SIR(lambda_E, lambda_I, beta, ts=ts, T_max=T_max)
        df_fit = SIR_result_to_dataframe(SIR_result)
        df_fit["time"] -= tau
        df_fit["N"] = df_fit[["S", "E", "I", "R"]].sum(axis=1)
        return df_fit

    def compute_I_max_R_inf(self, ts=0.1, fit_values=None, T_max=None):
        lambda_E, lambda_I, beta, tau = self._get_fit_values(fit_values)
        SIR_result = self._compute_result_SIR(lambda_E, lambda_I, beta, ts=ts, T_max=T_max)
        I_max = get_I_max(SIR_result_to_I(SIR_result))
        R_inf = get_R_inf(SIR_result_to_R(SIR_result))
        return I_max, R_inf

    def _remove_fixed(self, x):
        return np.array([xi for xi in x if not xi == "Fixed"])

    def _random_sample_fit_parameters(self, N_samples):
        mean = self._remove_fixed(self.get_fit_parameters().loc["mean"].values)
        cov = self.covariances
        if len(mean) != len(cov):
            raise AssertionError(
                f"mean and cov does not match in shape: mean = {mean}, cov = {cov}."
            )
        rng = np.random.default_rng()
        samples = []
        while len(samples) < N_samples:
            # sample = [lambda_E, lambda_I, beta, tau]
            sample = rng.multivariate_normal(mean, cov, 1)[0]
            if np.all(sample[:-1] >= 0):
                samples.append(sample)
        return samples

    def _random_samples_to_SIR_results(self, N_samples, T_max, ts):
        samples = self._random_sample_fit_parameters(N_samples)
        SIR_results = []
        for sample in samples:
            # if len(sample) == 2:
            #     lambda_E = self.cfg.lambda_E
            #     lambda_I = self.cfg.lambda_I
            #     beta, tau = sample
            # else:
            lambda_E, lambda_I, beta, tau = self._get_fit_values(sample)
            SIR_result = self._compute_result_SIR(lambda_E, lambda_I, beta, ts=ts, T_max=T_max)
            SIR_results.append(SIR_result)
        SIR_results = np.array(SIR_results)
        return SIR_results

    def make_monte_carlo_fits(self, N_samples, T_max=None, ts=0.1):
        SIR_results = self._random_samples_to_SIR_results(N_samples, T_max, ts)
        I_max_MC = np.zeros(N_samples)
        R_inf_MC = np.zeros(N_samples)
        for i, SIR_result in enumerate(SIR_results):
            t = SIR_result_to_time(SIR_result)
            I = SIR_result_to_I(SIR_result)
            R = SIR_result_to_R(SIR_result)
            I_max_MC[i] = get_I_max(I)
            R_inf_MC[i] = get_R_inf(R)
        return SIR_results, I_max_MC, R_inf_MC

    def plot_fit(self, ts=0.1, dt=0.01, T_max=None, xlim=(0, None)):
        t = self.t
        if T_max is None:
            T_max = max(t) * 1.1
        df_fit = self.calc_df_fit(ts=ts, T_max=T_max)

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.errorbar(t, self.y, self.sy, fmt=".", label="ABM")
        ax.plot(df_fit["time"], df_fit["I"], label="Fit")
        ax.set(xlim=xlim, title="Fit")
        # ax.text(0.1, 0.8, f"chi^2 = {self.chi2}")
        return fig, ax
