import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from pathlib import Path
from iminuit import Minuit
from collections import defaultdict
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform as sp_uniform

def uniform(a, b):
    loc = a
    scale = b-a
    return sp_uniform(loc, scale)


import extra_funcs
from importlib import reload
# import NewSpeedImprove_extra_funcs as extra_funcs2


savefig = False
# cfg = configuration.load()

ts = 0.1 # frequency of "observations". Now 1 pr. day
dt = 0.01 # stepsize in integration

FIT_MAX = 100

#%%

filenames = extra_funcs.get_filenames()
N_files = len(filenames)

all_fit_objects = defaultdict(list)

for filename in tqdm(filenames):

    cfg = extra_funcs.filename_to_dotdict(str(filename))
    parameters_as_string = extra_funcs.dict_to_str(cfg)
    # d = extra_funcs.string_to_dict(parameters_as_string)

    df, df_interpolated, time, t_interpolated = extra_funcs.pandas_load_file(filename)
    y_true = df_interpolated['I']
    Tmax = int(time.max())+1 # max number of days
    S0 = cfg.N0
    # y0 =  S, S0,                E1,E2,E3,E4,  I1,I2,I3,I4,  R, R0
    y0 = S0-cfg.Ninit,S0,   cfg.Ninit,0,0,0,      0,0,0,0,   0, cfg.Ninit

    # reload(extra_funcs)
    fit_object = extra_funcs.CustomChi2(time, t_interpolated, y_true, y0, Tmax, dt=dt, ts=ts, y_min=10)

    minuit = Minuit(fit_object, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, beta=cfg.beta, tau=0)

    minuit.migrad()
    fit_object.set_chi2(minuit)


    i_fit = 0
    # if (not minuit.get_fmin().is_valid) :
    if fit_object.chi2 / fit_object.N > 100:

        continue_fit = True
        while continue_fit:
            i_fit += 1

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
        all_fit_objects[parameters_as_string].append(fit_object)

    else:
        print(f"{filename} was discarded")

    # df_fit = fit_object.calc_df_fit(ts=0.01)
    # df_fit_parameters = fit_object.get_all_fit_pars()
    # df_correlations = fit_object.get_correlations()

import joblib
joblib.dump(all_fit_objects, 'all_fit_objects.joblib')


#%%

def cut_percentiles(x, p1, p2=None):
    if p2 is None:
        p1 = p1/2
        p2 = 100 - p1
    
    x = x[~np.isnan(x)]

    mask = (np.percentile(x, p1) < x) & (x < np.percentile(x, p2))
    return x[mask]



#%%

percentage1 = 10
percentage2 = 90
Nbins = 100

for parameters_as_string, fit_objects in all_fit_objects.items():

    # fit_objects = all_fit_objects[]
    d = extra_funcs.string_to_dict(parameters_as_string)


    fig = make_subplots(rows=3, cols=len(fit_objects[0].parameters), 
    subplot_titles=fit_objects[0].parameters,
    )


    for i_par, parameter in enumerate(fit_objects[0].parameters):
        i_par += 1

        N_files = len(fit_objects)

        means = np.zeros(N_files)
        stds = np.zeros(N_files)

        for i, fit_object in enumerate(fit_objects):
            means[i], stds[i] = fit_object.get_fit_par(parameter)


        fig.add_trace(go.Histogram(x=cut_percentiles(means, percentage1, percentage2), 
                                nbinsx=Nbins,
                                histnorm='probability', 
                                ),
                        row=1, col=i_par)


        fig.add_trace(go.Histogram(x=cut_percentiles(stds, percentage1, percentage2),
                                nbinsx=Nbins,
                                histnorm='probability', 
                                ),
                        row=2, col=i_par)

        fig.add_trace(go.Histogram(x=cut_percentiles(means/stds, percentage1, percentage2),
                                    nbinsx=Nbins,
                                # xbins=dict( # bins used for histogram
                                #         start=-20.0,
                                #         end=25.0,
                                #         size=2,
                                #         ),
                                histnorm='probability', 
                                ),
                        row=3, col=i_par)
        
        # fig.update_yaxes(title_text=f"Mu", row=1, col=i_par)

    fig.update_yaxes(title_text=f"Mu", row=1, col=1)
    fig.update_yaxes(title_text=f"Std", row=2, col=1)
    fig.update_yaxes(title_text=f"'Pull'", row=3, col=1)
        
    
        # fig.update_yaxes(title_text="Normalized Counts", row=1, col=1)

    k_scale = 1

    fig.update_layout(showlegend=False)

    # Edit the layout
    fig.update_layout(title=f"Histograms for 'Mrate1'={d['Mrate1']:.1f}",
                    height=600*k_scale, width=800*k_scale,
                    )

    fig.show()



#%%

fig = go.Figure()


df_fit = fit_object.calc_df_fit(ts=0.01, values=(2, 1, 15, 1))

# df_fit = fit_object.calc_df_fit(ts=0.01, values=(cfg.Mrate1, cfg.Mrate2, cfg.beta, 0))
# df_fit = fit_object.calc_df_fit(ts=0.01, values=(cfg.Mrate1, cfg.Mrate2, cfg.beta, 0))

for s in ['E', 'I', 'R']:
    fig.add_trace(go.Scatter(x=df['Time'], y=df[s], name=f'{s} raw network'))
    ss = f'{s}_sum' if s != 'R' else s
    fig.add_trace(go.Scatter(x=df_fit['Time'], y=df_fit[ss], name=f'{s} FIT'))

k_scale = 2/3
k_scale = 1

# Edit the layout
fig.update_layout(title=f'Simulation comparison',
                   xaxis_title='Time',
                   yaxis_title='Count',
                   height=600*k_scale, width=800*k_scale,
                   )

fig.update_yaxes(rangemode="tozero")

fig.show()
if savefig:
    fig.write_html(f"Figures/{filename.stem}.html")


# %%
