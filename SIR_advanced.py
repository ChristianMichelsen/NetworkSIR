import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from pathlib import Path
from iminuit import Minuit
import extra_funcs
from importlib import reload
import NewSpeedImprove_extra_funcs as extra_funcs2



savefig = False
# cfg = configuration.load()

ts = 0.1 # frequency of "observations". Now 1 pr. day
dt = 0.01 # stepsize in integration


#%%

filenames = extra_funcs.get_filenames()
N_files = len(filenames)


fit_objects = []
for filename in tqdm(filenames):

    cfg = extra_funcs2.filename_to_dotdict(str(filename))

    df, df_interpolated, time, t_interpolated = extra_funcs.pandas_load_file(filename)
    y_true = df_interpolated['I']
    Tmax = int(time.max())+1 # max number of days
    S0 = cfg.N0
    # y0 =  S, S0,                E1,E2,E3,E4,  I1,I2,I3,I4,  R, R0
    y0 = S0-cfg.Ninit,S0,   cfg.Ninit,0,0,0,      0,0,0,0,   0, cfg.Ninit

    # reload(extra_funcs)
    fit_object = extra_funcs.CustomChi2(time, t_interpolated, y_true, y0, Tmax, dt=dt, ts=ts, y_min=10)

    minuit = Minuit(fit_object, pedantic=False, print_level=0, Mrate1=cfg.Mrate1, Mrate2=cfg.Mrate2, mu0=cfg.mu, beta=cfg.beta, tau=0)
    minuit.migrad()
    if (not minuit.get_fmin().is_valid) :
        print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

    fit_object.set_minuit(minuit)
    fit_objects.append(fit_object)

    # df_fit = fit_object.calc_df_fit(ts=0.01)
    # df_fit_parameters = fit_object.get_all_fit_pars()
    # df_correlations = fit_object.get_correlations()

#%%


#%%

for parameter in fit_object.parameters:

    means = np.zeros(N_files)
    stds = np.zeros(N_files)

    for i, fit_object in enumerate(fit_objects):
        # df_fit_parameters = fit_object.get_all_fit_pars()
        means[i], stds[i] = fit_object.get_fit_par(parameter)


    fig = make_subplots(rows=1, cols=3, 
    subplot_titles=(f"mean {parameter}", f"std {parameter}", f"pull {parameter}"))

    fig.add_trace(go.Histogram(x=means, 
                            nbinsx=50,
                            histnorm='probability', 
                            name=parameter),
                    row=1, col=1)


    fig.add_trace(go.Histogram(x=stds, 
                               nbinsx=50,
                            # xbins=dict( # bins used for histogram
                            #         start=0.01,
                            #         end=0.075,
                            #         size=0.001,
                            #         ),
                            histnorm='probability', 
                            name=parameter),
                    row=1, col=2)

    fig.add_trace(go.Histogram(x=means/stds, 
                                nbinsx=50,
                            # xbins=dict( # bins used for histogram
                            #         start=-20.0,
                            #         end=25.0,
                            #         size=2,
                            #         ),
                            histnorm='probability', 
                            name=parameter),
                    row=1, col=3)

    for i in range(3):
        fig.update_xaxes(title_text=parameter, row=1, col=i+1)
    fig.update_yaxes(title_text="Normalized Counts", row=1, col=1)

    k_scale = 1


    # Edit the layout
    fig.update_layout(title=f'Histograms of fitted {parameter} values',
                    height=600*k_scale, width=800*k_scale,
                    )

    fig.show()


#%%

fig = go.Figure()

df_fit = fit_object.calc_df_fit(ts=0.01)

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
