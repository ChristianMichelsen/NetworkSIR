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
import joblib
import extra_funcs
from importlib import reload
# import NewSpeedImprove_extra_funcs as extra_funcs2

savefig = False

#%%

reload(extra_funcs)
filenames = extra_funcs.get_filenames()
N_files = len(filenames)

# filenames = filenames[:1000]

#%%

if __name__ == '__main__':

    fit_results = extra_funcs.get_fit_results(filenames, force_rerun=False, num_cores_max=30)
    all_fit_objects, discarded_files, N_refits_total = fit_results

    print(f"{N_refits_total=}, number of discarded files = {len(discarded_files)}", flush=True)


#%%



    #%%

        # reload(extra_funcs)

    percentage1 = 5
    percentage2 = 95
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


            fig.add_trace(go.Histogram(x=extra_funcs.cut_percentiles(means, percentage1, percentage2), 
                                    nbinsx=Nbins,
                                    histnorm='probability', 
                                    ),
                            row=1, col=i_par)


            fig.add_trace(go.Histogram(x=extra_funcs.cut_percentiles(stds, percentage1, percentage2),
                                    nbinsx=Nbins,
                                    histnorm='probability', 
                                    ),
                            row=2, col=i_par)

            fig.add_trace(go.Histogram(x=extra_funcs.cut_percentiles(means/stds, percentage1, percentage2),
                                        nbinsx=Nbins,
                                    # xbins=dict( # bins used for histogram
                                    #         start=-20.0,
                                    #         end=25.0,
                                    #         size=2,
                                    #         ),
                                    histnorm='probability', 
                                    ),
                            row=3, col=i_par)
            

        fig.update_yaxes(title_text=f"Mu", row=1, col=1)
        fig.update_yaxes(title_text=f"Std", row=2, col=1)
        fig.update_yaxes(title_text=f'"Pull"', row=3, col=1)

        fig.update_layout(showlegend=False)

        k_scale = 1
        # Edit the layout
        N0_str = extra_funcs.human_format(d['N0'])
        title = f"Histograms for N={N0_str}, Mrate1={d['Mrate1']:.1f}, Mrate2={d['Mrate2']:.1f}, beta={d['beta']:.1f}"
        fig.update_layout(title=title, height=600*k_scale, width=800*k_scale)

        fig.show()
        fig.write_html(f"Figures/fits_{parameters_as_string}.html")

    #%%

    fig = go.Figure()


    # df_fit = fit_object.calc_df_fit(ts=0.01, values=(2, 1, 15, 1))
    # df_fit = fit_object.calc_df_fit(ts=0.01, values=(cfg.Mrate1, cfg.Mrate2, cfg.beta, 0))
    df_fit = fit_object.calc_df_fit(ts=0.01)

    df, df_interpolated, time, t_interpolated = extra_funcs.pandas_load_file(filenames[-1])


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
    # if savefig:
    #     fig.write_html(f"Figures/{filename.stem}.html")


        # %%
