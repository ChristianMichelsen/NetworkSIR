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
# filename = filenames[0]


#%%

if __name__ == '__main__':

    fit_results = extra_funcs.get_fit_results(filenames, force_rerun=False, num_cores_max=30)
    all_fit_objects, discarded_files, N_refits_total = fit_results

    print(f"{N_refits_total=}, number of discarded files = {len(discarded_files)}\n\n", flush=True)

    I_maxs_true, I_maxs_normed = extra_funcs.get_fit_Imax_results(filenames, force_rerun=False, num_cores_max=30)
    I_maxs_times = np.linspace(0.05, 1-0.05, 9+1)    


    I_maxs_true = extra_funcs.fix_and_sort_index(pd.Series(I_maxs_true))
    # I_maxs_true.index = I_maxs_true.index.map(extra_funcs.filename_to_ID)
    # I_maxs_true = I_maxs_true.sort_index(ascending=True, inplace=False)

    I_maxs_normed = extra_funcs.fix_and_sort_index(pd.DataFrame(I_maxs_normed).T)
    I_maxs_normed.columns = I_maxs_times
    I_maxs_normed.loc['mean'] = I_maxs_normed.mean()
    I_maxs_normed.loc['std'] = I_maxs_normed.std()
    I_maxs_normed.loc['sdom'] = I_maxs_normed.std() / np.sqrt(len(I_maxs_normed)-2)
    


#%%

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=['Fits', 'Truth distriution'], 
                        column_widths=[0.7, 0.3])

    fig.add_trace(
        go.Scatter( x=I_maxs_normed.columns, 
                    y=I_maxs_normed.loc['mean'],
                    error_y=dict(
                        type='data', # value of error bar given in data coordinates
                        array=I_maxs_normed.loc['sdom'],
                        visible=True
                        ),
                    mode="markers",
                    ),
        row=1, col=1,
        )
    fig.update_xaxes(title_text=f"'Normalized Time'", row=1, col=1)
    fig.update_yaxes(title_text=f"I_max / I_max_truth", row=1, col=1)

    fig.add_trace(
        go.Histogram(x=I_maxs_true),
        row=1, col=2,
        )
    fig.update_xaxes(title_text=f"I_max_truth", row=1, col=2)
    fig.update_yaxes(title_text=f"Counts", row=1, col=2)


    k_scale = 1
    fit_pars_as_string = extra_funcs.filename_to_par_string(filenames[0])
    d_parameters = extra_funcs.string_to_dict(fit_pars_as_string)
    N0_str = extra_funcs.human_format(d_parameters['N0'])
    title = f"I_max fits and distribution for N={N0_str}, Mrate1={d_parameters['Mrate1']:.1f}, Mrate2={d_parameters['Mrate2']:.1f}, beta={d_parameters['beta']:.1f}"
    fig.update_layout(title=title, height=600*k_scale, width=800*k_scale, showlegend=False)

    fig.show()
    fig.write_html(f"Figures/fits_Imax_{fit_pars_as_string}.html")






#%%

    # reload(extra_funcs)
    fit_objects_by_pars = defaultdict(dict)
    for filename, fit_object in all_fit_objects.items():
        par_string = extra_funcs.filename_to_par_string(filename)
        ID = extra_funcs.filename_to_ID(filename)
        fit_objects_by_pars[par_string][ID] = fit_object


    # reload(extra_funcs)
    percentage1 = 5
    percentage2 = 95
    Nbins = 100

    for fit_pars_as_string, fit_objects_by_ID in fit_objects_by_pars.items():

        N_fits_for_parameter = len(fit_objects_by_ID)

        # fit_objects = all_fit_objects[]
        d_parameters = extra_funcs.string_to_dict(fit_pars_as_string)

        fit_pars = fit_objects_by_ID[0].parameters

        fig = make_subplots(rows=3, cols=len(fit_pars), subplot_titles=fit_pars)

        for i_fit_par, fit_par in enumerate(fit_pars, 1): # start enumerate at 1

            means = np.zeros(N_fits_for_parameter)
            stds = np.zeros(N_fits_for_parameter)

            for i_fit_object, fit_object in enumerate(fit_objects_by_ID.values()):
                means[i_fit_object], stds[i_fit_object] = fit_object.get_fit_par(fit_par)


            fig.add_trace(go.Histogram(x=extra_funcs.cut_percentiles(means, percentage1, percentage2), 
                                    nbinsx=Nbins,
                                    histnorm='probability', 
                                    ),
                            row=1, col=i_fit_par)


            fig.add_trace(go.Histogram(x=extra_funcs.cut_percentiles(stds, percentage1, percentage2),
                                    nbinsx=Nbins,
                                    histnorm='probability', 
                                    ),
                            row=2, col=i_fit_par)

            fig.add_trace(go.Histogram(x=extra_funcs.cut_percentiles(means/stds, percentage1, percentage2),
                                        nbinsx=Nbins,
                                    # xbins=dict( # bins used for histogram
                                    #         start=-20.0,
                                    #         end=25.0,
                                    #         size=2,
                                    #         ),
                                    histnorm='probability', 
                                    ),
                            row=3, col=i_fit_par)
            

        fig.update_yaxes(title_text=f"Mu", row=1, col=1)
        fig.update_yaxes(title_text=f"Std", row=2, col=1)
        fig.update_yaxes(title_text=f'"Pull"', row=3, col=1)

        fig.update_layout(showlegend=False)

        k_scale = 1
        # Edit the layout
        N0_str = extra_funcs.human_format(d_parameters['N0'])
        title = f"Histograms for N={N0_str}, Mrate1={d_parameters['Mrate1']:.1f}, Mrate2={d_parameters['Mrate2']:.1f}, beta={d_parameters['beta']:.1f}"
        fig.update_layout(title=title, height=600*k_scale, width=800*k_scale)

        fig.show()
        fig.write_html(f"Figures/fits_{fit_pars_as_string}.html")

#%%


    filename = filenames[0]


    fig = go.Figure()

    # reload(extra_funcs)
    fit_object = all_fit_objects[filename]

    df_fit = fit_object.calc_df_fit(ts=0.01)
    df, df_interpolated, time, t_interpolated = extra_funcs.pandas_load_file(filename)


    for s in ['E', 'I', 'R']:
        fig.add_trace(go.Scatter(x=df['Time'], y=df[s], name=f'{s} raw network'))
        ss = f'{s}_sum' if s != 'R' else s
        fig.add_trace(go.Scatter(x=df_fit['Time'], y=df_fit[ss], name=f'{s} FIT'))

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
