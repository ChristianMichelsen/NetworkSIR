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
import SimulateNetwork_extra_funcs
# import NewSpeedImprove_extra_funcs as extra_funcs2

savefig = False
do_animate = False
save_and_show_all_plots = False
plot_SIR_comparison = True if SimulateNetwork_extra_funcs.is_local_computer() else False

#%%

reload(extra_funcs)
filenames = extra_funcs.get_filenames()
N_files = len(filenames)

if plot_SIR_comparison:
    extra_funcs.plot_SIR_model_comparison(force_overwrite=False)


#%%

if do_animate:

    search_string = 'alpha_8_psi_0'
    filename_alpha_20 = [filename for filename in filenames if search_string in filename]

    for filename in filenames:
        if search_string in filename:
            break

    extra_funcs.animate_Imax_fit_filename(filename)

#%%

if __name__ == '__main__':

    I_maxs_truth, fit_objects_Imax, bin_centers_Imax = extra_funcs.get_fit_Imax_results(filenames, force_rerun=False, num_cores_max=30)
    # bins = np.linspace(0, 1, extra_funcs.N_peak_fits+1)
    # bin_centers_Imax = (bins[1:] + bins[:-1])/2

    fit_results = extra_funcs.get_fit_results(filenames, force_rerun=False, num_cores_max=30)
    all_fit_objects, discarded_files, N_refits_total = fit_results
    print(f"{N_refits_total=}, number of discarded files = {len(discarded_files)}\n\n", flush=True)


    x=x

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
    
    if save_and_show_all_plots:

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
            # N0_str = extra_funcs.human_format(d_parameters['N0'])
            title = extra_funcs.dict_to_title(d_parameters)
            fig.update_layout(title=title, height=600*k_scale, width=800*k_scale)

            fig.show()
            fig.write_html(f"Figures/fit_to_all/fits_{fit_pars_as_string}.html")
            fig.write_image(f"Figures/fit_to_all/fits_{fit_pars_as_string}.png")


#%%


    if save_and_show_all_plots:

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
        title = extra_funcs.filename_to_title(filename)
        fig.update_layout(title=title,
                        xaxis_title='Time',
                        yaxis_title='Count',
                        height=600*k_scale, width=800*k_scale,
                        )

        fig.update_yaxes(rangemode="tozero")

        fig.show()
        # if savefig:
        #     fig.write_html(f"Figures/{filename.stem}.html")



# %%

    # reload(extra_funcs)

    # first sort filenames into groups based on similar simulation parameters
    filenames_by_pars = defaultdict(list)
    for filename in I_maxs_truth.keys():
        par_string = extra_funcs.filename_to_par_string(filename)
        filenames_by_pars[par_string].append(filename)


    # reload(extra_funcs)
    I_max_truth_by_pars = {}
    I_max_normed_by_pars = {}
    I_max_relative_by_pars = {}
    betas_by_pars = {}
    betas_std_by_pars = {}

    for par_string in tqdm(filenames_by_pars.keys(), desc='Splitting I_max fits by simulation parameters'):
        filenames_to_use = filenames_by_pars[par_string]


        # I max truths 
        I_true_tmp = {k: I_maxs_truth[k] for k in filenames_to_use}
        I_maxs_true = extra_funcs.fix_and_sort_index(pd.Series(I_true_tmp))
        I_max_truth_by_pars[par_string] = I_maxs_true

        # fit_objects by par_string (contains all IDs)
        d_fit_objects_all_IDs = {k: fit_objects_Imax[k] for k in filenames_to_use}
        
        # normalized I_max from fits
        df_I_maxs_normed = extra_funcs.extract_normalized_Imaxs(d_fit_objects_all_IDs, 
                                                              I_maxs_truth, 
                                                              filenames_to_use, 
                                                              bin_centers_Imax)
        I_max_normed_by_pars[par_string] = df_I_maxs_normed


        # relative I_max from fits
        df_I_maxs_relative = extra_funcs.extract_relative_Imaxs(d_fit_objects_all_IDs, 
                                                              I_maxs_truth, 
                                                              filenames_to_use, 
                                                              bin_centers_Imax)
        I_max_relative_by_pars[par_string] = df_I_maxs_relative


        # extract betas
        df_betas, df_betas_std = extra_funcs.extract_fit_parameter('beta', d_fit_objects_all_IDs, filenames_to_use, bin_centers_Imax)
        betas_by_pars[par_string] = df_betas
        betas_std_by_pars[par_string] = df_betas_std


#%%

    do_mask_I_rel = True


    for par_string in tqdm(I_max_truth_by_pars.keys(), desc='Make Imax figures'):

        I_maxs_normed = I_max_normed_by_pars[par_string]
        I_maxs_relative = I_max_relative_by_pars[par_string]
        I_maxs_true = I_max_truth_by_pars[par_string]
        df_betas = betas_by_pars[par_string]
        df_betas_std = betas_std_by_pars[par_string]
        

        fig = make_subplots(rows=1, cols=5, 
                            subplot_titles=['Normalized Imax', 'Relative Imax', 'Beta', 'Beta std','Truth distriution'], 
                            column_widths=[0.225, 0.225, 0.225, 0.225, 0.1])

        I_maxs_normed = extra_funcs.mask_df(I_maxs_normed, 5)
        # subplot 1 - Normalized Imax
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


        
        I_maxs_relative = extra_funcs.mask_df(I_maxs_relative, 5)
        # subplot 2  Relative Imax
        fig.add_trace(
            go.Scatter( x=I_maxs_relative.columns, 
                        y=I_maxs_relative.loc['mean'],
                        error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=I_maxs_relative.loc['sdom'],
                            visible=True
                            ),
                        mode="markers",
                        ),
            row=1, col=2,
            )
        fig.update_xaxes(title_text=f"'Normalized Time'", row=1, col=2)
        fig.update_yaxes(title_text=f"Imax relative", row=1, col=2)


        # subplot 3
        fig.add_trace(
            go.Scatter( x=df_betas.columns, 
                        y=df_betas.loc['mean'],
                        error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=df_betas.loc['sdom'],
                            visible=True
                            ),
                        mode="markers",
                        ),
            row=1, col=3,
            )
        fig.update_xaxes(title_text=f"'Normalized Time'", row=1, col=3)
        fig.update_yaxes(title_text=f"Beta", row=1, col=3)


        df_betas_std = extra_funcs.mask_df(df_betas_std, 5)
        # subplot 4
        fig.add_trace(
            go.Scatter( x=df_betas_std.columns, 
                        y=df_betas_std.loc['mean'],
                        error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=df_betas_std.loc['sdom'],
                            visible=True
                            ),
                        mode="markers",
                        ),
            row=1, col=4,
            )
        fig.update_xaxes(title_text=f"'Normalized Time'", row=1, col=4)
        fig.update_yaxes(title_text=f"Beta std", row=1, col=4)


        # subplot 5
        fig.add_trace(
            go.Histogram(x=I_maxs_true),
            row=1, col=5,
            )
        fig.update_xaxes(title_text=f"I_max_truth", row=1, col=5)
        fig.update_yaxes(title_text=f"Counts", row=1, col=5)

        
        d_parameters = extra_funcs.string_to_dict(par_string)

        N_files = len(I_maxs_true)
        # N0_str = extra_funcs.human_format(d_parameters['N0'])

        title = extra_funcs.dict_to_title(d_parameters, N_files)

        k_scale = 1
        fig.update_layout(title=title, width=2400*k_scale, height=600*k_scale, showlegend=False)

        # fig.show()
        figname_html = Path(f"Figures/Imax_fits/html/fits_Imax_{par_string}.html")
        figname_png = Path(f"Figures/Imax_fits/png/fits_Imax_{par_string}.png")
        Path(figname_html).parent.mkdir(parents=True, exist_ok=True)
        Path(figname_png).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(figname_html))
        fig.write_image(str(figname_png))


# %%


    print("Finished running")






# %%


reload(extra_funcs)
filenames_beta_rest_default = extra_funcs.get_filenames_different_than_default('beta')

filenames_N0_rest_default = extra_funcs.get_filenames_different_than_default('N0')
