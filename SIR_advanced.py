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
from matplotlib.backends.backend_pdf import PdfPages
import extra_funcs
from importlib import reload
import SimulateDenmarkAgeHospitalization_extra_funcs
import matplotlib.patches as mpatches
import rc_params
rc_params.set_rc_params()

num_cores_max = 10

savefig = False
do_animate = False
save_and_show_all_plots = True
plot_SIR_comparison = False if SimulateDenmarkAgeHospitalization_extra_funcs.is_local_computer() else False

#%%

# reload(extra_funcs)
filenames = extra_funcs.get_filenames()
N_files = len(filenames)

if plot_SIR_comparison:

    extra_funcs.plot_SIR_model_comparison('I', force_overwrite=True)
    extra_funcs.plot_SIR_model_comparison('R', force_overwrite=True)

x=x

#%%

if False:

    # reload(extra_funcs)
    # parameter = 'rho'
    # do_log = False

    extra_funcs.plot_1D_scan('N_tot', do_log=True)
    extra_funcs.plot_1D_scan('N_init', do_log=True) 
    extra_funcs.plot_1D_scan('N_init', do_log=True, algo=1) 

    # extra_funcs.plot_1D_scan('N_ages', age_mixing=0)
    # extra_funcs.plot_1D_scan('N_ages', age_mixing=0.25)
    # extra_funcs.plot_1D_scan('N_ages', age_mixing=0.5)
    # extra_funcs.plot_1D_scan('N_ages', age_mixing=0.75)
    # extra_funcs.plot_1D_scan('N_ages', age_mixing=1) 
    # extra_funcs.plot_1D_scan('age_mixing', N_ages=1)
    # extra_funcs.plot_1D_scan('age_mixing', N_ages=3)
    # extra_funcs.plot_1D_scan('age_mixing', N_ages=10) 

    extra_funcs.plot_1D_scan('beta_scaling')
    extra_funcs.plot_1D_scan('beta_scaling', N_tot=5_800_000)

    extra_funcs.plot_1D_scan('rho') 
    extra_funcs.plot_1D_scan('rho', algo=1) 
    extra_funcs.plot_1D_scan('rho', N_tot=5_800_000) 
    extra_funcs.plot_1D_scan('epsilon_rho', rho=100)
    extra_funcs.plot_1D_scan('epsilon_rho', rho=100, algo=1)  
    
    extra_funcs.plot_1D_scan('mu')
    extra_funcs.plot_1D_scan('beta')
    extra_funcs.plot_1D_scan('sigma_beta') 
    extra_funcs.plot_1D_scan('sigma_beta', algo=1) 
    extra_funcs.plot_1D_scan('sigma_beta', rho=150) 
    extra_funcs.plot_1D_scan('sigma_beta', rho=150, algo=1) 
    extra_funcs.plot_1D_scan('sigma_mu') 
    extra_funcs.plot_1D_scan('sigma_mu', algo=1) 
    extra_funcs.plot_1D_scan('sigma_mu', rho=150) 
    extra_funcs.plot_1D_scan('sigma_mu', rho=150, algo=1) 
    extra_funcs.plot_1D_scan('lambda_E') 
    extra_funcs.plot_1D_scan('lambda_I') 


#%%

if __name__ == '__main__':

    fit_objects_all = extra_funcs.get_fit_Imax_results(filenames, force_rerun=False, num_cores_max=num_cores_max)


#%%


#%%

    def plot_fit_simulation_SIR_comparison(fit_objects_all, force_overwrite=False, verbose=False, do_log=True):

        pdf_name = f"Figures/Fits_IR.pdf"
        Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

        if Path(pdf_name).exists() and not force_overwrite:
            print(f"{pdf_name} already exists")
            return None

        with PdfPages(pdf_name) as pdf:

            # sim_par, fit_objects = list(fit_objects_all.items())[0]
            for sim_par, fit_objects in tqdm(fit_objects_all.items()):
                # break

                if len(fit_objects) == 0:
                    if verbose:
                        print(f"Skipping {sim_par}")
                    continue

                cfg = extra_funcs.string_to_dict(sim_par)

                fig, axes = plt.subplots(ncols=2, figsize=(18, 8), constrained_layout=True)
                fig.subplots_adjust(top=0.9)

                leg_loc = {'I': 'upper right', 'R': 'lower right'}
                d_ylabel = {'I': 'Infected', 'R': 'Recovered'}

                fit_values = defaultdict(list)
                fit_errors = defaultdict(list)
                max_y = defaultdict(int)
                for i, (filename, fit_object) in enumerate(fit_objects.items()):
                    
                    for fit_par in fit_object.parameters:
                        fit_values[fit_par].append(fit_object.fit_values[fit_par])
                        fit_errors[fit_par].append(fit_object.fit_errors[fit_par])

                    df = extra_funcs.pandas_load_file(filename, return_only_df=True)
                    T = df['Time'].values
                    Tmax = max(T)*1.5
                    df_fit = fit_object.calc_df_fit(ts=0.1, Tmax=Tmax)
                    
                    
                    T_min = fit_object.t_interpolated.min()
                    T_max = fit_object.t_interpolated.max()

                    lw = 0.1
                    for y, ax in zip(['I', 'R'], axes):
                        
                        label = 'Simulations' if i == 0 else None
                        ax.plot(T, df[y].to_numpy(int), 'k-', lw=lw, label=label)
                        max_y[y] = max(max_y[y], max(df[y].to_numpy(int)))

                        label_min = 'Min/Max' if i == 0 else None
                        ax.axvline(T_min, lw=lw, alpha=0.2, label=label_min)
                        ax.axvline(T_max, lw=lw, alpha=0.2)

                        label = 'Fits' if i == 0 else None
                        ax.plot(df_fit['Time'], df_fit[y], lw=lw, color='green', label=label)
                        max_y[y] = max(max_y[y], max(df_fit[y].to_numpy(int)))

                SIR_values = cfg.lambda_E, cfg.lambda_I, cfg.beta, 0
                df_SIR = fit_object.calc_df_fit(values=SIR_values, ts=0.1, Tmax=Tmax)

                # where I SIR is less than 50 and after the peak
                x_max = np.argmax((df_SIR['I'] < 50) & (np.argmax(df_SIR['I']) < df_SIR.index))
                x_max = df_SIR['Time'].iloc[x_max] * 1.5

                for y, ax in zip(['I', 'R'], axes):

                    ax.plot(df_SIR['Time'], df_SIR[y], lw=lw*30, color='red', label='SIR')

                    ax.set(ylim=(50, max_y[y]*2), 
                        #    xlim=(0, x_max),
                        )

                    if do_log:
                        ax.set_yscale('log', nonposy='clip')

                    ax.set(xlabel='Time', ylabel=d_ylabel[y])
                    ax.set_rasterized(True)
                    ax.set_rasterization_zorder(0)

                    leg = ax.legend(loc=leg_loc[y])
                    for legobj in leg.legendHandles:
                        legobj.set_linewidth(2.0)
                        legobj.set_alpha(1.0)

                title = extra_funcs.dict_to_title(cfg, len(fit_objects))
                fig.suptitle(title, fontsize=24)


                # # These are in unitless percentages of the figure size. (0,0 is bottom left)
                # left, bottom, width, height = [0.62, 0.76, 0.3*0.95, 0.08*0.9]

                # # background_box = [(0.51, 0.61), 0.47, 0.36]
                # # ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='lightgray', transform=ax.transAxes))

                # # delta_width = 0 * width / 100
                # ax2 = fig.add_axes([left, bottom, width, height])
                # ax2.hist(fit_values['beta'])

            
                pdf.savefig(fig, dpi=100)
                plt.close('all')


    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")
        plot_fit_simulation_SIR_comparison(fit_objects_all, force_overwrite=True)



#%%

    # %%
    print("Finished running")


    # %%

    import matplotlib as mpl
    mpl.rcParams

