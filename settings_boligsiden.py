#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:42:55 2018

@author: michelsen


        Settings for all programs
"""

import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

# %%============================================================================
#  initial parameters
# ==============================================================================

house_type =                'Villa'
# house_type =                'Ejerlejlighed'
max_num_cores =             20

force_rerun =               False   # if True, runs everything again. 
                                    # if False, loads data from disk 
force_replot =              False                                 
                                    
                                    
verbose =                   True
very_verbose =              True    # Print more than the bare results 

save_plots =                True  # Save these plots
create_plot =               True  # Plot results to screen
create_plot_all =           False   # Plot results to screen
create_plot_denmark =       False
create_plot_missingno =     True   # creates plots of missing values
create_plot_lin_MIC_corr =  True
create_shap_plot =          True 

add_rente =                 True

N_columns_to_keep =         100

is_quick_run =              False
quick_run_proportion =      0.01

do_forecasting =            True
do_tight_refit =            True
do_SHAP_vals =              True


cut =                       'all' # mic, lin, xgb, all, or auto
version =                   '19'

quick_run_name =            'run_quick' if is_quick_run else 'run_normal'

path =                      './data'
path_newest_data =         f'{path}/newest_data'
data_path =                f'{path}/boligsiden'
figures_path =              './figures/boligsiden'

Ncolsname =                 str(N_columns_to_keep) if cut != 'all' else 'all'
specs =                    f'{house_type}_v{version}_cut_{cut}_Ncols_{Ncolsname}_'
base_name =                f'{data_path}/' + specs
base_name_figures =        f'{figures_path}/' + specs
filename_org_data =        f'{path_newest_data}/DW_NBI_2019_09_03.csv'
filename_org_data_C20 =    f'{path_newest_data}/C20_indeks.xlsx'
filename_org_data_rente =  f'{path_newest_data}/Renteudvikling.xlsx'
filename_cluster_Ejer =    f'{path_newest_data}/output_data_Ejerlejlighed_(n_11_c_4).csv'
filename_cluster_Villa =   f'{path_newest_data}/output_data_Villa_(n_11_c_8).csv'



filename_correlation_MIC = f'{data_path}/{house_type}_correlation_MIC.pkl'
filename_correlation_lin = f'{data_path}/{house_type}_correlation_lin.pkl'
filename_correlation_xgb = f'{data_path}/{house_type}_correlation_xgb.pkl'
filename_correlation_auto= f'{data_path}/{house_type}_correlation_auto_N_{N_columns_to_keep}.npy'

filename_shap_interaction = lambda name: f'{base_name_figures}{name}_SHAP_vals_interaction.pdf'
filename_shap_summary     = lambda name: f'{base_name_figures}{name}_SHAP_vals_summary.pdf'
filename_shap_summary_all = lambda name: f'{base_name_figures}{name}_SHAP_vals_summary_all.pdf'
filename_shap_interaction_vaerdi0 = lambda name: f'{base_name_figures}{name}_SHAP_vals_interaction_Vaerdi0.pdf'


#%%

def check_python2():
    import sys
    is_python2 = (sys.version_info < (3, 0))
    if is_python2:
        raise ValueError("Must be running Python 3")

# %%

def detect_cluster(max_num_cores, num_cores_laptop=8):
    
    """
    Detects the number of cores on the system. If more than num_cores_laptop = 8,
    classifies it as cluster, otherwise not. 
    Returns the flag is_cluster and the number of cores.
    """
    
    if joblib.cpu_count() > num_cores_laptop:
        is_cluster = True
        cpu_n_jobs = max_num_cores
    else:
        is_cluster = False
        cpu_n_jobs = num_cores_laptop
    
    return is_cluster, cpu_n_jobs


is_cluster, cpu_n_jobs = detect_cluster(max_num_cores, num_cores_laptop=8)

# %%    

def pandas_and_seaborn_options(max_val = 10):
    """
    Sets pandas how many rows and columns pandas should print.
    Sets the seaborn context to poster.
    """
    
    pd.set_option('display.max_rows', max_val) # number of max rows to print for a DataFrame  
    pd.set_option('display.max_columns', max_val) # number of max rows to print for a DataFrame  
    sns.set_context("poster") # change some standard settings of plots

    # colors = 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'lime', 'cyan'
    # current_palette = sns.color_palette()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'light-blue', 'light-green', 'light-purple', 'light-red', 'light-orange'] #'brown', 'pink', 'grey', 'lime', 'cyan'
    
    plt.style.use("matplotlibrc")  # bmh

    current_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    color_dict = {}
    for i, color in enumerate(colors):
        color_dict[color] = current_palette[i]
    color_dict['black'] = color_dict['k'] = (0, 0, 0)

    return color_dict, current_palette

color_dict, current_palette = pandas_and_seaborn_options()


#%%

from scipy.stats import uniform, randint

#%% =============================================================================
#    
#                              X G B O O S T
#
# =============================================================================

# Put their parameter dictionaries in a dictionary with the estimator names as keys
" Hyperparameter-tune the different models. Notice the weird random uniform."


params_dict_xgb_random = {
                'colsample_bytree':      uniform(0.1, 0.9-0.1),
#                'gamma': uniform(0, 0.1),    
                'max_depth':             randint(1, 20),
                'min_child_weight':      randint(1, 30),
#                'n_estimators' :         randint(500, 2500),
                'reg_lambda':            uniform(0.1, 4-0.1),
                'reg_alpha':             uniform(0.1, 4-0.1), 
                'subsample':             uniform(0.1, 0.99-0.1),
}


params_dict_xgb_BO = {
                'colsample_bytree':     (0.1, 0.9),
#                'gamma':                (0, 0.1),    
                'max_depth':            (1, 20),
                'min_child_weight':     (1, 30),
#                'n_estimators' :        (500, 2500),
                'reg_lambda':           (0.1, 4),
                'reg_alpha':            (0.1, 4), 
                'subsample':            (0.1, 0.99),
}
int_list_xgb = ['max_depth', 'min_child_weight', 'n_estimators']

do_early_stopping_xgb = True
n_jobs_xgb = 1 # 1 for forests, cpu_n_jobs for other models
use_weights_xgb = True
kwargs_shap_xgb = {'N_rows': None, 'method':'normal', 'max_display': 20}


#%% ===========================================================================
#    
#                          L i g h t G B M
#
# =============================================================================



params_dict_lgb_random = {
#                'colsample_bytree': uniform(0.5, 0.9-0.5),
                'max_depth':        randint(10, 50),
#                'min_child_weight': randint(1, 20),
                'min_data_in_leaf': randint(10, 100),
                'num_leaves' :      randint(10, 100),
                'reg_lambda':       uniform(0.5, 3-0.5),
                'reg_alpha':        uniform(0.5, 3-0.5), 
                'subsample':        uniform(0.3, 0.9-0.5),
}

params_dict_lgb_BO = {
#                'colsample_bytree':     (0.5, 0.9),
                'max_depth':            (10, 50),
#                'min_child_weight':     (1, 20),
                'min_data_in_leaf':     (10, 100),
                'num_leaves' :          (10, 100),
                'reg_lambda':           (0.5, 3),
                'reg_alpha':            (0.5, 3), 
                'subsample':            (0.3, 0.9),
}

int_list_lgb = ['max_depth', 'min_data_in_leaf', 'num_leaves', 'n_estimators']

do_early_stopping_lgb = True
n_jobs_lgb = 1 # 1 for forests, cpu_n_jobs for other models

use_weights_lgb = True
kwargs_shap_lgb = kwargs_shap_xgb


#%% ===========================================================================
#    
#                          R i d g e
#
# =============================================================================

params_dict_lin_random = {'alpha': uniform(0, 0.1)}

epsilon = 1e-6
params_dict_lin_BO = {'alpha': (epsilon, 0.1)}
int_list_lin = []

do_early_stopping_lin = False
n_jobs_lin = 10 if is_cluster else cpu_n_jobs # 1 for forests, cpu_n_jobs for other models

use_weights_lin = True
kwargs_shap_lin = {'N_rows': 1000, 'method':'linear', 'max_display': 20}


#%% ===========================================================================
#    
#                     K   N e a r e s t  N e i g h b o r s 
#
# =============================================================================

params_dict_knn_random = {'n_neighbors': randint(1, 1000),
                          'p' : [1, 2]}
    

params_dict_knn_BO = {'n_neighbors': (1, 1000),
                      'p' : (1, 2.9)}

int_list_knn = ['n_neighbors', 'p']

do_early_stopping_knn = False
n_jobs_knn = 1 # 1 for forests, cpu_n_jobs for other models

sample_frac_knn = 0.2

use_weights_knn = False
kwargs_shap_knn = {'N_rows': 1000, 'method':None, 'max_display': 20}


#%% =============================================================================
#    
#              S u p p o r t   V e c t o r   R e g r e s s i o n
#
# =============================================================================


params_dict_svr_random = { 'base_estimator__C':     uniform(0, 1),
                           'base_estimator__gamma': uniform(0, 1),  
                    }

params_dict_svr_BO = { 'base_estimator__C':     (1e-6, 1),
                      'base_estimator__gamma': (1e-6, 1),  
                      }

int_list_svr = []
do_early_stopping_svr = False
n_jobs_svr = 1 # 1 for forests, cpu_n_jobs for other models

use_weights_svr = True
kwargs_shap_svr = {'N_rows': 1000, 'method':None, 'max_display': 20}



#%% ===========================================================================
#    
#                     R a n d o m   F o r e s t
#
# =============================================================================

params_dict_rf_random = {  'n_estimators': randint(10, 100),
                    'max_features': uniform(0, 0.3),
                    'max_depth':    randint(3, 5),
                                                   }

params_dict_rf_BO = { 'n_estimators': (10, 100),
                      'max_features': (1e-6, 0.3),
                      'max_depth':    (1, 5),
                                                }

int_list_rf = ['n_estimators', 'max_depth']
do_early_stopping_rf = False
n_jobs_rf = 1 # 1 for forests, cpu_n_jobs for other models

sample_frac_rf = 0.1
use_weights_rf = True
kwargs_shap_rf = {'N_rows': 1000, 'method':'normal', 'max_display': 20}




