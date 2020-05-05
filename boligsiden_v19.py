#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:21:38 2018

@author: christian

"""

import settings_boligsiden as s

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#if s.is_cluster:
 #   import matplotlib
 #   matplotlib.use('Agg')


from extra_functions import (
                                load_optimized_df,
                                load_feature_optimized_df,
                                
                                cut_df_original,
                                sagtype_convert,
                                add_fb_prophet_prediction,
                                #move_SalgsPris_to_end,
                                add_C20_information,
                                add_rente_information,
                                
                                load_columns_to_use,
                                
#                                print_np,
                                
                                train_test_split,
                                
                                plot_missing_values,
                                plot_overview_of_all_columns,
                                plot_map_of_Denmark,
                                plot_map_of_Denmark_old,
                                plot_tinglysning,
                                
                                calc_MAD,

                                convert_to_tight,
                                
                                test_weights_obj_log10,
                                get_weights,
                                plot_MAD_score_half_obj,
                                
                                this_should_run,
                                
                                ml_model,
                                get_ml_params,
                                load_ml_model,
                                load_ml_model_tight,
                                get_y_pred,
                                # get_udbudspris_time,
                                plot_ml_model,

                                calc_z_stat,
                                plot_multiple_models,
                                
                                
#                                obj_mse,
#                                obj_logcos,
#                                obj_welsch_paper,
#                                obj_fair,
#                                obj_cauchy_paper,
                                
                                
                            )

import extra_functions as Ext

import numpy as np                # numerical analysis (pip install numpy --user)
import matplotlib.pyplot as plt   # plotting package (Usually comes with Python, pip install matplotlib --user)
import pandas as pd               # excel-like dataframes  (pip install pandas --user)
#import seaborn as sns             # nicer standard plot looks (pip install seaborn --user)
#import xgboost as xgb_core        # machine learning, gradient boosting, package 
import os.path as path            # to check local folders for files
# from os.path import isfile
import os
#import joblib
import time
#from tqdm import tqdm
#import multiprocessing
#import dill
#import sys
from copy import deepcopy
import socket
import importlib
plt.close('all') # close all previously open figures

np.random.seed(42)


t0 = time.time()
s.check_python2()

house_type = s.house_type
cpu_n_jobs = s.cpu_n_jobs

make_ML_plots = True


print("\nRunning version", s.version)
print("Running on", socket.gethostname()[:5])
print(f"Running on {house_type}")
print(f"\nRunning on cpus={cpu_n_jobs} with PID={os.getpid()}")
print("Cut is: ", s.cut)
if s.is_quick_run:
    print(f"Quick Run(!) with {100*s.quick_run_proportion}% of all data")
else:
    print("Running as normal with all data")


#%%==============================================================================
#  Read original data, optimize it and plot missing values
#==============================================================================

print("\n----------------------------------------------------------------  ")
print("      Housing prices analysis")
print("---------------------------------------------------------------- \n ")


try:
    _ = _VSCode_defaultMatplotlib_Params
    is_VS_code = True
except NameError:
    is_VS_code = False

if is_VS_code:
    from IPython import get_ipython
    get_ipython().run_line_magic("matplotlib", "auto")

plt.style.use("matplotlibrc")  # bmh

plt.rcParams.update({"figure.max_open_warning": 30})
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# load optimized, original data frame
df_optimized = load_optimized_df()


if s.create_plot_missingno and not s.is_cluster:
    plot_missing_values(df_optimized)


if s.create_plot_all and s.save_plots and not s.is_cluster:
    # plot overview of data
    excl_colums = ['ID', 'Vejnavn'] # 'HusNr'
    plot_overview_of_all_columns(df_optimized, excl_colums)

if s.create_plot_denmark:

    df = df_optimized[['SalgsPris', 'ArealBolig', 'GisX_WGS84', 'GisY_WGS84']].dropna()
    df.loc[:, 'KvadratmeterPris'] = df['SalgsPris'] / df['ArealBolig']

    df = df.loc[df.eval("(10 <= ArealBolig <= 10_000) & (100_000 <= SalgsPris <= 10_000_000)")]

    Ext.plot_map_of_Denmark(df, x='GisX_WGS84', y='GisY_WGS84', z='SalgsPris', save_plots=s.save_plots, filename=f"{s.figures_path}/Denmark_Overview_SalesPrice")

    Ext.plot_map_of_Denmark(df, x='GisX_WGS84', y='GisY_WGS84', z='KvadratmeterPris', save_plots=s.save_plots, filename=f"{s.figures_path}/Denmark_Overview_SqmPrice")

    # Ext.plot_map_of_Denmark_old(df, resolution='h')


if s.create_plot_all and not s.is_cluster:
    df_tinglysning = plot_tinglysning(df_optimized)


if False:

    drop = ['ID', 'Vejnavn', 'Bygning_GOP_FredningKode', 'Bygning_AGP_HerafAffaldsrum', 'Bygning_MOP_AsbestMatr', 'Bygning_VOP_AfloebsTilladelse', 'Bygning_AHB_HerafUdvIsolering', 
    'GisX_ETRS89', 'GisY_ETRS89', 'PostHovedNr', 'Kontantpris', 'OISSalgspris', 'Enhed_Ejendomsnr', 'Bygning_GOP_AntalBoligMedKoekken', 'Bygning_AHB_SamletBygning', 'Afstand_MotorvejTilFraKoersel', 'Kommune_IntegreredeInstitutioner']

    drop = drop + [f'EjdVurdering_GrundVaerdi{x}' for x in range(5)]
    drop = drop + [f'EjdVurdering_StuehusVaerdi{x}' for x in range(5)]
    drop = drop + [f'EjdVurdering_StueGrundVaerdi{x}' for x in range(5)]

    df_corr = df_optimized.drop(drop, axis='columns').corr()

    # df_corr.abs()




#%%==============================================================================
#  Read feature optimized dataframe 
#==============================================================================

importlib.reload(Ext)


# load feature optimized data frame
df_original = load_feature_optimized_df(df_optimized)
del df_optimized


print("Dataframe loaded with dimension: ", df_original.shape)
print()


# filter out obious outliers
df_cut = cut_df_original(df_original)
print("\nDataframe dimensions after: outlier-removal: ", df_cut.shape)

# keep only specific sagtypenr's (villa = 100 = 0, Ejerlejlighed = 300 = 1): 
df_cut = sagtype_convert(df_cut, 'SagtypeNr', house_type)
print("Dataframe dimensions after:       sagtypenr: ", df_cut.shape)

# remove columns that have 90% or more NaNs
df_cut = df_cut.dropna(axis='columns', thresh=int(len(df_cut)*0.1))
print("\nDataframe dimensions after:     NaN-removal: ", df_cut.shape)

# if s.create_plot_all and not s.is_cluster:
    # df_tinglysning_cut = plot_tinglysning(df_cut)

# add FB Prophet prediction
df_cut = add_fb_prophet_prediction(df_cut)
print("Dataframe dimensions after: adding FB prophet: ", df_cut.shape)

if False:

    importlib.reload(Ext)
    fig_forecast, fig_trends = Ext.add_fb_prophet_prediction(df_cut, make_only_plot=True)

    if s.save_plots:

        fig_forecast.savefig(f"{s.base_name_figures}prophet_forecast.png", dpi=300) #, rasterized=True)
        fig_trends.savefig(f"{s.base_name_figures}prophet_trends.pdf", dpi=600)

if False:
    # add Simon's cluster data
    if house_type == 'Ejerlejlighed':
        filename_cluster = s.filename_cluster_Ejer
    else: 
        filename_cluster = s.filename_cluster_Villa

    df_cluster = pd.read_csv(filename_cluster, sep=';', low_memory=False, index_col=0)
    df_cut = df_cut.merge(df_cluster, on='ID', how='inner')


if False:
    # add C20.index and renter:
    if s.add_rente:
        df_cut['C20'] = add_C20_information(s.filename_org_data_C20, df_cut.SalgsDato)
        df_cut['Kort_rente'], df_cut['Lang_rente'] = add_rente_information(
                                        s.filename_org_data_rente, df_cut.SalgsDato).T

kontantpris_MAD = calc_MAD(df_cut.SalgsPris, df_cut.KontantprisOprettet)
kontantpris_z = (df_cut.KontantprisOprettet-df_cut.SalgsPris)/df_cut.SalgsPris


#only keep some columns for ML
columns_to_exclude_for_ML = list(np.genfromtxt('columns_to_exclude_for_ML.txt', 
                                                               dtype='str'))

# exclude the above columns and ensure that Salgspris is the last column
columns_to_keep_for_ML = [col for col in df_cut.columns 
                          if not ((col in columns_to_exclude_for_ML)
                              or (col == 'SalgsPris'))] + ['SalgsPris']

df_ML = df_cut[columns_to_keep_for_ML].copy()

print("Dataframe dimensions after:      ML columns: ", df_ML.shape)
print("")
print(f"Mæglers MAD for {house_type} is: {kontantpris_MAD:.4f}")




#%%==============================================================================
#  temp bla , initial data visualisation and MIC
#==============================================================================


#df_ML = df_ML.sample(10_000) 
columns_to_use = load_columns_to_use(df_ML)
#df_ML = df_cut[columns_to_keep_for_ML].copy() # 

if False:

    from tabulate import tabulate

    table = np.array([r"\code{"+col+r"}" for col in columns_to_use]).reshape((-1, 3))

    print(tabulate(table, tablefmt="latex_raw"))



## below is deprecated code
#if not s.is_cluster:
#    print(calc_MAD(df_cut.SalgsPris, df_cut.KontantprisOprettet))
#    
#    z = (df_cut.KontantprisOprettet - df_cut.SalgsPris) / df_cut.SalgsPris
#    
#    df = df_ML.join(z.to_frame(name='z')) #, left_index=True, right_index=True)
#    
#    df_z = df.loc[:, ['KontantprisOprettet', 'SalgsPris', 'z']]
#    
#    plt.figure(figsize=(14, 8))
#    plt.hist(z, 100, range=(-0.3, 0.8));


#%%==============================================================================
#  
#==============================================================================


train_test_data = train_test_split(df_ML, 
                                   s.is_cluster, 
#                                   N_small = 1000, #00, 
                                   keep_cols = columns_to_use, 
                                   is_quick_run = s.is_quick_run, 
                                   quick_run_proportion = s.quick_run_proportion)
(     df_ML_time_sorted, 
     (X_all, y_all), 
     (X_train, y_train), 
     (X_test, y_test), 
     (X_2019, y_2019)) = train_test_data

print("X_all shape", X_all.shape)
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
print("X_2019 shape", X_2019.shape)

(X_tight_train, y_tight_train, mask_tight_train,
 X_tight_test, y_tight_test, mask_tight_test, 
 X_tight_2019, y_tight_2019, mask_tight_2019) = convert_to_tight(
         X_train, y_train, X_test, y_test, X_2019, y_2019, percent=1)

print("X_tight_train shape", X_tight_train.shape)
print("X_tight_test shape", X_tight_test.shape)
print("X_tight_2019 shape", X_tight_2019.shape)

print("\n\nMæglers MAD:")
for key, val in {'All': X_all, 
                 'Train': X_train, 'Train_tight': X_tight_train, 
                 'Test': X_test, 'Test_tight': X_tight_test, 
                 '2019': X_2019, '2019_tight': X_tight_2019}.items():
    mad = calc_MAD(df_cut.loc[val.index]['SalgsPris'], df_cut.loc[val.index]['KontantprisOprettet'])
    print(f"{key:5s} \t {mad:.5f}")



# %%

res = test_weights_obj_log10(X_train, y_train, N_boost_round=10_000)
(halflife_years, obj_function, do_log10, N_estimators, MAD_scores, 
                                             MAD_scores_std, arg_mins) = res

if s.create_plot:
    plot_MAD_score_half_obj(MAD_scores, MAD_scores_std, arg_mins)

weights = get_weights(X_train, halflife_years=halflife_years)
df_weights = pd.DataFrame(weights, index=X_train.index)

base_score = np.median(np.log10(y_train)) if do_log10 else np.median(y_train)/1e6

#%%

if False:

    def get_weights(x, halflife_years=4):
        if halflife_years is not None:
            k = np.log(2) / (halflife_years)
            days = x
            weights = np.exp(k * (days - np.max(days)))
            weights /= weights.mean()  
            return weights
        else:
            return np.ones(X.shape[0])


    x = np.linspace(0, 10)

    # days = X_train['SalgsDato_siden0']
    fig, ax = plt.subplots(figsize=(5, 5))
    for weight in [2.5, 5, 10, 20, 1000]:
        weights = get_weights(x, weight)
        ax.plot(x, weights, '-', lw=2.5, label=r'$T_{\frac{1}{2}} = \infty$' if weight==1000 else r'$T_{\frac{1}{2}} = ' + f'{weight}' + r'$')
    
    ax.set(xlabel=r"$t$", ylabel=r"$w(t)$")
    ax.legend()

    box = ax.get_position()

    ax.set_position([box.x0, 
            box.y0 + box.height * 0.1, 
            box.width, 
            box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.01),
            fancybox=False, shadow=False, ncol=2, frameon=False,
            fontsize=20)

    fig.tight_layout()

    fig.savefig(s.base_name_figures+'half_life_weights.pdf', dpi=300, bbox_inches='tight')

    fig



#%% =============================================================================
# 
# =============================================================================

from sklearn.pipeline import Pipeline

if not s.is_quick_run:
    N_CV = 5
    N_iter = 100
    N_init_BO = 10
    N_iter_BO = N_iter - N_init_BO
else:
    N_iter = 2
    N_CV = 3  
    N_iter_BO = 2
    N_init_BO = 2


#%% 

do_TPOT = False
if do_TPOT:
    from tpot import TPOTRegressor
    
    pipeline_optimizer = TPOTRegressor(generations=10, 
                                       population_size=50, 
                                       cv=5,
                                       random_state=42, 
                                       verbosity=2,  
                                       n_jobs=cpu_n_jobs, 
                                       scoring='neg_mean_squared_error',
    				periodic_checkpoint_folder='./checkpoints'	)
    if do_log10:
        pipeline_optimizer.fit(X_train, np.log10(y_train))
    else:
        pipeline_optimizer.fit(X_train, y_train / 1e6)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
#    y_pred_tpot = pipeline_optimizer.predict(X_test)
else:
    print("Not doing TPOT")    



#%% =============================================================================
# Pandas Profiling
# =============================================================================


if False:
    
    import pandas_profiling
    
    df_tmp = df_ML.sample(100)#.iloc[:, :]
    pfr = pandas_profiling.ProfileReport(df_tmp, 
                                         bins=100,
                                         pool_size=s.cpu_n_jobs-1,
                                         check_correlation=False)
    pfr.to_file(f"{s.figures_path}/PandasProfile.html")
    
    
    columns = ['Prophet_index', 'AntalRum', 'BeregnetAreal', 'SalgsPris']
    tmp = df_ML[columns]
    masks = []
    for i in range(tmp.shape[1]):
        masks.append(((tmp.iloc[:, i].quantile(0.01) <= tmp.iloc[:, i]) &
                                    (tmp.iloc[:, i] <= tmp.iloc[:, i].quantile(0.99))).values)
    mask = np.logical_and.reduce(masks)
    tmp = tmp.iloc[mask, :]
    tmp.to_pickle('tmp.pkl')



#%% =============================================================================
#    
#                              X G B O O S T
#
# =============================================================================


from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib

# X_tight_train, y_tight_train, mask_tight_train,
#  X_tight_test, y_tight_test, mask_tight_test, 
#  X_tight_2019

from sklearn.model_selection import train_test_split as train_test_split_sklearn

N = len(X_tight_train)
# N = 1000

X_tight_all = pd.concat([X_tight_train, X_tight_test, X_tight_2019])
y_tight_all = pd.concat([y_tight_train, y_tight_test, y_tight_2019])
X_tight_all_train, X_tight_all_test, y_tight_all_train, y_tight_all_test = train_test_split_sklearn(X_tight_all, y_tight_all)


try:

    ngb = joblib.load('ngb2.pkl')

except:

    ngb = NGBRegressor(Base=default_tree_learner, Dist=Normal, Score=MLE, verbose=True, n_estimators=500)
    ngb.fit(X_tight_all_train.iloc[:N, :], y_tight_all_train[:N]/1e6)
    joblib.dump(ngb, 'ngb2.pkl')
    print('saved NGB model')

# N = len(X_train)
# %time LGBMRegressor().fit(X_train.iloc[:N, :], y_train[:N]/1e6)
# %time XGBRegressor().fit(X_train.iloc[:N, :], y_train[:N]/1e6)
# %time NGBRegressor().fit(X_train.iloc[:N, :], y_train[:N]/1e6)


Y_preds_train = ngb.predict(X_tight_all_train)
Y_preds_test  = ngb.predict(X_tight_all_test)
Y_dists_train = ngb.pred_dist(X_tight_all_train)
Y_dists_test = ngb.pred_dist(X_tight_all_test)


# Y_preds_train = ngb.predict(X_tight_train)
# Y_preds = ngb.predict(X_tight_test)
# Y_dists = ngb.pred_dist(X_tight_test)
# Y_dists_train = ngb.pred_dist(X_tight_train)
# Y_dists_2019 = ngb.pred_dist(X_tight_2019)


pull_train = (y_tight_all_train/1e6 - Y_dists_train.loc) / Y_dists_train.scale
pull_test = (y_tight_all_test/1e6 - Y_dists_test.loc) / Y_dists_test.scale
# pull_2019 = (y_tight_2019/1e6 - Y_dists_2019.loc) / Y_dists_2019.scale
# pull_test_2019 = np.r_[pull_test, pull_2019]



from scipy.stats import norm

x = np.linspace(0, 10, 1000)
i = -1
y = norm(Y_dists_train.loc[i], Y_dists_train.scale[i]).pdf(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.axvline(y_tight_all_train.iloc[i]/1e6, ls='--', color='k', lw=2, label='Sales Price')
ax.set(xlabel='Million DKK', ylabel='Probability Density', xlim=(0.5, 2.5), ylim=(0, 2))
fig.tight_layout()
ax.legend()
fig.savefig('single_pred_1.pdf', dpi=300)
ax.plot(x, y, '-', label='PDF')
ax.legend()
fig.savefig('single_pred_2.pdf', dpi=300)
fig


if True:
        
    xlim = (-3, 3)
    density = True

    fig, ax = plt.subplots(figsize=(8, 4))
    H = ax.hist(pull_test, 100, range=xlim, histtype='step', density=density, label='Test', color=colors[0])
    ax.set(xlim=xlim, xlabel=r'Pull, $z$', ylabel='Counts', ylim=(0, 0.7))
    ax.legend(loc='upper right')

    ax.text(0.05, 0.15, r"$z = \frac{y-\hat{y}}{\hat{\sigma}}$", 
            {'color': 'black', 'fontsize': 22, 'ha': 'center', 'va': 'center',
            'bbox': dict(boxstyle="round", fc="white", ec="grey", pad=0.4, alpha=0.8)})

    fig.tight_layout()
    fig.savefig('NGBoost_pullplot_1.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    
    mask = (xlim[0] < pull_test) & (pull_test < xlim[1])
    import probfit_custom

    def Gauss(x, mu, sigma):
        return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)

    unbinned_likelihood = probfit_custom.UnbinnedLH(Gauss, pull_test[mask])
    # x = 0.5*(H[1][1:] + H[1][:-1])
    # y = H[0]
    # sy = np.sqrt(y)
    # chi2_object = probfit_custom.Chi2Regression(Gauss, x, y, sy)

    from iminuit import Minuit
    minuit = Minuit(unbinned_likelihood, mu=0.1, sigma=1.1, pedantic=False)
    # minuit = Minuit(chi2_object, mu=0.1, sigma=1.1, pedantic=False)
    minuit.migrad()

    pull_mu, mu_sigma = minuit.values['mu'], minuit.values['sigma']
    pull_mu_std, mu_sigma_std = minuit.errors['mu'], minuit.errors['sigma']
    print(f"{pull_mu:.3f} +/- {pull_mu_std:.3f}")
    print(f"{mu_sigma:.3f} +/- {mu_sigma_std:.3f}")

    x = np.linspace(xlim[0], xlim[1], 1000)
    k = len(pull_test[mask]) * (xlim[1] - xlim[0]) / 100
    y = norm(pull_mu, mu_sigma).pdf(x) #* k
    ax.plot(x, y, '--', label='Test Fit', alpha=0.8, lw=3)
    ax.legend(loc='upper right')

    # s = r"$\hat{\mu} = -0.051 \pm 0.004$" + "\n" + r"$\hat{\sigma} = +0.898 \pm 0.003$"
    # ax.text(-1.9, 0.63, s, {'color': 'black', 'fontsize': 18, 'ha': 'center', 'va': 'center',
    #         'bbox': dict(boxstyle="round", fc="white", ec=colors[0], pad=0.2)})
    ss = r"$\hat{\mu} = -0.059 \pm 0.007$" + "\n" + r"$\hat{\sigma} = +0.901 \pm 0.005$"
    ax.text(-1.9, 0.55, ss, {'color': 'black', 'fontsize': 18, 'ha': 'center', 'va': 'center',
            'bbox': dict(boxstyle="round", fc="white", ec=colors[0], pad=0.2)})
    fig

    fig.savefig('NGBoost_pullplot_2.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)



    # H2 = ax.hist(pull_test, 100, range=xlim, histtype='step', density=density, label='Test', color=colors[1])
    # ax.legend(loc='upper right')
    # fig.savefig('NGBoost_pullplot_3.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)


    # mask_test = (xlim[0] < pull_test) & (pull_test < xlim[1])
    # unbinned_likelihood_test = probfit_custom.UnbinnedLH(Gauss, pull_test[mask_test])

    # minuit = Minuit(unbinned_likelihood_test, mu=0.1, sigma=1.1, pedantic=False)
    # # minuit = Minuit(chi2_object, mu=0.1, sigma=1.1, pedantic=False)
    # minuit.migrad()

    # pull_mu, mu_sigma = minuit.values['mu'], minuit.values['sigma']
    # pull_mu_std, mu_sigma_std = minuit.errors['mu'], minuit.errors['sigma']
    # print(f"{pull_mu:.3f} +/- {pull_mu_std:.3f}")
    # print(f"{mu_sigma:.3f} +/- {mu_sigma_std:.3f}")

    # x = np.linspace(xlim[0], xlim[1], 1000)
    # y = norm(pull_mu, mu_sigma).pdf(x) #* k
    # ax.plot(x, y, '--', label='Test Fit', alpha=0.8, lw=3, color=colors[1])
    # ax.legend(loc='upper right')

    # s = r"$\hat{\mu} = -0.059 \pm 0.007$" + "\n" + r"$\hat{\sigma} = +0.901 \pm 0.005$"
    # ax.text(-1.9, 0.44, s, {'color': 'black', 'fontsize': 18, 'ha': 'center', 'va': 'center',
    #         'bbox': dict(boxstyle="round", fc="white", ec=colors[1], pad=0.2)})
    # fig

    # fig.savefig('NGBoost_pullplot_4.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)


xlim=(0, 40)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(Y_dists_test.scale/Y_dists_test.loc*100, 100, range=xlim, histtype='step', label='Relative to prediction')
ax.hist(Y_dists_test.scale/(y_tight_all_test/1e6)*100, 100, range=xlim, histtype='step', label='Relative to price')
ax.legend()
ax.set(xlabel='Relative uncertainty (%)', ylabel='Counts', xlim=xlim)
fig.savefig('relative_uncertainty.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
# fig




x=x
assert False


# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, y_tight_test/1e6)
print('Test MSE', test_MSE)

MAD_ngb = calc_MAD(y_tight_test/1e6, Y_preds)
print('MAD NGB', MAD_ngb)


# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(y_tight_test.values.flatten()/1e6).mean()
print('Test NLL', test_NLL)

import umap
import hdbscan
import sklearn.cluster as cluster


n_components = 20
filename_cluster = f'clusterable_embedding_n_components={n_components}.pkl'

try:
    clusterable_embedding = joblib.load(filename_cluster)
except:
    clusterable_embedding = umap.UMAP(
        n_neighbors=100,
        min_dist=0.0,
        n_components=n_components,
        random_state=42,
    ).fit_transform(X_tight_train)
    joblib.dump(clusterable_embedding, filename_cluster)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
            c=np.log10(y_tight_train), s=0.1, cmap='gray');
fig


filename_cluster_labels = f'labels_n_components={n_components}.pkl'
try:
    labels = joblib.load(filename_cluster_labels)
except:

    labels = hdbscan.HDBSCAN(
        min_samples=30,
        min_cluster_size=100,
    ).fit_predict(clusterable_embedding)
    joblib.dump(labels, filename_cluster_labels)


labels_unique = np.unique(labels)
d_labels = {}
d_labels_counts = {}

x = []
y = []
sy = []

for label in labels_unique:
    if label >= 0:
        mask_i = (labels==label)
        # d_labels[label] = calc_MAD(y_tight_train.iloc[mask_i]/1e6, Y_preds_train[mask_i])
        z = (y_tight_train.iloc[mask_i]/1e6 - Y_preds_train[mask_i]) / (y_tight_train.iloc[mask_i]/1e6)
        d_labels[label] = np.std(z)
        d_labels_counts[label] = mask_i.sum()

        sigma_NGB_i_mean = np.mean(Y_dists_train.scale[mask_i])
        sigma_NGB_i_sdom = np.std(Y_dists_train.scale[mask_i]) / np.sqrt(mask_i.sum())
        sigma_cluster_i = np.std(z)

        x.append(sigma_cluster_i)
        y.append(sigma_NGB_i_mean)
        sy.append(sigma_NGB_i_sdom)

x = np.array(x)
y = np.array(y)
sy = np.array(sy)

s_labels = pd.Series(d_labels, name='sigma')
s_labels_counts = pd.Series(d_labels_counts, name='counts')
df_labels = pd.concat([s_labels, s_labels_counts], axis=1).sort_values('sigma')
df_labels


fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(x, y, sy, fmt='.')
fig




x=x

Y_dists = ngb.pred_dist(X_tight_test)
y_range = np.linspace(min(y_test), max(y_test), 200)/1e6
dist_values = Y_dists.pdf(y_range).transpose()


# see the probability distributions by visualising
y_range = np.linspace(0, 10, 1000)
dist_values = Y_dists.pdf(y_range).transpose()
# plot index 0 and 114
idx = 114

fig, ax = plt.subplots()
plt.plot(y_range,dist_values[idx])
plt.title(f"idx: {idx}")
plt.tight_layout()
plt.show()





obs_idx = [0,1]
dist = ngb.pred_dist(X_test.iloc[obs_idx, :])
print('P(y_0|x_0) is normally distributed with loc={:.2f} and scale={:.2f}'.format(dist.loc[0], dist.scale[0]))
print('P(y_1|x_1) is normally distributed with loc={:.2f} and scale={:.2f}'.format(dist.loc[1], dist.scale[1]))






#%%

    
print("\n----------------------------------------------------------------  ")
print("                          XGBOOST")
print("---------------------------------------------------------------- \n ")


from extra_functions import XGB_wrapper

model_name_xgb = 'xgb'


pipe_xgb = Pipeline([('clf_xgb', XGB_wrapper(n_estimators = N_estimators,
                                             learning_rate = 0.1,
                                             objective = obj_function,
                                             base_score = base_score,
                                             n_jobs = s.cpu_n_jobs,
                                             missing = -9999,
                                             random_state = 42,
                                             silent = True, ))])

model_base_xgb = ml_model(pipe_xgb, do_log10)


dict_xgb_random, dict_xgb_BO, dict_xgb_early = get_ml_params(  
                                            X_train, y_train, weights,
                                            s.n_jobs_xgb, 
                                            s.params_dict_xgb_random,
                                            s.params_dict_xgb_BO,
                                            N_iter, N_CV, N_iter_BO, N_init_BO,
                                            s.int_list_xgb,
                                            num_boost_round = 200_000,
                                            early_stopping_rounds = 10_000 if not s.is_quick_run else 100,
                                            use_weights = s.use_weights_xgb,
                                            )


#%%


# fit or load ML-model

(model_xgb_random, model_xgb_BO, model_xgb, 
results_xgb_random, results_xgb_BO, 
results_xgb) = load_ml_model( model_name_xgb, model_base_xgb,  
                              dict_xgb_random, dict_xgb_BO, dict_xgb_early, 
                              s.do_early_stopping_xgb)

(y_pred_xgb, y_pred_xgb_2019, 
 y_pred_xgb_train) = get_y_pred(model_xgb, model_name_xgb, 
                                                     X_test, X_2019, X_train)

if s.do_tight_refit:
    
    refit_params = [X_tight_train, 
                    y_tight_train, 
                    get_weights(X_tight_train, halflife_years)]
    
    (model_xgb_random_tight, model_xgb_BO_tight, 
     model_xgb_tight) = load_ml_model_tight(model_name_xgb, model_base_xgb, refit_params)

    (y_pred_xgb_tight, y_pred_xgb_tight_2019, 
     y_pred_xgb_tight_train) = get_y_pred(model_xgb_tight, model_name_xgb+'_tight', 
                                                         X_tight_test, X_tight_2019, X_tight_train)

# %%

import pickle
import sys
sys.path.insert(0, "/groups/hep/mnv794/work/QuarksVsGluons")

if True:

    import ExternalFunctionsbTagging as ExtExt
    importlib.reload(ExtExt)

    df_params = pd.read_excel('initial_HPO_overview.xlsx', house_type, usecols='A:F')

    labels, uniques = pd.factorize(df_params['objective'])

    df_params.loc[:, 'objective'] = labels

    order_cols_by = ['time', 'log10', 'N_trees', 'halflife', 'objective', r'$f_\mathrm{eval}$']

    fig_CV_viz_hpo_initial, ax_CV_viz  = ExtExt.plot_CV_viz_parallel_coords(df_params, score_col='MAD', score_name=r'$f_\mathrm{eval}$', sort_by='min', standardize_color=True, order_cols_by=order_cols_by, ticks_fontsize=16) # score or recursive or list

    fig_CV_viz_hpo_initial

    if s.save_plots:
        fig_CV_viz_hpo_initial.savefig(f"{s.base_name_figures}CV_viz_initial_HPO.pdf", dpi=600, bbox_inches='tight', pad_inches=0.15)
    fig_CV_viz_hpo_initial



import ExternalFunctionsbTagging as ExtExt

for df_hpo, name in zip([results_xgb_random, results_xgb_BO], ['RS', 'BO']):

    params = list(df_hpo['params'].values)
    params

    df_params = pd.DataFrame(params)
    df_params.columns = [x.replace('clf_xgb__', '') for x in df_params.columns]
    df_params['score'] = np.array(df_hpo['mean_test_score'], dtype=float)
    df_params

    order_cols_by = ['min_child_weight', 'colsample_bytree', 'subsample', 'reg_alpha', 'reg_lambda', 'max_depth', r'$f_\mathrm{eval}$']

    fig_CV_viz, ax_CV_viz  = ExtExt.plot_CV_viz_parallel_coords(df_params, score_col='score', score_name=r'$f_\mathrm{eval}$', sort_by='min', standardize_color=True, order_cols_by=order_cols_by, ticks_fontsize=14) # score or recursive or list

    if s.save_plots:
        fig_CV_viz.savefig(f"{s.base_name_figures}CV_viz_HPO_{name}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.15)
    fig_CV_viz

# %%

# plot results from models

# DO NOT DO THIS!
# importlib.reload(Ext)

ylim = (0.10, 0.18) if house_type=='Ejerlejlighed' else (0.15, 0.24)

if make_ML_plots:

    plot_ml_model(  model_name=model_name_xgb, 
                    house_type=house_type,
            
                    X_all=X_all, 
                    y_all=y_all, 
                    X_train=X_train, 
                    y_train=y_train, 
                    X_test=X_test, 
                    y_test=y_test,

                    X_tight_train=X_tight_train, 
                    y_tight_train=y_tight_train, 
                    X_tight_test=X_tight_test, 
                    y_tight_test=y_tight_test,
                    
                    results_random=results_xgb_random, 
                    results_BO=results_xgb_BO, 
                    results=results_xgb,

                    model_random=model_xgb_random, 
                    model_BO=model_xgb_BO, 
                    model=model_xgb, 
                    model_base=model_base_xgb,
                    
                    do_early_stopping=s.do_early_stopping_xgb, 
                    n_sigma=dict_xgb_early['n_sigma'],

                    df_ML_time_sorted=df_ML_time_sorted, 
                    halflife_years=halflife_years,
                    kwargs_shap=s.kwargs_shap_xgb,
                    figsize_RS=(10, 4), figsize_BO=(10, 4), figsize_ES=(10, 4),
                    ylim_RS=ylim, ylim_BO=ylim, ylim_ES=ylim,
                    )
    plt.close('all')


#%%


# model_name = model_name_xgb+'_tight'
# X = X_tight_train
# # shap.dependence_plot("rank(1)", shap_values, X, interaction_index="Education-Num")

# import shap
# import shap_plot

# filename = f'{s.base_name}{model_name}_shap_values.npy'

# # shap_values = shap.TreeExplainer(model_xgb).shap_values(X)
# shap_values = np.load(filename)

# shap_values_normed = np.abs(shap_values).sum(axis=0)
# shap_values_normed /= shap_values_normed.sum()
# # df_tmp = pd.Series(shap_values_normed, X.columns)
# feature_names = [f"{col} ({val:.2%})".replace('EjdVurdering_', '') for col, val in zip(X.columns, shap_values_normed)]


# importlib.reload(shap_plot)

# plt.figure()
# shap_plot.summary_plot(shap_values, X, 
#                                   feature_names=feature_names,
#                                   max_display=17, 
#                                   plot_type='dot')

# fig, ax = plt.gcf(), plt.gca()
# fig

#%%


# %%



# %%

d = {'Train': (y_train, y_pred_xgb_train),
     'Test': (y_test, y_pred_xgb),
     '2019': (y_2019, y_pred_xgb_2019),
}

d_tight = {'Train': (y_tight_train, y_pred_xgb_tight_train),
           'Test': (y_tight_test, y_pred_xgb_tight),
           '2019': (y_tight_2019, y_pred_xgb_tight_2019),
    }

d_tight = {'Train': (y_tight_train, model_xgb.predict(X_tight_train)),
           'Test': (y_tight_test, model_xgb.predict(X_tight_test)),
           '2019': (y_tight_2019, model_xgb.predict(X_tight_2019)),
    }



def numbers_to_latex_num(x):
    return r"\num{" + f"{100*x:.2f}" + r"}"

def no_formatting(x):
    return rf"${x}$"

def mean_sdom(row):
    return rf"{row['mean']:.5f} \pm {row['SD_ofthe_mean']:.5f}"

def predictions_to_table(d):
    z_stats = {}
    for key, val in d.items():
        z_stats[key] = calc_z_stat(*val)

    df_z_stats_tmp = pd.DataFrame(z_stats).T

    df_z_stats = df_z_stats_tmp.loc[:, ['MAD', '5%', '10%', '20%']]
    df_tmp = df_z_stats_tmp.loc[:, ['mean', 'SD_ofthe_mean']]
    # df_z_stats.loc[:, r'$\mu$'] = df_tmp.apply(mean_sdom, axis=1)

    formatters = [numbers_to_latex_num]*(df_z_stats.shape[1]) #+ [no_formatting]
    s_z_stats = df_z_stats.to_latex(formatters=formatters, escape=False)

    s_z_stats = s_z_stats.replace(r'\toprule\n', '')
    s_z_stats = s_z_stats.replace(r'\bottomrule', '')
    s_z_stats = s_z_stats.replace(r"\n\n", r"\n")

    return df_z_stats, s_z_stats

df_z_stats, s_z_stats = predictions_to_table(d)
df_z_stats_tight, s_z_stats_tight = predictions_to_table(d_tight)

if True:
    print(s_z_stats)
    print(s_z_stats_tight)

x=x

#%%

#%%


if True:

    import shap_plot

    if False:
        # importlib.reload(shap_plot)
        locs = [24325621, 11255391]  # evtns [9294240, 4300301]

        for loc in locs:

            fig_shap, ax_shap = shap_plot.shap_plot(clf=d_mc[njet]["clf_lgb"],
                                                    X=d_mc[njet]["X_test"].loc[loc],
                                                    identifier=loc)

            s = (f"./figures/shap_values-"
                f"{Ext.settings_to_string(d_settings)}-njet={njet}"
                f"loc={loc}.pdf")
            fig_shap.savefig(s, dpi=600)
        fig_shap



    # clf = model_lgb

    is_xgb = True

    if is_xgb:
        clf = model_xgb.get_clf().get_booster()
    else:
        clf = model_lgb.get_clf().booster_
    
    loc = 489266 if house_type=='Ejerlejlighed' else  336709 # 321787 # 
    X = X_train.loc[loc]
    y = y_train.loc[loc] / 1e6
    w = df_weights.loc[loc]

    # X = X_train.iloc[:1000]
    # y = y_train.iloc[:1000] / 1e6
    # w = df_weights.iloc[:1000]

    # base_score

    # clf.predict(X)
    # model_xgb.get_clf().predict(X)

    # clf.predict(X) + base_score
    # model_xgb.get_clf().predict(X) + base_score
    # y


    # importlib.reload(shap_plot)
    # do_log10

    import shap
    import warnings


    def cut_off_columns(values, colnames, N_max_cols):

        df_shap = pd.Series(values, colnames)

        most_important_cols = list(df_shap.abs().sort_values(ascending=False).index)
        df_shap_important = df_shap.loc[most_important_cols[:N_max_cols]].sort_values(ascending=False).copy()

        mask_pos = df_shap.loc[most_important_cols[N_max_cols:]] > 0
        leftover_pos = df_shap.loc[most_important_cols[N_max_cols:]].loc[mask_pos].sum()

        mask_neg = df_shap.loc[most_important_cols[N_max_cols:]] < 0
        leftover_neg = df_shap.loc[most_important_cols[N_max_cols:]].loc[mask_neg].sum()

        position_pos = int(np.argmin(df_shap_important.values > 0))

        tmp = df_shap_important.to_frame().T

        tmp.insert(loc=position_pos, column='Overflow', value=leftover_pos)
        tmp.insert(position_pos+1, 'Underflow', leftover_neg)

        df_shap_important = tmp.iloc[0, :]
        return df_shap_important.values, list(df_shap_important.index)




    def make_shap_plot(clf, X, y, ax=None):

        # colnames = ['bias'] + X.columns.to_list()
        # colnames = ['bias'] + X.index.to_list()

        shap_explainer = shap.TreeExplainer(clf)
        
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            shap_values = shap_explainer.shap_values(X)[0] # [0] TODO regression 
        bias = shap_explainer.expected_value# [0] #TODO
        bias += base_score # TODO

        values = np.r_[bias, shap_values]

        if False:
            values = -values # XXX Notice this switch, now its predicting signal

        colnames = ['Bias'] + X.columns.to_list()

        # values = values[:10]
        # colnames = colnames[:10]

        N_max_cols = 10
        values, colnames = cut_off_columns(values, colnames, N_max_cols)
        colnames = [s.replace('EjdVurdering_', '') for s in colnames]


        importlib.reload(shap_plot)

        fig_shap, ax_shap = shap_plot.plot(index=colnames, 
                                data=values, 
                                title=f"SHAP plot for #{loc}", 
                                formatting = "{:,.2f}", 
                                green_color=colors[2], 
                                red_color=colors[1], 
                                blue_color=colors[0],
                                rotation_value=20,
                                figsize = (10,4), fontsize=12, 
                                ax=None, #ax,
                                ml_type='regression', 
                                net_label=r'$\hat{y}$',
                                xticks_fontsize=10,
                                truth=y, truth_pos=(10, 13) if house_type=='Villa' else (10, 7))

        fig_shap

        if s.save_plots:
            fig_shap.savefig(s.base_name_figures+f'SHAP_fig_loc={loc}.pdf')

    # fig_shap
        if ax is None:
            return fig_shap, ax_shap

    make_shap_plot(clf, X, y, ax=None)


#%%


#%% ===========================================================================
#    
#                          L i g h t G B M
#
# =============================================================================


print("\n----------------------------------------------------------------  ")
print("                          LightGBM")
print("---------------------------------------------------------------- \n ")

from extra_functions import ml_model_lgb
from lightgbm_wrapper import LGBMRegressor as LGB_wrapper

model_name_lgb = 'lgb'

pipe_lgb = Pipeline([('clf_lgb',   LGB_wrapper(n_estimators = 10_000,
                                               objective=obj_function,# obj_welsch_paper, obj_fair, obj_cauchy_paper, obj_logcos,
                                               learning_rate=0.1,
                                               n_jobs = cpu_n_jobs,
                                               init_score = base_score,
                                               silent = True,
                                               random_state = 42,
                                               verbose = -1, # to suppress warnings
                                               ))])

model_base_lgb = ml_model_lgb(pipe_lgb, do_log10, missing=-9999)


dict_lgb_random, dict_lgb_BO, dict_lgb_early = get_ml_params(  
                                            X_train, y_train, weights,
                                            s.n_jobs_lgb, 
                                            s.params_dict_lgb_random,
                                            s.params_dict_lgb_BO,
                                            N_iter, N_CV, N_iter_BO, N_init_BO,
                                            s.int_list_lgb,
                                            num_boost_round = 200_000,
                                            early_stopping_rounds = 10_000,
                                            use_weights = s.use_weights_lgb,
                                            )



#%%

(model_lgb_random, model_lgb_BO, model_lgb, 
results_lgb_random, results_lgb_BO, 
results_lgb) = load_ml_model( model_name_lgb, model_base_lgb,  
                              dict_lgb_random, dict_lgb_BO, dict_lgb_early, 
                              s.do_early_stopping_lgb)

(y_pred_lgb, y_pred_lgb_2019, 
 y_pred_lgb_train) = get_y_pred(model_lgb, model_name_lgb, 
                                                     X_test, X_2019, X_train)

if s.do_tight_refit:
    
    refit_params = [X_tight_train, 
                    y_tight_train, 
                    get_weights(X_tight_train, halflife_years)]
    
    (model_lgb_random_tight, model_lgb_BO_tight, 
     model_lgb_tight) = load_ml_model_tight(model_name_lgb, model_base_lgb, refit_params)

    (y_pred_lgb_tight, y_pred_lgb_tight_2019, 
     y_pred_lgb_tight_train) = get_y_pred(model_lgb_tight, model_name_lgb+'_tight', 
                                                         X_test, X_2019, X_train)



# %%

if make_ML_plots:

    # plot results from models
    plot_ml_model(    model_name_lgb, house_type,
                
                        X_all, y_all, X_train, y_train, X_test, y_test,
                        X_tight_train, y_tight_train, X_tight_test, y_tight_test,
                        
                        results_lgb_random, results_lgb_BO, results_lgb,
                        model_lgb_random, model_lgb_BO, model_lgb, model_base_lgb,
                        
                        s.do_early_stopping_lgb, dict_lgb_early['n_sigma'],
                        df_ML_time_sorted, halflife_years,
                        kwargs_shap=s.kwargs_shap_lgb
                        )

    plt.close('all')


# %%



#%% =============================================================================
# 
# =============================================================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler # Imputer
from sklearn.pipeline import Pipeline

imp = SimpleImputer(missing_values=-9999, strategy='median', verbose=0, copy="True")
scaler = RobustScaler(quantile_range=(25, 75))

#from extra_functions import MICE_imputer
#imp_mice = MICE_imputer(missing_values=-9999, n_imputations=3, n_burn_in=2)
##X2 = imp_mice.fit_transform(X)


#%% ===========================================================================
#    
#                          R i d g e
#
# =============================================================================


print("\n----------------------------------------------------------------  ")
print("                          Ridge")
print("---------------------------------------------------------------- \n ")

from sklearn.linear_model import Ridge

model_name_lin = 'lin'

pipe_lin = Pipeline([('imputer', imp), 
                     ('scaler', scaler), 
                     ('clf_lin', Ridge(max_iter=10_000, tol=1e-4, 
                                       random_state=42))
                     ])

model_base_lin = ml_model(pipe_lin, do_log10)



dict_lin_random, dict_lin_BO, dict_lin_early = get_ml_params(  
                                            X_train, y_train, weights,
                                            s.n_jobs_lin, 
                                            s.params_dict_lin_random,
                                            s.params_dict_lin_BO,
                                            N_iter, N_CV, N_iter_BO, N_init_BO,
                                            s.int_list_lin,
                                            use_weights = s.use_weights_lin,
                                            )


#%%

(model_lin_random, model_lin_BO, model_lin, 
results_lin_random, results_lin_BO, 
results_lin) = load_ml_model( model_name_lin, model_base_lin,  
                              dict_lin_random, dict_lin_BO, dict_lin_early, 
                              s.do_early_stopping_lin)

(y_pred_lin, y_pred_lin_2019, 
 y_pred_lin_train) = get_y_pred(model_lin, model_name_lin, 
                                                     X_test, X_2019, X_train)

if s.do_tight_refit:
    
    refit_params = [X_tight_train, 
                    y_tight_train, 
                    get_weights(X_tight_train, halflife_years)]
    
    (model_lin_random_tight, model_lin_BO_tight, 
     model_lin_tight) = load_ml_model_tight(model_name_lin, model_base_lin, refit_params)

    (y_pred_lin_tight, y_pred_lin_tight_2019, 
     y_pred_lin_tight_train) = get_y_pred(model_lin_tight, model_name_lin+'_tight', 
                                                         X_test, X_2019, X_train)

# %%

if make_ML_plots:

    # plot results from models
    plot_ml_model(    model_name_lin, house_type,
                    
                        X_all, y_all, X_train, y_train, X_test, y_test,
                        X_tight_train, y_tight_train, X_tight_test, y_tight_test,
                        
                        results_lin_random, results_lin_BO, results_lin,
                        model_lin_random, model_lin_BO, model_lin, model_base_lin,
                        
                        s.do_early_stopping_lin, dict_lin_early['n_sigma'],
                        df_ML_time_sorted, halflife_years,
                        kwargs_shap=s.kwargs_shap_lin
                        )

    plt.close('all')


#%% ===========================================================================
#    
#                     K   N e a r e s t  N e i g h b o r s 
#
# =============================================================================

print("\n----------------------------------------------------------------  ")
print("                          KNN")
print("---------------------------------------------------------------- \n ")

from sklearn.neighbors import KNeighborsRegressor

model_name_knn = 'knn'

pipe_knn = Pipeline([('imputer', imp), 
                     ('scaler', scaler), 
                     ('clf_knn', KNeighborsRegressor(n_jobs = cpu_n_jobs))])

model_base_knn = ml_model(pipe_knn, do_log10)


sample_frac_knn = s.sample_frac_knn if house_type=='Villa' else None

dict_knn_random, dict_knn_BO, dict_knn_early = get_ml_params(  
                                            X_train, y_train, weights, 
                                            s.n_jobs_knn, 
                                            s.params_dict_knn_random,
                                            s.params_dict_knn_BO,
                                            N_iter, N_CV, N_iter_BO, N_init_BO,
                                            s.int_list_knn,
                                            sample_frac = sample_frac_knn,
                                            use_weights = s.use_weights_knn,
                                            )


#%%

(model_knn_random, model_knn_BO, model_knn, 
results_knn_random, results_knn_BO, 
results_knn) = load_ml_model( model_name_knn, model_base_knn,  
                              dict_knn_random, dict_knn_BO, dict_knn_early, 
                              s.do_early_stopping_knn)
(y_pred_knn, y_pred_knn_2019, 
 y_pred_knn_train) = get_y_pred(model_knn, model_name_knn, 
                                                     X_test, X_2019, X_train)

if s.do_tight_refit:
    
    refit_params = [X_tight_train, 
                    y_tight_train, 
                    get_weights(X_tight_train, halflife_years)]
    
    (model_knn_random_tight, model_knn_BO_tight, 
     model_knn_tight) = load_ml_model_tight(model_name_knn, model_base_knn, refit_params)

    (y_pred_knn_tight, y_pred_knn_tight_2019, 
     y_pred_knn_tight_train) = get_y_pred(model_knn_tight, model_name_knn+'_tight', 
                                                         X_test, X_2019, X_train)


# %%

if make_ML_plots:

    # plot results from models
    plot_ml_model(    model_name_knn, house_type,
                    
                        X_all, y_all, X_train, y_train, X_test, y_test,
                        X_tight_train, y_tight_train, X_tight_test, y_tight_test,
                        
                        results_knn_random, results_knn_BO, results_knn,
                        model_knn_random, model_knn_BO, model_knn, model_base_knn,
                        
                        s.do_early_stopping_knn, dict_knn_early['n_sigma'],
                        df_ML_time_sorted, halflife_years,
                        kwargs_shap=s.kwargs_shap_knn
                        )
    plt.close('all')

#%% =============================================================================
#    
#              S u p p o r t   V e c t o r   R e g r e s s i o n
#
# =============================================================================

if s.house_type.lower() != 'villa':
    
    print("\n----------------------------------------------------------------  ")
    print("                          SVR")
    print("---------------------------------------------------------------- \n ")
    
    from sklearn.svm import SVR
    from sklearn.ensemble import BaggingRegressor
    
    model_name_svr = 'svr'
    
    """ Train SVR by bagging to make fitting time more efficient by using all cores """
    svr = BaggingRegressor(SVR(kernel='rbf'), verbose = 0, n_jobs = cpu_n_jobs, bootstrap = False,
                           max_samples = 1.0 / cpu_n_jobs, n_estimators = cpu_n_jobs )
    
    pipe_svr = Pipeline([('imputer', imp), 
                         ('scaler', scaler), 
                         ('clf_svr', svr)])
    
    model_base_svr = ml_model(pipe_svr, do_log10)
    
    
    dict_svr_random, dict_svr_BO, dict_svr_early = get_ml_params(  
                                                X_train, y_train, weights,
                                                s.n_jobs_svr, 
                                                s.params_dict_svr_random,
                                                s.params_dict_svr_BO,
                                                N_iter, N_CV, N_iter_BO, N_init_BO,
                                                s.int_list_svr,
                                                use_weights = s.use_weights_svr)
    
    
    #%%
    
    (model_svr_random, model_svr_BO, model_svr, 
    results_svr_random, results_svr_BO, 
    results_svr) = load_ml_model( model_name_svr, model_base_svr,  
                                  dict_svr_random, dict_svr_BO, dict_svr_early, 
                                  s.do_early_stopping_svr)
    (y_pred_svr, y_pred_svr_2019, 
     y_pred_svr_train) = get_y_pred(model_svr, model_name_svr, 
                                                         X_test, X_2019, X_train)
    
    if s.do_tight_refit:
        
        refit_params = [X_tight_train, 
                        y_tight_train, 
                        get_weights(X_tight_train, halflife_years)]
        
        (model_svr_random_tight, model_svr_BO_tight, 
         model_svr_tight) = load_ml_model_tight(model_name_svr, model_base_svr, refit_params)
    
        (y_pred_svr_tight, y_pred_svr_tight_2019, 
         y_pred_svr_tight_train) = get_y_pred(model_svr_tight, model_name_svr+'_tight', 
                                                             X_test, X_2019, X_train)
    
    
    # %%
    
    # plot results from models
    if s.is_cluster:
        plot_ml_model(    model_name_svr, house_type,
                      
                          X_all, y_all, X_train, y_train, X_test, y_test,
                          X_tight_train, y_tight_train, X_tight_test, y_tight_test,
                          
                          results_svr_random, results_svr_BO, results_svr,
                          model_svr_random, model_svr_BO, model_svr, model_base_svr,
                          
                          s.do_early_stopping_svr, dict_svr_early['n_sigma'],
                          df_ML_time_sorted, halflife_years,
                          kwargs_shap=s.kwargs_shap_svr,
                          )
        plt.close('all')

#%% ===========================================================================
#    
#                     R a n d o m   F o r e s t
#
# =============================================================================

# print("\n----------------------------------------------------------------  ")
# print("                          RandomForest")
# print("---------------------------------------------------------------- \n ")

# from sklearn.ensemble import RandomForestRegressor

# model_name_rf = 'rf'

# pipe_rf = Pipeline([('clf_rf', RandomForestRegressor(
#                                                     n_jobs = cpu_n_jobs, 
#                                                     verbose = 0, 
#                                                     bootstrap=False, 
# #                                                   oob_score = True,
#                                                     min_samples_leaf = 10,
#                                                     criterion = 'mae'))])

# model_base_rf = ml_model(pipe_rf, do_log10)

# sample_frac_rf = s.sample_frac_knn if house_type=='Villa' else None



# dict_rf_random, dict_rf_BO, dict_rf_early = get_ml_params(  
#                                             X_train, y_train, weights,
#                                             s.n_jobs_rf, 
#                                             s.params_dict_rf_random,
#                                             s.params_dict_rf_BO,
#                                             N_iter, N_CV, N_iter_BO, N_init_BO,
#                                             s.int_list_rf,
#                                             sample_frac = sample_frac_rf,
#                                             use_weights = s.use_weights_rf)


# #%%

# (model_rf_random, model_rf_BO, model_rf, 
# results_rf_random, results_rf_BO, 
# results_rf) = load_ml_model( model_name_rf, model_base_rf,  
#                               dict_rf_random, dict_rf_BO, dict_rf_early, 
#                               s.do_early_stopping_rf)
# (y_pred_rf, y_pred_rf_2019, 
#  y_pred_rf_train) = get_y_pred(model_rf, model_name_rf, 
#                                                      X_test, X_2019, X_train)

# if s.do_tight_refit:
    
#     refit_params = [X_tight_train, 
#                     y_tight_train, 
#                     get_weights(X_tight_train, halflife_years)]
    
#     (model_rf_random_tight, model_rf_BO_tight, 
#      model_rf_tight) = load_ml_model_tight(model_name_rf, model_base_rf, refit_params)

#     (y_pred_rf_tight, y_pred_rf_tight_2019, 
#      y_pred_rf_tight_train) = get_y_pred(model_rf_tight, model_name_rf+'_tight', 
#                                                          X_test, X_2019, X_train)


# # %%

# # plot results from models
# if s.is_cluster:
#     plot_ml_model(    model_name_rf, house_type,
                  
#                       X_all, y_all, X_train, y_train, X_test, y_test,
#                       X_tight_train, y_tight_train, X_tight_test, y_tight_test,
                      
#                       results_rf_random, results_rf_BO, results_rf,
#                       model_rf_random, model_rf_BO, model_rf, model_base_rf,
                      
#                       s.do_early_stopping_rf, dict_rf_early['n_sigma'],
#                       df_ML_time_sorted, halflife_years,
#                       kwargs_shap=s.kwargs_shap_rf,
#                       )
#     plt.close('all')





#%%
#    
#from pygam import LinearGAM
#
#from pygam.utils import generate_X_grid
#
#X = X_train
#y = y_train
#gam = LinearGAM(n_splines=50, lam=0.6).gridsearch(X, y/1e6)
#gam.summary()
#
#
#
#XX = generate_X_grid(gam)
#plt.rcParams['figure.figsize'] = (18, 10)
#fig, axs = plt.subplots(2, 5)
#titles = X.columns
#for i, ax in enumerate(axs.flatten()):
#    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
#    ax.plot(XX[:, i], pdep)
#    ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
#    ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
#    ax.set_title(titles[i],fontsize=12)
#plt.tight_layout()
#plt.show()
#
#
#fig, axs = plt.subplots(2, 5)
#for i, ax in enumerate(axs.flatten()):
#    i = i+10
#    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
#    ax.plot(XX[:, i], pdep)
#    ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
#    ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
#    ax.set_title(titles[i],fontsize=12)
#plt.tight_layout()
#plt.show()


#%% =============================================================================
#    
#                          M E T A    L E A R N E R
#
# =============================================================================

print("\n----------------------------------------------------------------  ")
print("                          META")
print("---------------------------------------------------------------- \n ")


from sklearn.linear_model import LinearRegression


#from xgboost import XGBRegressor
from extra_functions import super_learner_class

if house_type.lower() == 'villa':
    ensemble_list = [('lin', model_lin), ('knn', model_knn), 
                     ('xgb', model_xgb), 
                     ('lgb', model_lgb)
                     ]
else:
    ensemble_list = [('lin', model_lin), ('knn', model_knn), ('svr', model_svr),
                     ('xgb', model_xgb), 
                     ('lgb', model_lgb),
                     ]
        
base_meta_xgb = XGB_wrapper(n_estimators = 1000,
                       n_jobs = cpu_n_jobs, 
                       learning_rate = 0.1,
                       objective = obj_function,
                       base_score = base_score,
                    #    missing = -9999,
                       random_state = 42,
                       silent = True, 
                       )

base_meta_lin = LinearRegression(fit_intercept=False, normalize=False)

meta_xgb = super_learner_class(ensemble_list, Pipeline([('clf_meta', base_meta_xgb)]))
meta_lin = super_learner_class(ensemble_list, Pipeline([('clf_meta', base_meta_lin)]))

filename_meta = s.base_name+f'model_meta'
filename_meta_lin = s.base_name+f'model_meta_lin'


if this_should_run(filename_meta, s.force_rerun, ignore_quick_run=True):    
    
    print("Fitting meta model")
    if s.is_cluster:
        N_CV_Z = 10
        meta_xgb.create_Z(X_train, y_train, weights, N_CV_Z)
        meta_xgb.save_model(filename_meta+'_Z')
        meta_xgb.fit_meta()

        meta_lin.set_Z(*meta_xgb.get_Z())
        meta_lin.fit_meta()
        
    else:

        N_CV_Z = 3
        N = 1000
        meta_xgb.create_Z(X_train.iloc[:N, :], y_train.iloc[:N], weights[:N], N_CV_Z)
        meta_xgb.fit_meta()
        # meta_xgb.fit(X_train.iloc[:N, :], y_train[:N], weights[:N], N_CV_Z)
    
    meta_xgb.save_model(filename_meta)
    meta_lin.save_model(filename_meta_lin)
else:
    meta_xgb.load_model(filename_meta)
    meta_lin.load_model(filename_meta_lin)


Z = meta_xgb.Z
Z_values = Z.values

y_train_values = y_train.values

from numba import njit, prange
from scipy.stats import norm as Gaussian
c = Gaussian.ppf(3/4.)

@njit
def least_squares_np(par):  # par is a numpy array here 
    # alpha = np.array(par)
    alpha = par
    y_Z_tmp = np.dot(Z_values, alpha) # / len(alpha)
    z_Z_tmp = (y_Z_tmp - y_train_values) / y_train_values
    return np.median(np.abs(z_Z_tmp)) / c



if False:

    M = len(Z.columns)
    alpha = np.ones(len(Z.columns))
    y_Z_tmp = Z.values @ alpha / M
    z_Z_tmp = (y_Z_tmp - y_train) / y_train
    calc_MAD(y_train, y_Z_tmp)

    
    


    from numba import njit, prange


    # @njit(parallel=False)
    # def para(P):
    #     s = 0
    #     val = 1
    #     res = np.array([1, 1, 1, 1, 1]) / 5
    #     # Without "parallel=True" in the jit-decorator
    #     # the prange statement is equivalent to range
    #     res = 0
    #     for i in prange(P.shape[0]):
    #         val_i = least_squares_np(P[i, :])
    #         if val_i < val:
    #             val = val_i
    #             # res = P[i, :]
    #     return res, val

    y_train_values = y_train.values.reshape((-1, 1))

    def least_squares_np_para(par):  # par is a numpy array here 
        # alpha = np.array(par)
        alpha = par #/ par.sum(axis=1).reshape((-1, 1))
        y_Z_tmp = np.dot(Z_values, alpha.T) # / len(alpha)
        z_Z_tmp = (y_Z_tmp - y_train_values) / y_train_values
        return np.median(np.abs(z_Z_tmp), axis=0) / c

    P = np.random.uniform(0, 1.2, 5*100).reshape((-1, 5)) / 5

    slice_0 = slice(0.0, 0.1, 0.01)
    slice_1 = slice(0.0, 0.1, 0.01)
    slice_2 = slice(0.0, 0.1, 0.01)
    slice_3 = slice(0.0, 1, 0.1)
    slice_4 = slice(0.0, 1, 0.1)

    if house_type == 'Villa':
        xy = np.mgrid[slice_0, slice_1, slice_3, slice_4].reshape(4,-1).T
    else:
        xy = np.mgrid[slice_0, slice_1, slice_2, slice_3, slice_4].reshape(5,-1).T
    
    print(xy.shape)
    val = least_squares_np_para(xy)
    print(val.min(), val.argmin(), xy[val.argmin()])
    par0 = xy[val.argmin()]

else:
    if house_type == 'Villa':
        par0 = np.array([0.002, 0, 0.813, 0.2])
    else:    
        par0 = np.array([0.002, 0, 0, 0.813, 0.2 ])

if False:
    par0 = meta_lin.meta.steps[0][1].coef_
    par0 = np.ones(len(Z.columns)) / len(Z.columns)
    par0 = np.array([0.01, 0.01, 0.01, 0.96, 0.01])
    par0 = np.array([0.0, 0.0, 0.0, 0.82, 0.2])
# meta_lin.meta.steps[0][1].intercept_*100

# m = Minuit.from_array_func(least_squares_np, p0, error=0.1, errordef=1)
# m.get_param_states()

from iminuit import minimize  # has same interface as scipy.optimize.minimize
res = minimize(least_squares_np, par0)
res

coef = res.x
coef_ = meta_lin.meta.steps[0][1].coef_

print(coef)
print(coef_)

y_pred_meta_lin = meta_lin.predict(X_test)


meta_lin.meta.steps[0][1].coef_ = coef

# meta_lin.meta.steps[0][1].coef_ = coef / np.sum(coef)
# y_pred_meta_lin3 = meta_lin.predict(X_test)

y_pred_meta_lin2 = meta_lin.predict(X_test)
y_pred_meta_2019_lin2 = meta_lin.predict(X_2019)


print(y_pred_meta_lin)
print(y_pred_meta_lin2)



if False:
    N_runs = 100

    np.random.seed(42)
    from tqdm import tqdm
    fun = 1
    res = 1
    for i in tqdm(range(N_runs)):
        p0 = np.random.uniform(0, 1.2, 5) / 5
        res_i = minimize(least_squares_np, p0)
        if res_i.fun < fun:
            res = res_i


filename_meta_pred = f'{s.base_name}y_pred_meta.npy'
filename_meta_pred_2019 = f'{s.base_name}y_pred_meta_2019.npy'
if not path.isfile(filename_meta_pred_2019):
    y_pred_meta = meta_xgb.predict(X_test)
    y_pred_meta_2019 = meta_xgb.predict(X_2019)
    
    np.save(filename_meta_pred, y_pred_meta)
    np.save(filename_meta_pred_2019, y_pred_meta_2019)
else:
    y_pred_meta = np.load(filename_meta_pred)
    y_pred_meta_2019 = np.load(filename_meta_pred_2019)

print("Meta results:")
print(calc_MAD(y_test, y_pred_xgb))
print(calc_MAD(y_test, y_pred_meta))
print(calc_MAD(y_test, y_pred_meta_lin))
print(calc_MAD(y_test, y_pred_meta_lin2))
# print(calc_MAD(y_test, y_pred_meta_lin3))


print("2019:")
print(calc_MAD(y_2019, y_pred_xgb_2019))
print(calc_MAD(y_2019, y_pred_meta_2019))
print(calc_MAD(y_2019, y_pred_meta_2019_lin2))


print("\n\n FINISHED RUNNING!!!") 


# %%

if house_type == 'Villa':
    d_y_preds = {'XGB': y_pred_xgb, 
                'LGB': y_pred_lgb, 
                'LIN': y_pred_lin, 
                'KNN': y_pred_knn, 
                # 'SVR': y_pred_svr, 
                # 'ENS': y_pred_meta,
                'ENS': y_pred_meta_lin2,
                }

else:
    d_y_preds = {'XGB': y_pred_xgb, 
                 'LGB': y_pred_lgb, 
                 'LIN': y_pred_lin, 
                 'KNN': y_pred_knn, 
                 'SVR': y_pred_svr, 
                #  'ENS': y_pred_meta,
                 'ENS': y_pred_meta_lin2,
                }

plot_multiple_models(d_y_preds, y_test, xlim = (-1, 1.5))

# %%



if False:

    import shap_plot

    clf = meta_xgb.meta.steps[0][1].get_booster()
    
    X = meta_xgb.Z / 1e6
    # X = X_train.loc[loc]
    y = y_train / 1e6
    w = weights

    shap_explainer = shap.TreeExplainer(clf)
    shap_values = shap_explainer.shap_values(X)

    shap_values_normed = np.abs(shap_values).sum(axis=0)
    shap_values_normed /= shap_values_normed.sum()
    # df_tmp = pd.Series(shap_values_normed, X.columns)
    feature_names = [f"{col.upper()} ({val:.2%})" for col, val in zip(X.columns, shap_values_normed)]

    plt.figure(figsize=(8, 14))
    shap.summary_plot(shap_values, X, 
                        feature_names=feature_names,
                        max_display=len(X.columns), 
                        plot_type='dot', 
                    #   plot_size=0.5,
                        )
    fig_shap_summary, ax_shap_summary = plt.gcf(), plt.gca()
    fig_shap_summary.tight_layout()
    if s.save_plots:
        fig_shap_summary.savefig(s.base_name_figures+f'SHAP_meta_summary.pdf', dpi=600)


    N = 72445 # biggest std
    N = 92189 # smallest min
    # loc = 489266 if house_type=='Ejerlejlighed' else  336709 # 321787 # 
    X = meta_xgb.Z.iloc[N, :] / 1e6
    # X = X_train.loc[loc]
    y = y_train.iloc[N] / 1e6
    w = weights[N]

    # X = X_train.iloc[:1000]
    # y = y_train.iloc[:1000] / 1e6
    # w = df_weights.iloc[:1000]

    # base_score

    # clf.predict(X)
    # model_xgb.get_clf().predict(X)

    # clf.predict(X) + base_score
    # model_xgb.get_clf().predict(X) + base_score
    # y


    # importlib.reload(shap_plot)
    # do_log10

    import shap
    import warnings

    def make_shap_plot(clf, X, y, ax=None):

        # colnames = ['bias'] + X.columns.to_list()
        # colnames = ['bias'] + X.index.to_list()

        shap_explainer = shap.TreeExplainer(clf)
        
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            shap_values = shap_explainer.shap_values(X)[0] # [0] TODO regression 
        bias = shap_explainer.expected_value# [0] #TODO
        # bias += base_score # TODO

        values = np.r_[bias, shap_values]

        colnames = ['Bias'] + X.columns.to_list()
        colnames = ['Bias', 'Lin', 'KNN', 'SVR', 'XGB', 'LGB']

        # values = values[:10]
        # colnames = colnames[:10]

        # N_max_cols = 10
        # values, colnames = cut_off_columns(values, colnames, N_max_cols)
        # colnames = [s.replace('EjdVurdering_', '') for s in colnames]


        importlib.reload(shap_plot)

        fig_shap, ax_shap = shap_plot.plot(index=colnames, 
                                data=values, 
                                title=f"SHAP plot for loc={N}", 
                                formatting = "{:,.2f}", 
                                green_color=colors[2], 
                                red_color=colors[1], 
                                blue_color=colors[0],
                                rotation_value=0,
                                figsize = (10,4), fontsize=14, 
                                ax=None, #ax,
                                ml_type='regression', 
                                net_label=r'$\hat{y}$',
                                xticks_fontsize=14,
                                truth=y, truth_pos=(5, 3) if house_type=='Villa' else (4.8, 0.5),
                                displace_x_ticks=False)

        fig_shap

        if s.save_plots:
            fig_shap.savefig(s.base_name_figures+f'SHAP_fig_loc={loc}_meta.pdf')

    # fig_shap
        if ax is None:
            return fig_shap, ax_shap

    make_shap_plot(clf, X, y, ax=None)


# %%
