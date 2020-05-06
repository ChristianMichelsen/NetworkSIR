import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt
from numba import njit#, set_num_threads
import numba

# numba.set_num_threads(7)

#%%

@njit
def haversine_2_inputs(lat_lon1, lat_lon2):
    lat1, lon1, lat2, lon2 = lat_lon1[0], lat_lon1[1], lat_lon2[0], lat_lon2[1]
    return haversine(lat1, lon1, lat2, lon2)


@njit
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = 6371 * c
    return km

#%%

filename = 'Data/DW_NBI_2019_09_03.csv'
cols = ['Sag_GisX_WGS84', 'Sag_GisY_WGS84']

df = pd.read_csv(filename, delimiter=',', usecols=cols)
df = df.dropna().drop_duplicates()
# df.to_csv('Data/GPS_coordinates.csv', index=False)

data = df.values
np.save('Data/GPS_coordinates.npy', data)

# %%

import datashader as ds
import datashader.transfer_functions as tf

def df_to_fig(df, plot_width=1000, plot_height=1000):

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                       x_range=[8, 15.5], 
                       y_range=[54.5, 57.8],
                       x_axis_type='linear', y_axis_type='linear',
                    )
    agg = canvas.points(df, 'Sag_GisX_WGS84', 'Sag_GisY_WGS84') # ds.count_cat('SK')

    # color_key = color_key_b_c_uds_g
    # img = tf.shade(agg, color_key=color_key, how='log') # eq_hist
    img = tf.shade(agg, how='eq_hist') # eq_hist color_key=color_key
    # spread = tf.dynspread(img, threshold=0.9, max_px=1)
    return img

df_to_fig(df)

# %%

coords = data[:1_000]

haversine(*coords[0], *coords[1])

haversine_2_inputs(coords[0], coords[1])


# %%

from scipy.spatial import distance
# %timeit distance.cdist(coords, coords, haversine_2_inputs)


#%%

from functools import wraps
from time import time
def measure_time(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_/1000 if end_ > 0 else 0} s")
    return _time_it


#%%

from numba import njit, prange

@njit
def get_triangular_indices(N):
    N_r = int(N*(N-1) / 2)
    indices = np.zeros((N_r, 2), np.int_)
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            indices[k] = (i, j)
            k += 1
    return indices

@measure_time
@njit(parallel=True)
def pairwise_dist(coords):
    N = len(coords)
    N_r = int(N*(N-1) / 2)
    indices = get_triangular_indices(N)
    r = np.zeros(N_r)
    for k in prange(len(indices)):
        i, j = indices[k]
        r[k] = haversine_2_inputs(coords[i], coords[j])
    return r


# @njit(parallel=True)
# def pairwise_dist(coords):
#     N = len(coords)
#     N_r = int(N*(N-1) / 2)
#     r = np.zeros(N_r)
#     k = 0
#     for i in prange(N):
#         for j in range(i+1, N):
#             r[k] = haversine_2_inputs(coords[i], coords[j])
#             k += 1
#     return r


#%%

N = 20_000
N = len(data)


print(f"Computing distances between {N} coordinates, please wait.", flush=True)

# r = pairwise_dist(data[:10])
r_dists = pairwise_dist(data[:N])
r_dists


print(f"Saving distances.", flush=True)

np.save(f'./Data/r_dists_N_{N}.npy', r_dists)

