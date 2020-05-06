import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt
from numba import njit#, set_num_threads
import numba
from numba.types import float32, float64, intp
import matplotlib.pyplot as plt

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


filename = 'Data/GPS_coordinates.csv'

if Path(filename).exists():
    print("Loading dataframe from file.")
    df = pd.read_csv(filename)
else:
    print("Reading original Housing Data")
    filename_org_data = 'Data/DW_NBI_2019_09_03.csv'
    cols = ['Sag_GisX_WGS84', 'Sag_GisY_WGS84']
    df = pd.read_csv(filename_org_data, 
                     delimiter=',', 
                     usecols=cols)
    df.columns = ['x', 'y']
    df = df.dropna().drop_duplicates()

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(df['x'], df['y'], s=1)

    df = df.query("(8<= x <= 13) & (54 <= y <= 58)")
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(df['x'], df['y'], s=1)

    df = df.sample(frac=1, random_state=42)

    print(f"Saving Housing Data to {filename}")
    df.to_csv(filename, index=False)
    np.save(filename.replace('csv', 'npy'), df.values)

data = df.values

np.random.seed(42)
np.random.shuffle(data)


# %%

import datashader as ds
import datashader.transfer_functions as tf

def df_to_fig(df, plot_width=1000, plot_height=1000):

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                       x_range=[8, 15.5], 
                       y_range=[54.5, 57.8],
                       x_axis_type='linear', y_axis_type='linear',
                    )
    agg = canvas.points(df, 'x', 'y') # ds.count_cat('SK')

    # color_key = color_key_b_c_uds_g
    # img = tf.shade(agg, color_key=color_key, how='log') # eq_hist
    img = tf.shade(agg, how='eq_hist') # eq_hist color_key=color_key
    # spread = tf.dynspread(img, threshold=0.9, max_px=1)
    return img

df_to_fig(df)

# %%

# coords = data[:1_000]

haversine(*data[0], *data[1])
haversine_2_inputs(data[0], data[1])


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
    indices = np.zeros((N_r, 2), np.uint32)
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            indices[k] = (i, j)
            k += 1
    return indices

@njit(parallel=True)
def calc_dist(indices, coords, start, stop, do_log): 
    L = stop - start
    r_loop = np.zeros(L, float32)

    for k in prange(L):
        i, j = indices[k+start]
        dist = haversine_2_inputs(coords[i], coords[j])
        if do_log:
            dist = np.log10(dist)
        r_loop[k] = dist
    return r_loop

@njit
def numba_hist(x, N_bins, r):
    return np.histogram(x, N_bins, r)


@measure_time
# @njit(parallel=False)
def pairwise_dist(coords, N_bins=100, max_GB=1, xminlog=-3, xmaxlog=3):

    N = len(coords)
    N_r = int(N*(N-1) / 2)
    indices = get_triangular_indices(N)
    N_r_max = int(max_GB * 1000 * 1000 * 1000 / 8)
    # N_r_max = 100

    N_loops = int(np.ceil(N_r / N_r_max))
    H_all = np.zeros((N_loops, N_bins), np.int_)

    # print(N_loops)

    i_counter = 0
    for i_loop in tqdm(range(N_loops)):
        # print(i_loop, N_loops)

        start = i_counter
        if i_loop != N_loops-1: # if not at last loop
            stop = i_counter + N_r_max
        else:
            stop = N_r
        
        r_loop = calc_dist(indices, coords, start, stop, do_log=True)
        i_counter += len(r_loop)

        H = numba_hist(r_loop, N_bins, (xminlog, xmaxlog))[0]
        H_all[i_loop] = H

    H_out = np.sum(H_all, axis=0) 

    return H_out

N = 20_000

x=x


# coords = data[:N]

print("Blabla", flush=True)
pairwise_dist(data[:N], max_GB=1)



print("Blabla", flush=True)
pairwise_dist(data[:N], max_GB=2)


x=x

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

# from sys import getsizeof

# @njit
# def return_r_32(N):
#     N_r = int(N*(N-1) / 2)
#     r = np.zeros(N_r, float32)
#     for i in range(N_r):
#         r[i] = i*2.2*i
#     return r

# @njit
# def return_r_64(N):
#     N_r = int(N*(N-1) / 2)
#     r = np.zeros(N_r, float64)
#     for i in range(N_r):
#         r[i] = i*2.2*i
#     return r

# # N = 100_000
# # N = len(data)
# # r_32 = return_r_32(N)
# # r_64 = return_r_64(N)

# # print(r_32.nbytes)
# # print(r_64.nbytes)


# def calc_size_in_gb(N, m):
#     N_r = int(N*(N-1) / 2)
#     if m == '32':
#         m = 4
#     elif m == '64': 
#         m = 8
#     return N_r * m / 1000 / 1000 / 1000


# print(calc_size_in_gb(N, '32'))
# print(calc_size_in_gb(N, '64'))

#%%

# a = [0] * 1024
# b = np.array(a)
# getsizeof(a)

#%%


# @njit
# def get_bin_edges(a, bins):
#     bin_edges = np.zeros(bins+1)
#     a_min = a.min()
#     a_max = a.max()
#     delta = (a_max - a_min) / bins
#     for i in range(bin_edges.shape[0]):
#         bin_edges[i] = a_min + i * delta

#     bin_edges[-1] = a_max  # Avoid roundoff error on last point
#     return bin_edges


# @njit
# def compute_bin(x, bin_edges):
#     # assuming uniform bins for now
#     n = bin_edges.shape[0] - 1
#     a_min = bin_edges[0]
#     a_max = bin_edges[-1]

#     # special case to mirror NumPy behavior for last bin
#     if x == a_max:
#         return n - 1 # a_max always in last bin

#     bin = int(n * (x - a_min) / (a_max - a_min))

#     if bin < 0 or bin >= n:
#         return None
#     else:
#         return bin




# @njit
# def numba_histogram(a, bins):
#     hist = np.zeros(bins, np.int_)
#     bin_edges = get_bin_edges(a, bins)

#     for x in a.flat:
#         bin = compute_bin(x, bin_edges)
#         if bin is not None:
#             hist[int(bin)] += 1

#     return hist, bin_edges

#%%

import fast_histogram


#%%

@measure_time
@njit
def bin_pairwise_dists(coords, N_bins):
    bins = np.linspace(0, 1000, N_bins+1)
    r_dists = pairwise_dist(coords)
    counts, bin_edges = numba_histogram(r_dists, bins)
    return counts, bin_edges, bins

N_bins = 100_000
N = 10_000
# N = len(data)

print(f"Computing distances between {N} coordinates, please wait.", flush=True)

@measure_time
def bin_pairwise_dists_fast_histogram(coords, N_bins):
    # bins = np.linspace(0, 1000, N_bins+1)
    r_dists = pairwise_dist(coords)
    H = fast_histogram.histogram1d(r_dists, bins=N_bins, range=(0, 1000))
    return H


@njit
def hist1d(v,b,r):
    return np.histogram(v, b, r)[0]

@measure_time
def bin_pairwise_dists_numba(coords, N_bins):
    # bins = np.linspace(0, 1000, N_bins+1)
    r_dists = pairwise_dist(coords)
    # H = fast_histogram.histogram1d(r_dists, bins=N_bins, range=(0, 1000))
    H = hist1d(r_dists, N_bins, (0, 1000))
    return H

print("fast_histogram")
H = bin_pairwise_dists_fast_histogram(data[:N], N_bins)
print("numba")
H = bin_pairwise_dists_numba(data[:N], N_bins)


print(f"Saving Counts.", flush=True)
np.save(f'./Data/H_N_{N}.npy', H)


print("Finished!")

# %%

bins = np.linspace(0, 1000, N_bins+1)
bin_edges = (bins[1:] + bins[:-1]) / 2
H = np.load('Data/H_N_100000.npy')

#%%

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(bin_edges, H, '-')
ax.set(title=f'Distribution of (all) distances between {N} randomly sampled houses/apartments', xlabel='km', ylabel='Counts', yscale='log')

fig.savefig('Figures/House_distribution.pdf')

# %%
