import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt
from numba import njit


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


filename = 'Data/DW_NBI_2019_09_03.csv'
cols = ['Sag_GisX_WGS84', 'Sag_GisY_WGS84']

df = pd.read_csv(filename, delimiter=',', usecols=cols)
df = df.dropna().drop_duplicates()

# df.to_csv('housing.csv', index=False)
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

haversine(*df.iloc[0:2], *df.iloc[1:3])