import numpy as np
import joblib
import pandas as pd
from memory_profiler import profile
import h5py
from tqdm import tqdm

filename = 'Data_animation/N_tot__5800000__N_init__100__N_ages__1__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__1.0__rho__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.0__frac_02__0.0__age_mixing__1.0__algo__2__ID__000.animation.joblib'

# i_day = 50

# fp=open('memory_profiler.log','w+')

# # @profile(stream=fp)
# def load(filename):
#     # return joblib.load(filename, mmap_mode='r')
#     return joblib.load(filename)

# @profile(stream=fp)
def to_df(which_state, coordinates, N_connections, i_day):
    df = pd.DataFrame(coordinates, columns=['x', 'y'])
    df['which_state_num'] = which_state[i_day]
    df['N_connections_num'] = N_connections[i_day]
    return df


# # @profile(stream=fp)
# def main(filename, i_day):
#     df = to_df(*load(filename), i_day)
#     return df

# def main():
#     # which_state, coordinates, N_connections = joblib.load(filename, mmap_mode='r')
#     which_state, coordinates, N_connections = joblib.load(filename)
#     N = len(which_state)
#     for i_day in tqdm(range(N)):
#         df = to_df(which_state, coordinates, N_connections, i_day)
#     print(i_day, df)

# main()

# which_state, coordinates, N_connections = load(filename)

which_state, coordinates, N_connections = joblib.load(filename)

f = h5py.File("test.hdf5")

dset = f.create_dataset("which_state", data=which_state, dtype='int8')
# dset.attrs

# dset = f.create_dataset("Images2", which_state.shape, dtype='int8', chunks=True)
# dset.chunks

with h5py.File("mytestfile.hdf5", "w") as f:
    f.create_dataset("which_state", data=which_state, dtype='int8')
    f.create_dataset("coordinates", data=coordinates, dtype='float64')
    f.create_dataset("N_connections", data=N_connections, dtype='int32')
    f.attrs

f = h5py.File('test.hdf5', 'r')

list(f.keys())
dset = f['which_state']

dset.shape
dset.dtype

dset[0, :]
dset[-1, :]

# print("Finished")