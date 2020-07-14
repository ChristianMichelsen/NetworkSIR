import numpy as np
import multiprocessing as mp
from pathlib import Path

import numba as nb
from numba import njit, prange, objmode, typeof # conda install -c numba/label/dev numba
from numba.typed import List, Dict

import simulation_utils
from simulation_utils import INTEGER_SIMULATION_PARAMETERS


def _is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

is_ipython = _is_ipython()


import platform
def is_local_computer(N_local_cores=8):
    if mp.cpu_count() <= N_local_cores and platform.system() == 'Darwin':
        return True
    else:
        return False


def get_num_cores(num_cores_max=None):
    num_cores = mp.cpu_count() - 1
    if num_cores_max and num_cores >= num_cores_max:
        num_cores = num_cores_max
    return num_cores


def delete_file(filename):
    try:
        Path(filename).unlink()
    except FileNotFoundError:
        pass

def make_sure_folder_exist(filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)



@njit
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a)) # [km]

@njit
def haversine_scipy(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    return haversine(lon1, lat1, lon2, lat2)


# def dict_to_filename_with_dir(cfg, ID, data_dir='ABN'):
#     filename = Path('Data') / data_dir
#     file_string = ''
#     for key, val in cfg.items():
#         file_string += f"{key}__{val}__"
#     file_string = file_string[:-2] # remove trailing _
#     filename = filename / file_string
#     file_string += f"__ID__{ID:03d}.csv"
#     filename = filename / file_string
#     return str(filename)


# def filename_to_dict(filename, normal_string=False, animation=False): # ,
#     cfg = {}

#     if normal_string:
#         keyvals = filename.split('__')
#     elif animation:
#         keyvals = filename.split('/')[-1].split('.animation')[0].split('__')
#     else:
#         keyvals = str(Path(filename).stem).split('__')

#     keyvals_chunks = [keyvals[i:i + 2] for i in range(0, len(keyvals), 2)]
#     for key, val in keyvals_chunks:
#         if not key == 'ID':
#             if key in INTEGER_SIMULATION_PARAMETERS:
#                 cfg[key] = int(val)
#             else:
#                 cfg[key] = float(val)
#     return DotDict(cfg)


@njit
def initialize_nested_lists(N, dtype):
    nested_list = List()
    for i in range(N): # prange
        tmp = List()
        tmp.append(dtype(-1))
        nested_list.append(tmp)
        nested_list[-1].pop(0) # trick to tell compiler which dtype
    return nested_list

@njit
def initialize_list_set(N, dtype):
    return [initialize_empty_set(dtype=dtype) for _ in range(N)]


@njit
def get_size_gb(x):
    return x.size * x.itemsize / 10**9


#%%
# Counters in Numba
from numba import types
from numba.experimental import jitclass

@jitclass({'_counter': types.DictType(types.int32, types.uint16)})
class Counter_int32_uint16():
    def __init__(self):
        self._counter = Dict.empty(key_type=types.int32, value_type=types.uint16)

    def _check_key(self, key):
        if not key in self._counter:
            self._counter[key] = 0

    def __getitem__(self, key):
        self._check_key(key)
        return self._counter[key]

    def __setitem__(self, key, val):
        self._check_key(key)
        self._counter[key] = val

    @property
    def d(self):
        return self._counter


@jitclass({'_counter': types.DictType(types.uint16, types.uint32)})
class Counter_uint16_uint32():
    def __init__(self):
        self._counter = Dict.empty(key_type=types.uint16, value_type=types.uint32)

    def _check_key(self, key):
        if not key in self._counter:
            self._counter[key] = 0

    def __getitem__(self, key):
        self._check_key(key)
        return self._counter[key]

    def __setitem__(self, key, val):
        self._check_key(key)
        self._counter[key] = val

    @property
    def d(self):
        return self._counter



@njit
def list_of_counters_to_numpy_array(counter_list, dtype=np.uint32):

    N = len(counter_list) # number of "days" in the list
    M = max([max(c.keys()) for c in counter_list]) # maximum key in the list

    res = np.zeros((N, M+1), dtype=dtype)
    for i_day in range(N):
        for key, val in counter_list[i_day].items():
            res[i_day, key] = val
    return res


@njit
def array_to_counter(arr):
    counter = Counter_uint16_uint32()
    for a in arr:
        counter[a] += 1
    return counter.d

#%%
# Cumulative Sums in numba

@njit
def numba_cumsum_2D(x, axis):
    y = np.zeros_like(x)
    n, m = np.shape(x)
    if axis==1:
        for i in range(n):
            y[i, :] = np.cumsum(x[i, :])
    elif axis==0:
        for j in range(m):
            y[:, j] = np.cumsum(x[:, j])
    return y

@njit
def numba_cumsum_3D(x, axis):
    y = np.zeros_like(x)
    n, m, p = np.shape(x)

    if axis==2:
        for i in range(n):
            for j in range(m):
                y[i, j, :] = np.cumsum(x[i, j, :])
    elif axis==1:
        for i in range(n):
            for k in range(p):
                y[i, :, k] = np.cumsum(x[i, :, k])
    elif axis==0:
        for j in range(m):
            for k in range(p):
                y[:, j, k] = np.cumsum(x[:, j, k])
    return y

from numba import generated_jit, types

# overload
# https://jcristharif.com/numba-overload.html

@generated_jit(nopython=True)
def numba_cumsum_shape(x, axis):
    if x.ndim == 1:
        return lambda x, axis: np.cumsum(x)
    elif x.ndim == 2:
        return lambda x, axis: numba_cumsum_2D(x, axis=axis)
    elif x.ndim == 3:
        return lambda x, axis: numba_cumsum_3D(x, axis=axis)

@njit
def numba_cumsum(x, axis=None):
    if axis is None and x.ndim != 1:
        print("numba_cumsum was used without any axis keyword set. Continuing using axis=0.")
        axis = 0
    return numba_cumsum_shape(x, axis)

#%%
# DotDict

class DotDict(dict):
    """
    Class that allows a dict to indexed using dot-notation.
    Example:
    >>> dotdict = DotDict({'first_name': 'Christian', 'last_name': 'Michelsen'})
    >>> dotdict.last_name
    'Michelsen'
    """

    def __getattr__(self, item):
        if item in self:
            return self.get(item)
        raise KeyError(f"'{item}' not in dict")

    def __setattr__(self, key, value):
        if not '__' in key:
            raise KeyError("Not allowed to change keys with dot notation, use brackets instead.")

    # make class pickle-able
    def __getstate__(self):
        return self.__dict__

    # make class pickle-able
    def __setstate__(self, state):
        self.__dict__ = state

#%%


class Filename:
    def __init__(self, filename):
        self.filename = filename
        self.d = self._string_to_dict()

    def __repr__(self):
        return str(self.d)

    def _string_to_dict(self):
        d = {}
        keyvals = str(Path(self.filename).stem).split('__')
        keyvals_chunks = [keyvals[i:i + 2] for i in range(0, len(keyvals), 2)]
        for key, val in keyvals_chunks:
            if key in INTEGER_SIMULATION_PARAMETERS + ['ID']:
                d[key] = int(val)
            else:
                d[key] = float(val)
        return DotDict(d)

    def to_dict(self): # ,
        return DotDict({key: val for key, val in self.d.items() if key != 'ID'})

    @property
    def simulation_parameters(self):
        return self.to_dict()

    def to_ID(self):
        return self.d['ID']

    @property
    def ID(self):
        return self.to_ID()
