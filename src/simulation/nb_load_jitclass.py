import numpy as np
import pandas as pd
import h5py
from importlib import reload
import pickle
import numba as nb
from numba import njit, prange, objmode, typeof
from numba.typed import List, Dict

from src.utils import utils
from src.simulation import nb_simulation

# hdf5_kwargs = dict(track_order=True)

#%%


def jitclass_to_hdf5_ready_dict(jitclass, skip="cfg"):

    if isinstance(skip, str):
        skip = [skip]

    typ = jitclass._numba_type_
    fields = typ.struct

    ints_floats_bool_set = (nb.types.Integer, nb.types.Float, nb.types.Boolean, nb.types.Set)

    d_out = {}
    for key, dtype in fields.items():
        val = getattr(jitclass, key)

        if key.lower() in skip:
            continue

        if isinstance(dtype, nb.types.ListType):
            if utils.is_nested_numba_list(val):
                d_out[key] = utils.NestedArray(val).to_dict()
            else:
                d_out[key] = list(val)
        elif isinstance(dtype, nb.types.Array):
            d_out[key] = np.array(val, dtype=dtype.dtype.name)
        elif isinstance(dtype, ints_floats_bool_set):
            d_out[key] = val
        else:
            print(f"Just ignoring {key} for now.")

    return d_out


# d_out = jitclass_to_hdf5_ready_dict(my, skip=["cfg", "infectious_states"])

#%%

# filename = "test.hdf5"


def save_jitclass_hdf5ready(f, jitclass_hdf5ready):

    # with h5py.File(filename, "w", **hdf5_kwargs) as f:
    for key, val in jitclass_hdf5ready.items():
        if isinstance(val, dict):
            group = f.create_group(key)
            for k, v in val.items():
                group.create_dataset(k, data=v)
        else:
            f.create_dataset(key, data=val)


# save_jitclass_hdf5ready(filename, d_out)

# %%

# filename = "test.hdf5"


def load_jitclass_to_dict(f):
    d_in = {}
    # with h5py.File(filename, "r") as f:
    for key, val in f.items():
        if isinstance(val, h5py.Dataset):
            d_in[key] = val[()]
        else:
            d_tmp = {}
            for k, v in val.items():
                d_tmp[k] = v[()]
            d_in[key] = d_tmp
    return d_in


# d_in = load_jitclass_to_dict(filename)

# %%


def load_My_from_dict(d_in, cfg):
    spec_my = nb_simulation.spec_my
    my = nb_simulation.initialize_My(cfg)
    for key, val in d_in.items():
        if isinstance(val, dict) and "content" in val and "offsets" in val:
            val = utils.NestedArray.from_dict(val).to_nested_numba_lists()

        # if read as numpy array from hdf5 but should be list, convert
        if isinstance(val, np.ndarray) and isinstance(spec_my[key], nb.types.ListType):
            val = List(val.tolist())
        setattr(my, key, val)
    return my


# def load_My_from_filename(filename):
#     cfg = utils.read_cfg_from_hdf5_file(filename)
#     d_in = load_jitclass_to_dict(filename)
#     return load_My_from_dict(d_in, cfg)
