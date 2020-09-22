import numpy as np
import inspect
from numba import jitclass, typeof
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type
from numba import types
import numba as nb


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # Class to make Jit Class pickable  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_init_argument(cls):
    return inspect.getargspec(cls.__init__)[0][1:]


class Jitpickler:
    """
    pickler
    """

    def __init__(self, jitobj):
        self.jitobj = jitobj

    def ParseNumbaOut(self, fields, obj):
        out = {}
        for key, dtype in fields.items():
            if isinstance(dtype, ListType):
                dtype = dtype.dtype
                out[key + "_List"] = np.array(getattr(obj, key), dtype=str(dtype))
            elif isinstance(dtype, DictType):
                d = {}
                for k, v in getattr(obj, key).items():
                    d[k] = v
                d["dtype"] = str(dtype.value_type)
                out[key + "_Dict"] = d
            else:
                out[key] = getattr(obj, key)
        return out

    def ParseNumbaIn(self, values_in):
        values_out = {}
        for key, val in values_in.items():
            if "_List" in key:
                key = key.replace("_List", "")
                if len(val) == 0:
                    dtype_str = str(val.dtype)
                    dtype = getattr(types, dtype_str)
                    values_out[key] = List.empty_list(dtype)
                else:
                    values_out[key] = List(val)
            elif "_Dict" in key:
                key = key.replace("_Dict", "")
                d = Dict()
                dtype_str = val.pop("dtype")
                dtype = getattr(np, dtype_str)
                for k, v in val.items():
                    d[k] = dtype(v)
                values_out[key] = d
            else:
                values_out[key] = val
        return values_out

    def __getstate__(self):
        obj = self.jitobj
        typ = obj._numba_type_
        fields = typ.struct
        out = self.ParseNumbaOut(fields, obj)
        # print(typ.classname, out)
        return typ.classname, out

    def __setstate__(self, state):
        name, values = state
        values = self.ParseNumbaIn(values)
        # print(f"{values=}")
        cls = globals()[name]
        init_args = get_init_argument(cls)

        inits = {}
        post = {}
        for key, val in values.items():
            if key in init_args:
                inits[key] = val
            else:
                post[key] = val

        jitobj = cls(**inits)
        for key, val in post.items():
            setattr(jitobj, key, val)
        self.__init__(jitobj)

    def unpickle(self):
        return self.jitobj


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # Alternative Code for pickling # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pickle


def dumps_jitclass(jc):
    typ = jc._numba_type_
    fields = typ.struct
    data = {"name": typ.classname, "struct": {k: getattr(jc, k) for k in fields}}

    return pickle.dumps(data)


def loads_jitclass(s):
    cls = globals()[data["name"]]
    instance = cls(**data["struct"])
    instance = cls(**data["struct"])
    return instance
