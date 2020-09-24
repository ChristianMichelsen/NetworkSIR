import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
import yaml

# conda install -c numba/label/dev numba
import numba as nb
from numba import njit, prange, objmode, typeof
from numba.typed import List, Dict
import platform
import datetime

# pip install awkward1
import awkward1 as ak


def _is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


is_ipython = _is_ipython()


def is_local_computer(N_local_cores=12):
    if mp.cpu_count() <= N_local_cores:  # and platform.system() == 'Darwin':
        return True
    else:
        return False


def get_num_cores(num_cores_max=None, subtract_cores=1):
    num_cores = mp.cpu_count() - subtract_cores
    if num_cores_max and num_cores >= num_cores_max:
        num_cores = num_cores_max
    return num_cores


def delete_file(filename):
    try:
        Path(filename).unlink()
    except FileNotFoundError:
        pass


def file_exists(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    return filename.exists()


def make_sure_folder_exist(filename, delete_file_if_exists=False):
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if delete_file_if_exists and filename.exists():
        filename.unlink()


def load_yaml(filename):
    with open(filename) as file:
        return DotDict(yaml.safe_load(file))


def format_time(t):
    return str(datetime.timedelta(seconds=t))


#%%


@njit
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = (
        np.radians(lon1),
        np.radians(lat1),
        np.radians(lon2),
        np.radians(lat2),
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367 * 2 * np.arcsin(np.sqrt(a))  # [km]


@njit
def haversine_scipy(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    return haversine(lon1, lat1, lon2, lat2)


@njit
def set_numba_random_seed(seed):
    np.random.seed(seed)


@njit
def initialize_nested_lists(N, dtype):
    nested_list = List()
    for i in range(N):  # prange
        tmp = List()
        tmp.append(dtype(-1))
        nested_list.append(tmp)
        nested_list[-1].pop(0)  # trick to tell compiler which dtype
    return nested_list


@njit
def initialize_empty_set(dtype):
    s = set()
    x = dtype(1)
    s.add(x)  # trick to tell compiler which dtype
    s.discard(x)
    return s


@njit
def initialize_list_set(N, dtype):
    return [initialize_empty_set(dtype=dtype) for _ in range(N)]


@njit
def get_size(x, unit="gb"):

    d_prefix_conversion = {
        "mb": 10 ** 6,
        "gb": 10 ** 9,
    }

    return x.size * x.itemsize / d_prefix_conversion[unit.lower()]


import re
from numba import typeof


def get_numba_list_dtype(x, as_string=False):
    s = str(typeof(x))

    if "int" in s:
        pat = "int"
    elif "float" in s:
        pat = "float"
    else:
        raise AssertionError('Neither "int", nor "float" in x')

    pat = r"(\w*%s\w*)" % pat  # Not thrilled about this line
    dtype = re.findall(pat, s)[0]

    if as_string:
        return dtype
    return getattr(np, dtype)


# @njit
# def sort_and_flatten_nested_list(nested_list):
#     res = List()
#     for lst in nested_list:
#         # sorted_indices = np.argsort(np.asarray(lst))
#         for index in sorted_indices:
#             res.append(lst[index])
#     return np.asarray(res)


@njit
def flatten_nested_list(nested_list, sort_nested_list=False):
    res = List()
    for lst in nested_list:
        if sort_nested_list:
            sorted_indices = np.argsort(np.asarray(lst))
            for index in sorted_indices:
                res.append(lst[index])
        else:
            for x in lst:
                res.append(x)
    return np.asarray(res)


@njit
def get_cumulative_indices(nested_list, index_dtype=np.int64):
    index = np.zeros(len(nested_list) + 1, index_dtype)
    for i, lst in enumerate(nested_list):
        index[i + 1] = index[i] + len(lst)
    return index


def nested_list_to_awkward_array(nested_list, return_lengths=False, sort_nested_list=False):
    content = ak.layout.NumpyArray(flatten_nested_list(nested_list, sort_nested_list))
    index = ak.layout.Index64(get_cumulative_indices(nested_list))
    listoffsetarray = ak.layout.ListOffsetArray64(index, content)
    array = ak.Array(listoffsetarray)

    if return_lengths:
        return (
            array,
            np.diff(index).astype(np.uint16),
        )  # get_lengths_of_nested_list(nested_list)
    else:
        return array


@njit
def get_lengths_of_nested_list(nested_list):
    N = len(nested_list)
    res = np.zeros(N, dtype=np.uint16)
    for i in range(N):
        res[i] = len(nested_list[i])
    return res


@njit
def binary_search(array, item):
    first = 0
    last = len(array) - 1
    found = False

    while first <= last and not found:
        index = (first + last) // 2
        if array[index] == item:
            found = True
        else:
            if item < array[index]:
                last = index - 1
            else:
                first = index + 1
    return found, index


@njit
def nested_lists_to_list_of_array(nested_list):
    out = List()
    for l in nested_list:
        out.append(np.asarray(l))
    return out


@njit
def list_of_arrays_to_list_of_lists(list_of_arrays):
    outer = List()
    for l in list_of_arrays:
        inner = List()
        for x in l:
            inner.append(x)
        outer.append(inner)
    return outer


# %timeit binary_search(my_connections[contact], 35818)
# %timeit np.searchsorted(my_connections[contact], 35818)

# @njit
# def get_lengths_of_nested_list2(nested_list, dtype=np.uint16):
#     res = List()
#     for i in range(len(nested_list)):
#         res.append(dtype(len(nested_list[i])))
#     return np.asarray(res)

#%%
# Counters in Numba
from numba import types
from numba.experimental import jitclass


@jitclass({"_counter": types.DictType(types.int32, types.uint16)})
class Counter_int32_uint16:
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


@jitclass({"_counter": types.DictType(types.uint16, types.uint32)})
class Counter_uint16_uint32:
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


def MetaClassNumbaCounter(key_type, value_type):
    @jitclass({"_counter": types.DictType(key_type, value_type)})
    class Counter:
        def __init__(self):
            self._counter = Dict.empty(key_type=key_type, value_type=value_type)

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

    return Counter()


@njit
def list_of_counters_to_numpy_array(counter_list, dtype=np.uint32):

    N = len(counter_list)  # number of "days" in the list
    M = max([max(c.keys()) for c in counter_list])  # maximum key in the list

    res = np.zeros((N, M + 1), dtype=dtype)
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


@njit
def array_to_counter2(arr):
    counter = MetaClassNumbaCounter(types.uint16, types.uint32)
    for a in arr:
        counter[a] += 1
    return counter.d


#%%
# Cumulative Sums in numba


@njit
def numba_cumsum_2D(x, axis):
    y = np.zeros_like(x)
    n, m = np.shape(x)
    if axis == 1:
        for i in range(n):
            y[i, :] = np.cumsum(x[i, :])
    elif axis == 0:
        for j in range(m):
            y[:, j] = np.cumsum(x[:, j])
    return y


@njit
def numba_cumsum_3D(x, axis):
    y = np.zeros_like(x)
    n, m, p = np.shape(x)

    if axis == 2:
        for i in range(n):
            for j in range(m):
                y[i, j, :] = np.cumsum(x[i, j, :])
    elif axis == 1:
        for i in range(n):
            for k in range(p):
                y[i, :, k] = np.cumsum(x[i, :, k])
    elif axis == 0:
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


# Counters in Numba
from numba import types
from numba.experimental import jitclass


def NumbaMutableArray(offsets, content, dtype):

    spec = [
        ("offsets", types.int64[:]),
        ("content", getattr(types, dtype)[:]),
    ]

    @jitclass(spec)
    class MetaNumbaMutableArray:
        def __init__(self, offsets, content):
            self.offsets = offsets
            self.content = content

        def __getitem__(self, i):
            return self.content[self.offsets[i] : self.offsets[i + 1]]

        def copy(self):
            return MetaNumbaMutableArray(self.offsets, self.content)

    return MetaNumbaMutableArray(offsets, content)


# "Nested/Mutable" Arrays are faster than list of arrays which are faster than lists of lists


class MutableArray:

    """The MutableArray is basically just a simple version of Awkward Array with _mutable_ data. Also allows the array to be used in zip/enumerate in numba code by using the .array property (not needed in awkward version >= 0.2.32)."""

    def __init__(self, arraylike_object):

        # if numba List
        if isinstance(arraylike_object, List):
            dtype = get_numba_list_dtype(arraylike_object, as_string=True)
            self._content = np.array(
                flatten_nested_list(arraylike_object), dtype=getattr(np, dtype)
            )  # float32
            self._offsets = np.array(get_cumulative_indices(arraylike_object), dtype=np.int64)
            self._awkward_array = None

        # if awkward array
        elif isinstance(arraylike_object, ak.Array):
            dtype = str(ak.type(arraylike_object)).split("* ")[-1]
            self._content = np.array(arraylike_object.layout.content, dtype=getattr(np, dtype))
            self._offsets = np.array(arraylike_object.layout.offsets, dtype=np.int64)
            self._awkward_array = arraylike_object

        else:
            raise AssertionError(
                f"arraylike_object is neither numba list or awkward arry, got {type(arraylike_object)}"
            )

        self.dtype = dtype
        self._initialize_numba_array()

    def _initialize_numba_array(self):
        self._array = NumbaMutableArray(self._offsets, self._content, self.dtype)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._content[self._offsets[i] : self._offsets[i + 1]]
        elif (
            isinstance(i, tuple) and len(i) == 2 and isinstance(i[0], int) and isinstance(i[1], int)
        ):
            x, y = i
            return self._content[self._offsets[x] : self._offsets[x + 1]][y]

    @property
    def array(self):
        return self._array

    def to_awkward(self, return_original_awkward_array=False):
        if return_original_awkward_array and self._awkward_array:
            return self._awkward_array
        elif return_original_awkward_array and not self._awkward_array:
            raise AssertionError(
                f"No original awkward array (possibly because it was loaded through pickle)"
            )

        offsets = ak.layout.Index64(self._offsets)
        content = ak.layout.NumpyArray(self._content)
        listarray = ak.layout.ListOffsetArray64(offsets, content)
        return ak.Array(listarray)

    def __repr__(self):
        return repr(self.to_awkward()).replace("Array", "MutableArray")

    def __len__(self):
        return len(self._offsets) - 1

    # make class pickle-able
    def __getstate__(self):
        d = {"_content": self._content, "_offsets": self._offsets, "dtype": self.dtype}
        return d

    # make class pickle-able
    def __setstate__(self, d):
        self._content = d["_content"]
        self._offsets = d["_offsets"]
        self.dtype = d["dtype"]
        self._initialize_numba_array()


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
        if key in self:
            self[key] = value
        elif not "__" in key:
            raise KeyError("Not allowed to change keys with dot notation, use brackets instead.")

    # make class pickle-able
    def __getstate__(self):
        return self.__dict__

    # make class pickle-able
    def __setstate__(self, state):
        self.__dict__ = state


#%%


INTEGER_SIMULATION_PARAMETERS = load_yaml("cfg/settings.yaml")["INTEGER_SIMULATION_PARAMETERS"]


def string_to_dict(string):
    # if path-like string

    if isinstance(string, Path):
        string = str(string)

    if isinstance(string, str) and "/" in string:
        string = Path(string).stem

    d = {}
    keyvals = string.split("__")
    keyvals_chunks = [keyvals[i : i + 2] for i in range(0, len(keyvals), 2)]
    for key, val in keyvals_chunks:
        if key in INTEGER_SIMULATION_PARAMETERS + ["ID"]:
            d[key] = int(val)
        elif key == "v":
            d["version"] = float(val)
        else:
            d[key] = float(val)
    return DotDict(d)


#%%


def get_parameter_to_latex():
    return load_yaml("cfg/parameter_to_latex.yaml")
    # parameter_to_latex = {
    #     "N_tot": r"N_\mathrm{tot}",
    #     "N_init": r"N_\mathrm{init}",
    #     "rho": r"\rho",
    #     "epsilon_rho": r"\epsilon_\rho",
    #     "mu": r"\mu",
    #     "sigma_mu": r"\sigma_\mu",
    #     "beta": r"\beta",
    #     "sigma_beta": r"\sigma_\beta",
    #     "lambda_E": r"\lambda_E",
    #     "lambda_I": r"\lambda_I",
    #     "algo": r"\mathrm{algo}",
    #     "ID": "ID",
    # }
    # return parameter_to_latex


def human_format(num, digits=3):
    num = float(f"{num:.{digits}g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "G", "T"][magnitude]
    )


def human_format_scientific(num, digits=3):
    num = float(f"{num:.{digits}g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return (
        "{}".format("{:f}".format(num).rstrip("0").rstrip(".")),
        f"{['', '10^3', '10^6', '10^9', '10^12'][magnitude]}",
    )


def dict_to_title(d, N=None, exclude=None, in_two_line=True):

    # important to make a copy since overwriting below
    cfg = DotDict(d)

    cfg.N_tot = human_format(cfg.N_tot)
    cfg.N_init = human_format(cfg.N_init)
    cfg.make_random_initial_infections = (
        r"\mathrm{" + str(bool(cfg.make_random_initial_infections)) + r"}"
    )

    # parameter_to_latex = get_parameter_to_latex()
    parameter_to_latex = load_yaml("cfg/parameter_to_latex.yaml")

    # make "exclude" a list of keys to ignore
    if isinstance(exclude, str) or exclude is None:
        exclude = [exclude]
    exclude.append("version")

    title = "$"
    for sim_par, val in cfg.items():
        if not sim_par in exclude:
            title += f"{parameter_to_latex[sim_par]} = {val}, \,"
    title += f"{parameter_to_latex['version']} = {cfg.version}, \,"

    if in_two_line:
        if "lambda_E" in title:
            title = title.replace(", \\,\\lambda_E", "$\n$\\lambda_E")
        else:
            title = title.replace(", \\,\\lambda_I", "$\n$\\lambda_I")

    if N:
        title += r"\#" + f"{N}, \,"

    title = title[:-4] + "$"

    return title


def string_to_title(s, N=None, exclude=None, in_two_line=True):
    d = string_to_dict(s)
    return dict_to_title(d, N, exclude, in_two_line)


# def format_uncertanties(median, errors, name='I'):

#     s_main = human_format(median, digits=3)
#     prefix = s_main[-1]

#     if name == 'I':
#         name = r'I_\mathrm{max}^\mathrm{fit}'
#     elif name == 'R':
#         name = r'R_\mathrm{inf}^\mathrm{fit}'
#     else:
#         raise AssertionError(f"name = {name} not defined")


#     d = {'K': 1000, 'M': 1000**2, 'G': 1000**3, 'T': 1000**4}
#     d2 = {'K': '10^3', 'M': '10^6', 'G': '10^9', 'T': '10^12'}

#     out = []
#     for error in errors:
#         try:
#             out.append(error / d[prefix])
#         except KeyError as e:
#             print(prefix, s_main)
#             raise e

#     s = r"$" + f"{name}" +  r" = " + f"{s_main[:-1]}" + r"_{" + f"-{out[0]:.1f}" + r"}^{+" + f"{out[1]:.1f}" + r"} \cdot " + f"{d2[prefix]}" + r"$"
#     return s


from decimal import Decimal


def round_to_uncertainty(value, uncertainty):
    # round the uncertainty to 1-2 significant digits
    u = Decimal(uncertainty).normalize()
    exponent = u.adjusted()  # find position of the most significant digit
    precision = u.as_tuple().digits[0] == 1  # is the first digit 1?
    u = u.scaleb(-exponent).quantize(Decimal(10) ** -precision)
    # round the value to remove excess digits
    return round(Decimal(value).scaleb(-exponent).quantize(u)), u, exponent


# import sigfig


def format_asymmetric_uncertanties(value, errors, name="I"):

    if name == "I":
        name = r"I_\mathrm{max}^\mathrm{fit}"
    elif name == "R":
        name = r"R_\infty^\mathrm{fit}"
    else:
        raise AssertionError(f"name = {name} not defined")

    mu, std_lower, exponent = round_to_uncertainty(value, errors[0])
    mu1, std_higher, exponent1 = round_to_uncertainty(value, errors[1])

    if mu != mu1 or exponent != exponent1:

        if np.abs(10 * mu - mu1) <= 9 and exponent - 1 == exponent1:
            mu = mu1
            # std_lower = Decimal(std_lower*10).normalize()
            std_lower = str(std_lower * 10).replace(".0", "")
            exponent = exponent1

        elif np.abs(mu - 10 * mu1) <= 9 and exponent == exponent1 - 1:
            mu = mu / 10
            std_lower = std_lower / 10
            std_higher = std_higher
            exponent = exponent1

        elif np.abs(100 * mu - mu1) <= 99 and exponent - 2 == exponent1:
            mu = mu1
            std_lower = str(std_lower * 100).replace(".0", "")
            exponent = exponent1

        elif np.abs(mu - 100 * mu1) <= 99 and exponent == exponent1 - 2:
            mu = mu / 100
            std_lower = std_lower / 100
            std_higher = std_higher
            exponent = exponent1

        elif np.abs(mu - mu1) == 1 and exponent == exponent1:
            if np.abs(mu * 10 ** exponent - value) < mu1 * 10 ** exponent1 - value:
                mu = mu
            else:
                mu = mu1

        else:
            raise AssertionError("The errors do not fit (not yet implemented)")

    if "E" in str(std_lower):
        assert False

    s = (
        r"$"
        + f"{name}"
        + r" = "
        + f"{mu}"
        + r"_{"
        + f"-{std_lower}"
        + r"}^{+"
        + f"{std_higher}"
        + r"} \cdot 10^{"
        + f"{exponent}"
        + r"}$"
    )

    return s


#%%


def get_column_dtypes(df, cols_to_str):
    kwargs = {}
    if cols_to_str is not None:
        if isinstance(cols_to_str, str):
            cols_to_str = [cols_to_str]
        kwargs["column_dtypes"] = {col: f"<S{int(df[col].str.len().max())}" for col in cols_to_str}
    return kwargs


def dataframe_to_hdf5_format(df, include_index=False, cols_to_str=None):
    kwargs = get_column_dtypes(df, cols_to_str)
    if include_index:
        kwargs["index_dtypes"] = f"<S{df.index.str.len().max()}"
    return np.array(df.to_records(index=True, **kwargs))


#%%


@njit
def numba_random_choice_list(l):
    return l[np.random.randint(len(l))]


#%%


from scipy.special import erf


def get_central_confidence_intervals(x, agg_func=np.median, N_sigma=1):
    agg = agg_func(x)
    sigma = 100 * erf(N_sigma / np.sqrt(2))
    p_lower = 50 - sigma / 2
    p_upper = 50 + sigma / 2
    lower_bound = np.percentile(x, p_lower)
    upper_bound = np.percentile(x, p_upper)
    errors = agg - lower_bound, upper_bound - agg
    return agg, errors


def SDOM(x):
    "standard deviation of the mean"
    return np.std(x) / np.sqrt(len(x))


#%%


# def load_df_coordinates():
#     GPS_filename = "Data/GPS_coordinates.feather"
#     df_coordinates = pd.read_feather(GPS_filename)
#     return df_coordinates


# def load_coordinates_from_indices(coordinate_indices):
#     return load_df_coordinates().iloc[coordinate_indices].reset_index(drop=True)


#%%


import numpy as np
from range_key_dict import RangeKeyDict  # pip install range-key-dict
from itertools import product
from numba import njit
from numba.typed import List, Dict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import numba as nb

# # try:
# # from src.utils import utils
# # except ImportError:
# from src.utils import utils

INTEGER_SIMULATION_PARAMETERS = load_yaml("cfg/settings.yaml")["INTEGER_SIMULATION_PARAMETERS"]


def get_cfg_default():
    """ Default Simulation Parameters """
    yaml_filename = "cfg/simulation_parameters_default.yaml"
    # if Path("").cwd().stem == "src":
    # yaml_filename = "../" + yaml_filename
    return load_yaml(yaml_filename)


def dict_to_filename_with_dir(cfg, ID, data_dir="ABM"):
    filename = Path("Data") / data_dir
    file_string = ""
    for key, val in cfg.items():
        if key == "version":
            key = "v"
        file_string += f"{key}__{val}__"
    file_string = file_string[:-2]  # remove trailing _
    filename = filename / file_string
    file_string += f"__ID__{ID:03d}.csv"
    filename = filename / file_string
    return str(filename)


def get_all_combinations(d_simulation_parameters):
    nameval_to_str = []
    for name, lst in reversed(d_simulation_parameters.items()):
        if isinstance(lst, (int, float)):
            lst = [lst]
        nameval_to_str.append([f"{name}__{x}" for x in lst])
    all_combinations = list(product(*nameval_to_str))
    return all_combinations


def generate_filenames(
    d_simulation_parameters, N_loops=10, force_rerun=False, N_tot_max=False, verbose=True
):
    filenames = []

    all_combinations = get_all_combinations(d_simulation_parameters)
    cfg = get_cfg_default()

    has_printed_once = False

    for combination in all_combinations:
        for s in combination:
            name, val = s.split("__")
            val = int(val) if name in INTEGER_SIMULATION_PARAMETERS else float(val)
            cfg[name] = val

        if not N_tot_max or cfg["N_tot"] < N_tot_max:

            # ID = 0
            for ID in range(N_loops):
                filename = dict_to_filename_with_dir(cfg, ID)

                not_existing = not Path(filename).exists()
                try:
                    zero_size = Path(filename).stat().st_size == 0
                except FileNotFoundError:
                    zero_size = True
                if not_existing or zero_size or force_rerun:
                    filenames.append(filename)

        else:
            if verbose and not has_printed_once:
                print(
                    f"Skipping some files since N_tot={human_format(cfg['N_tot'])} > N_tot_max={human_format(N_tot_max)}"
                )
                has_printed_once = True

    return filenames


d_num_cores_N_tot = RangeKeyDict(
    {
        (0, 1_000_001): 40,
        (1_000_001, 2_000_001): 30,
        (2_000_001, 5_000_001): 20,
        (5_000_001, 6_000_001): 12,
        (6_000_001, 10_000_001): 5,
    }
)


def extract_N_tot_max(d_simulation_parameters):
    if isinstance(d_simulation_parameters, dict) and "N_tot" in d_simulation_parameters.keys():
        if isinstance(d_simulation_parameters["N_tot"], int):
            return d_simulation_parameters["N_tot"]
        else:
            return max(d_simulation_parameters["N_tot"])
    else:
        return get_cfg_default()["N_tot"]


def get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max=None):
    N_tot_max = extract_N_tot_max(d_simulation_parameters)
    num_cores = d_num_cores_N_tot[N_tot_max]

    if num_cores > get_num_cores(num_cores_max):
        num_cores = get_num_cores(num_cores_max)

    return num_cores


def load_df_coordinates(N_tot, ID):
    # np.random.seed(ID)
    # coordinates = np.load(coordinates_filename)
    coordinates_filename = "Data/GPS_coordinates.feather"
    df_coordinates = (
        pd.read_feather(coordinates_filename)
        .sample(N_tot, replace=False, random_state=ID)
        .reset_index(drop=True)
    )
    return df_coordinates

    # coordinates = df_coordinates_to_coordinates(df_coordinates)

    # if N_tot > len(df_coordinates):
    #     raise AssertionError(
    #         "N_tot cannot be larger than coordinates (number of generated houses in DK)"
    #     )

    # index = np.arange(len(df_coordinates), dtype=np.uint32)
    # index_subset = np.random.choice(index, N_tot, replace=False)
    # return coordinates[index_subset], index_subset


# def load_coordinates_indices(coordinates_filename, N_tot, ID):
#     return load_coordinates(coordinates_filename, N_tot, ID)[1]


def df_coordinates_to_coordinates(df_coordinates):
    return df_coordinates[["Longitude", "Lattitude"]].values


#%%

EXCLUDE_PARAMETER_IN_INIT = load_yaml("cfg/settings.yaml")["EXCLUDE_PARAMETER_IN_INIT"]


class Filename:
    def __init__(self, filename):

        if isinstance(filename, dict):
            filename = generate_filenames(filename, N_loops=1, force_rerun=True)[0]

        self._filename = filename
        self.filename = self.filename_prefix + filename
        self.cfg = string_to_dict(filename.replace(".animation", ""))
        # self.cfg = self.simulation_parameters

    def __repr__(self):
        return str(self.cfg)

    # @property
    # def to_dict(self):  # ,
    # return DotDict({key: val for key, val in self.d.items() if key != "ID"})

    @property
    def simulation_parameters(self):
        return self.cfg

    @property
    def to_ID(self):
        return self.cfg["ID"]

    @property
    def ID(self):
        return self.to_ID

    @property
    def filename_prefix(self):
        filename_prefix = ""
        if str(Path.cwd()).endswith("src"):
            filename_prefix = "../"
        return filename_prefix

    def _filename_to_network(self, d, filename, extension):
        file_string = ""
        for key, val in d.items():
            file_string += f"{key}__{val}__"
        file_string = file_string[:-2]  # remove trailing _
        file_string += extension
        filename = filename / file_string
        filename = str(filename).replace("version", "v")
        return filename

    def get_filename_network_initialisation(self, extension=".hdf5"):
        variables_to_save_in_filename = []
        for parameter in self.cfg.keys():
            if not parameter in EXCLUDE_PARAMETER_IN_INIT:
                variables_to_save_in_filename.append(parameter)
        d = {key: self.cfg[key] for key in variables_to_save_in_filename}
        filename = Path(f"{self.filename_prefix}Data") / "network_initialization"
        return self._filename_to_network(d, filename, extension)

    filename_network_initialisation = property(get_filename_network_initialisation)

    def get_filename_network(self, extension=".hdf5"):
        filename = Path(f"{self.filename_prefix}Data") / "network"
        return self._filename_to_network(self.cfg, filename, extension)

    filename_network = property(get_filename_network)

    @property
    def memory_filename(self):
        filename = Path(f"{self.filename_prefix}Data") / "memory"
        return self._filename_to_network(self.cfg, filename, ".memory_file.txt")
        # return self.filename_prefix + self.filename.replace('.csv', '.memory_file.txt')

    @property
    def coordinates_filename(self):
        # return self.filename_prefix + "Data/GPS_coordinates.npy"
        return self.filename_prefix + "Data/GPS_coordinates.feather"

    @property
    def household_data_filenames(self):
        return (
            self.filename_prefix + "Data/PeopleInHousehold_NorthJutland.txt",
            self.filename_prefix + "Data/AgeDistributionPerPeopleInHousehold_NorthJutland.txt",
        )


#%%


@njit
def calculate_epsilon(alpha_age, N_ages):
    return 1 / N_ages * alpha_age


@njit
def calculate_age_proportions_1D(alpha_age, N_ages):
    """ Only used in v1 of simulation"""
    epsilon = calculate_epsilon(alpha_age, N_ages)
    x = epsilon * np.ones(N_ages, dtype=np.float32)
    x[0] = 1 - x[1:].sum()
    return x


@njit
def calculate_age_proportions_2D(alpha_age, N_ages):
    """ Only used in v1 of simulation"""
    epsilon = calculate_epsilon(alpha_age, N_ages)
    A = epsilon * np.ones((N_ages, N_ages), dtype=np.float32)
    for i in range(N_ages):
        A[i, i] = 1 - np.sum(np.delete(A[i, :], i))
    return A


@njit
def set_numba_random_seed(seed):
    np.random.seed(seed)


@njit
def _initialize_my_rates_nested_list(my_infection_weight, my_number_of_contacts):
    N_tot = len(my_infection_weight)
    res = List()
    for i in range(N_tot):
        x = np.full(
            my_number_of_contacts[i],
            fill_value=my_infection_weight[i],
            dtype=np.float64,
        )
        res.append(x)
    return res


def initialize_my_rates(my_infection_weight, my_number_of_contacts):
    return MutableArray(
        _initialize_my_rates_nested_list(my_infection_weight, my_number_of_contacts)
    )


# def initialize_my_rates(my_infection_weight, my_number_of_contacts):
#     return ak.Array(_initialize_my_rates_nested_list(my_infection_weight, my_number_of_contacts))


@njit
def initialize_non_infectable(N_tot, my_number_of_contacts):
    res = List()
    for i in range(N_tot):
        res.append(np.ones(my_number_of_contacts[i], dtype=np.bool_))
    return res


def initialize_SIR_transition_rates(N_states, N_infectious_states, cfg):
    SIR_transition_rates = np.zeros(N_states, dtype=np.float64)
    SIR_transition_rates[:N_infectious_states] = cfg.lambda_E
    SIR_transition_rates[N_infectious_states : 2 * N_infectious_states] = cfg.lambda_I
    return SIR_transition_rates


@njit
def _compute_agents_in_age_group(ages, N_ages):
    agents_in_age_group = initialize_nested_lists(N_ages, dtype=np.uint32)
    for idx, age in enumerate(ages):  # prange
        agents_in_age_group[age].append(np.uint32(idx))
    return agents_in_age_group


def compute_agents_in_age_group(ages, N_ages):
    agents_in_age_group = _compute_agents_in_age_group(ages, N_ages)
    agents_in_age_group = nested_list_to_awkward_array(agents_in_age_group)
    return agents_in_age_group


def get_hospitalization_variables(N_tot, N_ages=1):

    # Hospitalization track variables
    H_N_states = 6  # number of states
    H_state_total_counts = np.zeros(H_N_states, dtype=np.uint32)
    # H_my_state = -1*np.ones(N_tot, dtype=np.int8)
    H_my_state = np.full(N_tot, -1, dtype=np.int8)

    H_agents_in_state = initialize_nested_lists(H_N_states, dtype=np.uint32)
    H_probability_matrix = np.ones((N_ages, H_N_states), dtype=np.float32) / H_N_states
    H_probability_matrix_csum = numba_cumsum(H_probability_matrix, axis=1)

    H_move_matrix = np.zeros((H_N_states, H_N_states, N_ages), dtype=np.float32)
    H_move_matrix[0, 1] = 0.3
    H_move_matrix[1, 2] = 1.0
    H_move_matrix[2, 1] = 0.6
    H_move_matrix[1, 4] = 0.1
    H_move_matrix[2, 3] = 0.1
    H_move_matrix[3, 4] = 1.0
    H_move_matrix[3, 5] = 0.1

    H_move_matrix_sum = np.sum(H_move_matrix, axis=1)
    H_move_matrix_cumsum = numba_cumsum(H_move_matrix, axis=1)

    H_cumsum_move = np.zeros(H_N_states, dtype=np.float64)

    return (
        H_probability_matrix_csum,
        H_my_state,
        H_agents_in_state,
        H_state_total_counts,
        H_move_matrix_sum,
        H_cumsum_move,
        H_move_matrix_cumsum,
    )


#%%


def state_counts_to_df(time, state_counts):  #

    header = [
        "Time",
        "E1",
        "E2",
        "E3",
        "E4",
        "I1",
        "I2",
        "I3",
        "I4",
        "R",
        # 'H1', 'H2', 'ICU1', 'ICU2', 'R_H', 'D',
    ]

    df_time = pd.DataFrame(time, columns=header[0:1])
    df_states = pd.DataFrame(state_counts, columns=header[1:])
    # df_H_states = pd.DataFrame(H_state_total_counts, columns=header[10:])
    df = pd.concat([df_time, df_states], axis=1)  # .convert_dtypes()
    # assert sum(df_H_states.sum(axis=1) == df_states['R'])
    return df


#%%


def parse_memory_file(filename):

    change_points = {}

    d_time_mem = {}

    next_is_change_point = 0
    # zero_time = None

    import csv

    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for irow, row in enumerate(reader):

            # if new section
            if len(row) == 1:
                next_is_change_point = True
                s_change_points = row[0][1:]

            else:

                time = float(row[0])
                memory = float(row[1])

                # if zero_time is None:
                # zero_time = time
                # time -= zero_time

                d_time_mem[time] = memory

                if next_is_change_point:
                    change_points[time] = s_change_points
                    next_is_change_point = False

    s_change_points = pd.Series(change_points)
    df_time_memory = pd.DataFrame.from_dict(d_time_mem, orient="index")
    df_time_memory.columns = ["Memory"]

    df_time_memory["ChangePoint"] = s_change_points

    df_change_points = s_change_points.to_frame()
    df_change_points.columns = ["ChangePoint"]
    df_change_points["Time"] = df_change_points.index
    df_change_points = df_change_points.set_index("ChangePoint")
    df_change_points["TimeDiff"] = -df_change_points["Time"].diff(-1)
    df_change_points["TimeDiffRel"] = (
        df_change_points["TimeDiff"] / df_change_points["Time"].iloc[-1]
    )

    df_change_points["Memory"] = df_time_memory.loc[df_change_points["Time"]]["Memory"].values
    df_change_points["MemoryDiff"] = -df_change_points["Memory"].diff(-1)
    df_change_points["MemoryDiffRel"] = (
        df_change_points["MemoryDiff"] / df_change_points["Memory"].iloc[-1]
    )

    df_change_points.index.name = None

    return df_time_memory, df_change_points


def plot_memory_comsumption(
    df_time_memory,
    df_change_points,
    min_TimeDiffRel=0.1,
    min_MemoryDiffRel=0.1,
    time_unit="min",
):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [col for i, col in enumerate(colors) if i != 5]

    time_scale = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 60 * 60, "t": 60 * 60}

    fig, ax = plt.subplots()

    #
    df_change_points_non_compilation = df_change_points.query("index != 'Numba Compilation'")
    df_change_points_compilation = df_change_points.query("index == 'Numba Compilation'")
    # marker='s'

    ax.plot(
        df_time_memory.index / time_scale[time_unit],
        df_time_memory["Memory"],
        ".",
        c=colors[0],
        zorder=2,
        label="Data Points",
    )
    ax.scatter(
        df_change_points_non_compilation["Time"] / time_scale[time_unit],
        df_change_points_non_compilation["Memory"],
        s=200,
        c="white",
        edgecolors="k",
        zorder=3,
        label="Change Points",
    )
    ax.scatter(
        df_change_points_compilation["Time"] / time_scale[time_unit],
        df_change_points_compilation["Memory"],
        s=200,
        c="white",
        edgecolors="k",
        zorder=2,
        label="Numba Compilations",
        marker="s",
    )
    ax.set(xlabel=f"Time [{time_unit}]", ylabel="Memory [GiB]", ylim=(0, None))  # xlim=(0, None)

    ymax = ax.get_ylim()[1]
    i = 1
    for index, row in df_change_points_non_compilation.iterrows():
        # first_or_last = (i == 0) or (i == len(df_change_points)-1)
        last = index == df_change_points_non_compilation.index[-1]
        large_time_diff = row["TimeDiffRel"] > min_TimeDiffRel  # sec
        large_memory_diff = np.abs(row["MemoryDiffRel"]) > min_MemoryDiffRel  # GiB
        if any([last, large_time_diff, large_memory_diff]):
            t = row["Time"] / time_scale[time_unit]
            y = row["Memory"]
            col = colors[(i) % len(colors)]
            i += 1
            ax.plot([t, t], [0, y], ls="--", color=col, zorder=1, label=index)
            ax.plot([t, t], [y, ymax], ls="--", color=col, zorder=1, alpha=0.5)

            if row["TimeDiffRel"] > 0.01 or last:
                kwargs = dict(
                    rotation=90,
                    color=col,
                    fontsize=22,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="square", ec=col, fc="white"),
                )
                if y / ymax > 0.45:
                    ax.text(t, y / 2, index, **kwargs)
                else:
                    ax.text(t, (ymax + y) / 2, index, **kwargs)

    # ax.set_yscale('log')

    ax.legend()
    return fig, ax


#%%


def does_file_contains_string(filename, string):
    with open(filename) as f:
        if string in f.read():
            return True
    return False


def get_search_string_time(filename, search_string):
    is_search_string = False

    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for irow, row in enumerate(reader):

            # if new section
            if len(row) == 1:
                if search_string in row[0]:
                    is_search_string = True

            elif is_search_string and len(row) == 2:
                time = float(row[0])
                return time
    return 0


#%%


def get_simulation_parameters_1D_scan(parameter, non_defaults):
    """ Get a list of simulation parameters (as strings) for a given parameter to be used in a 1D-scan. Can take non-default values ('non_defaults')."""

    base_dir = Path("Data") / "ABM"
    simulation_parameters = sorted([x.name for x in base_dir.glob("*") if "N_tot" in x.name])
    d_simulation_parameters = {s: string_to_dict(s) for s in simulation_parameters}
    df_simulation_parameters = pd.DataFrame.from_dict(d_simulation_parameters, orient="index")

    parameters = get_cfg_default()
    for key, val in non_defaults.items():
        parameters[key] = val

    if isinstance(parameter, str):
        parameter = [parameter]

    query = ""
    for key, val in parameters.items():
        if not key in parameter:
            query += f"{key} == {val} & "
    query = query[:-3]

    df_different_than_default = df_simulation_parameters.query(query).sort_values(parameter)
    return list(df_different_than_default.index)


#%%


@njit
def numba_unique_with_counts(a):
    b = np.sort(a.flatten())
    unique = list(b[:1])
    counts = [1 for _ in unique]
    for x in b[1:]:
        if x != unique[-1]:
            unique.append(x)
            counts.append(1)
        else:
            counts[-1] += 1
    return np.array(unique), np.array(counts)


def PyDict2NumbaDict(d_python):
    "https://github.com/numba/numba/issues/4728"

    keys = list(d_python.keys())
    values = list(d_python.values())

    if isinstance(keys[0], str):
        key_type = nb.types.string
    elif isinstance(keys[0], int):
        key_type = nb.types.int64
    elif isinstance(keys[0], float):
        key_type = nb.types.float64
    else:
        raise AssertionError("Unknown Keytype")

    if isinstance(values[0], dict):
        d_numba = nb.typed.Dict.empty(key_type, nb.typeof(PyDict2NumbaDict(values[0])))
        for i, subDict in enumerate(values):
            subDict = PyDict2NumbaDict(subDict)
            d_numba[keys[i]] = subDict
        return d_numba

    if isinstance(values[0], int):
        d_numba = nb.typed.Dict.empty(key_type, nb.types.int64)

    elif isinstance(values[0], str):
        d_numba = nb.typed.Dict.empty(key_type, nb.types.string)

    elif isinstance(values[0], float):
        d_numba = nb.typed.Dict.empty(key_type, nb.types.float64)

    elif isinstance(values[0], np.ndarray):
        assert values[0].ndim == 1
        d_numba = nb.typed.Dict.empty(key_type, nb.types.float64[:])
    else:
        raise AssertionError("Unknown ValueType")

    for i, key in enumerate(keys):
        d_numba[key] = values[i]
    return d_numba


from numba.core import types
from numba.typed import Dict
from numba import generated_jit


@njit
def normalize_probabilities(p):
    return p / p.sum()


@njit
def rand_choice_nb_arr(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    prob = normalize_probabilities(prob)
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@njit
def rand_choice_nb(prob):
    """
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    prob = normalize_probabilities(prob)
    return np.searchsorted(np.cumsum(prob), np.random.random(), side="right")


@njit
def int_keys2array(d_prob):
    N = len(d_prob)
    arr = np.empty(N, dtype=np.int64)
    for i, x in enumerate(d_prob.keys()):
        arr[i] = x
    return arr


@njit
def values2array(d_prob):
    N = len(d_prob)
    x = np.empty(N, dtype=np.float64)
    for i, p in enumerate(d_prob.values()):
        x[i] = p
    return x


@njit
def draw_random_number_from_dict(d_prob):
    arr = int_keys2array(d_prob)
    prob = values2array(d_prob)
    return rand_choice_nb_arr(arr, prob)


@njit
def draw_random_index_based_on_array(prob):
    return rand_choice_nb(prob)


@generated_jit(nopython=True)
def draw_random_nb(x):
    if isinstance(x, types.DictType):
        return lambda x: draw_random_number_from_dict(x)
    elif isinstance(x, types.Array):
        return lambda x: draw_random_index_based_on_array(x)


#%%

from collections import defaultdict


def parse_household_data(filename, age_dist_as_dict=True):

    data = defaultdict(list)
    ages_groups = [20, 30, 40, 50, 60, 70, 80]

    with open(filename, "r") as file:
        N_persons = -1
        for line in file:
            line = line.strip()
            if line[0] == "#":
                N_persons = int(line[1:].split()[0])
            else:
                try:
                    x = int(line)
                except ValueError:
                    x = float(line)
                data[N_persons].append(x)

    # make sure all entries are normalized numpy arrays
    data = dict(data)
    for key, val in data.items():
        if len(val) == 1:
            data[key] = val[0]
        else:
            vals = np.array(val)
            vals = vals / np.sum(vals)
            if age_dist_as_dict:
                data[key] = {age: val for age, val in zip(ages_groups, vals)}
            else:
                data[key] = vals

    return PyDict2NumbaDict(data)


def parse_household_data_list(filename, convert_to_numpy=False):

    data = defaultdict(list)

    with open(filename, "r") as file:
        N_persons = -1
        for line in file:
            line = line.strip()
            if line[0] == "#":
                N_persons = int(line[1:].split()[0])
            else:
                try:
                    x = int(line)
                except ValueError:
                    x = float(line)
                data[N_persons].append(x)

    # make sure all entries are normalized numpy arrays
    data = dict(data)

    out = []
    for key, val in data.items():
        if len(val) == 1:
            out.append(val[0])
        else:
            vals = np.array(val)
            vals = vals / np.sum(vals)
            out.append(vals)
    if convert_to_numpy:
        out = np.array(out)
    return out


def load_household_data(household_data_filenames):
    people_in_household = parse_household_data_list(
        household_data_filenames[0], convert_to_numpy=True
    )
    age_distribution_per_people_in_household = parse_household_data_list(
        household_data_filenames[1], convert_to_numpy=True
    )
    return people_in_household, age_distribution_per_people_in_household


@njit
def nb_load_coordinates_Nordjylland(all_coordinates, N_tot=150_000, verbose=False):
    coordinates = List()
    for i in range(len(all_coordinates)):
        if all_coordinates[i][1] > 57.14:
            coordinates.append(all_coordinates[i])
            if len(coordinates) == N_tot:
                break
    if verbose:
        print(i)
    return coordinates


# def load_coordinates_Nordjylland(N_tot=150_000, verbose=False):
#     all_coordinates = np.load("../Data/GPS_coordinates.npy")
#     coordinates = nb_load_coordinates_Nordjylland(all_coordinates, N_tot, verbose)
#     return np.array(coordinates)


#%%

import geopandas as gpd  # conda install -c conda-forge geopandas

# Shapefiles
def load_kommune_shapefiles(shapefile_size, verbose=False):

    shp_file = {}
    shp_file["small"] = "Data/Kommuner/ADM_2M/KOMMUNE.shp"
    shp_file["medium"] = "Data/Kommuner/ADM_500k/KOMMUNE.shp"
    shp_file["large"] = "Data/Kommuner/ADM_10k/KOMMUNE.shp"

    if verbose:
        print(f"Loading {shapefile_size} kommune shape files")
    kommuner = gpd.read_file(shp_file[shapefile_size]).to_crs(
        {"proj": "latlong"}
    )  # convert to lat lon, compared to UTM32_EUREF89

    kommune_navn, kommune_idx = np.unique(kommuner["KOMNAVN"], return_inverse=True)
    name_to_idx = dict(zip(kommune_navn, range(len(kommune_navn))))
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    kommuner["idx"] = kommune_idx
    return kommuner, name_to_idx, idx_to_name
