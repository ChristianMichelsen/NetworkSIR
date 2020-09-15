import numpy as np
import multiprocessing as mp
from pathlib import Path
import yaml

import numba as nb
from numba import (
    njit,
    prange,
    objmode,
    typeof,
)  # conda install -c numba/label/dev numba
from numba.typed import List, Dict

import awkward1 as ak  # pip install awkward1

# try:
#     import simulation_utils
#     from simulation_utils import INTEGER_SIMULATION_PARAMETERS
# except ModuleNotFoundError:
#     try:
try:
    from src import simulation_utils
    from src.simulation_utils import INTEGER_SIMULATION_PARAMETERS
except ImportError:
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
        return yaml.safe_load(file)


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


class MutableArray:

    """The MutableArray is basically just a simple version of Awkward Array with _mutable_ data. Also allows the array to be used in zip/enumerate in numba code by using the .array property (not needed in awkward version >= 0.2.32)."""

    def __init__(self, arraylike_object):

        # if numba List
        if isinstance(arraylike_object, List):
            dtype = get_numba_list_dtype(arraylike_object, as_string=True)
            self._content = np.array(flatten_nested_list(arraylike_object), dtype=getattr(np, dtype))  # float32
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
        elif isinstance(i, tuple) and len(i) == 2 and isinstance(i[0], int) and isinstance(i[1], int):
            x, y = i
            return self._content[self._offsets[x] : self._offsets[x + 1]][y]

    @property
    def array(self):
        return self._array

    def to_awkward(self, return_original_awkward_array=False):
        if return_original_awkward_array and self._awkward_array:
            return self._awkward_array
        elif return_original_awkward_array and not self._awkward_array:
            raise AssertionError(f"No original awkward array (possibly because it was loaded through pickle)")

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
        if key in simulation_utils.INTEGER_SIMULATION_PARAMETERS + ["ID"]:
            d[key] = int(val)
        else:
            d[key] = float(val)
    return DotDict(d)


#%%


def get_d_translate():
    d_translate = {
        "N_tot": r"N_\mathrm{tot}",
        "N_init": r"N_\mathrm{init}",
        # 'N_ages': r'N_\mathrm{ages}',
        "rho": r"\rho",
        "epsilon_rho": r"\epsilon_\rho",
        "mu": r"\mu",
        "sigma_mu": r"\sigma_\mu",
        "beta": r"\beta",
        "sigma_beta": r"\sigma_\beta",
        "lambda_E": r"\lambda_E",
        "lambda_I": r"\lambda_I",
        # 'beta_scaling': r'\beta_\mathrm{scaling}',
        # 'age_mixing': r'\mathrm{age}_\mathrm{mixing}',
        "algo": r"\mathrm{algo}",
        # "ID": r"\mathrm{ID}",
        "ID": "ID",
    }
    return d_translate


def human_format(num, digits=3):
    num = float(f"{num:.{digits}g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "G", "T"][magnitude])


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

    d_translate = get_d_translate()

    # make "exclude" a list of keys to ignore
    if isinstance(exclude, str) or exclude is None:
        exclude = [exclude]

    title = "$"
    for sim_par, val in cfg.items():
        if not sim_par in exclude:
            title += f"{d_translate[sim_par]} = {val}, \,"

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
