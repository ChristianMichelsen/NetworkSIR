import numpy as np
from range_key_dict import RangeKeyDict # pip install range-key-dict
from itertools import product
from numba import njit
from numba.typed import List, Dict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import numba as nb

try:
    from src import utils
except ImportError:
    import utils

INTEGER_SIMULATION_PARAMETERS = ['N_tot', 'N_init',  'N_ages', 'algo']


def get_cfg_default():
    """ Default Simulation Parameters """
    cfg_default = dict(
                    N_tot = 580_000, # Total number of nodes!
                    N_init = 100, # Initial Infected
                    N_ages = 1, # Number of age categories
                    mu = 40.0,  # Average number of connections of a node (init: 20)
                    sigma_mu = 0.0, # Spread (skewness) in N connections
                    beta = 0.01, # Daily infection rate (SIR, init: 0-1, but beta = (2mu/N_tot)* betaSIR)
                    sigma_beta = 0.0, # Spread in rates, beta
                    rho = 0.0, # Spacial dependency. Average distance to connect with.
                    lambda_E = 1.0, # E->I, Lambda(from E states)
                    lambda_I = 1.0, # I->R, Lambda(from I states)
                    epsilon_rho = 0.04, # fraction of connections not depending on distance
                    beta_scaling = 1.0, # anmunt of beta scaling
                    age_mixing = 1.0,
                    algo = 2, # node connection algorithm
                    )
    return cfg_default


def dict_to_filename_with_dir(cfg, ID, data_dir='ABM'):
    filename = Path('Data') / data_dir
    file_string = ''
    for key, val in cfg.items():
        file_string += f"{key}__{val}__"
    file_string = file_string[:-2] # remove trailing _
    filename = filename / file_string
    file_string += f"__ID__{ID:03d}.csv"
    filename = filename / file_string
    return str(filename)


def generate_filenames(d_sim_pars, N_loops=10, force_overwrite=False):
    filenames = []

    nameval_to_str = [[f'{name}__{x}' for x in lst] for (name, lst) in d_sim_pars.items()]
    all_combinations = list(product(*nameval_to_str))

    cfg = get_cfg_default()
    # combination = all_combinations[0]
    for combination in all_combinations:
        for s in combination:
            name, val = s.split('__')
            val = int(val) if name in INTEGER_SIMULATION_PARAMETERS else float(val)
            cfg[name] = val

        # ID = 0
        for ID in range(N_loops):
            filename = dict_to_filename_with_dir(cfg, ID)

            not_existing = (not Path(filename).exists())
            try:
                zero_size = (Path(filename).stat().st_size == 0)
            except FileNotFoundError:
                zero_size = True
            if not_existing or zero_size or force_overwrite:
                filenames.append(filename)
    return filenames


d_num_cores_N_tot = RangeKeyDict({
        (0,         1_000_001):  40,
        (1_000_001, 2_000_001):  30,
        (2_000_001, 5_000_001):  20,
        (5_000_001, 10_000_001): 14,
    })


def get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max=None):
    num_cores = utils.get_num_cores(num_cores_max)

    if isinstance(d_simulation_parameters, dict) and 'N_tot' in d_simulation_parameters.keys():
        N_tot_max = max(d_simulation_parameters['N_tot'])
        num_cores = d_num_cores_N_tot[N_tot_max]

    if num_cores > utils.get_num_cores(num_cores_max):
        num_cores = utils.get_num_cores(num_cores_max)

    return num_cores





def load_coordinates(coordinates_filename, N_tot, ID):
    coordinates = np.load(coordinates_filename)
    if N_tot > len(coordinates):
        raise AssertionError("N_tot cannot be larger than coordinates (number of generated houses in DK)")

    np.random.seed(ID)
    index = np.arange(len(coordinates))
    index_subset = np.random.choice(index, N_tot, replace=False)
    return coordinates[index_subset]



#%%

class Filename:
    def __init__(self, filename):

        if isinstance(filename, dict):
            filename = generate_filenames(filename, N_loops=1, force_overwrite=True)[0]

        self._filename = filename
        self.filename = self.filename_prefix + filename
        self.d = self._string_to_dict
        self.cfg = self.simulation_parameters

    def __repr__(self):
        return str(self.d)

    @property
    def _string_to_dict(self):
        d = {}
        filename_stripped = self._filename.replace('.animation', '')
        keyvals = str(Path(filename_stripped).stem).split('__')
        keyvals_chunks = [keyvals[i:i + 2] for i in range(0, len(keyvals), 2)]
        for key, val in keyvals_chunks:
            if key in INTEGER_SIMULATION_PARAMETERS + ['ID']:
                d[key] = int(val)
            else:
                d[key] = float(val)
        return utils.DotDict(d)

    @property
    def to_dict(self): # ,
        return utils.DotDict({key: val for key, val in self.d.items() if key != 'ID'})

    @property
    def simulation_parameters(self):
        return self.to_dict


    @property
    def to_ID(self):
        return self.d['ID']

    @property
    def ID(self):
        return self.to_ID

    @property
    def filename_prefix(self):
        filename_prefix = ''
        if str(Path.cwd()).endswith('src'):
            filename_prefix = '../'
        return filename_prefix

    def _filename_to_network(self, d, filename, extension):
        file_string = ''
        for key, val in d.items():
            file_string += f"{key}__{val}__"
        file_string = file_string[:-2] # remove trailing _
        file_string += extension
        filename = filename / file_string
        return str(filename)


    def get_filename_network_initialisation(self, extension='.hdf5'):
        variables_to_save_in_filename = ['N_tot', 'N_ages', 'mu', 'sigma_mu', 'rho', 'epsilon_rho', 'algo', 'ID']
        d = {key: self.d[key] for key in variables_to_save_in_filename}
        filename = Path(f'{self.filename_prefix}Data') / 'network_initialization'
        return self._filename_to_network(d, filename, extension)
    filename_network_initialisation = property(get_filename_network_initialisation)


    def get_filename_network(self, extension='.hdf5'):
        filename = Path(f'{self.filename_prefix}Data') / 'network'
        return self._filename_to_network(self.d, filename, extension)
    filename_network = property(get_filename_network)


    @property
    def memory_filename(self):
        filename = Path(f'{self.filename_prefix}Data') / 'memory'
        return self._filename_to_network(self.d, filename, '.memory_file.txt')
        # return self.filename_prefix + self.filename.replace('.csv', '.memory_file.txt')

    @property
    def coordinates_filename(self):
        return self.filename_prefix + 'Data/GPS_coordinates.npy'


    @property
    def household_data_filenames(self):
        return (self.filename_prefix + 'Data/PeopleInHousehold_NorthJutland.txt',
                self.filename_prefix + 'Data/AgeDistributionPerPeopleInHousehold_NorthJutland.txt')


#%%


@njit
def calculate_epsilon(alpha_age, N_ages):
    return 1 / N_ages * alpha_age

@njit
def calculate_age_proportions_1D(alpha_age, N_ages):
    """ Only used in v1 of simulation"""
    epsilon = calculate_epsilon(alpha_age, N_ages)
    x = epsilon * np.ones(N_ages, dtype=np.float32)
    x[0] = 1-x[1:].sum()
    return x


@njit
def calculate_age_proportions_2D(alpha_age, N_ages):
    """ Only used in v1 of simulation"""
    epsilon = calculate_epsilon(alpha_age, N_ages)
    A = epsilon * np.ones((N_ages, N_ages), dtype=np.float32)
    for i in range(N_ages):
        A[i, i] = 1-np.sum(np.delete(A[i, :], i))
    return A


@njit
def set_numba_random_seed(seed):
    np.random.seed(seed)


@njit
def _initialize_my_rates_nested_list(my_infection_weight, my_number_of_contacts):
    N_tot = len(my_infection_weight)
    res = List()
    for i in range(N_tot):
        x = np.full(my_number_of_contacts[i], fill_value=my_infection_weight[i], dtype=np.float64)
        res.append(x)
    return res

def initialize_my_rates(my_infection_weight, my_number_of_contacts):
    return utils.MutableArray(_initialize_my_rates_nested_list(my_infection_weight, my_number_of_contacts))


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
    SIR_transition_rates[N_infectious_states:2*N_infectious_states] = cfg.lambda_I
    return SIR_transition_rates

@njit
def _compute_ages_in_state(ages, N_ages):
    ages_in_state = utils.initialize_nested_lists(N_ages, dtype=np.uint32)
    for idx, age in enumerate(ages): # prange
        ages_in_state[age].append(np.uint32(idx))
    return ages_in_state

def compute_ages_in_state(ages, N_ages):
    ages_in_state = _compute_ages_in_state(ages, N_ages)
    ages_in_state = utils.nested_list_to_awkward_array(ages_in_state)
    return ages_in_state


def get_hospitalization_variables(cfg):

    # Hospitalization track variables
    H_N_states = 6 # number of states
    H_state_total_counts = np.zeros(H_N_states, dtype=np.uint32)
    # H_my_state = -1*np.ones(N_tot, dtype=np.int8)
    H_my_state = np.full(cfg.N_tot, -1, dtype=np.int8)

    H_agents_in_state = utils.initialize_nested_lists(H_N_states, dtype=np.uint32)
    H_probability_matrix = np.ones((cfg.N_ages, H_N_states), dtype=np.float32) / H_N_states
    H_probability_matrix_csum = utils.numba_cumsum(H_probability_matrix, axis=1)

    H_move_matrix = np.zeros((H_N_states, H_N_states, cfg.N_ages), dtype=np.float32)
    H_move_matrix[0, 1] = 0.3
    H_move_matrix[1, 2] = 1.0
    H_move_matrix[2, 1] = 0.6
    H_move_matrix[1, 4] = 0.1
    H_move_matrix[2, 3] = 0.1
    H_move_matrix[3, 4] = 1.0
    H_move_matrix[3, 5] = 0.1

    H_move_matrix_sum = np.sum(H_move_matrix, axis=1)
    H_move_matrix_cumsum = utils.numba_cumsum(H_move_matrix, axis=1)

    H_cumsum_move = np.zeros(H_N_states, dtype=np.float64)

    return H_probability_matrix_csum, H_my_state, H_agents_in_state, H_state_total_counts, H_move_matrix_sum, H_cumsum_move, H_move_matrix_cumsum


#%%


def state_counts_to_df(time, state_counts): #

    header = [
            'Time',
            'E1', 'E2', 'E3', 'E4',
            'I1', 'I2', 'I3', 'I4',
            'R',
            # 'H1', 'H2', 'ICU1', 'ICU2', 'R_H', 'D',
            ]

    df_time = pd.DataFrame(time, columns=header[0:1])
    df_states = pd.DataFrame(state_counts, columns=header[1:])
    # df_H_states = pd.DataFrame(H_state_total_counts, columns=header[10:])
    df = pd.concat([df_time, df_states], axis=1)#.convert_dtypes()
    # assert sum(df_H_states.sum(axis=1) == df_states['R'])
    return df



#%%


def parse_memory_file(filename):

    change_points = {}

    d_time_mem = {}

    next_is_change_point = 0
    # zero_time = None

    import csv
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
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
    df_time_memory = pd.DataFrame.from_dict(d_time_mem, orient='index')
    df_time_memory.columns = ['Memory']

    df_time_memory['ChangePoint'] = s_change_points

    df_change_points = s_change_points.to_frame()
    df_change_points.columns = ['ChangePoint']
    df_change_points['Time'] = df_change_points.index
    df_change_points = df_change_points.set_index('ChangePoint')
    df_change_points['TimeDiff'] = -df_change_points['Time'].diff(-1)
    df_change_points['TimeDiffRel'] = df_change_points['TimeDiff'] / df_change_points['Time'].iloc[-1]

    df_change_points['Memory'] = df_time_memory.loc[df_change_points['Time']]['Memory'].values
    df_change_points['MemoryDiff'] = -df_change_points['Memory'].diff(-1)
    df_change_points['MemoryDiffRel'] = df_change_points['MemoryDiff'] / df_change_points['Memory'].iloc[-1]

    df_change_points.index.name = None

    return df_time_memory, df_change_points


def plot_memory_comsumption(df_time_memory, df_change_points, min_TimeDiffRel=0.1, min_MemoryDiffRel=0.1, time_unit='min'):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [col for i, col in enumerate(colors) if i != 5]

    time_scale = {'s': 1, 'sec': 1,
                  'm': 60, 'min': 60,
                  'h': 60*60, 't': 60*60}

    fig, ax = plt.subplots()

    #
    df_change_points_non_compilation = df_change_points.query("index != 'Numba Compilation'")
    df_change_points_compilation = df_change_points.query("index == 'Numba Compilation'")
    # marker='s'

    ax.plot(df_time_memory.index / time_scale[time_unit], df_time_memory['Memory'], '.', c=colors[0], zorder=2, label='Data Points')
    ax.scatter(df_change_points_non_compilation['Time'] / time_scale[time_unit], df_change_points_non_compilation['Memory'], s=200, c='white', edgecolors='k', zorder=3, label='Change Points')
    ax.scatter(df_change_points_compilation['Time'] / time_scale[time_unit], df_change_points_compilation['Memory'], s=200, c='white', edgecolors='k', zorder=2, label='Numba Compilations', marker='s')
    ax.set(xlabel=f'Time [{time_unit}]', ylabel='Memory [GiB]', ylim=(0, None)) # xlim=(0, None)

    ymax = ax.get_ylim()[1]
    i = 1
    for index, row in df_change_points_non_compilation.iterrows():
        # first_or_last = (i == 0) or (i == len(df_change_points)-1)
        last = (index == df_change_points_non_compilation.index[-1])
        large_time_diff = row['TimeDiffRel'] > min_TimeDiffRel # sec
        large_memory_diff = np.abs(row['MemoryDiffRel']) >  min_MemoryDiffRel # GiB
        if any([last, large_time_diff, large_memory_diff]):
            t = row['Time'] / time_scale[time_unit]
            y = row['Memory']
            col = colors[(i)%len(colors)]
            i += 1
            ax.plot([t, t], [0, y], ls='--',    color=col, zorder=1, label=index)
            ax.plot([t, t], [y, ymax], ls='--', color=col, zorder=1, alpha=.5)

            if row['TimeDiffRel'] > 0.01 or last:
                kwargs = dict(rotation=90, color=col, fontsize=22, ha='center', va='center', bbox=dict(boxstyle="square", ec=col, fc='white'))
                if y / ymax > 0.45:
                    ax.text(t, y/2, index, **kwargs)
                else:
                    ax.text(t, (ymax+y)/2, index, **kwargs)

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

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
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

    base_dir = Path('Data') / 'ABM'
    simulation_parameters = sorted([x.name for x in base_dir.glob('*') if 'N_tot' in x.name])
    d_simulation_parameters = {s: utils.string_to_dict(s) for s in simulation_parameters}
    df_simulation_parameters = pd.DataFrame.from_dict(d_simulation_parameters, orient='index')

    parameters = get_cfg_default()
    for key, val in non_defaults.items():
        parameters[key] = val

    if isinstance(parameter, str):
        parameter = [parameter]

    query = ''
    for key, val in parameters.items():
        if not key in parameter:
            query += f"{key} == {val} & "
    query = query[:-3]

    df_different_than_default = df_simulation_parameters.query(query).sort_values(parameter)
    return list(df_different_than_default.index)


#%%

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


    with open(filename, 'r') as file:
        N_persons = -1
        for line in file:
            line = line.strip()
            if line[0] == '#':
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

    with open(filename, 'r') as file:
        N_persons = -1
        for line in file:
            line = line.strip()
            if line[0] == '#':
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
    people_in_household = parse_household_data_list(household_data_filenames[0], convert_to_numpy=True)
    age_distribution_per_people_in_household = parse_household_data_list(household_data_filenames[1], convert_to_numpy=True)
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

def load_coordinates_Nordjylland(N_tot=150_000, verbose=False):
    all_coordinates = np.load('../Data/GPS_coordinates.npy')
    coordinates = nb_load_coordinates_Nordjylland(all_coordinates, N_tot, verbose)
    return np.array(coordinates)
