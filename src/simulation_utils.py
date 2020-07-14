import numpy as np
from range_key_dict import RangeKeyDict # pip install range-key-dict
from itertools import product
from numba import njit

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
                    epsilon_rho = 0.01, # fraction of connections not depending on distance
                    beta_scaling = 1.0, # anmunt of beta scaling
                    age_mixing = 1.0,
                    algo = 2, # node connection algorithm
                    )
    return cfg_default




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
        (2_000_001, 5_000_001):  15,
        (5_000_001, 10_000_001): 12,
    })


def get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max=None):
    num_cores = get_num_cores(num_cores_max)

    if isinstance(d_simulation_parameters, dict) and 'N_tot' in d_simulation_parameters.keys():
        N_tot_max = max(d_simulation_parameters['N_tot'])
        num_cores = d_num_cores_N_tot[N_tot_max]

    if num_cores > get_num_cores(num_cores_max):
        num_cores = get_num_cores(num_cores_max)

    return num_cores



@njit
def calculate_epsilon(alpha_age, N_ages):
    return 1 / N_ages * alpha_age

@njit
def calculate_age_proportions_1D(alpha_age, N_ages):
    epsilon = calculate_epsilon(alpha_age, N_ages)
    x = epsilon * np.ones(N_ages, dtype=np.float32)
    x[0] = 1-x[1:].sum()
    return x


@njit
def calculate_age_proportions_2D(alpha_age, N_ages):
    epsilon = calculate_epsilon(alpha_age, N_ages)
    A = epsilon * np.ones((N_ages, N_ages), dtype=np.float32)
    for i in range(N_ages):
        A[i, i] = 1-np.sum(np.delete(A[i, :], i))
    return A
