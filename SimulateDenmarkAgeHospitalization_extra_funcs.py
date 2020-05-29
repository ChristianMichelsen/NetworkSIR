import numpy as np
import pandas as pd
from numba import njit, prange
from pathlib import Path
import joblib
import multiprocessing as mp
from itertools import product
import matplotlib.pyplot as plt
import rc_params
rc_params.set_rc_params()

# conda install -c numba/label/dev numba

# conda install awkward
# conda install -c conda-forge pyarrow
import awkward

# N_Denmark = 535_806

do_fast_math = False
do_parallel_numba = False


def is_local_computer(N_local_cores=8):
    import platform
    if mp.cpu_count() <= N_local_cores and platform.system() == 'Darwin':
        return True
    else:
        return False

def get_cfg_default():
    cfg_default = dict(
                    # N_tot = 50_000 if is_local_computer() else 500_000, # Total number of nodes!
                    N_tot = 500_000, # Total number of nodes!
                    N_init = 100, # Initial Infected
                    mu = 20.0,  # Average number of connections of a node (init: 20)
                    sigma_mu = 0.0, # Spread (skewness) in N connections
                    rho = 0.0, # Spacial dependency. Average distance to connect with.
                    beta = 0.01, # Daily infection rate (SIR, init: 0-1, but beta = (2mu/N_tot)* betaSIR)
                    sigma_beta = 0.0, # Spread in rates, beta
                    lambda_E = 1.0, # E->I, Lambda(from E states)
                    lambda_I = 1.0, # I->R, Lambda(from I states)
                    epsilon_rho = 0.01, # fraction of connections not depending on distance
                    frac_02 = 0.0, # 0: as normal, 1: half of all (beta)rates are set to 0 the other half doubled
                    connect_algo = 1, # node connection algorithm
                    )
    return cfg_default

sim_pars_ints = ('N_tot', 'N_init', 'connect_algo')


def filename_to_ID(filename):
    return int(filename.split('ID__')[1].strip('.csv'))


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
            return
        raise KeyError(
            "Only allowed to change existing keys with dot notation. Use brackets instead."
        )


def filename_to_dotdict(filename, normal_string=False, animation=False):
    return DotDict(filename_to_dict(filename, normal_string=normal_string, animation=animation))


def get_num_cores(num_cores_max):
    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max
    return num_cores


def dict_to_filename_with_dir(cfg, ID):
    filename = Path('Data') / 'ABN' 
    file_string = ''
    for key, val in cfg.items():
        file_string += f"{key}__{val}__"
    file_string = file_string[:-2] # remove trailing _
    filename = filename / file_string
    file_string += f"__ID__{ID:03d}.csv"
    filename = filename / file_string
    return str(filename)


def filename_to_dict(filename, normal_string=False, animation=False): # , 
    cfg = {}

    if normal_string:
        keyvals = filename.split('__')
    elif animation:
        keyvals = filename.split('/')[-1].split('.animation')[0].split('__')
    else:
        keyvals = str(Path(filename).stem).split('__')

    keyvals_chunks = [keyvals[i:i + 2] for i in range(0, len(keyvals), 2)]
    for key, val in keyvals_chunks:
        if not key == 'ID':
            if key in sim_pars_ints:
                cfg[key] = int(val)
            else:
                cfg[key] = float(val)
    return DotDict(cfg)


def generate_filenames(d_sim_pars, N_loops=10, force_overwrite=False):
    filenames = []

    nameval_to_str = [[f'{name}__{x}' for x in lst] for (name, lst) in d_sim_pars.items()]
    all_combinations = list(product(*nameval_to_str))

    cfg = get_cfg_default()
    # combination = all_combinations[0]
    for combination in all_combinations:
        for s in combination:
            name, val = s.split('__')
            val = int(val) if name in sim_pars_ints else float(val)
            cfg[name] = val

        # ID = 0
        for ID in range(N_loops):
            filename = dict_to_filename_with_dir(cfg, ID)

            not_existing = (not Path(filename).exists())
            if not_existing or force_overwrite: 
                filenames.append(filename)
    return filenames

def get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max):
    num_cores = get_num_cores(num_cores_max)

    if isinstance(d_simulation_parameters, dict) and 'N_tot' in d_simulation_parameters.keys():
        N_tot_max = max(d_simulation_parameters['N_tot'])
        if 500_000 < N_tot_max <= 1_000_000:
            num_cores = 6
        elif 1_000_000 < N_tot_max <= 2_000_000:
            num_cores = 3
        elif 2_000_000 < N_tot_max:
            num_cores = 1

    if num_cores > num_cores_max:
        num_cores = num_cores_max
    
    return num_cores


# @njit(fastmath=do_fast_math)
# def deep_copy_1D_int(X):
#     outer = np.zeros(len(X), np.int_)
#     for ix in range(len(X)):
#         outer[ix] = X[ix]
#     return outer

# @njit(fastmath=do_fast_math)
# def deep_copy_2D_jagged_int(X, min_val=-1):
#     outer = []
#     n, m = X.shape
#     for ix in range(n):
#         inner = []
#         for jx in range(m):
#             if X[ix, jx] > min_val:
#                 inner.append(int(X[ix, jx]))
#         outer.append(inner)
#     return outer


# @njit(fastmath=do_fast_math)
# def deep_copy_2D_jagged(X, min_val=-1):
#     outer = []
#     n, m = X.shape
#     for ix in range(n):
#         inner = []
#         for jx in range(m):
#             if X[ix, jx] > min_val:
#                 inner.append(X[ix, jx])
#         outer.append(inner)
#     return outer


# @njit(fastmath=do_fast_math)
# def deep_copy_2D_int(X):
#     n, m = X.shape
#     outer = np.zeros((n, m), np.int_)
#     for ix in range(n):
#         for jx in range(m):
#             outer[ix, jx] = X[ix, jx]
#     return outer

@njit(fastmath=do_fast_math, parallel=do_parallel_numba)
def haversine(lon1, lat1, lon2, lat2):
    dlon = np.radians(lon2) - np.radians(lon1)
    dlat = np.radians(lat2) - np.radians(lat1) 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


#%%


@njit(fastmath=do_fast_math, parallel=do_parallel_numba)
def initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta, frac_02):

    connection_weight = np.ones(N_tot, dtype=np.float32)
    infection_weight = np.ones(N_tot, dtype=np.float32)

    for i in range(N_tot): # prange
        if (np.random.rand() < sigma_mu):
            connection_weight[i] = 0.1 - np.log(np.random.rand())# / 1.0
        else:
            connection_weight[i] = 1.1

        if (np.random.rand() < sigma_beta):
            infection_weight[i] = -np.log(np.random.rand())*beta
        else:
            infection_weight[i] = beta
        
        ra_R0_change = np.random.rand()
        if ra_R0_change < frac_02/2:
            infection_weight[i] = infection_weight[i]*2
        elif ra_R0_change > 1-frac_02/2:
            infection_weight[i] = 0
        else:
            pass
    
    return connection_weight, infection_weight


@njit
def initialize_in_state(N, dtype):
    nested_list = List()
    for i in range(N):
        tmp = List()
        tmp.append(dtype(-1))
        nested_list.append(tmp)
        nested_list[-1].pop(0) # trick to tell compiler which dtype
    return nested_list#, is_first_value


import numpy as np
import numba as nb
from numba import njit
from numba.typed import List

@njit(fastmath=do_fast_math, parallel=do_parallel_numba)
def initialize_ages(N_tot, N_ages, connection_weight):

    # N_tot = 10
    # N_ages = 3

    ages = -1*np.ones(N_tot, dtype=np.int8)
    ages_total_counts = np.zeros(N_ages, dtype=np.uint32)
    # ages_in_state = -1*np.ones((N_ages, N_tot), dtype=np.int32) # XXX nested
    ages_in_state = initialize_in_state(N_ages, dtype=np.int32) # XXX nested

    for i in range(N_tot): # prange
        age = np.random.randint(N_ages)
        ages[i] = age
        # ages_in_state[age, ages_total_counts[age]] = i # XXX nested
        ages_total_counts[age] += 1
        ages_in_state[age].append(i) 


    PT_ages = np.zeros(N_ages, dtype=np.float32)
    PC_ages = List()
    PP_ages = List()
    for i_age_group in range(N_ages): # prange
        indices = np.asarray(ages_in_state[i_age_group]) # , :ages_total_counts[i_age_group]] # XXX nested
        # indices = ages_in_state[i_age_group] # , :ages_total_counts[i_age_group]] # XXX nested
        connection_weight_ages = connection_weight[indices]
        PT_age = np.sum(connection_weight_ages)
        PC_age = np.cumsum(connection_weight_ages)
        PP_age = PC_age / PT_age

        PT_ages[i_age_group] = PT_age
        PC_ages.append(PC_age)
        PP_ages.append(PP_age)

    return ages, ages_total_counts, ages_in_state, PT_ages, PC_ages, PP_ages


@njit(fastmath=do_fast_math, parallel=do_parallel_numba)
def update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2):

    #  Make sure no element is present twice
    accept = True
    for i1 in range(N_connections[id1]):  # prange
        if which_connections[id1, i1] == id2:
            accept = False

    if (N_connections[id1] < N_AK_MAX) and (N_connections[id2] < N_AK_MAX) and (id1 != id2) and accept:
        r = haversine(coordinates[id1, 0], coordinates[id1, 1], coordinates[id2, 0], coordinates[id2, 1])
        if np.exp(-r*rho_tmp/rho_scale) > np.random.rand():
            
            individual_rates[id1, N_connections[id1]] = infection_weight[id1]
            individual_rates[id2, N_connections[id2]] = infection_weight[id2] # Changed from id1

            which_connections[id1, N_connections[id1]] = id2
            which_connections_reference[id1, N_connections[id1]] = id2                        
            which_connections[id2, N_connections[id2]] = id1 	
            which_connections_reference[id2, N_connections[id2]] = id1

            N_connections[id1] += 1 
            N_connections[id2] += 1
            N_connections_reference[id1] += 1 
            N_connections_reference[id2] += 1
            continue_run = False

    return continue_run


@njit(fastmath=do_fast_math)
def run_algo_2(PP_i, PP_j, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX):

    continue_run = True
    while continue_run:
        
        id1 = np.searchsorted(PP_i, np.random.rand())
        id2 = np.searchsorted(PP_j, np.random.rand())
        
        continue_run = update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2)


@njit(fastmath=do_fast_math)
def run_algo_1(PP_i, PP_j, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX):

    ra1 = np.random.rand()
    id1 = np.searchsorted(PP_i, ra1) 
    N_algo_1_tries = 0

    continue_run = True
    while continue_run:
        ra2 = np.random.rand()          
        id2 = np.searchsorted(PP_j, ra2)
        N_algo_1_tries += 1
        rho_tmp *= 0.9995 

        continue_run = update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2)
    
    return N_algo_1_tries


@njit(fastmath=do_fast_math)
def connect_nodes(mu, epsilon_rho, rho, connect_algo, PP_ages, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_scale, N_AK_MAX, N_ages, age_matrix, verbose):

    num_prints = 0

    for m_i in range(N_ages):
        for m_j in range(N_ages):
            for counter in range(int(age_matrix[m_i, m_j])): 

                if np.random.rand() > epsilon_rho:
                    rho_tmp = rho
                else:
                    rho_tmp = 0.0

                # PP_i, PP_j = PP_ages[0], PP_ages[1]
                
                if (connect_algo == 2):
                    run_algo_2(PP_ages[m_i], PP_ages[m_j], N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX)

                else:
                    N_algo_1_tries = run_algo_1(PP_ages[m_i], PP_ages[m_j], N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX)

                    if verbose and num_prints < 10:
                        # print(N_algo_1_tries, num_prints)
                        num_prints += 1


@njit(fastmath=do_fast_math)
def make_initial_infections(N_init, which_state, state_total_counts, agents_in_state, csMov, N_connections_reference, which_connections, which_connections_reference, N_connections, individual_rates, SIR_transition_rates, ages_in_state, initial_ages_exposed):

    TotMov = 0.0

    # XXX nested
    possible_idxs = List()
    for age_exposed in initial_ages_exposed:
        for agent in ages_in_state[age_exposed]:
            possible_idxs.append(agent)

    # possible_idxs = ages_in_state[initial_ages_exposed].flatten() # XXX nested
    # possible_idxs = possible_idxs[possible_idxs != -1]

    ##  Now make initial infections
    random_indices = np.random.choice(np.asarray(possible_idxs), size=N_init, replace=False)
    for idx in random_indices:
        new_state = np.random.randint(0, 4)
        which_state[idx] = new_state

        agents_in_state[new_state, state_total_counts[new_state]] = idx
        state_total_counts[new_state] += 1  
        TotMov += SIR_transition_rates[new_state]
        csMov[new_state:] += SIR_transition_rates[new_state]
        for i1 in range(N_connections_reference[idx]):
            Af = which_connections_reference[idx, i1]
            for i2 in range(N_connections[Af]):
                if which_connections[Af, i2] == idx:
                    for i3 in range(i2, N_connections[Af]):
                        which_connections[Af, i3] = which_connections[Af, i3+1] 
                        individual_rates[Af, i3] = individual_rates[Af, i3+1]
                    N_connections[Af] -= 1
                    break 
    return TotMov

@njit(fastmath=do_fast_math)
def run_simulation(N_tot, TotMov, csMov, state_total_counts, agents_in_state, which_state, csInf, N_states, InfRat, SIR_transition_rates, infectious_state, N_connections, individual_rates, N_connections_reference, which_connections_reference, which_connections, ages, H_probability_matrix_csum, H_which_state, H_agents_in_state, H_state_total_counts, H_move_matrix_sum, H_cumsum_move, nts, verbose):

    out_time = List()
    out_state_counts = List()
    out_which_state = List()
    out_N_connections = List()

    out_which_connections = which_connections.copy()
    out_individual_rates = individual_rates.copy()

    daily_counter = 0

    Tot = 0.0
    TotInf = 0.0
    click = 0 
    counter = 0
    Csum = 0.0 
    RT = 0.0

    H_tot_move = 0

    # Run the simulation ################################
    continue_run = True
    while continue_run:
        
        counter += 1 
        Tot = TotMov + TotInf + H_tot_move
        dt = - np.log(np.random.rand()) / Tot    
        RT = RT + dt
        Csum = 0.0
        ra1 = np.random.rand()


        #######/ Here we move infected between states
        AC = 0 
        if TotMov / Tot > ra1:
            x = csMov / Tot
            i1 = np.searchsorted(x, ra1)
            Csum = csMov[i1] / Tot
            for i2 in range(state_total_counts[i1]):
                Csum += SIR_transition_rates[i1] / Tot
                if Csum > ra1:
                    idx = agents_in_state[i1, i2]
                    AC = 1
                    break                
            
            # We have chosen idx to move -> here we move it
            agents_in_state[i1+1, state_total_counts[i1+1]] = idx
            for j in range(i2, state_total_counts[i1]):
                agents_in_state[i1, j] = agents_in_state[i1, j+1] 

            which_state[idx] += 1
            state_total_counts[i1] -= 1 
            state_total_counts[i1+1] += 1      
            TotMov -= SIR_transition_rates[i1] 
            TotMov += SIR_transition_rates[i1+1]     
            csMov[i1] -= SIR_transition_rates[i1]
            csMov[i1+1:N_states] += SIR_transition_rates[i1+1]-SIR_transition_rates[i1]
            csInf[i1] -= InfRat[idx]

            # Moves TO infectious State from non-infectious
            if which_state[idx] == infectious_state: 
                for i1 in range(N_connections[idx]): # Loop over row idx	  
                    if which_state[which_connections[idx, i1]] < 0:
                        TotInf += individual_rates[idx, i1]
                        InfRat[idx] += individual_rates[idx, i1]
                        csInf[which_state[idx]:N_states] += individual_rates[idx, i1]
           
            # If this moves to Recovered state
            if which_state[idx] == N_states-1:
                for i1 in range(N_connections[idx]): # Loop over row idx
                    TotInf -= individual_rates[idx, i1] 
                    InfRat[idx] -= individual_rates[idx, i1]
                    csInf[which_state[idx]:N_states] -= individual_rates[idx, i1]


                # XXX HOSPITAL
                # # Now in hospital track
                # H_state = np.searchsorted(H_probability_matrix_csum[ages[idx]], np.random.rand())

                # H_which_state[idx] = H_state
                # H_agents_in_state[H_state, H_state_total_counts[H_state]] = idx 
                # H_state_total_counts[H_state] += 1
                
                # H_tot_move += H_move_matrix_sum[H_state, ages[idx]]
                # H_cumsum_move[H_state:] += H_move_matrix_sum[H_state, ages[idx]] 



        # Here we infect new states
        # elif (TotMov + TotInf) / Tot > ra1:  # XXX HOSPITAL
        else: # XXX HOSPITAL
            x = TotMov/Tot + csInf/Tot
            i1 = np.searchsorted(x, ra1)
            Csum = TotMov/Tot + csInf[i1]/Tot
            for i2 in range(state_total_counts[i1]):
                idy = agents_in_state[i1, i2]
                for i3 in range(N_connections[idy]): 
                    Csum += individual_rates[idy][i3]/Tot
                    if Csum > ra1:
                        idx = which_connections[idy, i3]	      
                        which_state[idx] = 0 
                        agents_in_state[0, state_total_counts[0]] = idx	      
                        state_total_counts[0] += 1
                        TotMov += SIR_transition_rates[0]	      
                        csMov += SIR_transition_rates[0]
                        AC = 1
                        break                    
                if AC == 1:
                    break

            # Here we update infection lists      
            for i1 in range(N_connections_reference[idx]):
                Af = which_connections_reference[idx, i1]
                for i2 in range(N_connections[Af]):
                    if which_connections[Af, i2] == idx:
                        if (which_state[Af] >= infectious_state) and (which_state[Af] < N_states-1):	      
                            TotInf -= individual_rates[Af, i2]
                            InfRat[Af] -= individual_rates[Af, i2]
                            csInf[which_state[Af]:N_states] -= individual_rates[Af, i2]
                        for i3 in range(i2, N_connections[Af]):
                            which_connections[Af, i3] = which_connections[Af, i3+1]
                            individual_rates[Af, i3] = individual_rates[Af, i3+1]
                        N_connections[Af] -= 1 
                        break


        # ## move between hospital tracks
        # else:
        #     x = (TotMov + TotInf + H_cumsum_move) / Tot
        #     i1 = np.searchsorted(x, ra1)
        #     Csum = (TotMov + TotInf + H_cumsum_move[i1]) / Tot
        #     for i2 in range(H_state_total_counts[i1]):
        #         Csum += H_move_matrix_sum[i1] / Tot
        #         if Csum > ra1:
        #             idx = agents_in_state[i1, i2]
        #             AC = 1
        #             H_ra = np.random.rand()

        #             H_tmp = H_move_matrix_cumsum[H_which_state[idx], :, ages[idx]] / H_move_matrix_sum[H_which_state[idx], ages[idx]]
        #             H_new_state = np.searchsorted(H_tmp, H_ra)

        #             # for nested list pop element
        #             # We have chosen idx to move -> here we move it
        #             for h in range(i2, H_state_total_counts[i1]):
        #                 H_agents_in_state[i1, h] = H_agents_in_state[i1, h+1] 

        #             H_which_state[idx] = H_new_state
        #             H_agents_in_state[H_new_state, H_state_total_counts[H_new_state]] = idx 
        #             H_state_total_counts[H_new_state] += 1

        #             break                


        #     # which_state[idx] += 1
        #     # state_total_counts[i1] -= 1 
        #     # state_total_counts[i1+1] += 1      
        #     # TotMov -= SIR_transition_rates[i1] 
        #     # TotMov += SIR_transition_rates[i1+1]     
        #     # csMov[i1] -= SIR_transition_rates[i1]
        #     # csMov[i1+1:N_states] += SIR_transition_rates[i1+1]-SIR_transition_rates[i1]
        #     # csInf[i1] -= InfRat[idx]

            
        ################

        if nts*click < RT:

            daily_counter += 1
            out_time.append(RT)
            out_state_counts.append(state_total_counts.copy())

            if daily_counter >= 10:
                daily_counter = 0

                out_which_state.append(which_state.copy())
                out_N_connections.append(N_connections.copy())

            click += 1 

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        continue_run, TotMov, TotInf = do_bug_check(counter, continue_run, TotInf, TotMov, verbose, state_total_counts, N_states, N_tot, AC, csMov)

    return out_time, out_state_counts, out_which_state, out_N_connections, out_which_connections, out_individual_rates



@njit(fastmath=do_fast_math)
def do_bug_check(counter, continue_run, TotInf, TotMov, verbose, state_total_counts, N_states, N_tot, AC, csMov):

    if counter > 100_000_000: 
        # if verbose:
        print("counter > 100_000_000")
        continue_run = False
    
    if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001): 
        continue_run = False
        if verbose:
            print("Equilibrium")
    
    if state_total_counts[N_states-1] > N_tot-10:      
        if verbose:
            print("2/3 through")
        continue_run = False

    # Check for bugs
    if AC == 0: 
        print("No Chosen rate", csMov)
        continue_run = False
    
    if (TotMov < 0) and (TotMov > -0.001):
        TotMov = 0 
        
    if (TotInf < 0) and (TotInf > -0.001):
        TotInf = 0 
        
    if (TotMov < 0) or (TotInf < 0): 
        print("\nNegative Problem", TotMov, TotInf)
        continue_run = False

    return continue_run, TotMov, TotInf





from numba.typed import List
import time

@njit
def get_size_gb(x):
    return x.size * x.itemsize / 10**9


#%%

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

# @njit(fastmath=do_fast_math, parallel=do_parallel_numba)
@njit(fastmath=do_fast_math)
def single_run_numba(N_tot, N_init, mu, sigma_mu, rho, beta, sigma_beta, lambda_E, lambda_I, connect_algo, epsilon_rho, frac_02, ID, coordinates, verbose=False):
    
    # N_tot = 10000 # Total number of nodes!
    # N_init = 100 # Initial Infected
    # mu = 20.0  # Average number of connections of a node (init: 20)
    # sigma_mu = 0.0 # Spread (skewness) in N connections
    # rho = 0 # Spacial dependency. Average distance to connect with.
    # beta = 0.01 # Daily infection rate (SIR, init: 0-1, but beta = (2mu/N_tot)* betaSIR)
    # sigma_beta = 0.0 # Spread in rates, beta (beta_eff = beta - sigma_beta+2*sigma_beta*rand[0,1])... could be exponential?
    # lambda_E = 1.0 # E->I, Lambda(from E states)
    # lambda_I = 1.0 # I->R, Lambda(from I states)
    # connect_algo = 1 # node connection algorithm
    # epsilon_rho = 0.01 # fraction of connections not depending on distance
    # frac_02 = 0.0 # 0: as normal, 1: half of all (beta)rates are set to 0 the other half doubled
    # ID = 0
    # coordinates = np.load('Data/GPS_coordinates.npy')[:N_tot]
    # verbose = True

    np.random.seed(ID)

    nts = 0.1 # Time step (0.1 - ten times a day)
    N_states = 9 # number of states
    rho_scale = 1000 # scale factor of rho
    N_ages = 3 # TODO XXX
    initial_ages_exposed = np.array([0, 1])

    N_AK_MAX = 1_000 # TODO XXX

    # For generating Network
    which_connections = -1*np.ones((N_tot, N_AK_MAX), dtype=np.int32)  # TODO: Nested list
    which_connections_reference = -1*np.ones((N_tot, N_AK_MAX), dtype=np.int32)  # TODO: Nested list
    # get_size_gb(which_connections)
    individual_rates = -1*np.ones((N_tot, N_AK_MAX), dtype=np.float32)  # TODO: Nested list
    agents_in_state = -1*np.ones((N_states, N_tot), dtype=np.int32) # TODO: Nested list

    state_total_counts = np.zeros(N_states, dtype=np.int32)
    SIR_transition_rates = np.zeros(N_states, dtype=np.float32)

    N_connections = np.zeros(N_tot, dtype=np.int32)
    N_connections_reference = np.zeros(N_tot, dtype=np.int32)
    
    which_state = -1*np.ones(N_tot, dtype=np.int8)
    
    csMov = np.zeros(N_states, dtype=np.float64)
    csInf = np.zeros(N_states, dtype=np.float64)
    InfRat = np.zeros(N_tot, dtype=np.float64)

    infectious_state = 4 # This means the 5'th state

    SIR_transition_rates[:4] = lambda_E
    SIR_transition_rates[4:8] = lambda_I

    # age variables
    
    age_matrix_rel = np.ones((N_ages, N_ages), dtype=np.float64) / N_ages 
    age_connections_rel = np.ones(N_ages, dtype=np.float64) / N_ages
    age_matrix = age_matrix_rel * age_connections_rel * mu * N_tot
    

    # Hospitalization track variables
    H_N_states = 6 # number of states
    H_state_total_counts = np.zeros(H_N_states, dtype=np.int32)
    H_which_state = -1*np.ones(N_tot, dtype=np.int8)
    H_agents_in_state = -1*np.ones((H_N_states, N_tot), dtype=np.int32)
    H_probability_matrix = np.ones((N_ages, H_N_states), dtype=np.float32) / H_N_states
    H_probability_matrix_csum = numba_cumsum(H_probability_matrix, axis=1)

    H_move_matrix = np.zeros((H_N_states, H_N_states, N_ages), dtype=np.float32)
    H_move_matrix[0, 1] = 0.3
    H_move_matrix[1, 2] = 1.0
    H_move_matrix[1, 4] = 0.1
    H_move_matrix[2, 3] = 0.1
    H_move_matrix[3, 4] = 1.0
    H_move_matrix[3, 5] = 0.1

    H_move_matrix_sum = np.sum(H_move_matrix, axis=1) 
    H_move_matrix_cumsum = numba_cumsum(H_move_matrix, axis=1) 

    H_cumsum_move = np.zeros(H_N_states, dtype=np.float64)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RATES AND CONNECTIONS # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("Make rates and connections")

    connection_weight, infection_weight = initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta, frac_02)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # AGES  # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("Make ages")

    ages, ages_total_counts, ages_in_state, PT_ages, PC_ages, PP_ages = initialize_ages(N_tot, N_ages, connection_weight)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # CONNECT NODES # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    if verbose:
        print("CONNECT NODES")

    connect_nodes(mu, epsilon_rho, rho, connect_algo, PP_ages, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_scale, N_AK_MAX, N_ages, age_matrix, verbose)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("INITIAL INFECTIONS")

    TotMov = make_initial_infections(N_init, which_state, state_total_counts, agents_in_state, csMov, N_connections_reference, which_connections, which_connections_reference, N_connections, individual_rates, SIR_transition_rates, ages_in_state, initial_ages_exposed)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RUN SIMULATION  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("RUN SIMULATION")

    res = run_simulation(N_tot, TotMov, csMov, state_total_counts, agents_in_state, which_state, csInf, N_states, InfRat, SIR_transition_rates, infectious_state, N_connections, individual_rates, N_connections_reference, which_connections_reference, which_connections, ages, H_probability_matrix_csum, H_which_state, H_agents_in_state, H_state_total_counts, H_move_matrix_sum, H_cumsum_move, nts, verbose)

    return res



def single_run_and_save(filename, verbose=False):


    filename = 'Data/ABN/N_tot__10000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1/N_tot__10000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1__ID__000.csv'
    verbose=True

    cfg = filename_to_dict(filename)
    ID = filename_to_ID(filename)


    coordinates = np.load('Data/GPS_coordinates.npy')
    if cfg.N_tot > len(coordinates):
        raise AssertionError("N_tot cannot be larger than coordinates (number of generated houses in DK)")

    np.random.seed(ID)
    index = np.arange(len(coordinates))
    index_subset = np.random.choice(index, cfg.N_tot, replace=False)
    coordinates = coordinates[index_subset]

    res = single_run_numba(**cfg, ID=ID, coordinates=coordinates, verbose=verbose)
    # out_single_run, SIRfile_which_state, SIRfile_P1, SIRfile_N_connections, SIRfile_AK_initial, SIRfile_Rate_initial, out_time, out_state_counts = res
    out_time, out_state_counts, out_which_state, out_N_connections, out_which_connections, out_individual_rates = res

    header = [
             'Time', 
            'E1', 'E2', 'E3', 'E4', 
            'I1', 'I2', 'I3', 'I4', 
            'R',
            ]

    df_time = pd.DataFrame(np.array(out_time), columns=header[0:1])
    df_states = pd.DataFrame(np.array(out_state_counts), columns=header[1:])
    df_raw = pd.concat([df_time, df_states], axis=1)#.convert_dtypes()
    # df_raw = pd.DataFrame(out_time, columns=header).convert_dtypes()

    # make sure parent folder exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # save csv file
    df_raw.to_csv(filename, index=False)

    # save which_state, coordinates, and N_connections, once for each set of parameters
    if ID == 0:
        out_which_state = np.array(out_which_state, dtype=np.int8)
        out_N_connections = np.array(out_N_connections, dtype=np.int32)
        
        filename_animation = str(Path('Data_animation') / Path(filename).stem) + '.animation.joblib'

        Path(filename_animation).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump([out_which_state, coordinates, out_N_connections], filename_animation)
        # pickle.dump([SIRfile_which_state, SIRfile_P1, SIRfile_N_connections], open(filename_animation.replace('joblib', 'pickle'), "wb"))

        out_which_connections = awkward.fromiter(out_which_connections).astype(np.int32)
        filename_which_connections = filename_animation.replace('animation.joblib', 'which_connections.parquet')
        awkward.toparquet(filename_which_connections, out_which_connections)

        out_individual_rates = awkward.fromiter(out_individual_rates)
        filename_rates = filename_which_connections.replace('which_connections.parquet', 'rates.parquet')
        awkward.toparquet(filename_rates, out_individual_rates)

    return None



# %%


# %%

if False:

    filename = 'Data/ABN/N_tot__10000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1/N_tot__10000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1__ID__000.csv'
    verbose=True

    cfg = filename_to_dict(filename)
    ID = filename_to_ID(filename)

    cfg.N_tot = 50_000

    coordinates = np.load('Data/GPS_coordinates.npy')
    if cfg.N_tot > len(coordinates):
        raise AssertionError("N_tot cannot be larger than coordinates (number of generated houses in DK)")

    np.random.seed(ID)
    index = np.arange(len(coordinates))
    index_subset = np.random.choice(index, cfg.N_tot, replace=False)
    coordinates = coordinates[index_subset]

    res = single_run_numba(**cfg, ID=ID, coordinates=coordinates, verbose=verbose)
    # out_single_run, SIRfile_which_state, SIRfile_P1, SIRfile_N_connections, SIRfile_AK_initial, SIRfile_Rate_initial, out_time, out_state_counts = res
    out_time, out_state_counts, out_which_state, out_N_connections, out_which_connections, out_individual_rates = res

    header = [
                'Time', 
            'E1', 'E2', 'E3', 'E4', 
            'I1', 'I2', 'I3', 'I4', 
            'R',
            ]

    df_time = pd.DataFrame(np.array(out_time), columns=header[0:1])
    df_states = pd.DataFrame(np.array(out_state_counts), columns=header[1:])
    df_raw = pd.concat([df_time, df_states], axis=1)#.convert_dtypes()

    print(df_raw)



#     # %timeit single_run_numba(**cfg, ID=ID, coordinates=coordinates, verbose=False)
#     @njit(parallel=True)
#     def test(N_tot, N_init, mu, sigma_mu, rho, beta, sigma_beta, lambda_E, lambda_I, connect_algo, epsilon_rho, frac_02, coordinates, verbose=False):
#         for ID in prange(6):
#             print(ID)
#             single_run_numba(N_tot, N_init, mu, sigma_mu, rho, beta, sigma_beta, lambda_E, lambda_I, connect_algo, epsilon_rho, frac_02, ID, coordinates, verbose=verbose)
#         return None

#     test(**cfg, coordinates=coordinates, verbose=False)
#     # %time test(**cfg, coordinates=coordinates, verbose=False)



# def unpack_test(ID):

#     filename = 'Data/ABN/N_tot__10000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1/N_tot__10000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1__ID__000.csv'
#     verbose=True

#     cfg = filename_to_dict(filename)
#     cfg.N_tot = 50_000
#     # ID = filename_to_ID(filename)

#     coordinates = np.load('Data/GPS_coordinates.npy')
#     if cfg.N_tot > len(coordinates):
#         raise AssertionError("N_tot cannot be larger than coordinates (number of generated houses in DK)")

#     np.random.seed(ID)
#     index = np.arange(len(coordinates))
#     index_subset = np.random.choice(index, cfg.N_tot, replace=False)
#     coordinates = coordinates[index_subset]

#     res = single_run_numba(**cfg, ID=ID, coordinates=coordinates, verbose=verbose)
#     # out_single_run, SIRfile_which_state, SIRfile_P1, SIRfile_N_connections, SIRfile_AK_initial, SIRfile_Rate_initial, out_time, out_state_counts = res


# from tqdm import tqdm
# import multiprocessing as mp

# if __name__ == '__main__':
#     with mp.Pool(6) as p:
#         list(tqdm(p.imap_unordered(unpack_test, range(6)), total=6))

# %%
