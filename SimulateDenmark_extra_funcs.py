import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import joblib
import multiprocessing as mp
from itertools import product
import matplotlib.pyplot as plt
import rc_params
rc_params.set_rc_params()

# conda install awkward
# conda install -c conda-forge pyarrow
import awkward

# N_Denmark = 535_806

def is_local_computer(N_local_cores=8):
    import platform
    if mp.cpu_count() <= N_local_cores and platform.system() == 'Darwin':
        return True
    else:
        return False

def get_cfg_default():
    cfg_default = dict(
                    N_tot = 50_000 if is_local_computer() else 500_000, # Total number of nodes!
                    N_init = 100, # Initial Infected
                    mu = 20.0,  # Average number of connections of a node (init: 20)
                    sigma_mu = 0.0, # Spread (skewness) in N connections
                    rho = 0, # Spacial dependency. Average distance to connect with.
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

        print(cfg)

        # ID = 0
        for ID in range(N_loops):
            filename = dict_to_filename_with_dir(cfg, ID)

            not_existing = (not Path(filename).exists())
            if not_existing or force_overwrite: 
                filenames.append(filename)
    return filenames


# def get_filenames_iter(all_sim_pars, force_animation, N_loops):
#     all_filenames = []
#     if not force_animation:
#         d_sim_pars = []
#         for d_simulation_parameters in all_sim_pars:
#             filenames = generate_filenames(d_simulation_parameters, N_loops, force_animation=force_animation)
#             all_filenames.append(filenames)
#             d_sim_pars.append(d_simulation_parameters)
#     else:
#         for d_simulation_parameters in all_sim_pars:
#             all_filenames.extend(generate_filenames(d_simulation_parameters, N_loops, force_animation=force_animation))
#         all_filenames = [all_filenames]
#         d_sim_pars = ['all configurations']
#     return all_filenames, d_sim_pars


def get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max):
    num_cores = get_num_cores(num_cores_max)

    if isinstance(d_simulation_parameters, dict) and 'N_tot' in d_simulation_parameters.keys():
        if max(d_simulation_parameters['N_tot']) > 2_000_000:
            num_cores = 1
        elif 600_000 <= max(d_simulation_parameters['N_tot']) <= 2_000_000:
            num_cores = 2
            
    elif isinstance(d_simulation_parameters, str):
        num_cores = 1
    
    return num_cores


@njit
def deep_copy_1D_int(X):
    outer = np.zeros(len(X), np.int_)
    for ix in range(len(X)):
        outer[ix] = X[ix]
    return outer

@njit
def deep_copy_2D_jagged_int(X, min_val=-1):
    outer = []
    n, m = X.shape
    for ix in range(n):
        inner = []
        for jx in range(m):
            if X[ix, jx] > min_val:
                inner.append(int(X[ix, jx]))
        outer.append(inner)
    return outer


@njit
def deep_copy_2D_jagged(X, min_val=-1):
    outer = []
    n, m = X.shape
    for ix in range(n):
        inner = []
        for jx in range(m):
            if X[ix, jx] > min_val:
                inner.append(X[ix, jx])
        outer.append(inner)
    return outer


@njit
def deep_copy_2D_int(X):
    n, m = X.shape
    outer = np.zeros((n, m), np.int_)
    for ix in range(n):
        for jx in range(m):
            outer[ix, jx] = X[ix, jx]
    return outer

@njit
def haversine(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2) - np.radians(lon1)
    dlat = np.radians(lat2) - np.radians(lat1) 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


# def foo(N_connections_reference, idx, which_connections_reference, N_connections):
#     # Here we update infection lists      
#     for i1 in range(N_connections_reference[idx]):
#         acc = 0
#         Af = which_connections_reference[idx, i1]
#         for i2 in range(N_connections[Af]):
#             if which_connections[Af, i2] == idx:
#                 if (which_state[Af] >= infectious_state) and (which_state[Af] < N_states-1):	      
#                     TotInf -= individual_rates[Af, i2]
#                     InfRat[Af] -= individual_rates[Af, i2]
#                     csInf[which_state[Af]:N_states] -= individual_rates[Af, i2]
#                 for i3 in range(i2, N_connections[Af]):
#                     which_connections[Af, i3] = which_connections[Af, i3+1]
#                     individual_rates[Af, i3] = individual_rates[Af, i3+1]
#                 N_connections[Af] -= 1 
#                 break



@njit
def single_run_numba(N_tot, N_init, mu, sigma_mu, rho, beta, sigma_beta, lambda_E, lambda_I, connect_algo, epsilon_rho, frac_02, ID, coordinates, verbose=False):
    
    # N_tot = 50_000 # Total number of nodes!
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

    N_AK_MAX = 1000

    # For generating Network
    which_connections = -1*np.ones((N_tot, N_AK_MAX), np.uint16)
    which_connections_reference = -1*np.ones((N_tot, N_AK_MAX), np.uint16)

    N_connections = np.zeros(N_tot, np.int_)
    N_connections_reference = np.zeros(N_tot, np.int_)
    
    connection_weight = np.ones(N_tot)
    infection_weight = np.ones(N_tot)
    which_state = -1*np.ones(N_tot, np.uint8)
    
    individual_rates = -1*np.ones((N_tot, N_AK_MAX))
    agents_in_state = -1*np.ones((N_states, N_tot), np.uint16)
    state_total_counts = np.zeros(N_states, np.int_)
    SIR_transition_rates = np.zeros(N_states)

    csMov = np.zeros(N_states)
    csInf = np.zeros(N_states)
    InfRat = np.zeros(N_tot)

    infectious_state = 4 # This means the 5'th state

    SIR_transition_rates[:4] = lambda_E
    SIR_transition_rates[4:8] = lambda_I

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RATES AND CONNECTIONS # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    if verbose:
        print("Make rates and connections")

    for i in range(N_tot):
        ra = np.random.rand()
        if (ra < sigma_mu):
            connection_weight[i] = 0.1 - np.log(np.random.rand())# / 1.0
        else:
            connection_weight[i] = 1.1

        ra = np.random.rand()
        if (ra < sigma_beta):
            rat = -np.log(np.random.rand())*beta
            infection_weight[i] = rat
        else:
            infection_weight[i] = beta
        
        ra_R0_change = np.random.rand()
        if ra_R0_change < frac_02/2:
            infection_weight[i] = infection_weight[i]*2
        elif ra_R0_change > 1-frac_02/2:
            infection_weight[i] = 0
        else:
            pass

    PT = np.sum(connection_weight)
    PC = np.cumsum(connection_weight)
    PP = PC/PT

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # CONNECT NODES # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    if verbose:
        print("CONNECT NODES")
    
    if (connect_algo == 2):
        for c in range(int(mu*N_tot)):
            accra = 0
            while accra == 0:
                ra1 = np.random.rand()
                ra2 = np.random.rand()            
                id1 = np.searchsorted(PP, ra1)
                id2 = np.searchsorted(PP, ra2)
                acc = 1
                for i1 in range(N_connections[id1]):         #  Make sure no element is present twice
                    if which_connections[id1, i1] == id2:
                        acc = 0         
                if (N_connections[id1] < N_AK_MAX) and (N_connections[id2] < N_AK_MAX) and (id1 != id2) and (acc == 1):
                    
                    # r = np.sqrt((coordinates[id1, 0] - coordinates[id2, 0])**2 + (coordinates[id1, 1] - coordinates[id2, 1])**2)
                    r = haversine(coordinates[id1, 0], coordinates[id1, 1], coordinates[id2, 0], coordinates[id2, 1])

                    ra = np.random.rand()
                    if np.exp(-r*rho/rho_scale) > ra:
                        individual_rates[id1, N_connections[id1]] = infection_weight[id1]
                        individual_rates[id2, N_connections[id2]] = infection_weight[id1]

                        which_connections[id1, N_connections[id1]] = id2	        
                        which_connections_reference[id1, N_connections[id1]] = id2                        
                        which_connections[id2, N_connections[id2]] = id1 	
                        which_connections_reference[id2, N_connections[id2]] = id1

                        N_connections[id1] += 1 
                        N_connections[id2] += 1
                        N_connections_reference[id1] += 1 
                        N_connections_reference[id2] += 1
                        accra = 1                    
    else:
        # N_cac = 100
        num_prints = 0
        for c in range(int(mu*N_tot)):
            ra1 = np.random.rand()
            id1 = np.searchsorted(PP, ra1) 
            accra = 0
            cac = 0

            ra_rho = np.random.rand()
            if ra_rho >= epsilon_rho:
                rho_tmp = rho
            else:
                rho_tmp = 0

            while accra == 0:
                ra2 = np.random.rand()          
                id2 = np.searchsorted(PP, ra2)
                acc = 1
                cac += 1

                rho_tmp *= 0.9995 # 1.0005 # 

                #  Make sure no element is present twice
                for i1 in range(N_connections[id1]):               
                    if which_connections[id1, i1] == id2:
                        acc = 0
                if (N_connections[id1] < N_AK_MAX) and (N_connections[id2] < N_AK_MAX) and (id1 != id2) and (acc == 1):
                    # r = np.sqrt((coordinates[id1, 0] - coordinates[id2, 0])**2 + (coordinates[id1, 1] - coordinates[id2, 1])**2)
                    r = haversine(coordinates[id1, 0], coordinates[id1, 1], coordinates[id2, 0], coordinates[id2, 1])

                    ra = np.random.rand()
                    if np.exp(-r*rho/rho_scale) > ra:
                    
                        individual_rates[id1, N_connections[id1]] = infection_weight[id1]
                        individual_rates[id2, N_connections[id2]] = infection_weight[id1]

                        which_connections[id1, N_connections[id1]] = id2
                        which_connections_reference[id1, N_connections[id1]] = id2
                        which_connections[id2, N_connections[id2]] = id1
                        which_connections_reference[id2, N_connections[id2]] = id1

                        N_connections[id1] += 1
                        N_connections[id2] += 1
                        N_connections_reference[id1] += 1
                        N_connections_reference[id2] += 1
                        accra = 1
                        # print(c)

                        
    if verbose:
        print(cac, num_prints)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    if verbose:
        print("INITIAL INFECTIONS")


    on = 1  
    Tot = 0  
    TotMov = 0 
    TotInf = 0  
    click = 0 
    c = 0  
    Csum = 0 
    RT = 0 

    ##  Now make initial infectious
    for iin in range(N_init):
        idx = iin*10
        # which_state[idx] = 0
        new_state = np.random.randint(0, 4)
        # new_state = 0
        if verbose:
            print("new_state", new_state)
        which_state[idx] = new_state

        agents_in_state[new_state, state_total_counts[new_state]] = idx
        state_total_counts[new_state] += 1  
        # DK[idx] = 1  
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


    #   #############/

    SIRfile = []
    SIRfile_which_state = []
    SIRfile_N_connections = []
    SIRfile_which_connections = deep_copy_2D_jagged_int(which_connections)
    SIRfile_individual_rates = deep_copy_2D_jagged(individual_rates)
    
    SIRfile_daily_counter = 0


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RUN SIMULATION  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    if verbose:
        print("RUN SIMULATION")

    # Run the simulation ################################
    while on == 1:
        
        c += 1 
        Tot = TotMov + TotInf
        ra1 = np.random.rand()   
        dt = - np.log(ra1)/Tot    
        RT = RT + dt
        Csum = 0 
        ra1 = np.random.rand()
        #######/ Here we move infected between states

        AC = 0 
        if TotMov/Tot > ra1:
            x = csMov/Tot
            i1 = np.searchsorted(x, ra1)
            Csum = csMov[i1]/Tot
            for i2 in range(state_total_counts[i1]):
                Csum += SIR_transition_rates[i1]/Tot
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
            csMov[i1+1:N_states] += (SIR_transition_rates[i1+1]-SIR_transition_rates[i1])
            csInf[i1] -= InfRat[idx]

            if which_state[idx] == infectious_state: # Moves TO infectious State from non-infectious
                for i1 in range(N_connections[idx]): # Loop over row idx	  
                    if which_state[which_connections[idx, i1]] < 0:
                        TotInf += individual_rates[idx, i1]
                        InfRat[idx] += individual_rates[idx, i1]
                        csInf[which_state[idx]:N_states] += individual_rates[idx, i1]
            if which_state[idx] == N_states-1: # If this moves to Recovered state
                for i1 in range(N_connections[idx]): # Loop over row idx
                    TotInf -= individual_rates[idx, i1] 
                    InfRat[idx] -= individual_rates[idx, i1]
                    csInf[which_state[idx]:N_states] -= individual_rates[idx, i1]


        # Here we infect new states
        else:
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
                        # NrDInf += 1
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
                acc = 0
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

        ################

        if nts*click < RT:
            SIRfile_tmp = np.zeros(N_states + 1)
            icount = 0
            SIRfile_tmp[icount] = RT
            for s in state_total_counts:
                icount += 1
                SIRfile_tmp[icount] = s #<< "\t"
            SIRfile.append(SIRfile_tmp)
            SIRfile_daily_counter += 1

            if SIRfile_daily_counter >= 10:
                SIRfile_daily_counter = 0

                # deepcopy
                SIRfile_which_state.append(deep_copy_1D_int(which_state))
                SIRfile_N_connections.append(deep_copy_1D_int(N_connections))

            click += 1 

    
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        if c > 100_000_000: 
            # if verbose:
            print("c > 100_000_000")
            on = 0
        
        if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001): 
            on = 0 
            if verbose:
                print("Equilibrium")
        
        if state_total_counts[N_states-1] > N_tot-10:      
            if verbose:
                print("2/3 through")
            on = 0

        # Check for bugs
        if AC == 0: 
            print("No Chosen rate", csMov)
            on = 0
        
        if (TotMov < 0) and (TotMov > -0.001):
            TotMov = 0 
            
        if (TotInf < 0) and (TotInf > -0.001):
            TotInf = 0 
            
        if (TotMov < 0) or (TotInf < 0): 
            print("\nNegative Problem", TotMov, TotInf)
            print(rho, beta, sigma_mu)
            on = 0 
    
    return SIRfile, SIRfile_which_state, coordinates, SIRfile_N_connections, SIRfile_which_connections, SIRfile_individual_rates




def single_run_and_save(filename, verbose=False):

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
    out_single_run, SIRfile_which_state, SIRfile_P1, SIRfile_N_connections, SIRfile_AK_initial, SIRfile_Rate_initial = res

    header = ['Time', 
            'E1', 'E2', 'E3', 'E4', 
            'I1', 'I2', 'I3', 'I4', 
            'R',
            ]

    df_raw = pd.DataFrame(out_single_run, columns=header).convert_dtypes()

    # make sure parent folder exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # save csv file
    df_raw.to_csv(filename, index=False)

    # save which_state, coordinates, and N_connections, once for each set of parameters
    if ID == 0:
        SIRfile_which_state = np.array(SIRfile_which_state, dtype=int)
        SIRfile_P1 = np.array(SIRfile_P1)
        SIRfile_N_connections = np.array(SIRfile_N_connections, dtype=int)
        
        filename_animation = str(Path('Data_animation') / Path(filename).stem) + '.animation.joblib'

        Path(filename_animation).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump([SIRfile_which_state, SIRfile_P1, SIRfile_N_connections], filename_animation)
        # pickle.dump([SIRfile_which_state, SIRfile_P1, SIRfile_N_connections], open(filename_animation.replace('joblib', 'pickle'), "wb"))

        SIRfile_AK_initial = awkward.fromiter(SIRfile_AK_initial).astype(np.int32)
        filename_AK = filename_animation.replace('animation.joblib', 'AK_initial.parquet')
        awkward.toparquet(filename_AK, SIRfile_AK_initial)

        SIRfile_Rate_initial = awkward.fromiter(SIRfile_Rate_initial)
        filename_Rate = filename_AK.replace('AK_initial.parquet', 'Rate_initial.parquet')
        awkward.toparquet(filename_Rate, SIRfile_Rate_initial)

    return None


# def convert_df(df_raw):

#     for state in ['E', 'I']:
#         df_raw[state] = sum([df_raw[col] for col in df_raw.columns if state in col and len(col) == 2])
#     # only keep relevant columns
#     df = df_raw[['Time', 'E', 'I', 'R']].copy()
#     return df
