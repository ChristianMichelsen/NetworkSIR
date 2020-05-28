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


#%%

import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types
import numba as nb

spec = {
        'value': int32,               # a simple scalar field
        'array': float32[:],          # an array field
        }

@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] = val
        return self.array

# mybag = Bag(123)
# print(mybag.value, mybag.array)
# mybag.increment(3)
# print(mybag.value, mybag.array)


#%%

N_tot = 100_000
coordinates = np.load('Data/GPS_coordinates.npy')[:N_tot]


#%%

from numba import types, typed

spec = {
        'N_tot':  nb.int32,
        'N_init':  nb.int32,
        'mu':  nb.float32,
        'sigma_mu':  nb.float32,
        'rho':  nb.float32,
        'beta':  nb.float32,
        'sigma_beta':  nb.float32,
        'lambda_E':  nb.float32,
        'lambda_I':  nb.float32,
        'connect_algo':  nb.uint8,
        'epsilon_rho':  nb.float32,
        'frac_02': nb.float32,
        'ID': nb.uint16,
        'coordinates': nb.float64[:, :],
        'verbose': nb.boolean,


        'nts': nb.float32, # Time step (0.1 - ten times a day)
        'N_states': nb.uint8, # number of states
        'rho_scale': nb.float32, # scale factor of rho

        'N_AK_MAX': nb.uint16, 


        'which_connections': nb.int32[:, :],
        'which_connections_reference': nb.int32[:, :],
        'N_connections': nb.uint16[:],
        'N_connections_reference': nb.uint16[:],
        'connection_weight': nb.float32[:],
        'infection_weight': nb.float32[:],
        'which_state': nb.int8[:],
        'individual_rates': nb.float32[:, :],
        'agents_in_state': nb.uint16[:, :],
        'state_total_counts': nb.int32[:],
        'SIR_transition_rates': nb.float32[:],
        'csMov': nb.float32[:],
        'csInf': nb.float32[:],
        'InfRat': nb.float32[:],
        'infectious_state': nb.uint8,
        
        'PT': nb.float32, 
        'PC': nb.float32[:], 
        'PP': nb.float32[:], 

        'N_tot_mu': nb.uint64,
        'rho_tmp': nb.float32,
        'accra': nb.boolean,
        'acc': nb.boolean,
        'id1': nb.int32,
        'id2': nb.int32,
        'r': nb.float32,
        'num_prints': nb.int32,
        'cac': nb.int32,

        'random_indices': nb.int32[:],
        'idx': nb.int32,
        'Af': nb.uint16,
        'TotMov': nb.float32,
        
        'Tot': nb.float32,
        'TotInf': nb.float32,
        'click': nb.float32,
        'c': nb.int32,
        'Csum': nb.float32,
        'RT': nb.float64,
        'ra1': nb.float32,
        'AC': nb.boolean,
        'x':  nb.float32[:],
        'i1': nb.int32,
        'continue_running': nb.boolean,
        'idy': nb.uint16,
        'idx2': nb.int32,

        'out_time':  types.ListType(types.float64),
        # 'out_state_counts':  types.ListType(types.int32[:]),
        'out_state_counts': types.List(types.Array(types.int32, 1, 'C')),
        'first_run': nb.boolean,
        'first_run_daily': nb.boolean,
        'daily_counter': nb.int16,
}


from numba import typed, typeof
# typeof(out_single_run)

# l = typed.List()
# l.append(np.arange(5))
# l.append(np.arange(2))
# typeof(l)


# @jitclass(spec)
# class ABN(object):
#     def __init__(self, N_tot, N_init, mu, sigma_mu, rho, beta, sigma_beta, lambda_E, lambda_I, connect_algo, epsilon_rho, frac_02, ID, coordinates, verbose=False):

#         self.N_tot = N_tot
#         self.N_init = N_init
#         self.mu = mu
#         self.sigma_mu = sigma_mu
#         self.rho = rho
#         self.beta = beta
#         self.sigma_beta = sigma_beta
#         self.lambda_E = lambda_E
#         self.lambda_I = lambda_I
#         self.connect_algo = connect_algo
#         self.epsilon_rho = epsilon_rho
#         self.frac_02 = frac_02
#         self.ID = ID
#         self.coordinates = coordinates
#         self.verbose = verbose

#         np.random.seed(ID)
#         self.nts = 0.1 # Time step (0.1 - ten times a day)
#         self.N_states = 9 # number of states
#         self.rho_scale = 1000.0 # scale factor of rho


#         # For generating Network
#         self.N_AK_MAX = 1000
        
#         self.which_connections = np.full((N_tot, self.N_AK_MAX), fill_value=-1, dtype=np.int32)
#         self.which_connections_reference = np.full((N_tot, self.N_AK_MAX), fill_value=-1, dtype=np.int32)

#         self.N_connections = np.zeros(N_tot, dtype=np.uint16)
#         self.N_connections_reference = np.zeros(N_tot, dtype=np.uint16)
        
#         self.connection_weight = np.ones(N_tot, dtype=np.float32)
#         self.infection_weight = np.ones(N_tot, dtype=np.float32)
#         self.which_state = np.full(N_tot, fill_value=-1, dtype=np.int8)
        
#         self.individual_rates = np.full((N_tot, self.N_AK_MAX), fill_value=-1, dtype=np.float32)
#         self.agents_in_state = np.full((self.N_states, N_tot), fill_value=-1, dtype=np.uint16)
#         self.state_total_counts = np.zeros(self.N_states, dtype=np.int32)
#         self.SIR_transition_rates = np.zeros(self.N_states, dtype=np.float32)
#         self.SIR_transition_rates[:4] = lambda_E
#         self.SIR_transition_rates[4:8] = lambda_I


#         self.TotMov = 0.0
#         self.csMov = np.zeros(self.N_states, dtype=np.float32)
#         self.csInf = np.zeros(self.N_states, dtype=np.float32)
#         self.InfRat = np.zeros(N_tot, dtype=np.float32)

#         self.infectious_state = 4 # This means the 5'th state

        
#         self.out_time = typed.List.empty_list(types.float64)
#         # self.out_state_counts = typed.List.empty_list(types.Array(types.int32, 1, 'C'))
#         self.out_state_counts = [np.array([0], dtype=np.int32)]

        

#         # outfile = []
#         # out_which_state = []
#         # out_N_connections = []
#         # # out_which_connections = deep_copy_2D_jagged_int(which_connections)
#         # # out_individual_rates = deep_copy_2D_jagged(individual_rates)
#         # out_daily_counter = 0


#         self.Tot = 0.0
#         self.TotInf = 0.0  
#         self.click = 0.0
#         self.c = 0
#         self.Csum = 0.0
#         self.RT = 0.0


#     def make_rates_and_connections(self):
#         if self.verbose:
#             print("Make rates and connections")

#         for i in range(self.N_tot):
#             if (np.random.rand() < self.sigma_mu):
#                 self.connection_weight[i] = 0.1 - np.log(np.random.rand())# / 1.0
#             else:
#                 self.connection_weight[i] = 1.1

#             if (np.random.rand() < self.sigma_beta):
#                 self.infection_weight[i] = -np.log(np.random.rand())*self.beta
#             else:
#                 self.infection_weight[i] = self.beta
            
#             ra_R0_change = np.random.rand()
#             if ra_R0_change < self.frac_02/2:
#                 self.infection_weight[i] = self.infection_weight[i]*2
#             elif ra_R0_change > 1-self.frac_02/2:
#                 self.infection_weight[i] = 0.0
#             else:
#                 pass

#         self.PT = np.sum(self.connection_weight)
#         self.PC = np.cumsum(self.connection_weight)
#         self.PP = self.PC / self.PT
#         return None

    
#     def _get_rho_tmp(self):
#         if (self.epsilon_rho > 0) and (np.random.rand() < self.epsilon_rho): 
#             return 0
#         else:
#             return self.rho


#     def _update_node_connections(self, id1, id2, rho_tmp):

#         acc = True
#         accra = True

#         #  Make sure no element is present twice
#         for i1 in range(self.N_connections[id1]):         
#             if self.which_connections[id1, i1] == id2:
#                 acc = False

#         if (self.N_connections[id1] < self.N_AK_MAX) and (self.N_connections[id2] < self.N_AK_MAX) and (id1 != id2) and acc:
#             r = haversine(self.coordinates[id1, 0], self.coordinates[id1, 1], self.coordinates[id2, 0], self.coordinates[id2, 1])
            
#             if np.exp(-r*rho_tmp/self.rho_scale) > np.random.rand():
#                 self.individual_rates[id1, self.N_connections[id1]] = self.infection_weight[id1]
#                 self.individual_rates[id2, self.N_connections[id2]] = self.infection_weight[id1]

#                 self.which_connections[id1, self.N_connections[id1]] = id2	        
#                 self.which_connections_reference[id1, self.N_connections[id1]] = id2                        
#                 self.which_connections[id2, self.N_connections[id2]] = id1 	
#                 self.which_connections_reference[id2, self.N_connections[id2]] = id1

#                 self.N_connections[id1] += 1 
#                 self.N_connections[id2] += 1
#                 self.N_connections_reference[id1] += 1 
#                 self.N_connections_reference[id2] += 1
#                 accra = False
#         return accra


#     def _run_connect_algo_1(self):
#         N_tot_mu = int(self.mu*self.N_tot)
#         num_prints = 0
#         for c in range(N_tot_mu):
#             id1 = np.searchsorted(self.PP, np.random.rand()) 
#             cac = 0
#             rho_tmp = self._get_rho_tmp()
#             accra = True
#             while accra:
#                 id2 = np.searchsorted(self.PP, np.random.rand())
#                 cac += 1
#                 rho_tmp *= 0.9995 # 1.0005 # 
#                 accra = self._update_node_connections(id1, id2, rho_tmp)

#             num_prints += 1
#             if self.verbose and num_prints < 10:
#                 print(cac, num_prints)                
#         return None

#     def _run_connect_algo_2(self):
#         N_tot_mu = int(self.mu*self.N_tot)
#         for c in range(N_tot_mu):
#             rho_tmp = self._get_rho_tmp()
#             accra = True
#             while accra:
#                 id1 = np.searchsorted(self.PP, np.random.rand())
#                 id2 = np.searchsorted(self.PP, np.random.rand())
#                 accra = self._update_node_connections(id1, id2, rho_tmp)
#         return None


#     def connect_nodes(self):
#         if self.verbose:
#             print("Connect nodes with algo", self.connect_algo)

#         N_tot_mu = int(self.mu*self.N_tot)
        
#         if self.connect_algo == 1:
#             self._run_connect_algo_1()

#         else:
#             self._run_connect_algo_2()

#         return None

    
#     def make_initial_infections(self):

#         if self.verbose:
#             print("INITIAL INFECTIONS")

#         ##  Now make initial infectious
#         # random_indices = np.random.choice(self.N_tot, size=self.N_init, replace=False)  # TODO check that this is correct, XXX
#         random_indices = np.arange(self.N_tot, dtype=np.int32)
#         np.random.shuffle(random_indices)
#         random_indices = random_indices[:self.N_init]
#         # print(random_indices)

#         # for iin in range(self.N_init):
#             # idx = iin*10
#         for idx in random_indices:
#             new_state = np.random.randint(0, 4)
#             # if self.verbose and iin < 10:
#                 # print("new_state", new_state)
#             self.which_state[idx] = new_state

#             self.agents_in_state[new_state, self.state_total_counts[new_state]] = idx
#             self.state_total_counts[new_state] += 1  
#             self.TotMov += self.SIR_transition_rates[new_state]
#             self.csMov[new_state:] += self.SIR_transition_rates[new_state]
#             for i1 in range(self.N_connections_reference[idx]):
#                 Af = self.which_connections_reference[idx, i1]
#                 for i2 in range(self.N_connections[Af]):
#                     if self.which_connections[Af, i2] == idx:
#                         for i3 in range(i2, self.N_connections[Af]):
#                             self.which_connections[Af, i3] = self.which_connections[Af, i3+1] 
#                             self.individual_rates[Af, i3] = self.individual_rates[Af, i3+1]
#                         self.N_connections[Af] -= 1 
#                         break 

    
#     def _move_from_non_infectious_to_infectious_state(self, idx):
#         for j1 in range(self.N_connections[idx]): # Loop over row idx	  
#             if self.which_state[self.which_connections[idx, j1]] < 0:
#                 self.TotInf += self.individual_rates[idx, j1]
#                 self.InfRat[idx] += self.individual_rates[idx, j1]
#                 self.csInf[self.which_state[idx]:self.N_states] += self.individual_rates[idx, j1]

#     def _move_from_infectious_to_recovered_state(self, idx):
#         for j1 in range(self.N_connections[idx]): # Loop over row idx
#             self.TotInf -= self.individual_rates[idx, j1] 
#             self.InfRat[idx] -= self.individual_rates[idx, j1]
#             self.csInf[self.which_state[idx]:self.N_states] -= self.individual_rates[idx, j1]


#     def _move_infected_between_states(self, ra1):
#         x = self.csMov / self.Tot
#         i1 = np.searchsorted(x, ra1)
#         self.Csum = self.csMov[i1] / self.Tot
#         for i2 in range(self.state_total_counts[i1]):
#             self.Csum += self.SIR_transition_rates[i1] / self.Tot
#             if self.Csum > ra1:
#                 idx = self.agents_in_state[i1, i2]
#                 self.AC = True
#                 break
        
#         # We have chosen idx to move -> here we move it
#         self.agents_in_state[i1+1, self.state_total_counts[i1+1]] = idx
#         for j in range(i2, self.state_total_counts[i1]):
#             self.agents_in_state[i1, j] = self.agents_in_state[i1, j+1] 

#         self.which_state[idx] += 1
#         self.state_total_counts[i1] -= 1 
#         self.state_total_counts[i1+1] += 1      
#         self.TotMov -= self.SIR_transition_rates[i1] 
#         self.TotMov += self.SIR_transition_rates[i1+1]     
#         self.csMov[i1] -= self.SIR_transition_rates[i1]
#         self.csMov[i1+1:self.N_states] += self.SIR_transition_rates[i1+1] - self.SIR_transition_rates[i1]
#         self.csInf[i1] -= self.InfRat[idx]

#         # Moves TO infectious State from non-infectious
#         if self.which_state[idx] == self.infectious_state: 
#             self._move_from_non_infectious_to_infectious_state(idx)

#         # If this moves to Recovered state
#         if self.which_state[idx] == self.N_states-1: 
#             self._move_from_infectious_to_recovered_state(idx)


#     def _update_infection_lists(self, idx2):
#         for j1 in range(self.N_connections_reference[idx2]):
#             Af = self.which_connections_reference[idx2, j1]
#             for i2 in range(self.N_connections[Af]):
#                 if self.which_connections[Af, i2] == idx2:
#                     if (self.which_state[Af] >= self.infectious_state) and (self.which_state[Af] < self.N_states-1): 
#                         self.TotInf -= self.individual_rates[Af, i2]
#                         self.InfRat[Af] -= self.individual_rates[Af, i2]
#                         self.csInf[self.which_state[Af]:self.N_states] -= self.individual_rates[Af, i2]
#                     for i3 in range(i2, self.N_connections[Af]):
#                         self.which_connections[Af, i3] = self.which_connections[Af, i3+1]
#                         self.individual_rates[Af, i3] = self.individual_rates[Af, i3+1]
#                     self.N_connections[Af] -= 1 
#                     break


#     def _infect_new_states(self, ra1):
#         x = self.TotMov / self.Tot + self.csInf / self.Tot
#         i1 = np.searchsorted(x, ra1)
#         self.Csum = self.TotMov / self.Tot + self.csInf[i1] / self.Tot
#         for i2 in range(self.state_total_counts[i1]):
#             idy = self.agents_in_state[i1, i2]
#             for i3 in range(self.N_connections[idy]): 
#                 self.Csum += self.individual_rates[idy][i3] / self.Tot
#                 if self.Csum > ra1:
#                     idx2 = self.which_connections[idy, i3]	      
#                     self.which_state[idx2] = 0 
#                     self.agents_in_state[0, self.state_total_counts[0]] = idx2
#                     self.state_total_counts[0] += 1
#                     self.TotMov += self.SIR_transition_rates[0]	      
#                     self.csMov += self.SIR_transition_rates[0]
#                     self.AC = True
#                     break 

#             if self.AC:
#                 break

#         # Here we update infection lists      
#         self._update_infection_lists(idx2)

#     def run_simulation(self):

#         if self.verbose:
#             print("RUN SIMULATION")

#         # Run the simulation ################################
#         daily_counter = 0
#         first_run = True
#         continue_running = True
#         while continue_running:
            
#             self.c += 1 
#             self.Tot = self.TotMov + self.TotInf
#             dt = - np.log(np.random.rand()) / self.Tot    
#             self.RT = self.RT + dt
#             ra1 = np.random.rand()

#             #######/ Here we move infected between states

#             self.AC = False
#             if self.TotMov / self.Tot > ra1:
#                 self._move_infected_between_states(ra1)
#                 # print("move", self.AC)

#             # Here we infect new states
#             else:
#                 self._infect_new_states(ra1)
#                 # print("infect", self.AC)

#             ################

#             if self.nts*self.click < self.RT:
#                 # print(self.RT)

#                 self.out_time.append(self.RT)
#                 if first_run:
#                     self.out_state_counts[0] = self.state_total_counts.copy()
#                     first_run = False
#                 else:
#                     self.out_state_counts.append(self.state_total_counts.copy())

#             #     SIRfile_tmp = np.zeros(N_states + 1)
#             #     icount = 0
#             #     SIRfile_tmp[icount] = RT
#             #     for s in state_total_counts:
#             #         icount += 1
#             #         SIRfile_tmp[icount] = s #<< "\t"
#             #     SIRfile.append(SIRfile_tmp)
#                 daily_counter += 1

#                 if daily_counter >= 10:
#                     daily_counter = 0

#                     # deepcopy
#                     # SIRfile_which_state.append(deep_copy_1D_int(which_state))
#                     # SIRfile_N_connections.append(deep_copy_1D_int(N_connections))

#                 self.click += 1 



#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#             # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#             if self.c > 100_000_000: 
#                 # if verbose:   
#                 print("c > 100_000_000")
#                 continue_running = False
            
#             if (self.TotInf + self.TotMov < 0.0001) and (self.TotMov + self.TotInf > -0.00001): 
#                 continue_running = False 
#                 if self.verbose:
#                     print("Equilibrium")
            
#             if self.state_total_counts[self.N_states-1] > self.N_tot-10:      
#                 if self.verbose:
#                     print("2/3 through")
#                 continue_running = False

#             # Check for bugs
#             if not self.AC: 
#                 print("No Chosen rate", self.csMov)
#                 continue_running = False
            
#             if (self.TotMov < 0) and (self.TotMov > -0.001):
#                 self.TotMov = 0 
                
#             if (self.TotInf < 0) and (self.TotInf > -0.001):
#                 self.TotInf = 0 
                
#             if (self.TotMov < 0) or (self.TotInf < 0): 
#                 print("\nNegative Problem", self.TotMov, self.TotInf)
#                 print(self.rho, self.beta, self.sigma_mu)
#                 continue_running = False

    

#%%

# if False:

#     abn_model = ABN(
#                     N_tot = N_tot, # Total number of nodes!
#                     N_init = 100, # Initial Infected
#                     mu = 20.0,  # Average number of connections of a node (init: 20)
#                     sigma_mu = 0.0, # Spread (skewness) in N connections
#                     rho = 0, # Spacial dependency. Average distance to connect with.
#                     beta = 0.01, # Daily infection rate (SIR, init: 0-1, but beta = (2mu/N_tot)* betaSIR)
#                     sigma_beta = 0.0, # Spread in rates, beta (beta_eff = beta - sigma_beta+2*sigma_beta*rand[0,1]
#                     lambda_E = 1.0, # E->I, Lambda(from E states)
#                     lambda_I = 1.0, # I->R, Lambda(from I states)
#                     connect_algo = 2, # node connection algorithm
#                     epsilon_rho = 0.01, # fraction of connections not depending on distance
#                     frac_02 = 0.0, # 0: as normal, 1: half of all (beta)rates are set to 0 the other half doubled
#                     ID = 0,
#                     coordinates = coordinates,
#                     verbose = True,
#                     )

#     abn_model.rho
#     abn_model.which_connections
#     abn_model.N_connections



#     abn_model.make_rates_and_connections()
#     abn_model.connection_weight
#     abn_model.PT
#     abn_model.PP

#     abn_model.connect_nodes()
#     abn_model.N_connections


#     abn_model.make_initial_infections()

#     # fig, ax = plt.subplots()
#     # ax.hist(abn_model.N_connections);


#     # abn_model.test()
#     abn_model.run_simulation()

#     # np.array(abn_model.out_time)
#     out_state_counts = np.array(abn_model.out_state_counts)


# # fig, ax = plt.subplots()
# # ax.plot(out_state_counts[:, -1]);


#%%


#%%#%%




#%%


@njit
def initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta, frac_02):

    connection_weight = np.ones(N_tot, dtype=np.float32)
    infection_weight = np.ones(N_tot, dtype=np.float32)

    for i in range(N_tot):
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
def initialize_ages(N_tot, N_ages, connection_weight):

    ages = np.zeros(N_tot, np.uint8)
    ages_total_counts = np.zeros(N_ages, np.uint32)
    ages_in_state = -1*np.ones((N_ages, N_tot), np.int32)

    for i in range(N_tot):
        age = np.random.randint(N_ages)
        ages[i] = age
        ages_in_state[age, ages_total_counts[age]] = i
        ages_total_counts[age] += 1


    PT_ages = []
    PC_ages = []
    PP_ages = []
    for i_age_group in range(N_ages):
        indices = ages_in_state[i_age_group, :ages_total_counts[i_age_group]]
        connection_weight_ages = connection_weight[indices]
        PT_age = np.sum(connection_weight_ages)
        PC_age = np.cumsum(connection_weight_ages)
        PP_age = PC_age / PT_age

        PT_ages.append(PT_age)
        PC_ages.append(PC_age)
        PP_ages.append(PP_age)

    return ages, ages_total_counts, ages_in_state, PT_ages, PC_ages, PP_ages


@njit
def update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2):

    #  Make sure no element is present twice
    accept = True
    for i1 in range(N_connections[id1]):        
        if which_connections[id1, i1] == id2:
            accept = False

    if (N_connections[id1] < N_AK_MAX) and (N_connections[id2] < N_AK_MAX) and (id1 != id2) and accept:
        r = haversine(coordinates[id1, 0], coordinates[id1, 1], coordinates[id2, 0], coordinates[id2, 1])
        if np.exp(-r*rho_tmp/rho_scale) > np.random.rand():
            
            individual_rates[id1, N_connections[id1]] = infection_weight[id1]
            individual_rates[id2, N_connections[id2]] = infection_weight[id2] # XXX changed from id1

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


# @njit
# def run_algo_2(PP, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX):

#     continue_run = True
#     while continue_run:
        
#         id1 = np.searchsorted(PP, np.random.rand())
#         id2 = np.searchsorted(PP, np.random.rand())
        
#         continue_run = update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2)


# @njit
# def run_algo_1(PP, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX):

#     ra1 = np.random.rand()
#     id1 = np.searchsorted(PP, ra1) 
#     N_algo_1_tries = 0

#     continue_run = True
#     while continue_run:
#         ra2 = np.random.rand()          
#         id2 = np.searchsorted(PP, ra2)
#         N_algo_1_tries += 1
#         rho_tmp *= 0.9995 # 1.0005 # 

#         continue_run = update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2)
    
#     return N_algo_1_tries


@njit
def run_algo_2(PP_i, PP_j, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX):

    continue_run = True
    while continue_run:
        
        id1 = np.searchsorted(PP_i, np.random.rand())
        id2 = np.searchsorted(PP_j, np.random.rand())
        
        continue_run = update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2)


@njit
def run_algo_1(PP_i, PP_j, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX):

    ra1 = np.random.rand()
    id1 = np.searchsorted(PP_i, ra1) 
    N_algo_1_tries = 0

    continue_run = True
    while continue_run:
        ra2 = np.random.rand()          
        id2 = np.searchsorted(PP_j, ra2)
        N_algo_1_tries += 1
        rho_tmp *= 0.9995 # 1.0005 # 

        continue_run = update_node_connections(N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX, continue_run, id1, id2)
    
    return N_algo_1_tries



# @njit
# def connect_nodes(mu, N_tot, epsilon_rho, rho, connect_algo, PP, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_scale, N_AK_MAX, verbose):

#     # for m_i in range(N_ages):
#     #     for m_j in range(N_ages):
#     # id1 = np.searchsorted(PP_ages[m_i], np.random.rand())
#     # id2 = np.searchsorted(PP_ages[m_j], np.random.rand())

#     num_prints = 0
#     for c in range(int(mu*N_tot)): # age_matrix[m_i, m_j]

#         if np.random.rand() > epsilon_rho:
#             rho_tmp = rho
#         else:
#             rho_tmp = 0.0
        
#         if (connect_algo == 2):
#             run_algo_2(PP, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX)

#         else:
#             N_algo_1_tries = run_algo_1(PP, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX)

#             if verbose and num_prints < 10:
#                 # print(N_algo_1_tries, num_prints)
#                 num_prints += 1


@njit
def connect_nodes(mu, N_tot, epsilon_rho, rho, connect_algo, PP_ages, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_scale, N_AK_MAX, N_ages, age_matrix, verbose):

    num_prints = 0

    for m_i in range(N_ages):
        for m_j in range(N_ages):
            # print(m_i, m_j)
            # m_i, m_j = 0, 1
            for c in range(int(age_matrix[m_i, m_j])): 

                if np.random.rand() > epsilon_rho:
                    rho_tmp = rho
                else:
                    rho_tmp = 0.0
                
                if (connect_algo == 2):
                    # print("connect algo 2", c)
                    run_algo_2(PP_ages[m_i], PP_ages[m_j], N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX)

                else:
                    # print("connect algo 1", c)
                    N_algo_1_tries = run_algo_1(PP_ages[m_i], PP_ages[m_j], N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_tmp, rho_scale, N_AK_MAX)

                    if verbose and num_prints < 10:
                        # print(N_algo_1_tries, num_prints)
                        num_prints += 1


@njit
def make_initial_infections(N_tot, N_init, which_state, state_total_counts, agents_in_state, csMov, N_connections_reference, which_connections, which_connections_reference, N_connections, individual_rates, SIR_transition_rates, ages_in_state, initial_ages_exposed):

    TotMov = 0.0


    # possible_idxs_age_groups = ages_in_state[initial_ages_exposed] 
    # possible_idxs = List()
    # for i in range(len(possible_idxs_age_groups)):
    #     for xi in possible_idxs_age_groups[i]:
    #         if xi >= 0:
    #             possible_idxs.append(xi)
    # possible_idxs_np = np.zeros(len(possible_idxs))
    # for i, xi in enumerate(possible_idxs):
    #     possible_idxs_np[i] = xi


    possible_idxs = ages_in_state[initial_ages_exposed] # TODO add possible more boxes
    possible_idxs = possible_idxs[0, :np.argmin(possible_idxs)]

    ##  Now make initial infectious
    # random_indices = np.random.choice(N_tot, size=N_init, replace=False)
    random_indices = np.random.choice(possible_idxs, size=N_init, replace=False)
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

@njit
def run_simulation(TotMov, csMov, state_total_counts, agents_in_state, which_state, csInf, N_states, InfRat, SIR_transition_rates, infectious_state, N_connections, individual_rates, N_connections_reference, which_connections_reference, which_connections, nts):

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

    # Run the simulation ################################
    continue_run = True
    while continue_run:
        
        counter += 1 
        Tot = TotMov + TotInf
        dt = - np.log(np.random.rand()) / Tot    
        RT = RT + dt
        Csum = 0.0
        ra1 = np.random.rand()


        #######/ Here we move infected between states
        AC = 0 
        if TotMov/Tot > ra1:
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


        # Here we infect new states
        elif (TotMov + TotInf) / Tot > ra1:
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


        # move between hospital tracks
        else:



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




@njit 
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


@njit # TODO REMEMBER THIS
def single_run_numba(N_tot, N_init, mu, sigma_mu, rho, beta, sigma_beta, lambda_E, lambda_I, connect_algo, epsilon_rho, frac_02, ID, coordinates, verbose=False):
    
    N_tot = 10_000 # Total number of nodes!
    N_init = 100 # Initial Infected
    mu = 20.0  # Average number of connections of a node (init: 20)
    sigma_mu = 0.0 # Spread (skewness) in N connections
    rho = 0 # Spacial dependency. Average distance to connect with.
    beta = 0.01 # Daily infection rate (SIR, init: 0-1, but beta = (2mu/N_tot)* betaSIR)
    sigma_beta = 0.0 # Spread in rates, beta (beta_eff = beta - sigma_beta+2*sigma_beta*rand[0,1])... could be exponential?
    lambda_E = 1.0 # E->I, Lambda(from E states)
    lambda_I = 1.0 # I->R, Lambda(from I states)
    connect_algo = 1 # node connection algorithm
    epsilon_rho = 0.01 # fraction of connections not depending on distance
    frac_02 = 0.0 # 0: as normal, 1: half of all (beta)rates are set to 0 the other half doubled
    ID = 0
    coordinates = np.load('Data/GPS_coordinates.npy')[:N_tot]
    verbose = True

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
    
    which_state = -1*np.ones(N_tot, np.int8)
    
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

    N_ages = 10
    age_matrix_rel = np.ones((N_ages, N_ages)) / N_ages 
    age_connections_rel = np.ones(N_ages) / N_ages
    age_matrix = age_matrix_rel * age_connections_rel * mu * N_tot
    # age_matrix = age_matrix.astype(int)

    initial_ages_exposed = np.array([0])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RATES AND CONNECTIONS # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("Make rates and connections")

    connection_weight, infection_weight = initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta, frac_02)
    # PT = np.sum(connection_weight)
    # PC = np.cumsum(connection_weight)
    # PP = PC / PT

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

    connect_nodes(mu, N_tot, epsilon_rho, rho, connect_algo, PP_ages, N_connections, individual_rates, which_connections, which_connections_reference, coordinates, infection_weight, N_connections_reference, rho_scale, N_AK_MAX, N_ages, age_matrix, verbose)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("INITIAL INFECTIONS")

    TotMov = make_initial_infections(N_tot, N_init, which_state, state_total_counts, agents_in_state, csMov, N_connections_reference, which_connections, which_connections_reference, N_connections, individual_rates, SIR_transition_rates, ages_in_state, initial_ages_exposed)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RUN SIMULATION  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    if verbose:
        print("RUN SIMULATION")

    return run_simulation(TotMov, csMov, state_total_counts, agents_in_state, which_state, csInf, N_states, InfRat, SIR_transition_rates, infectious_state, N_connections, individual_rates, N_connections_reference, which_connections_reference, which_connections, nts)



def single_run_and_save(filename, verbose=False):


    filename = 'Data/ABN/N_tot__1000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1/N_tot__1000__N_init__100__mu__20.0__sigma_mu__0.0__rho__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__frac_02__0.0__connect_algo__1__ID__000.csv'
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