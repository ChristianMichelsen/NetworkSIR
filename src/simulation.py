import numpy as np
import pandas as pd

from pathlib import Path
import multiprocessing as mp
import matplotlib.pyplot as plt
import time as Time
import h5py
import psutil
import warnings
from importlib import reload
from contexttimer import Timer
import os
from IPython.display import display
from contexttimer import Timer


import numba as nb
from numba import njit, prange, objmode, typeof # conda install -c numba/label/dev numba
from numba.typed import List, Dict
# from numba.types import Set
from numba.core.errors import NumbaTypeSafetyWarning, NumbaExperimentalFeatureWarning, NumbaPendingDeprecationWarning


import awkward as awkward0 # conda install awkward0, conda install -c conda-forge pyarrow
import awkward1 as ak # pip install awkward1

try:
    from src import utils
    from src import simulation_utils
except ImportError:
    import utils
    import simulation_utils

np.set_printoptions(linewidth=200)


@njit
def initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta, beta_scaling):

    with objmode():
        track_memory('Rates and Connections')

    connection_weight = np.ones(N_tot, dtype=np.float32)
    infection_weight = np.ones(N_tot, dtype=np.float32)

    for agent in range(N_tot):

        if (np.random.rand() < sigma_mu):
            connection_weight[agent] = 0.1 - np.log(np.random.rand())# / 1.0
        else:
            connection_weight[agent] = 1.1

        if (np.random.rand() < sigma_beta):
            infection_weight[agent] = -np.log(np.random.rand())*beta
        else:
            infection_weight[agent] = beta

        f = 0.1
        beta_scaling_up = 1 / (f + (1-f)/beta_scaling)
        beta_scaling_down = beta_scaling_up / beta_scaling

        ra_R0_change = np.random.rand()
        if ra_R0_change < f:
            infection_weight[agent] = infection_weight[agent]*beta_scaling_up
        else:
            infection_weight[agent] = infection_weight[agent]*beta_scaling_down

    return connection_weight, infection_weight


@njit
def initialize_ages(N_tot, N_ages, connection_weight):

    with objmode():
        track_memory('Ages')

    ages = np.full(N_tot, fill_value=-1, dtype=np.int8)
    ages_total_counts = np.zeros(N_ages, dtype=np.uint32)
    ages_in_state = utils.initialize_nested_lists(N_ages, dtype=np.int32)

    for agent in range(N_tot): # prange
        age = np.random.randint(N_ages)
        ages[agent] = age
        ages_total_counts[age] += 1
        ages_in_state[age].append(np.int32(agent))


    PT_ages = np.zeros(N_ages, dtype=np.float32)
    PC_ages = List()
    PP_ages = List()
    for age_group in range(N_ages): # prange
        indices = np.asarray(ages_in_state[age_group])
        connection_weight_ages = connection_weight[indices]
        PT_age = np.sum(connection_weight_ages)
        PC_age = np.cumsum(connection_weight_ages)
        PP_age = PC_age / PT_age

        PT_ages[age_group] = PT_age
        PC_ages.append(PC_age)
        PP_ages.append(PP_age)

    return ages, ages_total_counts, ages_in_state, PT_ages, PC_ages, PP_ages


@njit
def update_node_connections(which_connections, coordinates, rho_tmp, rho_scale, continue_run, agent1, agent2):
    if agent1 != agent2:
        # r = utils.haversine_scipy(coordinates[agent1, :], coordinates[agent2, :])
        r = utils.haversine(coordinates[agent1, 0], coordinates[agent1, 1], coordinates[agent2, 0], coordinates[agent2, 1])
        if np.exp(-r*rho_tmp/rho_scale) > np.random.rand():

            which_connections[agent1].append(agent2)
            which_connections[agent2].append(agent1)
            continue_run = False

    return continue_run


@njit
def run_algo_2(PP_ages, m_i, m_j, which_connections, coordinates, rho_tmp, rho_scale, ages_in_state):

    continue_run = True
    while continue_run:

        agent1 = np.searchsorted(PP_ages[m_i], np.random.rand())
        agent2 = np.searchsorted(PP_ages[m_j], np.random.rand())

        agent1 = ages_in_state[m_i][agent1]
        agent2 = ages_in_state[m_j][agent2]

        continue_run = update_node_connections(which_connections, coordinates, rho_tmp, rho_scale, continue_run, agent1, agent2)


@njit
def run_algo_1(PP_ages, m_i, m_j, which_connections, coordinates, rho_tmp, rho_scale, ages_in_state):

    ra1 = np.random.rand()
    agent1 = np.searchsorted(PP_ages[m_i], ra1)
    agent1 = ages_in_state[m_i][agent1]

    continue_run = True
    while continue_run:
        ra2 = np.random.rand()
        agent2 = np.searchsorted(PP_ages[m_j], ra2)
        agent2 = ages_in_state[m_j][agent2]

        rho_tmp *= 0.9995

        continue_run = update_node_connections(which_connections, coordinates, rho_tmp, rho_scale, continue_run, agent1, agent2)


@njit
def connect_nodes(epsilon_rho, rho, algo, PP_ages, which_connections, coordinates, rho_scale, N_ages, age_matrix, ages_in_state):

    with objmode():
        track_memory('Connecting Nodes')

    if (algo == 2):
        run_algo = run_algo_2
    else:
        run_algo = run_algo_1

    with objmode():
        track_memory()

    for m_i in range(N_ages):
        for m_j in range(N_ages):
            N_max = int(age_matrix[m_i, m_j])
            for counter in range(N_max):

                if np.random.rand() > epsilon_rho:
                    rho_tmp = rho
                else:
                    rho_tmp = 0.0

                run_algo(PP_ages, m_i, m_j, which_connections, coordinates, rho_tmp, rho_scale, ages_in_state)

                if (counter % (N_max//30)) == 0:
                    with objmode():
                        track_memory()
    with objmode():
        track_memory()


@njit
def make_initial_infections(N_init, which_state, state_total_counts, agents_in_state, csMov, which_connections, N_connections, individual_rates, SIR_transition_rates, ages_in_state, initial_ages_exposed, cs_move_individual, N_infectious_states):

    with objmode():
        track_memory('Initial Infections')

    TotMov = 0.0
    non_infectable_agents = set()

    possible_agents = List()
    for age_exposed in initial_ages_exposed:
        for agent in ages_in_state[age_exposed]:
            possible_agents.append(agent)

    ##  Now make initial infections
    random_agents = np.random.choice(np.asarray(possible_agents), size=N_init, replace=False)
    for i, agent in enumerate(random_agents):
        new_state = np.random.randint(0, N_infectious_states)
        which_state[agent] = new_state

        agents_in_state[new_state].append(np.uint32(agent))
        state_total_counts[new_state] += 1
        TotMov += SIR_transition_rates[new_state]
        csMov[new_state:] += SIR_transition_rates[new_state]

        non_infectable_agents.add(agent)

    with objmode():
        track_memory()

    return TotMov, non_infectable_agents


#%%

@njit
def run_simulation(N_tot, TotMov, csMov, state_total_counts, agents_in_state, which_state, csInf, N_states, InfRat, SIR_transition_rates, N_infectious_states, N_connections, individual_rates, which_connections, ages, individual_infection_counter, cs_move_individual, H_probability_matrix_csum, H_which_state, H_agents_in_state, H_state_total_counts, H_move_matrix_sum, H_cumsum_move, H_move_matrix_cumsum, nts, verbose, non_infectable_agents):


    with objmode():
        track_memory('Simulation')


    out_time = List()
    out_state_counts = List()
    out_which_state = List()
    # out_infection_counter = List()
    # out_N_connections = List()
    out_H_state_total_counts = List()

    daily_counter = 0

    Tot = 0.0
    TotInf = 0.0
    click = 0
    counter = 0
    Csum = 0.0
    RT = 0.0

    H_tot_move = 0
    H_counter = 0

    intervention_switch = False
    intervention_day0 = 0.0
    closed_contacts = utils.initialize_nested_lists(N_tot, dtype=np.int32)
    closed_contacts_rate = utils.initialize_nested_lists(N_tot, dtype=np.float32)

    bug_move = 0.0
    bug_inf = 0.0
    bug_hos = 0.0
    time_inf = np.zeros(N_tot, np.float32)
    bug_contacts = np.zeros(N_tot, np.int32)

    active_agents = set()

    total_printed = 0

    s_counter = np.zeros(4)

    with objmode():
        track_memory()

    # Run the simulation ################################
    continue_run = True
    while continue_run:

        s = 0

        counter += 1
        Tot = TotMov + TotInf #+ H_tot_move XXX Hospitals
        dt = - np.log(np.random.rand()) / Tot
        RT = RT + dt
        Csum = 0.0
        ra1 = np.random.rand()

        csInf0 = csInf.copy()

        #######/ Here we move infected between states
        accept = False
        if TotMov / Tot > ra1:

            s = 1
            if click >= 1608:
                print(s)

            bug_move += Tot / TotMov

            x = csMov / Tot
            state_now = np.searchsorted(x, ra1)
            state_after = state_now + 1
            random_agent_id = np.random.randint(state_total_counts[state_now])
            agent = agents_in_state[state_now][random_agent_id]
            # agent = np.random.choice(agents_in_state[state_now])

            # We have chosen agent to move -> here we move it
            agents_in_state[state_after].append(agent)
            agents_in_state[state_now].remove(agent)

            which_state[agent] += 1
            state_total_counts[state_now] -= 1
            state_total_counts[state_after] += 1
            TotMov -= SIR_transition_rates[state_now]
            TotMov += SIR_transition_rates[state_after]
            csMov[state_now] -= SIR_transition_rates[state_now]
            csMov[state_after:N_states] += SIR_transition_rates[state_after]-SIR_transition_rates[state_now]
            csInf[state_now] -= InfRat[agent]
            accept = True

            # Moves TO infectious State from non-infectious
            if which_state[agent] == N_infectious_states:
                for contact, rate in zip(which_connections[agent], individual_rates[agent]): # Loop over row agent
                    if contact not in non_infectable_agents:
                        TotInf += rate
                        InfRat[agent] += rate
                        csInf[which_state[agent]:] += rate
                        time_inf[agent] = RT
                        bug_contacts[agent] = N_connections[agent]
                active_agents.add(agent)


            # If this moves to Recovered state
            if which_state[agent] == N_states-1:
                for contact, rate in zip(which_connections[agent], individual_rates[agent]):
                    if contact not in non_infectable_agents:
                        TotInf -= rate
                        InfRat[agent] -= rate
                        csInf[which_state[agent]:] -= rate
                        time_inf[agent] = RT - time_inf[agent]
                active_agents.remove(agent)

                # XXX HOSPITAL
                # Now in hospital track
                H_state = np.searchsorted(H_probability_matrix_csum[ages[agent]], np.random.rand())

                H_which_state[agent] = H_state
                H_agents_in_state[H_state].append(agent)
                H_state_total_counts[H_state] += 1

                H_tot_move += H_move_matrix_sum[H_state, ages[agent]]
                H_cumsum_move[H_state:] += H_move_matrix_sum[H_state, ages[agent]]




        # Here we infect new states
        elif (TotMov + TotInf) / Tot > ra1:  # XXX HOSPITAL
        # else: # XXX HOSPITAL
            s = 2
            if click >= 1608:
                print(s)

            bug_inf += Tot / TotInf

            TotInf0 = TotInf

            x = TotMov/Tot + csInf/Tot
            state_now = np.searchsorted(x, ra1)
            Csum = TotMov/Tot + csInf[state_now-1]/Tot # important change from [state_now] to [state_now-1]

            for agent in agents_in_state[state_now]:
                Csum2 = Csum + InfRat[agent] / Tot

                if Csum2 > ra1:
                    for rate, contact in zip(individual_rates[agent], which_connections[agent]):
                        if contact not in non_infectable_agents:
                            Csum += rate / Tot
                            if Csum > ra1:
                                which_state[contact] = 0
                                agents_in_state[0].append(contact)
                                state_total_counts[0] += 1
                                TotMov += SIR_transition_rates[0]
                                csMov += SIR_transition_rates[0]
                                accept = True
                                individual_infection_counter[agent] += 1
                                non_infectable_agents.add(contact)
                                break
                else:
                    Csum = Csum2

                if accept:
                    break

            if click >= 1608:
                print("bla", s)
                print(which_connections[contact])


            # Here we update infection lists
            for step_cousin in which_connections[contact]:
                if click >= 1608:
                    print(step_cousin)
                    print(active_agents)
                    # print(nb.typeof(step_cousin))
                    # print(nb.typeof(active_agents))
                    # yyy = {14433}
                    # print(yyy)
                    # print(1006 in yyy)
                    # print(1006 in active_agents)
                    # xxx = (step_cousin in active_agents)
                    # print(xxx)
                    # return step_cousin, active_agents
                #     print(step_cousin in active_agents)

                if step_cousin in active_agents:
                    # if click >= 1608:
                    #     print(which_connections[step_cousin])
                    #     print(individual_rates[step_cousin])
                    for step_cousins_contacts, rate in zip(which_connections[step_cousin],  individual_rates[step_cousin]):
                        if step_cousins_contacts == contact:
                            TotInf -= rate
                            InfRat[step_cousin] -= rate
                            csInf[which_state[step_cousin]:] -= rate
                            break
                else:
                    continue

            # if click >= 1608:
            #     print("bla2", s)

        ## move between hospital tracks
        else:
            s = 3
            # if click >= 1608:
            #     print(s)

            bug_hos += Tot / H_tot_move

            x = (TotMov + TotInf + H_cumsum_move) / Tot
            H_old_state = np.searchsorted(x, ra1)
            Csum = (TotMov + TotInf + H_cumsum_move[H_old_state-1]) / Tot # important change from [H_old_state] to [H_old_state-1]
            for idx_H_state in range(len(H_agents_in_state[H_old_state])):

                agent = H_agents_in_state[H_old_state][idx_H_state]
                Csum += H_move_matrix_sum[H_old_state, ages[agent]] / Tot

                if Csum > ra1:

                    accept = True
                    H_ra = np.random.rand()

                    H_tmp = H_move_matrix_cumsum[H_which_state[agent], :, ages[agent]] / H_move_matrix_sum[H_which_state[agent], ages[agent]]
                    H_new_state = np.searchsorted(H_tmp, H_ra)

                    # We have chosen agent to move -> here we move it
                    H_agents_in_state[H_old_state].pop(idx_H_state)

                    H_which_state[agent] = H_new_state
                    H_agents_in_state[H_new_state].append(agent)
                    H_state_total_counts[H_old_state] -= 1
                    H_state_total_counts[H_new_state] += 1

                    H_tot_move += H_move_matrix_sum[H_new_state, ages[agent]] - H_move_matrix_sum[H_old_state, ages[agent]]

                    # moving forward
                    if H_old_state < H_new_state:
                        H_cumsum_move[H_old_state:H_new_state] -= H_move_matrix_sum[H_old_state, ages[agent]]
                        H_cumsum_move[H_new_state:] += H_move_matrix_sum[H_new_state, ages[agent]] - H_move_matrix_sum[H_old_state, ages[agent]]

                    #moving backwards
                    else:
                        H_cumsum_move[H_new_state:H_old_state] += H_move_matrix_sum[H_old_state, ages[agent]]
                        H_cumsum_move[H_new_state:] += H_move_matrix_sum[H_new_state, ages[agent]] - H_move_matrix_sum[H_old_state, ages[agent]]

                    break


        ################

        if nts*click < RT:

            # print(click, RT, counter, s, state_total_counts)

            daily_counter += 1
            out_time.append(RT)
            out_state_counts.append(state_total_counts.copy())
            out_H_state_total_counts.append(H_state_total_counts.copy())
            # out_infection_counter.append(utils.array_to_counter(individual_infection_counter))

            if daily_counter >= 10:

                daily_counter = 0

                # out_N_connections.append(utils.array_to_counter(N_connections))
                out_which_state.append(which_state.copy())

                with objmode():
                    track_memory()

            click += 1


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        continue_run, TotMov, TotInf = do_bug_check(counter, continue_run, TotInf, TotMov, verbose, state_total_counts, N_states, N_tot, accept, csMov, ra1, s, x, bug_move, bug_inf, bug_hos, csInf)

        s_counter[s] += 1

    if verbose:
        print("Simulation counter, ", counter)
        print("s_counter", s_counter)

    return out_time, out_state_counts, out_which_state, out_H_state_total_counts


#%%

@njit
def do_bug_check(counter, continue_run, TotInf, TotMov, verbose, state_total_counts, N_states, N_tot, accept, csMov, ra1, s, x, bug_move, bug_inf, bug_hos, csInf):

    if counter > 100_000_000:
        # if verbose:
        print("counter > 100_000_000")
        continue_run = False

    if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001):
        continue_run = False
        if verbose:
            print("Equilibrium")
            # print(bug_move/counter, bug_inf/counter, bug_hos/counter)

    if state_total_counts[N_states-1] > N_tot-10:
        if verbose:
            print("2/3 through")
        continue_run = False

    # Check for bugs
    if not accept:
        print("\nNo Chosen rate")
        print("s: \t", s)
        print("TotInf: \t", TotInf)
        print("csInf: \t", csInf)
        print("csMov: \t", csMov)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    if (TotMov < 0) and (TotMov > -0.001):
        TotMov = 0

    if (TotInf < 0) and (TotInf > -0.001):
        TotInf = 0

    if (TotMov < 0) or (TotInf < 0):
        print("\nNegative Problem", TotMov, TotInf)
        print("s: \t", s)
        print("TotInf: \t", TotInf)
        print("csInf: \t", csInf)
        print("csMov: \t", csMov)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    return continue_run, TotMov, TotInf



#%%


class Simulation:

    def __init__(self, filename, verbose=False, do_track_memory=True):

        self.verbose = verbose

        self._Filename = Filename = simulation_utils.Filename(filename)

        self.cfg = Filename.simulation_parameters
        self.ID = Filename.ID

        self.filenames = {}
        self.filename = self.filenames['filename'] = Filename.filename
        self.filenames['network_initialisation'] = Filename.filename_network_initialisation
        self.filenames['network_network'] = Filename.filename_network

        utils.set_numba_random_seed(self.ID)

        self._prepare_memory_tracking(do_track_memory)


    def _prepare_memory_tracking(self, do_track_memory=True):
        self.filenames['memory'] = memory_file = self._Filename.memory_filename
        self.do_track_memory = do_track_memory # if self.ID == 0 else False
        self.time_start = Time.time()

        search_string = "Saving network initialization"

        if utils.file_exists(memory_file) and simulation_utils.does_file_contains_string(memory_file, search_string):
            self.time_start -= simulation_utils.get_search_string_time(memory_file, search_string)
            self.track_memory('Appending to previous network initialization')
        else:
            utils.make_sure_folder_exist(self.filenames['memory'], delete_file_if_exists=True) # make sure parent folder exists

        global track_memory
        track_memory = self.track_memory

    @property
    def current_memory_usage(self):
        "Returns current memory usage of entire process in GiB"
        process = psutil.Process()
        return process.memory_info().rss / 2**30


    def track_memory(self, s=None):
        if self.do_track_memory:
            with open(self.filenames['memory'], 'a') as file:
                if s:
                    print("#"+s, file=file)
                time = Time.time()-self.time_start
                print(time, self.current_memory_usage, file=file, sep='\t')  # GiB


    def _initialize_network(self):

        cfg = self.cfg
        self.track_memory('Loading Coordinates')
        self.coordinates = simulation_utils.load_coordinates(self._Filename.coordinates_filename, cfg.N_tot, self.ID)

        if self.verbose:
            print("INITIALIZE NETWORK")
        self.track_memory('Initialize Network')

        which_connections = utils.initialize_nested_lists(cfg.N_tot, dtype=np.uint32) # initialize_list_set

        rho_scale = 1000 # scale factor of rho
        # cfg.mu /= 2 # fix to factor in that both nodes have connections with each other

        # age variables
        age_matrix_relative_interactions = simulation_utils.calculate_age_proportions_1D(1.0, cfg.N_ages)
        age_relative_proportions = simulation_utils.calculate_age_proportions_2D(cfg.age_mixing, cfg.N_ages)
        age_matrix = age_matrix_relative_interactions * age_relative_proportions * (cfg.mu / 2) * cfg.N_tot

        if self.verbose:
            print("MAKE RATES AND CONNECTIONS")
        self.track_memory('Numba Compilation')

        connection_weight, infection_weight = initialize_connections_and_rates(cfg.N_tot, cfg.sigma_mu, cfg.beta, cfg.sigma_beta, cfg.beta_scaling)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # AGES  # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if self.verbose:
            print("MAKE AGES")
        self.track_memory('Numba Compilation')

        ages, ages_total_counts, ages_in_state, PT_ages, PC_ages, PP_ages = initialize_ages(cfg.N_tot, cfg.N_ages, connection_weight)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # CONNECT NODES # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if self.verbose:
            print("CONNECT NODES")
        self.track_memory('Numba Compilation')

        connect_nodes(cfg.epsilon_rho, cfg.rho, cfg.algo, PP_ages, which_connections, self.coordinates, rho_scale, cfg.N_ages, age_matrix, ages_in_state)

        return which_connections, ages


    def _save_network_initalization(self, ages, N_connections, which_connections, time_elapsed):
        self.track_memory('Saving network initialization')
        utils.make_sure_folder_exist(self.filenames['network_initialisation'])
        with h5py.File(self.filenames['network_initialisation'], "w") as f: #
            f.create_dataset("cfg_str", data=str(self.cfg)) # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("ages", data=ages)
            f.create_dataset("N_connections", data=N_connections)
            awkward0.hdf5(f)["which_connections"] = which_connections
            for key, val in self.cfg.items():
                f.attrs[key] = val
            f.create_dataset("time_elapsed", data=time_elapsed)


    def _load_network_initalization(self):
        self.track_memory('Loading network initialization')
        with h5py.File(self.filenames['network_initialisation'], 'r') as f:
            ages = f["ages"][()]
            N_connections = f["N_connections"][()]
            which_connections = awkward0.hdf5(f)["which_connections"]
        self.track_memory('Loading Coordinates')
        self.coordinates = simulation_utils.load_coordinates(self._Filename.coordinates_filename, self.cfg.N_tot, self.ID)
        return ages, N_connections, ak.from_awkward0(which_connections)


    def initialize_network(self, force_rerun=False, save_initial_network=True):
        utils.set_numba_random_seed(self.ID)

        if Path(self.filenames['network_initialisation']).exists() and not force_rerun:
            if self.verbose:
                print(f"{self.filenames['network_initialisation']} exists")
            ages, N_connections, which_connections = self._load_network_initalization()

        else:
            if self.verbose:
                print(f"{self.filenames['network_initialisation']} does not exist, creating it")
            with Timer() as t:
                which_connections, ages = self._initialize_network()
            which_connections, N_connections = utils.nested_list_to_awkward_array(which_connections, return_lengths=True, sort_nested_list=True)
            if save_initial_network:
                self._save_network_initalization(ages=ages,
                                                 N_connections=N_connections,
                                                 which_connections=ak.to_awkward0(which_connections),
                                                 time_elapsed=t.elapsed)

        self.ages = ages
        self.which_connections = utils.RaggedArray(which_connections)
        self.N_connections = N_connections


    def make_initial_infections(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("INITIAL INFECTIONS")
        self.track_memory('Numba Compilation')

        cfg = self.cfg

        np.random.seed(self.ID)

        self.nts = 0.1 # Time step (0.1 - ten times a day)
        self.N_states = 9 # number of states
        self.N_infectious_states = 4 # This means the 5'th state
        self.initial_ages_exposed = np.arange(cfg.N_ages)

        self.individual_rates = simulation_utils.initialize_individual_rates(cfg.N_tot, cfg.beta, cfg.sigma_beta, self.N_connections, self.ID)

        self.which_state = np.full(cfg.N_tot, -1, dtype=np.int8)
        self.state_total_counts = np.zeros(self.N_states, dtype=np.uint32)
        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.csMov = np.zeros(self.N_states, dtype=np.float64)
        self.csInf = np.zeros(self.N_states, dtype=np.float64)
        self.InfRat = np.zeros(cfg.N_tot, dtype=np.float64)

        self.cs_move_individual = utils.initialize_nested_lists(self.N_states, dtype=np.float64)

        self.SIR_transition_rates = simulation_utils.initialize_SIR_transition_rates(self.N_states, self.N_infectious_states, cfg)

        self.ages_in_state = simulation_utils.compute_ages_in_state(self.ages, cfg.N_ages)

        self.TotMov, self.non_infectable_agents = make_initial_infections(cfg.N_init, self.which_state, self.state_total_counts, self.agents_in_state, self.csMov, self.which_connections.array, self.N_connections, self.individual_rates.array, self.SIR_transition_rates, self.ages_in_state, self.initial_ages_exposed, self.cs_move_individual, self.N_infectious_states)


    def run_simulation(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("RUN SIMULATION")
        self.track_memory('Numba Compilation')

        cfg = self.cfg

        self.individual_infection_counter = np.zeros(cfg.N_tot, dtype=np.uint16)

        H = simulation_utils.get_hospitalization_variables(cfg)
        H_probability_matrix_csum, H_which_state, H_agents_in_state, H_state_total_counts, H_move_matrix_sum, H_cumsum_move, H_move_matrix_cumsum = H

        res = run_simulation(cfg.N_tot, self.TotMov, self.csMov, self.state_total_counts, self.agents_in_state, self.which_state, self.csInf, self.N_states, self.InfRat, self.SIR_transition_rates, self.N_infectious_states, self.N_connections, self.individual_rates.array, self.which_connections.array, self.ages, self.individual_infection_counter, self.cs_move_individual, H_probability_matrix_csum, H_which_state, H_agents_in_state, H_state_total_counts, H_move_matrix_sum, H_cumsum_move, H_move_matrix_cumsum, self.nts, self.verbose, self.non_infectable_agents)

        out_time, out_state_counts, out_which_state, out_H_state_total_counts = res

        track_memory('Arrays Conversion')

        self.time = np.array(out_time)
        self.state_counts = np.array(out_state_counts)
        self.which_state = np.array(out_which_state)
        # self.N_connections = utils.list_of_counters_to_numpy_array(out_N_connections)
        self.H_state_total_counts = np.array(out_H_state_total_counts)
        # out_infection_counter = utils.list_of_counters_to_numpy_array(out_infection_counter)


    def make_dataframe(self):

        self.track_memory('Make DataFrame')
        self.df = df = simulation_utils.state_counts_to_df(self.time, self.state_counts, self.H_state_total_counts)

        self.track_memory('Save CSV')
        utils.make_sure_folder_exist(self.filename)
        # save csv file
        df.to_csv(self.filename, index=False)
        return df


    def save_simulation_results(self, save_only_ID_0=False, time_elapsed=None):

        if save_only_ID_0 and self.ID != 0:
            return None

        utils.make_sure_folder_exist(self.filenames['network_network'], delete_file_if_exists=True)

        self.track_memory('Saving HDF5 File')
        with h5py.File(self.filenames['network_network'], "w") as f: #
            f.create_dataset("coordinates", data=self.coordinates)
            f.create_dataset("which_state", data=self.which_state)
            f.create_dataset("N_connections", data=self.N_connections)
            f.create_dataset("ages", data=self.ages)
            f.create_dataset("cfg_str", data=str(self.cfg)) # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))

            if time_elapsed:
                f.create_dataset("time_elapsed", data=time_elapsed)

            if self.do_track_memory:
                memory_file = self.filenames["memory"]
                self.df_time_memory, self.df_change_points = simulation_utils.parse_memory_file(memory_file)
                df_time_memory_hdf5 = utils.dataframe_to_hdf5_format(self.df_time_memory, cols_to_str=['ChangePoint'])
                df_change_points_hdf5 = utils.dataframe_to_hdf5_format(self.df_change_points, include_index=True)

                f.create_dataset("memory_file", data=Path(memory_file).read_text())
                f.create_dataset("df_time_memory", data=df_time_memory_hdf5)
                f.create_dataset("df_change_points", data=df_change_points_hdf5)

            # if do_include_awkward:
            #     if do_track_memory:
            #         track_memory('Awkward')
            #     g = awkward0.hdf5(f)
            #     g["which_connections"] = awkward0.fromiter(out_which_connections).astype(np.int32)
            #     g["individual_rates"] = awkward0.fromiter(out_individual_rates)

            for key, val in self.cfg.items():
                f.attrs[key] = val

        # self.track_memory('Finished')
        # if verbose:
        #     print(f"Run took in total: {t.elapsed:.1f}s.")

        if self.verbose:
            print("\n\n")
            print("coordinates", utils.get_size(self.coordinates))
            print("which_state", utils.get_size(self.which_state))
            print("N_connections", utils.get_size(self.N_connections))
            print("ages", utils.get_size(self.ages))


    def save_memory_figure(self, savefig=True):
        if self.do_track_memory:
            fig, ax = simulation_utils.plot_memory_comsumption(self.df_time_memory, self.df_change_points, min_TimeDiffRel=0.1, min_MemoryDiffRel=0.1, time_unit='s')
            if savefig:
                fig.savefig(self.filenames["memory"].replace('.txt', '.pdf'))


#%%

def run_full_simulation(filename, verbose=False, force_rerun=False, only_initialize_network=False):

    with Timer() as t, warnings.catch_warnings():
        if not verbose:
            warnings.simplefilter('ignore', NumbaTypeSafetyWarning)
            warnings.simplefilter('ignore', NumbaExperimentalFeatureWarning)
            warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)

        simulation = Simulation(filename, verbose)
        simulation.initialize_network(force_rerun=force_rerun)
        if only_initialize_network:
            return None

        simulation.make_initial_infections()
        simulation.run_simulation()
        simulation.make_dataframe()
        simulation.save_simulation_results(time_elapsed=t.elapsed)
        simulation.save_memory_figure()

        if verbose and simulation.ID == 0:
            print(f"\n\n{simulation.cfg}\n")
            print(simulation.df_change_points)

    if verbose:
        print("\n\nFinished!!!")

reload(utils)
reload(simulation_utils)

verbose = True
force_rerun = False
filename = 'Data/ABN/N_tot__58000__N_init__100__N_ages__1__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__rho__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__beta_scaling__1.0__age_mixing__1.0__algo__2/N_tot__58000__N_init__100__N_ages__1__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__rho__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__beta_scaling__1.0__age_mixing__1.0__algo__2__ID__000.csv'
# filename = filename.replace('58000', '580000')
filename = filename.replace('ID__000', 'ID__002')
filename = filename.replace('rho__0.0__', 'rho__25.0__')

if True:
    simulation = Simulation(filename, verbose)
    simulation.initialize_network(force_rerun=force_rerun)
    simulation.make_initial_infections()
    simulation.run_simulation()
    df = simulation.make_dataframe()
    display(df)

# # if verbose:
# #     print("\n\n")
# #     print("coordinates", utils.get_size(simulation.coordinates, 'mb'))
# #     print("which_state", utils.get_size(simulation.which_state, 'mb'))
# #     print("N_connections", utils.get_size(simulation.N_connections, 'mb'))
# #     print("ages", utils.get_size(simulation.ages, 'mb'))

# # # simulation.save_simulation_results()
# # # simulation.save_memory_figure()
# # # print(f"\n\n{simulation.cfg}\n")
# # # print(simulation.df_change_points)
