import numpy as np
from pathlib import Path
import os
import numba as nb
from numba.experimental import jitclass
from numba import njit, typeof
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type

from src.utils import utils


#      ██ ██ ████████      ██████ ██       █████  ███████ ███████ ███████ ███████
#      ██ ██    ██        ██      ██      ██   ██ ██      ██      ██      ██
#      ██ ██    ██        ██      ██      ███████ ███████ ███████ █████   ███████
# ██   ██ ██    ██        ██      ██      ██   ██      ██      ██ ██           ██
#  █████  ██    ██         ██████ ███████ ██   ██ ███████ ███████ ███████ ███████
#
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Regular&t=Version%202%0A%20


spec_cfg = {
    "version": nb.float32,
    "N_tot": nb.uint32,
    "rho": nb.float32,
    "epsilon_rho": nb.float32,
    "mu": nb.float32,
    "sigma_mu": nb.float32,
    "beta": nb.float32,
    "sigma_beta": nb.float32,
    "algo": nb.uint8,
    "N_init": nb.uint16,
    "lambda_E": nb.float32,
    "lambda_I": nb.float32,
    "make_random_initial_infections": nb.boolean,
    "N_connect_retries": nb.uint32,
    "ID": nb.uint16,
}


@jitclass(spec_cfg)
class Config(object):
    def __init__(self):
        self.version = 1.0
        self.N_tot = 580_000
        self.rho = 0.0
        self.epsilon_rho = 0.04
        self.mu = 40.0
        self.sigma_mu = 0.0
        self.beta = 0.01
        self.sigma_beta = 0.0
        self.algo = 2
        self.N_init = 100
        self.lambda_E = 1.0
        self.lambda_I = 1.0
        self.make_random_initial_infections = 1
        self.N_connect_retries = 0
        self.ID = 0

    def print(self):
        print(
            *("version: ", self.version, "\n"),
            *("N_tot: ", self.N_tot, "\n"),
            *("rho: ", self.rho, "\n"),
            *("epsilon_rho: ", self.epsilon_rho, "\n"),
            *("mu: ", self.mu, "\n"),
            *("sigma_mu: ", self.sigma_mu, "\n"),
            *("beta: ", self.beta, "\n"),
            *("sigma_beta: ", self.sigma_beta, "\n"),
            *("algo: ", self.algo, "\n"),
            *("N_init: ", self.N_init, "\n"),
            *("lambda_E: ", self.lambda_E, "\n"),
            *("lambda_I: ", self.lambda_I, "\n"),
            *("make_random_initial_infections: ", self.make_random_initial_infections, "\n"),
            *("ID: ", self.ID, "\n"),
        )


def initialize_nb_cfg(cfg):
    config = Config()
    for key, val in cfg.items():
        setattr(config, key, val)
    return config


nb_cfg_type = Config.class_type.instance_type

spec = {
    "age": nb.uint8[:],
    "connections": ListType(ListType(nb.uint32)),
    "connections_type": ListType(ListType(nb.uint8)),
    "coordinates": nb.float32[:, :],
    "connection_weight": nb.float32[:],
    "infection_weight": nb.float64[:],
    "number_of_contacts": nb.uint16[:],
    "state": nb.int8[:],
    "cfg": nb_cfg_type,
}


# "Nested/Mutable" Arrays are faster than list of arrays which are faster than lists of lists
@jitclass(spec)
class My(object):
    def __init__(self, nb_cfg):
        N_tot = nb_cfg.N_tot
        self.age = np.zeros(N_tot, dtype=np.uint8)
        self.coordinates = np.zeros((N_tot, 2), dtype=np.float32)
        self.connections = utils.initialize_nested_lists(N_tot, np.uint32)
        self.connections_type = utils.initialize_nested_lists(N_tot, np.uint8)
        self.connection_weight = np.ones(N_tot, dtype=np.float32)
        self.infection_weight = np.ones(N_tot, dtype=np.float64)
        self.number_of_contacts = np.zeros(N_tot, dtype=nb.uint16)
        self.state = np.full(N_tot, -1, dtype=np.int8)
        self.cfg = nb_cfg

    def dist(self, agent1, agent2):
        point1 = self.coordinates[agent1]
        point2 = self.coordinates[agent2]
        return utils.haversine_scipy(point1, point2)

    def dist_accepted(self, agent1, agent2, rho_tmp):
        if np.exp(-self.dist(agent1, agent2) * rho_tmp) > np.random.rand():
            return True
        else:
            return False


def initialize_My(cfg):
    nb_cfg = initialize_nb_cfg(cfg)
    return My(nb_cfg)


spec_g = {
    "N_tot": nb.uint32,
    "N_states": nb.uint8,
    "total_sum": nb.float64,
    "total_sum_infections": nb.float64,
    "total_sum_of_state_changes": nb.float64,
    "cumulative_sum": nb.float64,
    "cumulative_sum_of_state_changes": nb.float64[:],
    "cumulative_sum_infection_rates": nb.float64[:],
    "rates": ListType(nb.float64[::1]),  # ListType[array(float64, 1d, C)] (C vs. A)
    "sum_of_rates": nb.float64[:],
}


@jitclass(spec_g)
class Gillespie(object):
    def __init__(self, my, N_states):
        self.N_states = N_states
        self.total_sum = 0.0
        self.total_sum_infections = 0.0
        self.total_sum_of_state_changes = 0.0
        self.cumulative_sum = 0.0
        self.cumulative_sum_of_state_changes = np.zeros(N_states, dtype=np.float64)
        self.cumulative_sum_infection_rates = np.zeros(N_states, dtype=np.float64)
        self._initialize_rates(my)

    def _initialize_rates(self, my):
        rates = List()
        for i in range(my.cfg.N_tot):
            x = np.full(
                shape=my.number_of_contacts[i],
                fill_value=my.infection_weight[i],
                dtype=np.float64,
            )
            rates.append(x)
        self.rates = rates
        self.sum_of_rates = np.zeros(my.cfg.N_tot, dtype=np.float64)

    def update_rates(self, my, rate, agent):
        self.total_sum_infections += rate
        self.sum_of_rates[agent] += rate
        self.cumulative_sum_infection_rates[my.state[agent] :] += rate


# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██      ██
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██     ███
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      ██
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██      ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████      ██
#


@njit
def v1_initialize_my(my, coordinates_raw):
    for agent in range(my.cfg.N_tot):
        set_connection_weight(my, agent)
        set_infection_weight(my, agent)
        my.coordinates[agent] = coordinates_raw[agent]


@njit
def v1_run_algo_1(my, PP, rho_tmp):
    """ Algo 1: density independent connection algorithm """
    agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
    while True:
        agent2 = np.uint32(np.searchsorted(PP, np.random.rand()))
        rho_tmp *= 0.9995
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=-1,
            code_version=1,
        )
        if do_stop:
            break


@njit
def v1_run_algo_2(my, PP, rho_tmp):
    """ Algo 2: increases number of connections in high-density ares """
    while True:
        agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
        agent2 = np.uint32(np.searchsorted(PP, np.random.rand()))
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=-1,
            code_version=1,
        )
        if do_stop:
            break


@njit
def v1_connect_nodes(my):
    """ v1 of connecting nodes. No age dependence, and a specific choice of Algo """
    if my.cfg.algo == 2:
        run_algo = v1_run_algo_2
    else:
        run_algo = v1_run_algo_1
    PP = np.cumsum(my.connection_weight) / np.sum(my.connection_weight)
    for _ in range(my.cfg.mu / 2 * my.cfg.N_tot):
        if np.random.rand() > my.cfg.epsilon_rho:
            rho_tmp = my.cfg.rho
        else:
            rho_tmp = 0.0
        run_algo(my, PP, rho_tmp)


# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██     ██████
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██          ██
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      █████
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██     ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████     ███████


#%%

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # INITIALIZATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def set_connection_weight(my, agent):
    """ How introvert / extrovert you are. How likely you are at having many contacts in your network."""
    if np.random.rand() < my.cfg.sigma_mu:
        my.connection_weight[agent] = 0.1 - np.log(np.random.rand())
    else:
        my.connection_weight[agent] = 1.1


@njit
def set_infection_weight(my, agent):
    " How much of a super sheader are you?"
    if np.random.rand() < my.cfg.sigma_beta:
        my.infection_weight[agent] = -np.log(np.random.rand()) * my.cfg.beta
    else:
        my.infection_weight[agent] = my.cfg.beta


@njit
def computer_number_of_cluster_retries(my, agent1, agent2):
    """Number of times to (re)try to connect two agents.
    A higher cluster_retries gives higher cluster coeff."""
    connectivity_factor = 1
    for contact in my.connections[agent1]:
        if contact in my.connections[agent2]:
            connectivity_factor += my.cfg.N_connect_retries
    return connectivity_factor


@njit
def cluster_retry_succesful(my, agent1, agent2, rho_tmp):
    """" (Re)Try to connect two agents. Returns True if succesful, else False"""
    if my.cfg.N_connect_retries == 0:
        return False
    connectivity_factor = computer_number_of_cluster_retries(my, agent1, agent2)
    for _ in range(connectivity_factor):
        if my.dist_accepted(agent1, agent2, rho_tmp):
            return True
    return False


@njit
def update_node_connections(
    my,
    rho_tmp,
    agent1,
    agent2,
    connection_type,
    code_version=2,
):
    """Returns True if two agents should be connected, else False"""

    if agent1 == agent2:
        return False

    dist_accepted = rho_tmp == 0 or my.dist_accepted(agent1, agent2, rho_tmp)
    if not dist_accepted:
        # try and reconnect to increase clustering effect
        if not cluster_retry_succesful(my, agent1, agent2, rho_tmp):
            return False

    alread_added = agent1 in my.connections[agent2] or agent2 in my.connections[agent1]
    if alread_added:
        return False

    my.connections[agent1].append(np.uint32(agent2))
    my.connections[agent2].append(np.uint32(agent1))

    if code_version >= 2:
        connection_type = np.uint8(connection_type)
        my.connections_type[agent1].append(connection_type)
        my.connections_type[agent2].append(connection_type)

    my.number_of_contacts[agent1] += 1
    my.number_of_contacts[agent2] += 1

    return True


@njit
def place_and_connect_families(
    my, people_in_household, age_distribution_per_people_in_household, coordinates_raw
):

    N_tot = my.cfg.N_tot

    all_indices = np.arange(N_tot, dtype=np.uint32)
    np.random.shuffle(all_indices)

    N_dim_people_in_household, N_ages = age_distribution_per_people_in_household.shape
    assert N_dim_people_in_household == len(people_in_household)
    people_index_to_value = np.arange(1, N_dim_people_in_household + 1)

    counter_ages = np.zeros(N_ages, dtype=np.uint16)
    agents_in_age_group = utils.initialize_nested_lists(N_ages, dtype=np.uint32)

    mu_counter = 0
    agent = 0
    do_continue = True
    while do_continue:

        agent0 = agent

        house_index = all_indices[agent]

        N_people_in_house_index = utils.rand_choice_nb(people_in_household)
        N_people_in_house = people_index_to_value[N_people_in_house_index]

        # if N_in_house would increase agent to over N_tot,
        # set N_people_in_house such that it fits and break loop
        if agent + N_people_in_house >= N_tot:
            N_people_in_house = N_tot - agent
            do_continue = False

        for _ in range(N_people_in_house):

            age_index = utils.rand_choice_nb(
                age_distribution_per_people_in_household[N_people_in_house_index]
            )

            age = age_index  # just use age index as substitute for age
            my.age[agent] = age
            counter_ages[age_index] += 1
            agents_in_age_group[age_index].append(np.uint32(agent))

            my.coordinates[agent] = coordinates_raw[house_index]

            set_connection_weight(my, agent)
            set_infection_weight(my, agent)

            agent += 1

        # add agents to each others networks (connections)
        for agent1 in range(agent0, agent0 + N_people_in_house):
            for agent2 in range(agent1, agent0 + N_people_in_house):
                if agent1 != agent2:
                    my.connections[agent1].append(np.uint32(agent2))
                    my.connections[agent2].append(np.uint32(agent1))
                    my.connections_type[agent1].append(np.uint8(0))
                    my.connections_type[agent2].append(np.uint8(0))
                    my.number_of_contacts[agent1] += 1
                    my.number_of_contacts[agent2] += 1
                    mu_counter += 1

    agents_in_age_group = utils.nested_lists_to_list_of_array(agents_in_age_group)

    return mu_counter, counter_ages, agents_in_age_group


@njit
def run_algo_other(my, agents_in_age_group, age1, age2, rho_tmp):
    while True:
        # agent1 = np.searchsorted(PP_ages[m_i], np.random.rand())
        # agent1 = agents_in_age_group[m_i][agent1]
        # TODO: Add connection weights
        agent1 = np.random.choice(agents_in_age_group[age1])
        agent2 = np.random.choice(agents_in_age_group[age2])
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=2,
            code_version=2,
        )
        if do_stop:
            break


@njit
def run_algo_work(my, agents_in_age_group, age1, age2, rho_tmp):
    # agent1 = np.searchsorted(PP_ages[m_i], np.random.rand())
    # agent1 = agents_in_age_group[m_i][agent1]
    # TODO: Add connection weights
    agent1 = np.random.choice(agents_in_age_group[age1])

    while True:
        agent2 = np.random.choice(agents_in_age_group[age2])
        rho_tmp *= 0.9995
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=1,
            code_version=2,
        )

        if do_stop:
            break


@njit
def find_two_age_groups(N_ages, matrix):
    a = 0
    ra = np.random.rand()
    for i in range(N_ages):
        for j in range(N_ages):
            a += matrix[i, j]
            if a > ra:
                age1, age2 = i, j
                return age1, age2
    raise AssertionError("find_two_age_groups couldn't find two age groups")


@njit
def connect_work_and_others(
    my,
    N_ages,
    mu_counter,
    work_other_ratio,
    matrix_work,
    matrix_other,
    agents_in_age_group,
):

    while mu_counter < my.cfg.mu / 2 * my.cfg.N_tot:

        ra_work_other = np.random.rand()
        if ra_work_other < work_other_ratio:
            matrix = matrix_work
            run_algo = run_algo_work
        else:
            matrix = matrix_other
            run_algo = run_algo_other

        age1, age2 = find_two_age_groups(N_ages, matrix)

        if np.random.rand() > my.cfg.epsilon_rho:
            rho_tmp = my.cfg.rho
        else:
            rho_tmp = 0.0

        run_algo(
            my,
            agents_in_age_group,
            age1,
            age2,
            rho_tmp,
        )
        mu_counter += 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def single_random_choice(x):
    return np.random.choice(x, size=1)[0]


@njit
def compute_initial_agents_to_infect(my, possible_agents):

    ##  Standard outbreak type, infecting randomly
    if my.cfg.make_random_initial_infections:
        return np.random.choice(possible_agents, size=my.cfg.N_init, replace=False)

    # Local outbreak type, infecting around a point:
    else:

        rho_init_local_outbreak = 0.1

        outbreak_agent = single_random_choice(possible_agents)  # this is where the outbreak starts

        initial_agents_to_infect = List()
        initial_agents_to_infect.append(outbreak_agent)

        while len(initial_agents_to_infect) < my.cfg.N_init:
            proposed_agent = single_random_choice(possible_agents)

            if my.dist_accepted(outbreak_agent, proposed_agent, rho_init_local_outbreak):
                if proposed_agent not in initial_agents_to_infect:
                    initial_agents_to_infect.append(proposed_agent)
        return np.asarray(initial_agents_to_infect, dtype=np.uint32)


@njit
def make_initial_infections(
    my,
    g,
    state_total_counts,
    agents_in_state,
    SIR_transition_rates,
    agents_in_age_group,
    initial_ages_exposed,
    N_infectious_states,
):

    # version 2 has age groups
    if my.cfg.version >= 2:
        possible_agents = List()
        for age_exposed in initial_ages_exposed:
            for agent in agents_in_age_group[age_exposed]:
                possible_agents.append(agent)
        possible_agents = np.asarray(possible_agents, dtype=np.uint32)
    # version 1 has no age groups
    else:
        possible_agents = np.arange(my.cfg.N_tot, dtype=np.uint32)

    initial_agents_to_infect = compute_initial_agents_to_infect(my, possible_agents)

    ##  Now make initial infections
    for _, agent in enumerate(initial_agents_to_infect):
        new_state = np.random.randint(N_infectious_states)  # E1, E2, E3 or E4
        my.state[agent] = new_state

        agents_in_state[new_state].append(np.uint32(agent))
        state_total_counts[new_state] += 1

        g.total_sum_of_state_changes += SIR_transition_rates[new_state]
        g.cumulative_sum_of_state_changes[new_state:] += SIR_transition_rates[new_state]

    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # PRE SIMULATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def daily_tent_test(
    my,
    g,
    N_daily_tests,
    f_test_succes,
    N_infectious_states,
    N_states,
    infectious_states,
):

    n_positive_tested = 0

    for _ in range(N_daily_tests):
        agent = np.random.randint(my.cfg.N_tot)

        # only if in I state and  un-noticed
        if (my.state[agent] in infectious_states) and (np.random.rand() < f_test_succes):

            # agent_tested_positive.append(agent)
            n_positive_tested += 1

            for i in range(my.number_of_contacts[agent]):
                contact = my.connections[agent][i]
                rate = g.rates[agent][i]
                connection_type = my.connections_type[agent][i]

                # only close work/other contacts
                if my.state[contact] == -1 and connection_type > -1:
                    g.update_rates(my, -rate, agent)
                    g.rates[agent][i] = 0
                    # g.total_sum_infections -= rate
                    # g.sum_of_rates[agent] -= rate
                    # g.cumulative_sum_infection_rates[my.state[agent] :] -= rate

    return n_positive_tested


@njit
def do_bug_check(
    my,
    g,
    step_number,
    continue_run,
    verbose,
    state_total_counts,
    N_states,
    accept,
    ra1,
    s,
    x,
):

    if step_number > 100_000_000:
        print("step_number > 100_000_000")
        continue_run = False

    if (g.total_sum_infections + g.total_sum_of_state_changes < 0.0001) and (
        g.total_sum_of_state_changes + g.total_sum_infections > -0.00001
    ):
        continue_run = False
        if verbose:
            print("Equilibrium")

    if state_total_counts[N_states - 1] > my.cfg.N_tot - 10:
        if verbose:
            print("2/3 through")
        continue_run = False

    # Check for bugs
    if not accept:
        print("\nNo Chosen rate")
        print("s: \t", s)
        print("g.total_sum_infections: \t", g.total_sum_infections)
        print("g.cumulative_sum_infection_rates: \t", g.cumulative_sum_infection_rates)
        print("g.cumulative_sum_of_state_changes: \t", g.cumulative_sum_of_state_changes)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    if (g.total_sum_of_state_changes < 0) and (g.total_sum_of_state_changes > -0.001):
        g.total_sum_of_state_changes = 0

    if (g.total_sum_infections < 0) and (g.total_sum_infections > -0.001):
        g.total_sum_infections = 0

    if (g.total_sum_of_state_changes < 0) or (g.total_sum_infections < 0):
        print("\nNegative Problem", g.total_sum_of_state_changes, g.total_sum_infections)
        print("s: \t", s)
        print("g.total_sum_infections: \t", g.total_sum_infections)
        print("g.cumulative_sum_infection_rates: \t", g.cumulative_sum_infection_rates)
        print("g.cumulative_sum_of_state_changes: \t", g.cumulative_sum_of_state_changes)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    return continue_run


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # SIMULATION  # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def run_simulation(
    my,
    g,
    state_total_counts,
    agents_in_state,
    N_states,
    SIR_transition_rates,
    N_infectious_states,
    nts,
    verbose,
):

    out_time = List()
    out_state_counts = List()
    out_my_state = List()
    # out_infection_counter = List()
    # out_my_number_of_contacts = List()
    N_positive_tested = List()

    daily_counter = 0
    click = 0
    step_number = 0
    real_time = 0.0

    s_counter = np.zeros(4)

    infectious_states = {4, 5, 6, 7}  # TODO: fix

    # Run the simulation ################################
    continue_run = True
    while continue_run:

        s = 0

        step_number += 1
        g.total_sum = g.total_sum_of_state_changes + g.total_sum_infections

        dt = -np.log(np.random.rand()) / g.total_sum
        real_time += dt

        g.cumulative_sum = 0.0
        ra1 = np.random.rand()

        #######/ Here we move between infected between states
        accept = False
        if g.total_sum_of_state_changes / g.total_sum > ra1:

            s = 1

            x = g.cumulative_sum_of_state_changes / g.total_sum
            state_now = np.searchsorted(x, ra1)
            state_after = state_now + 1

            agent = utils.numba_random_choice_list(agents_in_state[state_now])

            # We have chosen agent to move -> here we move it
            agents_in_state[state_after].append(agent)
            agents_in_state[state_now].remove(agent)

            my.state[agent] += 1

            state_total_counts[state_now] -= 1
            state_total_counts[state_after] += 1

            g.total_sum_of_state_changes -= SIR_transition_rates[state_now]
            g.total_sum_of_state_changes += SIR_transition_rates[state_after]

            g.cumulative_sum_of_state_changes[state_now] -= SIR_transition_rates[state_now]
            g.cumulative_sum_of_state_changes[state_after:] += (
                SIR_transition_rates[state_after] - SIR_transition_rates[state_now]
            )

            g.cumulative_sum_infection_rates[state_now] -= g.sum_of_rates[agent]

            accept = True

            # Moves TO infectious State from non-infectious
            if my.state[agent] == N_infectious_states:
                for contact, rate in zip(
                    my.connections[agent], g.rates[agent]
                ):  # Loop over row agent
                    # if contact is susceptible
                    if my.state[contact] == -1:
                        g.update_rates(my, +rate, agent)
                        # g.total_sum_infections += rate
                        # g.sum_of_rates[agent] += rate
                        # g.cumulative_sum_infection_rates[my.state[agent] :] += rate

            # If this moves to Recovered state
            if my.state[agent] == N_states - 1:
                for contact, rate in zip(my.connections[agent], g.rates[agent]):
                    # if contact is susceptible
                    if my.state[contact] == -1:
                        g.update_rates(my, -rate, agent)
                        # g.total_sum_infections -= rate
                        # g.sum_of_rates[agent] -= rate
                        # g.cumulative_sum_infection_rates[my.state[agent] :] -= rate

        #######/ Here we infect new states
        else:
            s = 2

            x = (g.total_sum_of_state_changes + g.cumulative_sum_infection_rates) / g.total_sum
            state_now = np.searchsorted(x, ra1)
            g.cumulative_sum = (
                g.total_sum_of_state_changes + g.cumulative_sum_infection_rates[state_now - 1]
            ) / g.total_sum  # important change from [state_now] to [state_now-1]

            agent_getting_infected = -1
            for agent in agents_in_state[state_now]:

                # suggested cumulative sum
                suggested_cumulative_sum = g.cumulative_sum + g.sum_of_rates[agent] / g.total_sum

                if suggested_cumulative_sum > ra1:
                    for rate, contact in zip(g.rates[agent], my.connections[agent]):

                        # if contact is susceptible
                        if my.state[contact] == -1:

                            g.cumulative_sum += rate / g.total_sum

                            # here agent infect contact
                            if g.cumulative_sum > ra1:
                                my.state[contact] = 0
                                agents_in_state[0].append(np.uint32(contact))
                                state_total_counts[0] += 1
                                g.total_sum_of_state_changes += SIR_transition_rates[0]
                                g.cumulative_sum_of_state_changes += SIR_transition_rates[0]
                                accept = True
                                agent_getting_infected = contact
                                break
                else:
                    g.cumulative_sum = suggested_cumulative_sum

                if accept:
                    break

            if agent_getting_infected == -1:
                print(
                    "Error! Not choosing any agent getting infected.",
                    "\naccept:",
                    accept,
                    "\nagent_getting_infected: ",
                    agent_getting_infected,
                    "\nstep_number",
                    step_number,
                    "\ncfg",
                )
                my.cfg.print()
                break

            # Here we update infection lists so that newly infected cannot be infected again

            # loop over contacts of the newly infected agent in order to:
            # 1) remove newly infected agent from contact list (find_myself) by setting rate to 0
            # 2) remove rates from contacts gillespie sums (only if they are in infections state (I))
            for contact_of_agent_getting_infected in my.connections[agent_getting_infected]:

                # loop over indexes of the contact to find_myself and set rate to 0
                for ith_contact_of_agent_getting_infected in range(
                    my.number_of_contacts[contact_of_agent_getting_infected]
                ):

                    ith_contact_of_agent_getting_infected = np.uint64(
                        ith_contact_of_agent_getting_infected
                    )

                    find_myself = my.connections[contact_of_agent_getting_infected][
                        ith_contact_of_agent_getting_infected
                    ]

                    # check if the contact found is myself
                    if find_myself == agent_getting_infected:

                        rate = g.rates[contact_of_agent_getting_infected][
                            ith_contact_of_agent_getting_infected
                        ]

                        # set rates to myself to 0 (I cannot get infected again)
                        g.rates[contact_of_agent_getting_infected][
                            ith_contact_of_agent_getting_infected
                        ] = 0

                        # if the contact can infect, then remove the rates from the overall gillespie accounting
                        if my.state[contact_of_agent_getting_infected] in infectious_states:
                            g.update_rates(my, -rate, contact_of_agent_getting_infected)
                            # g.total_sum_infections -= rate
                            # g.sum_of_rates[contact_of_agent_getting_infected] -= rate
                            # g.cumulative_sum_infection_rates[
                            # my.state[contact_of_agent_getting_infected] :
                            # ] -= rate

                        break

                # else:
                #     continue

        ################

        if nts * click < real_time:

            daily_counter += 1
            out_time.append(real_time)
            out_state_counts.append(state_total_counts.copy())

            if daily_counter >= 10:

                daily_counter = 0
                out_my_state.append(my.state.copy())

                # tent testing
                # N_daily_tests = 10_000
                N_daily_tests = 0
                f_test_succes = 0.8

                n_positive_tested = daily_tent_test(
                    my,
                    g,
                    N_daily_tests,
                    f_test_succes,
                    N_infectious_states,
                    N_states,
                    infectious_states,
                )
                N_positive_tested.append(n_positive_tested)

            click += 1

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        continue_run = do_bug_check(
            my,
            g,
            step_number,
            continue_run,
            verbose,
            state_total_counts,
            N_states,
            accept,
            ra1,
            s,
            x,
        )

        s_counter[s] += 1

    if verbose:
        print("Simulation step_number, ", step_number)
        print("s_counter", s_counter)
        print("N_daily_tests", N_daily_tests)
        print("N_positive_tested", N_positive_tested)

    return out_time, out_state_counts, out_my_state


# ███    ███  █████  ██████  ████████ ██ ███    ██ ██    ██
# ████  ████ ██   ██ ██   ██    ██    ██ ████   ██  ██  ██
# ██ ████ ██ ███████ ██████     ██    ██ ██ ██  ██   ████
# ██  ██  ██ ██   ██ ██   ██    ██    ██ ██  ██ ██    ██
# ██      ██ ██   ██ ██   ██    ██    ██ ██   ████    ██
#

#%%


@njit
def compute_my_cluster_coefficient(my):
    """calculates my cluster cooefficent
    (np.mean of the first output gives cluster coeff for whole network )
    """

    cluster_coefficient = np.zeros(my.cfg.N_tot, dtype=np.float32)
    for agent1 in range(my.cfg.N_tot):
        counter = 0
        total = 0
        for j, contact in enumerate(my.connections[agent1]):
            for k in range(j + 1, my.number_of_contacts[agent1]):
                if contact in my.connections[my.connections[agent1][k]]:
                    counter += 1
                    break
                total += 1
        cluster_coefficient[agent1] = counter / total
    return cluster_coefficient
