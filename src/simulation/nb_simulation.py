import numpy as np
from pathlib import Path
import os
import numba as nb
from numba.experimental import jitclass
from numba import njit, typeof
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type

from src.utils import utils

#%%

#      ██ ██ ████████      ██████ ██       █████  ███████ ███████ ███████ ███████
#      ██ ██    ██        ██      ██      ██   ██ ██      ██      ██      ██
#      ██ ██    ██        ██      ██      ███████ ███████ ███████ █████   ███████
# ██   ██ ██    ██        ██      ██      ██   ██      ██      ██ ██           ██
#  █████  ██    ██         ██████ ███████ ██   ██ ███████ ███████ ███████ ███████
#
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Regular&t=Version%202%0A%20

#%%

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # cfg Config file # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
            *(" version: ", self.version, "\n"),
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # My object # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#%%

spec_my = {
    "age": nb.uint8[:],
    "connections": ListType(ListType(nb.uint32)),
    "connections_type": ListType(ListType(nb.uint8)),
    "coordinates": nb.float32[:, :],
    "connection_weight": nb.float32[:],
    "infection_weight": nb.float64[:],
    "number_of_contacts": nb.uint16[:],
    "state": nb.int8[:],
    "tent": nb.uint16[:],
    "kommune": nb.uint8[:],
    "infectious_states": ListType(nb.int64),
    "cfg": nb_cfg_type,
}


# "Nested/Mutable" Arrays are faster than list of arrays which are faster than lists of lists
@jitclass(spec_my)
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
        self.state = np.full(N_tot, fill_value=-1, dtype=np.int8)
        self.tent = np.zeros(N_tot, dtype=np.uint16)
        self.kommune = np.zeros(N_tot, dtype=np.uint8)
        self.infectious_states = List([4, 5, 6, 7])
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

    def agent_is_susceptable(self, agent):
        return self.state[agent] == -1

    def agent_is_infectious(self, agent):
        return self.state[agent] in self.infectious_states


def initialize_My(cfg):
    nb_cfg = initialize_nb_cfg(cfg)
    return My(nb_cfg)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # Gillespie Algorithm # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#%%

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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # Intervention Class  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#%%
#%%

spec_intervention = {
    "N_tot": nb.uint32,
    "N_daily_tests": nb.uint32,
    "labels": nb.uint8[:],  # affilitation? XXX
    "label_counter": nb.uint32[:],
    "N_labels": nb.uint32,
    "interventions_to_apply": ListType(nb.int64),
    "day_found_infected": nb.int32[:],
    "reason_for_test": nb.int8[:],
    "positive_test_counter": nb.uint32[:],
    "clicks_when_tested": nb.int32[:],
    "clicks_when_tested_result": nb.int32[:],
    "test_delay_in_clicks": nb.uint32[:],
    "results_delay_in_clicks": nb.uint32[:],
    "chance_of_finding_infected": nb.float32[:],
    "days_looking_back": nb.float32,
    "masking_rate_reduction": nb.float32[:, ::1],  # to make the type C instead if A
    "lockdown_rate_reduction": nb.float32[:, ::1],  # to make the type C instead if A
    "isolation_rate_reduction": nb.float32[:],
    "tracking_rates": nb.float32[:],
    "types": nb.uint8[:],
    "started_as": nb.uint8[:],
    "verbose": nb.boolean,
}


@jitclass(spec_intervention)
class Intervention(object):
    """
    - N_tot: Number of agents
    - N_daily_tests: Number of total daily tests scaled relative to a full population

    - N_labels: Number of labels. "Label" here can refer to either tent or kommune.
    - labels: a label or ID which is either the nearest tent or the kommune which the agent belongs to
    - label_counter: count how many agent belong to a particular label

    - interventions_to_apply:
        1: Lockdown (jobs and schools)
        2: Cover (with face masks)
        3: Tracking (infected and their connections)
        4: Test people with symptoms
        5: Isolate (if you get a positive test, isolate yourself from your contacts (isolation_rate_reduction))
        6: Random Testing
        0/None: Do nothing

    - day_found_infected: -1 if not infected, otherwise the day of infection

    - reason_for_test:
         0: symptoms
         1: random_test
         2: tracing,
        -1: No reason yet (or no impending tests). You can still be tested again later on (if negative test)

    - positive_test_counter: counter of how many were found tested positive due to reasom 0, 1 or 2

    - clicks_when_tested: When you were tested measured in clicks (10 clicks = 1 day)

    - clicks_when_tested_result: When you get your test results measured in clicks

    - test_delay_in_clicks: clicks until test. [symptoms, random_test, tracing]

    - results_delay_in_clicks: clicks from test until results. [symptoms, random_test, tracing]

    - chance_of_finding_infected: When people moves into the ith I state, what is the chance to detect them in a test

    - days_looking_back: When looking for local outbreaks, how many days are we looking back, e.g. number of people infected within the last 7 days

    # Reductions.
        The rate reductions are list of list, first and second entry are rate reductions for the groups [family, job, other]. The third entry is the chance of choosing the first set. As such you can have some follow the lockdown and some not or some one group being able to work from home and another isn't.
    - masking_rate_reduction: Rate reduction for the groups [family, job, other]
    - lockdown_rate_reduction: Rate reduction for the groups [family, job, other]
    - isolation_rate_reduction: Rate reduction for the groups [family, job, other]
    - tracking_rates: fraction of connections we track for the groups [family, job, other]

    - types: array to keep count of which intervention are at place at which label
        0: Do nothing
        1: lockdown (jobs and schools)
        2: Track (infected and their connections),
        3: Cover (with face masks)

    - started_as: describes whether or not an intervention has been applied. If 0, no intervention has been applied.

    - verbose: Prints status of interventions and removal of them

    """

    def __init__(self, N_tot, N_daily_tests, labels, interventions_to_apply, verbose=False):
        self.N_tot = N_tot
        self.N_daily_tests = int(N_daily_tests * N_tot / 5_800_000)

        self._initialize_labels(labels)

        self._initialize_interventions_to_apply(interventions_to_apply)

        self.day_found_infected = np.full(N_tot, fill_value=-1, dtype=np.int32)
        self.reason_for_test = np.full(N_tot, fill_value=-1, dtype=np.int8)
        self.positive_test_counter = np.zeros(3, dtype=np.uint32)
        self.clicks_when_tested = np.full(N_tot, fill_value=-1, dtype=np.int32)
        self.clicks_when_tested_result = np.full(N_tot, fill_value=-1, dtype=np.int32)
        self.test_delay_in_clicks = np.array([0, 0, 25], dtype=np.uint32)
        self.results_delay_in_clicks = np.array([5, 10, 5], dtype=np.uint32)
        self.chance_of_finding_infected = np.array([0.0, 0.15, 0.15, 0.15, 0.0], dtype=np.float32)
        self.days_looking_back = 7.0

        self.masking_rate_reduction = np.array([[0, 0, 0.0], [0, 0, 0.8]], dtype=np.float32)
        self.lockdown_rate_reduction = np.array([[0, 1, 0.6], [0, 0.6, 0.6]], dtype=np.float32)
        self.isolation_rate_reduction = np.array([0.2, 1, 1], dtype=np.float32)
        self.tracking_rates = np.array([1, 0.8, 0], dtype=np.float32)

        self.types = np.zeros(self.N_labels, dtype=np.uint8)
        self.started_as = np.zeros(self.N_labels, dtype=np.uint8)

        self.verbose = verbose

    def _initialize_interventions_to_apply(self, interventions_to_apply=None):
        if interventions_to_apply is None:
            self.interventions_to_apply = List([np.int64(0)])
        else:
            lst = List()
            for intervention in interventions_to_apply:
                lst.append(np.int64(intervention))
            self.interventions_to_apply = lst

    def _initialize_labels(self, labels):
        self.labels = np.asarray(labels, dtype=np.uint8)
        unique, counts = utils.numba_unique_with_counts(labels)
        self.label_counter = np.asarray(counts, dtype=np.uint32)
        self.N_labels = len(unique)

    def agent_has_not_been_tested(self, agent):
        return self.day_found_infected[agent] == -1

    def do_apply_intervention_on_label(self):
        return 1 in self.interventions_to_apply or 2 in self.interventions_to_apply

    def do_tracking(self):
        return 3 in self.interventions_to_apply

    def do_test_people_with_symptoms(self):
        return 4 in self.interventions_to_apply

    def do_isolate(self):
        return 5 in self.interventions_to_apply

    def do_random_test(self):
        return 6 in self.interventions_to_apply

    def any_testing(self):
        return self.do_tracking() or self.do_test_people_with_symptoms() or self.do_random_test()


#%%
# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██      ██
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██     ███
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      ██
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██      ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████      ██
#
#%%


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


#%%
# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██     ██████
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██          ██
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      █████
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██     ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████     ███████
#
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
def do_bug_check(
    my,
    g,
    step_number,
    day,
    continue_run,
    verbose,
    state_total_counts,
    N_states,
    accept,
    ra1,
    s,
    x,
):

    if day > 2_000:
        print("day exceeded 2000")
        continue_run = False

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
    intervention,
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
    # N_positive_tested = List()

    daily_counter = 0
    day = 0
    click = 0
    step_number = 0
    real_time = 0.0

    s_counter = np.zeros(4)

    # infectious_states = {4, 5, 6, 7}  # TODO: fix

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

            if intervention.do_test_people_with_symptoms():

                # test infectious people
                if my.state[agent] >= N_infectious_states:

                    randomly_selected = (
                        np.random.rand()
                        < intervention.chance_of_finding_infected[my.state[agent] - 4]
                    )
                    not_tested_before = intervention.clicks_when_tested[agent] == -1

                    if randomly_selected and not_tested_before:
                        # testing in n_clicks for symptom checking
                        intervention.clicks_when_tested[agent] = (
                            click + intervention.test_delay_in_clicks[0]
                        )
                        # set the reason for testing to symptoms (0)
                        intervention.reason_for_test[agent] = 0

            # Moves TO infectious State from non-infectious
            if my.state[agent] == N_infectious_states:
                for contact, rate in zip(my.connections[agent], g.rates[agent]):
                    # update rates if contact is susceptible
                    if my.agent_is_susceptable(contact):
                        g.update_rates(my, +rate, agent)

            # If this moves to Recovered state
            if my.state[agent] == N_states - 1:
                for contact, rate in zip(my.connections[agent], g.rates[agent]):
                    # update rates if contact is susceptible
                    if my.agent_is_susceptable(contact):
                        g.update_rates(my, -rate, agent)

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
                    *("\naccept:", accept),
                    *("\nagent_getting_infected: ", agent_getting_infected),
                    *("\nstep_number", step_number),
                    "\ncfg:",
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

                    # tmp = my.connections[contact_of_agent_getting_infected]
                    # find_myself = np.int64(tmp[np.int64(ith_contact_of_agent_getting_infected)])
                    # find_myself = np.uint64(tmp[ith_contact_of_agent_getting_infected])
                    # find_myself = tmp[np.uint64(ith_contact_of_agent_getting_infected)]
                    # agent_getting_infected = np.uint64(agent_getting_infected)
                    find_myself = my.connections[contact_of_agent_getting_infected][
                        ith_contact_of_agent_getting_infected
                    ]
                    # agent_getting_infected = np.uint64(agent_getting_infected)

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
                        if my.agent_is_infectious(contact_of_agent_getting_infected):
                            g.update_rates(my, -rate, contact_of_agent_getting_infected)

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
                day += 1
                out_my_state.append(my.state.copy())

                if intervention.do_apply_intervention_on_label():

                    if intervention.do_random_test():

                        # choose N_daily_test people at random to test
                        random_people_for_test = np.random.choice(
                            np.arange(my.cfg.N_tot, dtype=np.uint32), intervention.N_daily_tests
                        )
                        intervention.clicks_when_tested[random_people_for_test] = (
                            click + intervention.test_delay_in_clicks[1]
                        )
                        # count that random test is the reason for test
                        intervention.reason_for_test[random_people_for_test] = 1

                    test_if_label_needs_intervention(
                        intervention, day, intervention_type_to_init=1, threshold=0.004
                    )

                    test_if_intervention_on_labels_can_be_removed(
                        my, g, intervention, day, threshold=0.001
                    )

                    for ith_label, intervention_type in enumerate(intervention.types):

                        if intervention_type in intervention.interventions_to_apply:
                            intervention_has_not_been_applied = (
                                intervention.started_as[ith_label] == 0
                            )

                            apply_lockdown = intervention_type == 1
                            if apply_lockdown and intervention_has_not_been_applied:
                                intervention.started_as[ith_label] = 1
                                lockdown_on_label(
                                    my,
                                    g,
                                    intervention,
                                    label=ith_label,
                                    rate_reduction=intervention.lockdown_rate_reduction,
                                )

            if intervention.any_testing():

                # test everybody whose counter say we should test
                for agent in range(my.cfg.N_tot):
                    # testing everybody who should be tested
                    if intervention.clicks_when_tested[agent] == click:
                        test_a_person(my, g, intervention, agent, click, day)

                    # getting results for people
                    if intervention.clicks_when_tested_result[agent] == click:
                        intervention.clicks_when_tested_result[agent] = -1
                        intervention.day_found_infected[agent] = day
                        if intervention.do_isolate():
                            cut_rates_of_agent(
                                my,
                                g,
                                intervention,
                                agent,
                                rate_reduction=intervention.isolation_rate_reduction,
                            )

            click += 1

        continue_run = do_bug_check(
            my,
            g,
            step_number,
            day,
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
        print("positive_test_counter", intervention.positive_test_counter)
        # print("N_daily_tests", intervention.N_daily_tests)
        # print("N_positive_tested", N_positive_tested)

    return out_time, out_state_counts, out_my_state


#%%
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


@njit
def initialize_tents(my, N_tents):

    """Pick N_tents tents in random positions and compute which tent each
    person is nearest to and connect that person to that tent.
    """

    N_tot = my.cfg.N_tot

    tent_positions = np.zeros((N_tents, 2), np.float32)
    for i in range(N_tents):
        tent_positions[i] = my.coordinates[np.random.randint(N_tot)]

    tent_counter = np.zeros(N_tents, np.uint16)
    for agent in range(N_tot):
        distances = np.array(
            [utils.haversine_scipy(my.coordinates[agent], p_tent) for p_tent in tent_positions]
        )
        closest_tent = np.argmin(distances)
        my.tent[agent] = closest_tent
        tent_counter[closest_tent] += 1

    return tent_positions, tent_counter


# @njit
def initialize_kommuner(my, df_coordinates):
    my.kommune = np.array(df_coordinates["idx"].values, dtype=np.uint8)
    kommune_counter = df_coordinates["idx"].value_counts().sort_index()
    kommune_counter = np.array(kommune_counter, dtype=np.uint16)
    return kommune_counter


@njit
def test_if_label_needs_intervention(
    intervention,
    day,
    intervention_type_to_init,
    threshold=0.004,  # threshold is the fraction that need to be positive.
):
    infected_per_label = np.zeros_like(intervention.label_counter, dtype=np.uint32)

    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > max(0, day - intervention.days_looking_back):
            infected_per_label[intervention.labels[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        if N_infected / N_inhabitants > threshold and my_intervention_type == 0:
            if intervention.verbose:
                print(
                    *("lockdown at label", i_label),
                    *("at day", day),
                    *("the num of infected is", N_infected),
                    *("/", N_inhabitants),
                )

            intervention.types[i_label] = intervention_type_to_init

    return None


@njit
def reset_rates_of_agent(my, g, agent, connection_type_weight=None):

    if connection_type_weight is None:
        # reset infection rate to origin times this number for [home, job, other]
        connection_type_weight = np.ones(3, dtype=np.float32)

    agent_update_rate = 0.0
    for ith_contact, contact in enumerate(my.connections[agent]):

        infection_rate = (
            my.infection_weight[agent]
            * connection_type_weight[my.connections_type[agent][ith_contact]]
        )

        rate = infection_rate - g.rates[agent][ith_contact]
        g.rates[agent][ith_contact] = infection_rate

        if my.agent_is_infectious(agent) and my.agent_is_susceptable(contact):
            agent_update_rate += rate

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact, possible_agent in enumerate(my.connections[contact]):

            # check if the contact found is myself
            if agent == possible_agent:

                # update rates from contact to agent.
                c_rate = my.infection_weight[contact] - g.rates[contact][ith_contact_of_contact]
                g.rates[contact][ith_contact_of_contact] = my.infection_weight[contact]

                # updates to gillespie sums, if contact is infectious and agent is susceptible
                if my.agent_is_infectious(contact) and my.agent_is_susceptable(agent):
                    g.update_rates(my, c_rate, contact)
                break

    # actually updates to gillespie sums
    g.update_rates(my, +agent_update_rate, agent)
    return None


@njit
def remove_intervention_at_label(my, g, intervention, ith_label):
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == ith_label:
            reset_rates_of_agent(my, g, agent, connection_type_weight=None)
    return None


@njit
def test_if_intervention_on_labels_can_be_removed(my, g, intervention, day, threshold=0.001):

    infected_per_label = np.zeros(intervention.N_labels, dtype=np.int32)
    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > day - intervention.days_looking_back:
            infected_per_label[intervention.labels[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        if (N_infected / N_inhabitants) < threshold and my_intervention_type != 0:

            remove_intervention_at_label(my, g, intervention, i_label)

            intervention.types[i_label] = 0
            intervention.started_as[i_label] = 0
            if intervention.verbose:
                print(
                    *("remove lockdown at num of infected", i_label),
                    *("at day", day),
                    *("the num of infected is", N_infected),
                    *("/", N_inhabitants),
                )

    return None


@njit
def loop_update_rates_of_contacts(
    my, g, intervention, agent, contact, rate, agent_update_rate, rate_reduction
):

    # updates to gillespie sums, if agent is infected and contact is susceptible
    if my.agent_is_infectious(agent) and my.agent_is_susceptable(contact):
        agent_update_rate += rate

    # loop over indexes of the contact to find_myself and set rate to 0
    for ith_contact_of_contact, possible_agent in enumerate(my.connections[contact]):

        # check if the contact found is myself
        if agent == possible_agent:

            # update rates from contact to agent. Rate_reduction makes it depending on connection type
            c_rate = (
                g.rates[contact][ith_contact_of_contact]
                * rate_reduction[my.connections_type[contact][ith_contact_of_contact]]
            )
            g.rates[contact][ith_contact_of_contact] -= c_rate

            # updates to gillespie sums, if contact is infectious and agent is susceptible
            if my.agent_is_infectious(contact) and my.agent_is_susceptable(agent):
                g.update_rates(my, -c_rate, contact)
            break

    return agent_update_rate


@njit
def cut_rates_of_agent(my, g, intervention, agent, rate_reduction):

    agent_update_rate = 0.0

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):

        # update rates from agent to contact. Rate_reduction makes it depending on connection type

        rate = g.rates[agent][ith_contact] * rate_reduction[my.connections_type[agent][ith_contact]]
        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None


@njit
def reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction):
    # rate reduction is 2 3-vectors. is used for masking interventions
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        if np.random.rand() < remove_rates[my.connections_type[agent][ith_contact]]:
            act_rate_reduction = np.array([0, 0, 0], dtype=np.float32)
        else:
            act_rate_reduction = reduce_rates

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )

        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None


@njit
def remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction):
    # rate reduction is 2 3-vectors. is used for masking interventions
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        act_rate_reduction = reduce_rates
        if np.random.rand() < remove_rates[my.connections_type[agent][ith_contact]]:
            act_rate_reduction = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )

        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=act_rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None


@njit
def lockdown_on_label(my, g, intervention, label, rate_reduction):
    # lockdown on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to set to 0.
    # second is the fraction of reduction of the remaining [home, job, others] rates.
    # ie: [[0,0.8,0.8],[0,0.8,0.8]] means that 80% of your contacts on job and other is set to 0, and the remaining 20% is reduced by 80%.
    # loop over all agents
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == label:
            remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def masking_on_label(my, g, intervention, label, rate_reduction):
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie: [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == label:
            reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def test_a_person(my, g, intervention, agent, click, day):
    # if agent is infectious and hasn't been tested before
    if my.agent_is_infectious(agent) and intervention.agent_has_not_been_tested(agent):
        intervention.clicks_when_tested_result[agent] = (
            click + intervention.results_delay_in_clicks[intervention.reason_for_test[agent]]
        )
        intervention.positive_test_counter[
            intervention.reason_for_test[agent]
        ] += 1  # count reason found infected

        # check if tracking is on
        if intervention.do_tracking():
            # loop over contacts
            for ith_contact, contact in enumerate(my.connections[agent]):
                if (
                    np.random.rand()
                    < intervention.tracking_rates[my.connections_type[agent][ith_contact]]
                    and intervention.clicks_when_tested[contact] == -1
                ):
                    intervention.reason_for_test[contact] = 2
                    intervention.clicks_when_tested[contact] = (
                        click + intervention.test_delay_in_clicks[2]
                    )

    intervention.clicks_when_tested[agent] = -1
    intervention.reason_for_test[agent] = -1

    return None