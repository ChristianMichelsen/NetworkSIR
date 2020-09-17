import numpy as np
from numba import njit
from numba.typed import List
from pathlib import Path
import os
from src.utils import utils


def test():
    print("test")


# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██      ██
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██     ███
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      ██
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██      ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████      ██
#
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Regular&t=Version%202%0A%20


@njit
def v1_initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta):
    my_connection_weight = np.ones(N_tot, dtype=np.float32)
    my_infection_weight = np.ones(N_tot, dtype=np.float32)
    for agent in range(N_tot):
        set_connections_weight(my_connection_weight, agent, sigma_mu)
        set_infection_weight(my_infection_weight, agent, sigma_beta, beta)
    my_number_of_contacts = np.zeros(N_tot, dtype=np.uint16)
    my_connections_type = utils.initialize_nested_lists(1, dtype=np.uint8)
    return my_connection_weight, my_infection_weight, my_number_of_contacts, my_connections_type


@njit
def v1_run_algo_2(
    PP,
    my_connections,
    my_connections_type,
    my_number_of_contacts,
    coordinates,
    rho_tmp,
):
    while True:
        agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
        agent2 = np.uint32(np.searchsorted(PP, np.random.rand()))
        do_stop = update_node_connections(
            my_connections,
            coordinates,
            rho_tmp,
            agent1,
            agent2,
            my_number_of_contacts,
            my_connections_type,
            connection_type=-1,
            code_version=1,
        )
        if do_stop:
            break


@njit
def v1_run_algo_1(
    PP,
    my_connections,
    my_connections_type,
    my_number_of_contacts,
    coordinates,
    rho_tmp,
):
    agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
    while True:
        agent2 = np.uint32(np.searchsorted(PP, np.random.rand()))
        rho_tmp *= 0.9995
        do_stop = update_node_connections(
            my_connections,
            coordinates,
            rho_tmp,
            agent1,
            agent2,
            my_number_of_contacts,
            my_connections_type,
            connection_type=-1,
            code_version=1,
        )
        if do_stop:
            break


@njit
def v1_connect_nodes(
    N_tot,
    mu,
    rho,
    epsilon_rho,
    algo,
    my_connection_weight,
    my_connections,
    my_connections_type,
    my_number_of_contacts,
    coordinates,
):
    if algo == 2:
        run_algo = v1_run_algo_2
    else:
        run_algo = v1_run_algo_1
    PP = np.cumsum(my_connection_weight) / np.sum(my_connection_weight)
    for _ in range(mu / 2 * N_tot):
        if np.random.rand() > epsilon_rho:
            rho_tmp = rho
        else:
            rho_tmp = 0.0
        run_algo(PP, my_connections, my_connections_type, my_number_of_contacts, coordinates, rho_tmp)


# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██     ██████
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██          ██
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      █████
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██     ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████     ███████


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # INITIALIZATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




@njit
def set_connections_weight(my_connection_weight, agent, sigma_mu):
    """ How introvert / extrovert you are. How likely you are at having many contacts in your network."""
    if np.random.rand() < sigma_mu:
        my_connection_weight[agent] = 0.1 - np.log(np.random.rand())
    else:
        my_connection_weight[agent] = 1.1


@njit
def set_infection_weight(my_infection_weight, agent, sigma_beta, beta):
    " How much of a super sheader are you?"
    if np.random.rand() < sigma_beta:
        my_infection_weight[agent] = -np.log(np.random.rand()) * beta
    else:
        my_infection_weight[agent] = beta


@njit
def update_node_connections(
    my_connections,
    coordinates,
    rho_tmp,
    agent1,
    agent2,
    my_number_of_contacts,
    my_connections_type,
    connection_type,
    code_version=2,
):
    connect_and_stop = False
    if agent1 != agent2:

        if rho_tmp == 0:
            connect_and_stop = True
        else:
            r = utils.haversine_scipy(coordinates[agent1], coordinates[agent2])
            if np.exp(-r * rho_tmp) > np.random.rand():
                connect_and_stop = True

        if connect_and_stop:

            if agent1 not in my_connections[agent2] and agent2 not in my_connections[agent1]:

                my_connections[agent1].append(np.uint32(agent2))
                my_connections[agent2].append(np.uint32(agent1))

                if code_version >= 2:
                    my_connections_type[agent1].append(np.uint8(connection_type))
                    my_connections_type[agent2].append(np.uint8(connection_type))

                my_number_of_contacts[agent1] += 1
                my_number_of_contacts[agent2] += 1

    return connect_and_stop


@njit
def place_and_connect_families(
    N_tot,
    people_in_household,
    age_distribution_per_people_in_household,
    coordinates,
    sigma_mu,
    sigma_beta,
    beta,
):

    all_indices = np.arange(N_tot)
    np.random.shuffle(all_indices)

    my_age = np.zeros(N_tot, dtype=np.uint8)
    my_connections = utils.initialize_nested_lists(N_tot, dtype=np.uint32)
    my_connections_type = utils.initialize_nested_lists(N_tot, dtype=np.uint8)
    my_coordinates = np.zeros((N_tot, 2), dtype=np.float32)

    my_connection_weight = np.ones(N_tot, dtype=np.float32)
    my_infection_weight = np.ones(N_tot, dtype=np.float64)

    my_number_of_contacts = np.zeros(N_tot, dtype=np.uint16)

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

            age_index = utils.rand_choice_nb(age_distribution_per_people_in_household[N_people_in_house_index])

            age = age_index  # just use age index as substitute for age
            my_age[agent] = age
            counter_ages[age_index] += 1
            agents_in_age_group[age_index].append(np.uint32(agent))

            my_coordinates[agent] = coordinates[house_index]

            set_connections_weight(my_connection_weight, agent, sigma_mu)
            set_infection_weight(my_infection_weight, agent, sigma_beta, beta)

            agent += 1

        # add agents to each others networks (connections)
        for agent1 in range(agent0, agent0 + N_people_in_house):
            for agent2 in range(agent1, agent0 + N_people_in_house):
                if agent1 != agent2:
                    my_connections[agent1].append(np.uint32(agent2))
                    my_connections[agent2].append(np.uint32(agent1))
                    my_connections_type[agent1].append(np.uint8(0))
                    my_connections_type[agent2].append(np.uint8(0))
                    # my_index_in_contact[agent2].append(my_number_of_contacts[agent1])
                    my_number_of_contacts[agent1] += 1
                    my_number_of_contacts[agent2] += 1
                    mu_counter += 1

    agents_in_age_group = utils.nested_lists_to_list_of_array(agents_in_age_group)

    return (
        my_age,
        my_connections,
        my_coordinates,
        my_connection_weight,
        my_infection_weight,
        mu_counter,
        counter_ages,
        agents_in_age_group,
        my_connections_type,
        my_number_of_contacts,
    )


@njit
def run_algo_other(
    agents_in_age_group,
    age1,
    age2,
    my_connections,
    my_connections_type,
    my_number_of_contacts,
    coordinates,
    rho_tmp,
):
    while True:
        # agent1 = np.searchsorted(PP_ages[m_i], np.random.rand())
        # agent1 = agents_in_age_group[m_i][agent1]
        # TODO: Add connection weights
        agent1 = np.random.choice(agents_in_age_group[age1])
        agent2 = np.random.choice(agents_in_age_group[age2])
        do_stop = update_node_connections(
            my_connections,
            coordinates,
            rho_tmp,
            agent1,
            agent2,
            my_number_of_contacts,
            my_connections_type,
            connection_type=2,
            code_version=2,
        )
        if do_stop:
            break


@njit
def run_algo_work(
    agents_in_age_group,
    age1,
    age2,
    my_connections,
    my_connections_type,
    my_number_of_contacts,
    coordinates,
    rho_tmp,
):
    # ra1 = np.random.rand()
    # agent1 = np.searchsorted(PP_ages[m_i], ra1)
    # agent1 = agents_in_age_group[m_i][agent1]
    # TODO: Add connection weights
    agent1 = np.random.choice(agents_in_age_group[age1])

    while True:
        agent2 = np.random.choice(agents_in_age_group[age2])
        rho_tmp *= 0.9995
        do_stop = update_node_connections(
            my_connections,
            coordinates,
            rho_tmp,
            agent1,
            agent2,
            my_number_of_contacts,
            my_connections_type,
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
    N_tot,
    N_ages,
    mu_counter,
    mu,
    work_other_ratio,
    matrix_work,
    matrix_other,
    rho,
    epsilon_rho,
    coordinates,
    agents_in_age_group,
    my_connections,
    my_connections_type,
    my_number_of_contacts,
):

    while mu_counter < mu / 2 * N_tot:

        ra_work_other = np.random.rand()
        if ra_work_other < work_other_ratio:
            matrix = matrix_work
            run_algo = run_algo_work
        else:
            matrix = matrix_other
            run_algo = run_algo_other

        age1, age2 = find_two_age_groups(N_ages, matrix)

        if np.random.rand() > epsilon_rho:
            rho_tmp = rho
        else:
            rho_tmp = 0.0

        run_algo(
            agents_in_age_group,
            age1,
            age2,
            my_connections,
            my_connections_type,
            my_number_of_contacts,
            coordinates,
            rho_tmp,
        )
        mu_counter += 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def make_initial_infections(
    N_init,
    my_state,
    state_total_counts,
    agents_in_state,
    g_cumulative_sum_of_state_changes,
    my_connections,
    my_number_of_contacts,
    my_rates,
    SIR_transition_rates,
    agents_in_age_group,
    initial_ages_exposed,
    N_infectious_states,
    coordinates,
    make_random_initial_infections,
    code_version=2,
):

    g_total_sum_of_state_changes = 0.0
    N_tot = len(my_number_of_contacts)

    if code_version >= 2:
        possible_agents = List()
        for age_exposed in initial_ages_exposed:
            for agent in agents_in_age_group[age_exposed]:
                possible_agents.append(np.uint32(agent))
        possible_agents = np.asarray(possible_agents)
    else:
        possible_agents = np.arange(N_tot, dtype=np.uint32)

    ##  Standard outbreak type, infecting randomly
    if make_random_initial_infections:
        initial_agents_to_infect = np.random.choice(possible_agents, size=N_init, replace=False)

    # Local outbreak type, infecting around a point:
    else:

        rho_init_local_outbreak = 0.1

        outbreak_agent = np.random.randint(N_tot)  # this is where the outbreak starts

        initial_agents_to_infect = List()
        initial_agents_to_infect.append(np.uint32(outbreak_agent))

        while len(initial_agents_to_infect) < N_init:
            proposed_agent = np.random.randint(N_tot)

            r = utils.haversine_scipy(coordinates[outbreak_agent], coordinates[proposed_agent])
            if np.exp(-r * rho_init_local_outbreak) > np.random.rand():
                initial_agents_to_infect.append(np.uint32(proposed_agent))
        initial_agents_to_infect = np.asarray(initial_agents_to_infect, dtype=np.uint32)

    ##  Now make initial infections
    for _, agent in enumerate(initial_agents_to_infect):
        new_state = np.random.randint(N_infectious_states)  # E1, E2, E3 or E4
        my_state[agent] = new_state

        agents_in_state[new_state].append(np.uint32(agent))
        state_total_counts[new_state] += 1

        g_total_sum_of_state_changes += SIR_transition_rates[new_state]  # 'g_' = gillespie variable
        g_cumulative_sum_of_state_changes[new_state:] += SIR_transition_rates[new_state]

    return g_total_sum_of_state_changes


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # PRE SIMULATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def daily_tent_test(
    N_daily_tests,
    f_test_succes,
    N_tot,
    my_state,
    N_infectious_states,
    N_states,
    my_number_of_contacts,
    my_connections,
    my_rates,
    my_connections_type,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):

    n_positive_tested = 0

    for _ in range(N_daily_tests):
        agent = np.random.randint(N_tot)

        # only if in I state and  un-noticed
        if (my_state[agent] in infectious_states) and (np.random.rand() < f_test_succes):

            # agent_tested_positive.append(agent)
            n_positive_tested += 1

            for i in range(my_number_of_contacts[agent]):
                contact = my_connections[agent][i]
                rate = my_rates[agent][i]
                connection_type = my_connections_type[agent][i]

                # only close work/other contacts
                if my_state[contact] == -1 and connection_type > -1:
                    g_total_sum_infections -= rate
                    my_sum_of_rates[agent] -= rate
                    g_cumulative_sum_infection_rates[my_state[agent] :] -= rate
                    my_rates[agent][i] = 0

    return g_total_sum_infections, n_positive_tested


@njit
def do_bug_check(
    step_number,
    continue_run,
    g_total_sum_infections,
    g_total_sum_of_state_changes,
    verbose,
    state_total_counts,
    N_states,
    N_tot,
    accept,
    g_cumulative_sum_of_state_changes,
    ra1,
    s,
    x,
    g_cumulative_sum_infection_rates,
):

    if step_number > 100_000_000:
        print("step_number > 100_000_000")
        continue_run = False

    if (g_total_sum_infections + g_total_sum_of_state_changes < 0.0001) and (
        g_total_sum_of_state_changes + g_total_sum_infections > -0.00001
    ):
        continue_run = False
        if verbose:
            print("Equilibrium")

    if state_total_counts[N_states - 1] > N_tot - 10:
        if verbose:
            print("2/3 through")
        continue_run = False

    # Check for bugs
    if not accept:
        print("\nNo Chosen rate")
        print("s: \t", s)
        print("g_total_sum_infections: \t", g_total_sum_infections)
        print("g_cumulative_sum_infection_rates: \t", g_cumulative_sum_infection_rates)
        print("g_cumulative_sum_of_state_changes: \t", g_cumulative_sum_of_state_changes)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    if (g_total_sum_of_state_changes < 0) and (g_total_sum_of_state_changes > -0.001):
        g_total_sum_of_state_changes = 0

    if (g_total_sum_infections < 0) and (g_total_sum_infections > -0.001):
        g_total_sum_infections = 0

    if (g_total_sum_of_state_changes < 0) or (g_total_sum_infections < 0):
        print("\nNegative Problem", g_total_sum_of_state_changes, g_total_sum_infections)
        print("s: \t", s)
        print("g_total_sum_infections: \t", g_total_sum_infections)
        print("g_cumulative_sum_infection_rates: \t", g_cumulative_sum_infection_rates)
        print("g_cumulative_sum_of_state_changes: \t", g_cumulative_sum_of_state_changes)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    return continue_run, g_total_sum_of_state_changes, g_total_sum_infections


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # SIMULATION  # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def run_simulation(
    N_tot,
    g_total_sum_of_state_changes,
    g_cumulative_sum_of_state_changes,
    state_total_counts,
    agents_in_state,
    my_state,
    g_cumulative_sum_infection_rates,
    N_states,
    my_sum_of_rates,
    SIR_transition_rates,
    N_infectious_states,
    my_number_of_contacts,
    my_rates,
    my_connections,
    my_age,
    individual_infection_counter,
    nts,
    verbose,
    my_connections_type,
):

    out_time = List()
    out_state_counts = List()
    out_my_state = List()
    # out_infection_counter = List()
    # out_my_number_of_contacts = List()
    N_positive_tested = List()

    daily_counter = 0

    g_total_sum = 0.0
    g_total_sum_infections = 0.0
    g_cumulative_sum = 0.0

    click = 0
    step_number = 0

    real_time = 0.0
    time_inf = np.zeros(N_tot, np.float32)
    bug_contacts = np.zeros(N_tot, np.int32)

    s_counter = np.zeros(4)

    infectious_states = {4, 5, 6, 7}  # TODO: fix

    # Run the simulation ################################
    continue_run = True
    while continue_run:

        s = 0

        step_number += 1
        g_total_sum = g_total_sum_of_state_changes + g_total_sum_infections

        dt = -np.log(np.random.rand()) / g_total_sum
        real_time += dt

        g_cumulative_sum = 0.0
        ra1 = np.random.rand()

        #######/ Here we move between infected between states
        accept = False
        if g_total_sum_of_state_changes / g_total_sum > ra1:

            s = 1

            x = g_cumulative_sum_of_state_changes / g_total_sum
            state_now = np.searchsorted(x, ra1)
            state_after = state_now + 1

            agent = utils.numba_random_choice_list(agents_in_state[state_now])

            # We have chosen agent to move -> here we move it
            agents_in_state[state_after].append(agent)
            agents_in_state[state_now].remove(agent)

            my_state[agent] += 1

            state_total_counts[state_now] -= 1
            state_total_counts[state_after] += 1

            g_total_sum_of_state_changes -= SIR_transition_rates[state_now]
            g_total_sum_of_state_changes += SIR_transition_rates[state_after]

            g_cumulative_sum_of_state_changes[state_now] -= SIR_transition_rates[state_now]
            g_cumulative_sum_of_state_changes[state_after:] += (
                SIR_transition_rates[state_after] - SIR_transition_rates[state_now]
            )

            g_cumulative_sum_infection_rates[state_now] -= my_sum_of_rates[agent]

            accept = True

            # Moves TO infectious State from non-infectious
            if my_state[agent] == N_infectious_states:
                for contact, rate in zip(my_connections[agent], my_rates[agent]):  # Loop over row agent
                    # if contact is susceptible
                    if my_state[contact] == -1:
                        g_total_sum_infections += rate
                        my_sum_of_rates[agent] += rate
                        g_cumulative_sum_infection_rates[my_state[agent] :] += rate

            # If this moves to Recovered state
            if my_state[agent] == N_states - 1:
                for contact, rate in zip(my_connections[agent], my_rates[agent]):
                    # if contact is susceptible
                    if my_state[contact] == -1:
                        g_total_sum_infections -= rate
                        my_sum_of_rates[agent] -= rate
                        g_cumulative_sum_infection_rates[my_state[agent] :] -= rate

        #######/ Here we infect new states
        else:
            s = 2

            x = (g_total_sum_of_state_changes + g_cumulative_sum_infection_rates) / g_total_sum
            state_now = np.searchsorted(x, ra1)
            g_cumulative_sum = (
                g_total_sum_of_state_changes + g_cumulative_sum_infection_rates[state_now - 1]
            ) / g_total_sum  # important change from [state_now] to [state_now-1]

            agent_getting_infected = -1
            for agent in agents_in_state[state_now]:

                # suggested cumulative sum
                suggested_cumulative_sum = g_cumulative_sum + my_sum_of_rates[agent] / g_total_sum

                if suggested_cumulative_sum > ra1:
                    for rate, contact in zip(my_rates[agent], my_connections[agent]):

                        # if contact is susceptible
                        if my_state[contact] == -1:

                            g_cumulative_sum += rate / g_total_sum

                            # here agent infect contact
                            if g_cumulative_sum > ra1:
                                my_state[contact] = 0
                                agents_in_state[0].append(np.uint32(contact))
                                state_total_counts[0] += 1
                                g_total_sum_of_state_changes += SIR_transition_rates[0]
                                g_cumulative_sum_of_state_changes += SIR_transition_rates[0]
                                accept = True
                                agent_getting_infected = contact
                                break
                else:
                    g_cumulative_sum = suggested_cumulative_sum

                if accept:
                    break

            if agent_getting_infected == -1:
                print("Error here", accept, agent_getting_infected, step_number)
                break

            # Here we update infection lists so that newly infected cannot be infected again

            # loop over contacts of the newly infected agent in order to:
            # 1) remove newly infected agent from contact list (find_myself) by setting rate to 0
            # 2) remove rates from contacts gillespie sums (only if they are in infections state (I))
            for contact_of_agent_getting_infected in my_connections[agent_getting_infected]:

                # loop over indexes of the contact to find_myself and set rate to 0
                for ith_contact_of_agent_getting_infected in range(
                    my_number_of_contacts[contact_of_agent_getting_infected]
                ):

                    find_myself = my_connections[contact_of_agent_getting_infected][
                        ith_contact_of_agent_getting_infected
                    ]

                    # check if the contact found is myself
                    if find_myself == agent_getting_infected:

                        rate = my_rates[contact_of_agent_getting_infected][ith_contact_of_agent_getting_infected]

                        # set rates to myself to 0 (I cannot get infected again)
                        my_rates[contact_of_agent_getting_infected][ith_contact_of_agent_getting_infected] = 0

                        # if the contact can infect, then remove the rates from the overall gillespie accounting
                        if my_state[contact_of_agent_getting_infected] in infectious_states:
                            g_total_sum_infections -= rate
                            my_sum_of_rates[contact_of_agent_getting_infected] -= rate
                            g_cumulative_sum_infection_rates[my_state[contact_of_agent_getting_infected] :] -= rate

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
                out_my_state.append(my_state.copy())

                # tent testing
                # N_daily_tests = 10_000
                N_daily_tests = 0
                f_test_succes = 0.8

                g_total_sum_infections, n_positive_tested = daily_tent_test(
                    N_daily_tests,
                    f_test_succes,
                    N_tot,
                    my_state,
                    N_infectious_states,
                    N_states,
                    my_number_of_contacts,
                    my_connections,
                    my_rates,
                    my_connections_type,
                    infectious_states,
                    g_total_sum_infections,
                    my_sum_of_rates,
                    g_cumulative_sum_infection_rates,
                )
                N_positive_tested.append(n_positive_tested)

            click += 1

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        (continue_run, g_total_sum_of_state_changes, g_total_sum_infections,) = do_bug_check(
            step_number,
            continue_run,
            g_total_sum_infections,
            g_total_sum_of_state_changes,
            verbose,
            state_total_counts,
            N_states,
            N_tot,
            accept,
            g_cumulative_sum_of_state_changes,
            ra1,
            s,
            x,
            g_cumulative_sum_infection_rates,
        )

        s_counter[s] += 1

    if verbose:
        print("Simulation step_number, ", step_number)
        print("s_counter", s_counter)
        print("N_daily_tests", N_daily_tests)
        print("N_positive_tested", N_positive_tested)

    return out_time, out_state_counts, out_my_state
