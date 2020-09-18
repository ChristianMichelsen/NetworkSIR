from matplotlib.pyplot import xticks
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
from numba import (
    njit,
    prange,
    objmode,
    typeof,
)  # conda install -c numba/label/dev numba
from numba.typed import List, Dict

from numba.types import Set
from numba.core.errors import (
    NumbaTypeSafetyWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning,
)


import awkward as awkward0  # conda install awkward0, conda install -c conda-forge pyarrow
import awkward1 as ak  # pip install awkward1

try:
    from src import utils
    from src import simulation_utils
except ImportError:
    import utils
    import simulation_utils

np.set_printoptions(linewidth=200)
do_memory_tracking = False


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


#%%


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

        N_people_in_house_index = simulation_utils.rand_choice_nb(people_in_household)
        N_people_in_house = people_index_to_value[N_people_in_house_index]

        # if N_in_house would increase agent to over N_tot,
        # set N_people_in_house such that it fits and break loop
        if agent + N_people_in_house >= N_tot:
            N_people_in_house = N_tot - agent
            do_continue = False

        for _ in range(N_people_in_house):

            age_index = simulation_utils.rand_choice_nb(
                age_distribution_per_people_in_household[N_people_in_house_index]
            )

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


#%%


@njit
def initialize_tents(coordinates, N_tot, N_tents):

    tent_position = np.zeros((N_tents, 2), np.float32)
    for i in range(N_tents):
        tent_position[i] = coordinates[np.random.randint(N_tot)]

    my_closest_tent = np.zeros(N_tot, np.int16)
    people_per_tent = np.zeros(N_tents, np.int32)
    for agent in range(N_tot):
        closest_tent = -1
        r_min = 10e10
        for i_tent, tent_position in enumerate(tent_position):
            r = utils.haversine_scipy(coordinates[agent], tent_position)
            if r < r_min:
                r_min = r
                closest_tent = i_tent
        my_closest_tent[agent] = closest_tent
        people_per_tent[closest_tent] += 1

    return my_closest_tent, tent_position, people_per_tent


@njit
def initialize_kommuner(N_tot, my_kommune, kommune_names):
    my_label = np.zeros(N_tot, np.int32)
    people_per_kommune = np.zeros(len(kommune_names), np.int32)
    for agent, agent_kommune in enumerate(my_kommune):
        for ith_kommune, kommune_name in enumerate(kommune_names):
            if agent_kommune == kommune_name:
                people_per_kommune[ith_kommune] += 1
                my_label[agent] = ith_kommune
                break
    return my_label, np.arange(len(kommune_names)), people_per_kommune



#%%


@njit
def nb_run_simulation(
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
    H_probability_matrix_csum,
    H_my_state,
    H_agents_in_state,
    H_state_total_counts,
    H_move_matrix_sum,
    H_cumsum_move,
    H_move_matrix_cumsum,
    nts,
    verbose,
    my_connections_type,
    coordinates,
    my_infection_weight,
    my_label,
    label_index,
    people_per_label,
):

    # if do_memory_tracking:
    #     with objmode():
    #         track_memory('Simulation')

    N_daily_tests = 20000  # TODO make Par?
    N_daily_tests = int(N_daily_tests * N_tot / 5_800_000)  # number of random people tested per day
    # day_found_infected is a list of integers with length N_tot. It represent if each agent have been found infected.
    # If an entry is- = 1 the agent has not been found not being infected, else the Integer represent the day the agent was found infected
    # this is the only needed data to save about the infected, and #infected/Tent can be generated from that
    my_day_found_infected = np.ones(N_tot, dtype=np.int32) * -1
    my_reason_for_test = (
        np.ones(N_tot, dtype=np.int32) * -1
    )  # reasons for test: 0: symptoms 1:random_test 2:tracing
    positive_test_reasons = np.zeros(3, dtype=np.int32)  #
    my_clicks_when_test = np.ones(N_tot, dtype=np.int32) * -1
    my_clicks_when_test_result = np.ones(N_tot, dtype=np.int32) * -1
    test_delay_in_clicks = np.array(
        [0, 0, 25], dtype=np.int32
    )  # clicks until test for entry 0: symptoms 1:random_test 2: tracing  10 clicks = 1 day
    results_delay_in_clicks = np.array(
        [5, 10, 5], dtype=np.int32
    )  # clicks from test until results, results works for isolation for entry 0: symptoms 1: random_test 2:tracing 10 clicks = 1 day

    # p_infected_list = List() # each entry is the fraction of people in local area newlyfound infected for each infected person
    chance_of_finding_infected = [
        0.0,
        0.15,
        0.15,
        0.15,
        0.0,
    ]  # When people moves into the ith I state, what is the chance to test them
    days_looking_back = 7  # When looking for local outbreaks, how many days are we looking back
    distance_cut = 10  # defines distance for when another person is local

    # Time and type of intervention against outbreak:
    intervention_type = [
        0
    ]  # 0: Do nothing, 1: lockdown (jobs and schools), 2: Track (infected and their connections), 3: Cover (with face masks)
    intervention_day = 1000  # If -1, this is done by simulated testing (tents!)
    intervention_testbased = False  # Boolean flag if something is going wrong
    is_in_lockdown = False  # Boolean flag if we are in lockdown
    is_wearing_masks = False  # Boolean flag if mask enforcing are in place
    tracking_rate = 0.5  # Fraction of connection we track
    center_of_intervention = (
        coordinates[1, 0],
        coordinates[1, 1],
    )  # We intervene in a circle with this center
    r_intervention = 10000  # and this radius
    # the rate reductions are list of list, first and second entry are rate reductions for the groups [family, job, other]
    # the third entry is the chance of choosing the first set. As such you can have some follow the lockdown and some not or
    # some one group being able to work from home and another isn't
    masking_rate_reduction = [
        [0, 0, 0.0],
        [0, 0, 0.8],
    ]  # rate reduction for the groups [family, job, other]
    lockdown_rate_reduction = [
        [0, 1, 0.6],
        [0, 0.6, 0.6],
    ]  # rate reduction for the groups [family, job, other]
    isolation_rate_reduction = [
        [0.2, 1.0, 1.0],
        [0.0, 0.2, 0.2],
    ]  # rate reduction for the groups [family, job, other]
    tracking_rates = [
        1,
        0.8,
        0,
    ]  # fraction of connections we track for the groups [family, job, other]
    frac_following_mask_rules = 0.8  # fraction donning the masks
    isolate = False  # Do people who test positive isolate from all but family
    lockdown_into_masking = True
    N_labels = len(people_per_label)
    intervention_type_at_tent = np.zeros(
        N_labels, dtype=np.int32
    )  # array to keep count of which intervention is at place at which tent # 0: Do nothing, 1: lockdown (jobs and schools), 2: Track (infected and their connections), 3: Cover (with face masks)
    intervention_started = np.zeros(
        N_labels, dtype=np.int32
    )  # array to keep count of which intervention is at place at which tent # 0: Do nothing, 1: lockdown (jobs and schools), 2: Track (infected and their connections), 3: Cover (with face masks)

    #    5  10 15  20  25  30
    # age_test_bias = [50,50,100,250,250,250,250,250,250,250,250,250,200,200,200,200,250,250]/10000 # number of test per 10000

    out_time = List()
    out_state_counts = List()
    out_my_state = List()

    daily_counter = 0
    day = 0

    g_total_sum = 0.0
    g_total_sum_infections = 0.0
    g_cumulative_sum = 0.0

    click = 0
    step_number = 0

    real_time = 0.0

    H_tot_move = 0

    time_inf = np.zeros(N_tot, np.float32)
    bug_contacts = np.zeros(N_tot, np.int32)

    s_counter = np.zeros(4)

    infectious_states = {4, 5, 6, 7}  # TODO: fix

    # if do_memory_tracking:
    #     with objmode():
    #         track_memory()

    # Run the simulation ################################
    continue_run = True
    while continue_run:
        assert len(my_clicks_when_test_result) == N_tot, "my_clicks_when_test_result"
        assert len(my_day_found_infected) == N_tot, "my_day problem"
        assert len(my_reason_for_test) == N_tot, "my_reason_for_test prob"

        step_number += 1
        g_total_sum = (
            g_total_sum_of_state_changes + g_total_sum_infections
        )  # + H_tot_move XXX Hospitals

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

            # test people with symptoms
            if my_state[agent] >= N_infectious_states:

                assert len(my_clicks_when_test) == N_tot
                assert (my_state[agent] - 4) < len(chance_of_finding_infected)
                if (
                    np.random.rand() < chance_of_finding_infected[my_state[agent] - 4]
                    and my_clicks_when_test[agent] == -1
                ):
                    my_clicks_when_test[agent] = (
                        click + test_delay_in_clicks[0]
                    )  # testing in n_clicks for symptom checking
                    my_reason_for_test[agent] = 0  # set the reason for testing to symptoms

            # Moves TO infectious State from non-infectious
            if my_state[agent] == N_infectious_states:
                for contact, rate in zip(
                    my_connections[agent], my_rates[agent]
                ):  # Loop over row agent
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
                                agents_in_state[0].append(contact)
                                state_total_counts[0] += 1
                                g_total_sum_of_state_changes += SIR_transition_rates[0]
                                g_cumulative_sum_of_state_changes += SIR_transition_rates[0]
                                accept = True
                                # non_infectable_agents[contact] = True
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
                for ith_contact_of_agent_getting_infected, possible_agent in enumerate(
                    my_connections[contact_of_agent_getting_infected]
                ):

                    # check if the contact found is myself
                    if possible_agent == agent_getting_infected:

                        rate = my_rates[contact_of_agent_getting_infected][
                            ith_contact_of_agent_getting_infected
                        ]
                        # set rates to myself to 0 (I cannot get infected again)
                        my_rates[contact_of_agent_getting_infected][
                            ith_contact_of_agent_getting_infected
                        ] = 0

                        # if the contact can infect, then remove the rates from the overall gillespie accounting
                        if my_state[contact_of_agent_getting_infected] in infectious_states:
                            g_total_sum_infections -= rate
                            my_sum_of_rates[contact_of_agent_getting_infected] -= rate
                            g_cumulative_sum_infection_rates[
                                my_state[contact_of_agent_getting_infected] :
                            ] -= rate

                        break

                # else:

                #     continue

        if nts * click < real_time:

            daily_counter += 1
            out_time.append(real_time)
            out_state_counts.append(state_total_counts.copy())

            if daily_counter >= 10:

                daily_counter = 0
                day += 1
                out_my_state.append(my_state.copy())

                random_people_for_test = np.random.choice(N_tot, N_daily_tests)
                my_clicks_when_test[random_people_for_test] = (
                    click + test_delay_in_clicks[1]
                )  # choose N_daily_test people at random to test
                my_reason_for_test[
                    random_people_for_test
                ] = 1  # count that random test is the reason for test
                intervention_type_at_tent = test_if_label_needs_intervention(
                    my_state,
                    my_day_found_infected,
                    my_label,
                    label_index,
                    people_per_label,
                    N_tot,
                    day,
                    days_looking_back,
                    intervention_type_at_tent,
                )
                (
                    intervention_type_at_tent,
                    intervention_started,
                    my_rates,
                    g_total_sum_infections,
                    my_sum_of_rates,
                    g_cumulative_sum_infection_rates,
                ) = test_if_intervention_on_labels_can_be_removed(
                    my_state,
                    my_day_found_infected,
                    intervention_testbased,
                    day,
                    people_per_label,
                    days_looking_back,
                    my_label,
                    N_tot,
                    infectious_states,
                    my_connections,
                    my_connections_type,
                    my_rates,
                    my_infection_weight,
                    g_total_sum_infections,
                    my_sum_of_rates,
                    g_cumulative_sum_infection_rates,
                    intervention_type_at_tent,
                    intervention_started,
                )
                for ith_tent, intervention in enumerate(intervention_type_at_tent):
                    if intervention == 1 and intervention_started[ith_tent] == 0:
                        intervention_started[ith_tent] = 1
                        rate_reduction = lockdown_rate_reduction
                        (
                            my_rates,
                            g_total_sum_infections,
                            my_sum_of_rates,
                            g_cumulative_sum_infection_rates,
                        ) = lockdown_on_label(
                            N_tot,
                            my_label,
                            my_state,
                            my_rates,
                            my_connections,
                            my_connections_type,
                            rate_reduction,
                            ith_tent,
                            infectious_states,
                            g_total_sum_infections,
                            my_sum_of_rates,
                            g_cumulative_sum_infection_rates,
                        )

                # if do_memory_tracking:
                #     with objmode():
                #         track_memory()
            for agent in range(N_tot):  # test everybody whose counter say we should test
                # testing everybody who should be tested
                if my_clicks_when_test[agent] == click:
                    (
                        my_reason_for_test,
                        my_clicks_when_test,
                        my_clicks_when_test_result,
                        my_rates,
                        g_total_sum_infections,
                        my_sum_of_rates,
                        g_cumulative_sum_infection_rates,
                    ) = test_a_person(
                        agent,
                        my_day_found_infected,
                        test_delay_in_clicks,
                        results_delay_in_clicks,
                        intervention_type,
                        tracking_rates,
                        positive_test_reasons,
                        my_reason_for_test,
                        my_state,
                        my_rates,
                        my_connections,
                        my_connections_type,
                        my_clicks_when_test,
                        my_clicks_when_test_result,
                        infectious_states,
                        g_total_sum_infections,
                        my_sum_of_rates,
                        g_cumulative_sum_infection_rates,
                        click,
                        day,
                    )
                # getting results for people
                if my_clicks_when_test_result[agent] == click:
                    my_clicks_when_test_result[agent] = -1
                    my_day_found_infected[agent] = day
                    if isolate:
                        (
                            my_rates,
                            g_total_sum_infections,
                            my_sum_of_rates,
                            g_cumulative_sum_infection_rates,
                        ) = cut_rates_of_agent(
                            agent,
                            my_state,
                            my_rates,
                            my_connections,
                            my_connections_type,
                            isolation_rate_reduction[0],
                            infectious_states,
                            g_total_sum_infections,
                            my_sum_of_rates,
                            g_cumulative_sum_infection_rates,
                        )
            click += 1

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        (continue_run, g_total_sum_of_state_changes, g_total_sum_infections,) = nb_do_bug_check(
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
            day,
        )
        s_counter[s] += 1
    print(
        "cluster_coefficient",
        np.mean(compute_my_cluster_coefficient(N_tot, my_connections, my_day_found_infected)[0]),
    )
    n_infected = np.sum(my_state > np.zeros(N_tot))
    if verbose:
        print("Simulation step_number, ", step_number)
        print("s_counter", s_counter)
        print("N_daily_tests", N_daily_tests)
    print("sim ended at day:", day)
    print("the fraction of the pop infected was:", n_infected / N_tot)
    print(
        "the fraction of infected found was:",
        len(my_day_found_infected[my_day_found_infected >= 0]) / n_infected,
    )
    print("Test reasons", positive_test_reasons)
    return out_time, out_state_counts, out_my_state  # , out_H_state_total_counts


@njit
def test_if_label_needs_intervention(
    my_state,
    my_day_found_infected,
    my_label,
    label_index,
    people_per_label,
    N_tot,
    day,
    days_looking_back,
    intervention_type_at_tent,
    intervention_type_to_init=1,
    threshold=0.004,
):  # threshold is the fraction that need to be positive.
    infected_per_tent = np.zeros(len(people_per_label), dtype=np.int32)
    for agent, day_found in enumerate(my_day_found_infected):
        if day_found > max(0, day - days_looking_back):
            infected_per_tent[my_label[agent]] += 1
    for infected, inhabitants, intervention_type, ith_tent in zip(
        infected_per_tent,
        people_per_label,
        intervention_type_at_tent,
        range(len(intervention_type_at_tent)),
    ):
        if (1.0 * infected / inhabitants) > threshold and intervention_type == 0:
            print(
                "lockdown at tent",
                ith_tent,
                "at day",
                day,
                "the num of infected is",
                infected,
                "/",
                inhabitants,
            )

            intervention_type_at_tent[ith_tent] = intervention_type_to_init
    return intervention_type_at_tent


@njit
def test_if_intervention_on_labels_can_be_removed(
    my_state,
    my_day_found_infected,
    intervention_testbased,
    day,
    people_per_label,
    days_looking_back,
    my_label,
    N_tot,
    infectious_states,
    my_connections,
    my_connections_type,
    my_rates,
    my_infection_weight,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
    intervention_type_at_tent,
    intervention_started,
    threshold=0.001,
):
    infected_per_tent = np.zeros(len(people_per_label), dtype=np.int32)
    for agent, day_found in enumerate(my_day_found_infected):
        if day_found > day - days_looking_back:
            infected_per_tent[my_label[agent]] += 1

    for infected, inhabitants, intervention_type, ith_tent in zip(
        infected_per_tent,
        people_per_label,
        intervention_type_at_tent,
        range(len(intervention_type_at_tent)),
    ):
        if (1.0 * infected / inhabitants) < threshold and intervention_type != 0:
            (
                my_rates,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            ) = remove_intervention_at_label(
                N_tot,
                my_state,
                my_label,
                infectious_states,
                my_connections,
                my_connections_type,
                my_rates,
                my_infection_weight,
                ith_tent,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            )
            intervention_type_at_tent[ith_tent] = 0
            intervention_started[ith_tent] = 0
            print(
                "remove lockdown at num of infected",
                ith_tent,
                "at day",
                day,
                "the num of infected is",
                infected,
                "/",
                inhabitants,
            )

    return (
        intervention_type_at_tent,
        intervention_started,
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def test_a_person(
    agent,
    my_day_found_infected,
    test_delay_in_clicks,
    results_delay_in_clicks,
    intervention_type,
    tracking_rates,
    positive_test_reasons,
    my_reason_for_test,
    my_state,
    my_rates,
    my_connections,
    my_connections_type,
    my_clicks_when_test,
    my_clicks_when_test_result,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
    click,
    day,
):
    if my_state[agent] in infectious_states and my_day_found_infected[agent] == -1:
        my_clicks_when_test_result[agent] = (
            click + results_delay_in_clicks[my_reason_for_test[agent]]
        )
        assert my_reason_for_test[agent] in (0, 1, 2)
        positive_test_reasons[my_reason_for_test[agent]] += 1  # count reason found infected

        if 2 in intervention_type:  # check if tracking is on
            for ith_contact, contact in enumerate(my_connections[agent]):  # loop over contacts
                if (
                    np.random.rand() < tracking_rates[my_connections_type[agent][ith_contact]]
                    and my_clicks_when_test[contact] == -1
                ):
                    my_reason_for_test[contact] = 2
                    my_clicks_when_test[contact] = click + test_delay_in_clicks[2]
    my_clicks_when_test[agent] = -1
    my_reason_for_test[agent] = -1
    return (
        my_reason_for_test,
        my_clicks_when_test,
        my_clicks_when_test_result,
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def cut_rates_of_agent(
    agent,
    my_state,
    my_rates,
    my_connections,
    my_connections_type,
    rate_reduction,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):
    # rate reduction is a 3 vector. All rates for agent is reduced by [home, job, other ]. Is used eq. for isolation
    agent_infected = my_state[agent] in infectious_states
    agent_infectable = my_state[agent] == -1
    agent_update_rate = 0.0
    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my_connections[agent]):
        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        rate = (
            my_rates[agent][ith_contact] * rate_reduction[my_connections_type[agent][ith_contact]]
        )
        my_rates[agent][ith_contact] -= rate

        # updates to gillespie sums, if agent is infected and contact is susceptible
        if agent_infected and my_state[contact] == -1:
            agent_update_rate += rate

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact, possible_agent in enumerate(my_connections[contact]):

            # check if the contact found is myself
            if agent == possible_agent:

                # update rates from contact to agent. Rate_reduction makes it depending on connection type
                c_rate = (
                    my_rates[contact][ith_contact_of_contact]
                    * rate_reduction[my_connections_type[contact][ith_contact_of_contact]]
                )
                my_rates[contact][ith_contact_of_contact] -= c_rate

                # updates to gillespie sums, if contact is infected and agent is susceptible
                if my_state[contact] in infectious_states and agent_infectable:
                    g_total_sum_infections -= c_rate
                    my_sum_of_rates[contact] -= c_rate
                    g_cumulative_sum_infection_rates[my_state[contact] :] -= c_rate
                break

    # actually updates to gillespie sums
    g_total_sum_infections -= agent_update_rate
    my_sum_of_rates[agent] -= agent_update_rate
    g_cumulative_sum_infection_rates[my_state[agent] :] -= agent_update_rate

    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def remove_and_reduce_rates_of_agent(
    agent,
    my_state,
    my_rates,
    my_connections,
    my_connections_type,
    rate_reduction,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):
    # rate reduction is a 2 3-vectors. is used in lockdown intervention
    agent_infected = my_state[agent] in infectious_states
    agent_infectable = my_state[agent] == -1
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my_connections[agent]):
        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        act_rate_reduction = reduce_rates
        if np.random.rand() < remove_rates[my_connections_type[agent][ith_contact]]:
            act_rate_reduction = [1.0, 1.0, 1.0]

        rate = (
            my_rates[agent][ith_contact]
            * act_rate_reduction[my_connections_type[agent][ith_contact]]
        )
        my_rates[agent][ith_contact] -= rate

        # updates to gillespie sums, if agent is infected and contact is susceptible
        if agent_infected and my_state[contact] == -1:
            agent_update_rate += rate

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact, possible_agent in enumerate(my_connections[contact]):

            # check if the contact found is myself
            if agent == possible_agent:

                # update rates from contact to agent. Rate_reduction makes it depending on connection type
                c_rate = (
                    my_rates[contact][ith_contact_of_contact]
                    * act_rate_reduction[my_connections_type[contact][ith_contact_of_contact]]
                )
                my_rates[contact][ith_contact_of_contact] -= c_rate

                # updates to gillespie sums, if contact is infected and agent is susceptible
                if my_state[contact] in infectious_states and agent_infectable:
                    g_total_sum_infections -= c_rate
                    my_sum_of_rates[contact] -= c_rate
                    g_cumulative_sum_infection_rates[my_state[contact] :] -= c_rate
                break

    # actually updates to gillespie sums
    g_total_sum_infections -= agent_update_rate
    my_sum_of_rates[agent] -= agent_update_rate
    g_cumulative_sum_infection_rates[my_state[agent] :] -= agent_update_rate

    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def reduce_frac_rates_of_agent(
    agent,
    my_state,
    my_rates,
    my_connections,
    my_connections_type,
    rate_reduction,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):
    # rate reduction is 2 3-vectors. is used for masking interventions
    agent_infected = my_state[agent] in infectious_states
    agent_infectable = my_state[agent] == -1
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]
    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my_connections[agent]):
        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        if np.random.rand() < remove_rates[my_connections_type[agent][ith_contact]]:
            act_rate_reduction = [0, 0, 0]
        else:
            act_rate_reduction = reduce_rates
        rate = (
            my_rates[agent][ith_contact]
            * act_rate_reduction[my_connections_type[agent][ith_contact]]
        )
        my_rates[agent][ith_contact] -= rate

        # updates to gillespie sums, if agent is infected and contact is susceptible
        if agent_infected and my_state[contact] == -1:
            agent_update_rate += rate

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact, possible_agent in enumerate(my_connections[contact]):

            # check if the contact found is myself
            if agent == possible_agent:

                # update rates from contact to agent. Rate_reduction makes it depending on connection type
                c_rate = (
                    my_rates[contact][ith_contact_of_contact]
                    * rate_reduction[my_connections_type[contact][ith_contact_of_contact]]
                )
                my_rates[contact][ith_contact_of_contact] -= c_rate

                # updates to gillespie sums, if contact is infected and agent is susceptible
                if my_state[contact] in infectious_states and agent_infectable:
                    g_total_sum_infections -= c_rate
                    my_sum_of_rates[contact] -= c_rate
                    g_cumulative_sum_infection_rates[my_state[contact] :] -= c_rate
                break

    # actually updates to gillespie sums
    g_total_sum_infections -= agent_update_rate
    my_sum_of_rates[agent] -= agent_update_rate
    g_cumulative_sum_infection_rates[my_state[agent] :] -= agent_update_rate

    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def reset_rates_of_agent(
    agent,
    my_state,
    infectious_states,
    my_connections,
    my_connections_type,
    my_rates,
    my_infection_weight,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
    connection_type_weight=np.ones(
        3, dtype=np.float32
    ),  # [home, job, other] reset infection rate to ori times this number
):
    agent_infected = my_state[agent] in infectious_states
    agent_infectable = my_state[agent] == -1
    agent_update_rate = 0.0
    for ith_contact, contact in enumerate(my_connections[agent]):
        infection_rate = (
            my_infection_weight[agent]
            * connection_type_weight[my_connections_type[agent][ith_contact]]
        )
        rate = infection_rate - my_rates[agent][ith_contact]
        my_rates[agent][ith_contact] = infection_rate

        if agent_infected and my_state[contact] == -1:
            agent_update_rate += rate

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact, possible_agent in enumerate(my_connections[contact]):

            # check if the contact found is myself
            if agent == possible_agent:

                # update rates from contact to agent.
                c_rate = my_infection_weight[contact] - my_rates[contact][ith_contact_of_contact]
                my_rates[contact][ith_contact_of_contact] = my_infection_weight[contact]

                # updates to gillespie sums, if contact is infected and agent is susceptible
                if my_state[contact] in infectious_states and agent_infectable:
                    g_total_sum_infections += c_rate
                    my_sum_of_rates[contact] += c_rate
                    g_cumulative_sum_infection_rates[my_state[contact] :] += c_rate
                break

    # actually updates to gillespie sums
    g_total_sum_infections += agent_update_rate
    my_sum_of_rates[agent] += agent_update_rate
    g_cumulative_sum_infection_rates[my_state[agent] :] += agent_update_rate

    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def lockdown_on_label(
    N_tot,
    my_label,
    my_state,
    my_rates,
    my_connections,
    my_connections_type,
    rate_reduction,
    label,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):
    # lockdown on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to set to 0.
    # second is the fraction of reduction of the remaining [home, job, others] rates.
    # ie: [[0,0.8,0.8],[0,0.8,0.8]] means that 80% of your contacts on job and other is set to 0, and the remaining 20% is reduced by 80%.
    # loop over all agents
    for agent in range(N_tot):
        # calculate
        if my_label[agent] == label:
            (
                my_rates,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            ) = remove_and_reduce_rates_of_agent(
                agent,
                my_state,
                my_rates,
                my_connections,
                my_connections_type,
                rate_reduction,
                infectious_states,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            )
    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def masking_on_label(
    N_tot,
    my_label,
    my_state,
    my_rates,
    my_connections,
    my_connections_type,
    rate_reduction,
    label,
    infectious_states,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):

    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie: [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(N_tot):
        # calculate
        if my_label[agent] == label:
            (
                my_rates,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            ) = reduce_frac_rates_of_agent(
                agent,
                my_state,
                my_rates,
                my_connections,
                my_connections_type,
                rate_reduction,
                infectious_states,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            )
    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def remove_intervention_at_label(
    N_tot,
    my_state,
    my_label,
    infectious_states,
    my_connections,
    my_connections_type,
    my_rates,
    my_infection_weight,
    ith_tent,
    g_total_sum_infections,
    my_sum_of_rates,
    g_cumulative_sum_infection_rates,
):
    for agent in range(N_tot):
        if my_label[agent] == ith_tent:
            (
                my_rates,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            ) = reset_rates_of_agent(
                agent,
                my_state,
                infectious_states,
                my_connections,
                my_connections_type,
                my_rates,
                my_infection_weight,
                g_total_sum_infections,
                my_sum_of_rates,
                g_cumulative_sum_infection_rates,
            )
    return (
        my_rates,
        g_total_sum_infections,
        my_sum_of_rates,
        g_cumulative_sum_infection_rates,
    )


@njit
def nb_do_bug_check(
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
    day,
):
    if day > 1200:
        continue_run = False

    if step_number > 100_000_000:
        print("step_number > 100_000_000")
        continue_run = False

    if (g_total_sum_infections + g_total_sum_of_state_changes < 0.0001) and (
        g_total_sum_of_state_changes + g_total_sum_infections > -0.00001
    ):
        continue_run = False
        print("Equilibrium")

    if state_total_counts[N_states - 1] > N_tot - 10:
        if True:
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
        print("total g sum problem")
        g_total_sum_of_state_changes = 0

    if (g_total_sum_infections < 0) and (g_total_sum_infections > -0.001):
        print("total total sum infection problem")
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


#%%


class Simulation:
    def __init__(self, filename, verbose=False, do_track_memory=True):

        self.verbose = verbose

        self._Filename = Filename = simulation_utils.Filename(filename)

        self.cfg = Filename.simulation_parameters
        self.ID = Filename.ID

        self.filenames = {}
        self.filename = self.filenames["filename"] = Filename.filename
        self.filenames["network_initialisation"] = Filename.filename_network_initialisation
        self.filenames["network_network"] = Filename.filename_network

        utils.set_numba_random_seed(self.ID)

        self._prepare_memory_tracking(do_track_memory)

    def _prepare_memory_tracking(self, do_track_memory=True):
        self.filenames["memory"] = memory_file = self._Filename.memory_filename
        self.do_track_memory = do_track_memory  # if self.ID == 0 else False
        self.time_start = Time.time()

        search_string = "Saving network initialization"

        if utils.file_exists(memory_file) and simulation_utils.does_file_contains_string(
            memory_file, search_string
        ):
            self.time_start -= simulation_utils.get_search_string_time(memory_file, search_string)
            self.track_memory("Appending to previous network initialization")
        else:
            utils.make_sure_folder_exist(self.filenames["memory"], delete_file_if_exists=True)

        global track_memory
        track_memory = self.track_memory

    @property
    def current_memory_usage(self):
        "Returns current memory usage of entire process in GiB"
        process = psutil.Process()
        return process.memory_info().rss / 2 ** 30

    def track_memory(self, s=None):
        if self.do_track_memory:
            with open(self.filenames["memory"], "a") as file:
                if s:
                    print("#" + s, file=file)
                time = Time.time() - self.time_start
                print(time, self.current_memory_usage, file=file, sep="\t")  # GiB

    def _initialize_network(self):

        cfg = self.cfg
        self.track_memory("Loading Coordinates")

        self.coordinates, self.coordinate_indices = simulation_utils.load_coordinates(
            self._Filename.coordinates_filename, cfg.N_tot, self.ID
        )

        if self.verbose:
            print("INITIALIZE NETWORK")
        self.track_memory("Initialize Network")

        rho_scale = 1000  # scale factor of rho

        (
            people_in_household,
            age_distribution_per_people_in_household,
        ) = simulation_utils.load_household_data(self._Filename.household_data_filenames)
        (
            N_dim_people_in_household,
            N_ages,
        ) = age_distribution_per_people_in_household.shape

        if self.verbose:
            print("Families")
        (
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
        ) = place_and_connect_families(
            cfg.N_tot,
            people_in_household,
            age_distribution_per_people_in_household,
            self.coordinates,
            cfg.sigma_mu,
            cfg.sigma_beta,
            cfg.beta,
        )

        if self.verbose:
            print("Using uniform work and other matrices")
        # matrix_work = np.random.random((N_ages, N_ages))
        matrix_work = np.ones((N_ages, N_ages))
        matrix_work = matrix_work * counter_ages * counter_ages.reshape((-1, 1))
        matrix_work = matrix_work / matrix_work.sum()

        # matrix_other = np.random.random((N_ages, N_ages))
        matrix_other = np.ones((N_ages, N_ages))
        matrix_other = matrix_other * counter_ages * counter_ages.reshape((-1, 1))
        matrix_other = matrix_other / matrix_other.sum()

        work_other_ratio = 0.3  # 20% work, 80% other

        if self.verbose:
            print("Connecting work and others, currently slow, please wait")
        # below one of three algorithms can be chosen by calling: connect_work_and_others, connect_work_and_others_clusters_reroll, or connect_work_and_others_cluster_prop
        # they take the same inputs.
        connect_work_and_others_clusters_reroll(
            cfg.N_tot,
            N_ages,
            mu_counter,
            cfg.mu,
            work_other_ratio,
            matrix_work,
            matrix_other,
            cfg.rho,
            rho_scale,
            cfg.epsilon_rho,
            self.coordinates,
            agents_in_age_group,
            my_connections,
            my_connections_type,
            my_number_of_contacts,
        )

        self.counter_ages = counter_ages
        self.agents_in_age_group = agents_in_age_group
        self.my_number_of_contacts = my_number_of_contacts
        self.my_infection_weight = my_infection_weight

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # Find closests test tents  # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if self.verbose:
            print("CONNECT TENTS")
        self.track_memory("Connect tents")

        tents_as_labels = False  # True makes N_labels random tents/positions to use as labels. If False kommuner is used as labels. label_lab
        if tents_as_labels:
            my_label, label_index, people_per_label = initialize_tents(
                self.coordinates, self.cfg.N_tot, N_labels=20
            )
        else:
            GPS_filename = "Data/GPS_coordinates.feather"
            df_coordinates = (
                pd.read_feather(GPS_filename).iloc[self.coordinate_indices].reset_index(drop=True)
            )
            my_kommune = df_coordinates["kommune"].tolist()
            kommune_names = list(set(my_kommune))
            print(kommune_names)
            my_label, label_index, people_per_label = initialize_kommuner(
                cfg.N_tot, my_kommune, kommune_names
            )

        return (
            my_connections,
            my_connections_type,
            my_number_of_contacts,
            my_infection_weight,
            my_age,
            agents_in_age_group,
            my_label,
            label_index,
            people_per_label,
        )

    def _save_network_initalization(
        self,
        my_age,
        agents_in_age_group,
        my_number_of_contacts,
        my_infection_weight,
        my_label,
        label_index,
        people_per_label,
        my_connections,
        my_connections_type,
        time_elapsed,
    ):
        self.track_memory("Saving network initialization")
        utils.make_sure_folder_exist(self.filenames["network_initialisation"])
        with h5py.File(self.filenames["network_initialisation"], "w") as f:  #
            f.create_dataset(
                "cfg_str", data=str(self.cfg)
            )  # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("my_age", data=my_age)
            f.create_dataset("my_number_of_contacts", data=my_number_of_contacts)
            f.create_dataset("my_infection_weight", data=my_infection_weight)
            f.create_dataset("my_label", data=my_label)
            f.create_dataset("people_per_label", data=people_per_label)
            f.create_dataset("label_index", data=label_index)
            awkward0.hdf5(f)["my_connections"] = ak.to_awkward0(my_connections)
            awkward0.hdf5(f)["my_connections_type"] = ak.to_awkward0(my_connections_type)
            awkward0.hdf5(f)["agents_in_age_group"] = ak.to_awkward0(agents_in_age_group)
            for key, val in self.cfg.items():
                f.attrs[key] = val
            f.create_dataset("time_elapsed", data=time_elapsed)

    def _load_network_initalization(self):
        self.track_memory("Loading network initialization")
        with h5py.File(self.filenames["network_initialisation"], "r") as f:
            my_age = f["my_age"][()]
            my_number_of_contacts = f["my_number_of_contacts"][()]
            my_infection_weight = f["my_infection_weight"][()]
            my_label = f["my_label"][()]
            people_per_label = f["people_per_label"][()]
            label_index = f["label_index"][()]
            my_connections = awkward0.hdf5(f)["my_connections"]
            my_connections_type = awkward0.hdf5(f)["my_connections_type"]
            agents_in_age_group = awkward0.hdf5(f)["agents_in_age_group"]
        self.track_memory("Loading Coordinates")
        self.coordinates, self.coordinate_indices = simulation_utils.load_coordinates(
            self._Filename.coordinates_filename, self.cfg.N_tot, self.ID
        )
        return (
            my_age,
            ak.from_awkward0(agents_in_age_group),
            my_number_of_contacts,
            my_infection_weight,
            my_label,
            label_index,
            people_per_label,
            ak.from_awkward0(my_connections),
            ak.from_awkward0(my_connections_type),
        )

    def initialize_network(self, force_rerun=False, save_initial_network=True):
        utils.set_numba_random_seed(self.ID)

        OSError_flag = False

        # try to load file (except if forced to rerun)
        if not force_rerun:
            try:
                (
                    my_age,
                    agents_in_age_group,
                    my_number_of_contacts,
                    my_infection_weight,
                    my_label,
                    label_index,
                    people_per_label,
                    my_connections,
                    my_connections_type,
                ) = self._load_network_initalization()
                if self.verbose:
                    print(f"{self.filenames['network_initialisation']} exists")
            except OSError:
                if self.verbose:
                    print(f"{self.filenames['network_initialisation']} had OSError, creating it")
                OSError_flag = True

        # if ran into OSError above or forced to rerun:
        if OSError_flag or force_rerun:

            if self.verbose and not OSError_flag:
                print(f"{self.filenames['network_initialisation']} does not exist, creating it")

            with Timer() as t:
                (
                    my_connections,
                    my_connections_type,
                    my_number_of_contacts,
                    my_infection_weight,
                    my_age,
                    agents_in_age_group,
                    my_label,
                    label_index,
                    people_per_label,
                ) = self._initialize_network()
            my_connections = utils.nested_list_to_awkward_array(my_connections)
            my_connections_type = utils.nested_list_to_awkward_array(my_connections_type)

            if save_initial_network:
                try:
                    self._save_network_initalization(
                        my_age=my_age,
                        agents_in_age_group=agents_in_age_group,
                        my_number_of_contacts=ak.to_awkward0(my_number_of_contacts),
                        my_infection_weight=my_infection_weight,
                        my_label=my_label,
                        label_index=label_index,
                        people_per_label=people_per_label,
                        my_connections=ak.to_awkward0(my_connections),
                        my_connections_type=ak.to_awkward0(my_connections_type),
                        time_elapsed=t.elapsed,
                    )
                except OSError as e:
                    print(
                        f"\nSkipped saving network initialization for {self.filenames['network_initialisation']}"
                    )
                    # print(e)

        self.my_age = my_age
        self.agents_in_age_group = agents_in_age_group
        self.N_ages = len(self.agents_in_age_group)
        self.my_connections = utils.MutableArray(my_connections)
        self.my_connections_type = utils.MutableArray(my_connections_type)
        self.my_number_of_contacts = my_number_of_contacts
        self.my_infection_weight = my_infection_weight
        self.my_label = my_label
        self.label_index = label_index
        self.people_per_label = people_per_label

    def make_initial_infections(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("INITIAL INFECTIONS")
        self.track_memory("Numba Compilation")

        cfg = self.cfg

        np.random.seed(self.ID)

        self.nts = 0.1  # Time step (0.1 - ten times a day)
        self.N_states = 9  # number of states
        self.N_infectious_states = 4  # This means the 5'th state
        self.initial_ages_exposed = np.arange(self.N_ages)  # means that all ages are exposed
        make_random_infections = True

        self.my_rates = simulation_utils.initialize_my_rates(
            self.my_infection_weight, self.my_number_of_contacts
        )

        self.my_state = np.full(cfg.N_tot, -1, dtype=np.int8)
        self.state_total_counts = np.zeros(self.N_states, dtype=np.uint32)
        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.g_cumulative_sum_of_state_changes = np.zeros(self.N_states, dtype=np.float64)
        self.g_cumulative_sum_infection_rates = np.zeros(self.N_states, dtype=np.float64)
        self.my_sum_of_rates = np.zeros(cfg.N_tot, dtype=np.float64)

        self.SIR_transition_rates = simulation_utils.initialize_SIR_transition_rates(
            self.N_states, self.N_infectious_states, cfg
        )

        self.g_total_sum_of_state_changes = nb_make_initial_infections(
            cfg.N_init,
            self.my_state,
            self.state_total_counts,
            self.agents_in_state,
            self.g_cumulative_sum_of_state_changes,
            self.my_connections.array,
            self.my_number_of_contacts,
            self.my_rates.array,
            self.SIR_transition_rates,
            self.agents_in_age_group,
            self.initial_ages_exposed,
            self.N_infectious_states,
            self.coordinates,
            make_random_infections,
        )

    def run_simulation(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("RUN SIMULATION")
        self.track_memory("Numba Compilation")

        cfg = self.cfg

        self.individual_infection_counter = np.zeros(cfg.N_tot, dtype=np.uint16)

        H = simulation_utils.get_hospitalization_variables(cfg.N_tot)
        (
            H_probability_matrix_csum,
            H_my_state,
            H_agents_in_state,
            H_state_total_counts,
            H_move_matrix_sum,
            H_cumsum_move,
            H_move_matrix_cumsum,
        ) = H

        res = nb_run_simulation(
            cfg.N_tot,
            self.g_total_sum_of_state_changes,
            self.g_cumulative_sum_of_state_changes,
            self.state_total_counts,
            self.agents_in_state,
            self.my_state,
            self.g_cumulative_sum_infection_rates,
            self.N_states,
            self.my_sum_of_rates,
            self.SIR_transition_rates,
            self.N_infectious_states,
            self.my_number_of_contacts,
            self.my_rates.array,
            self.my_connections.array,
            self.my_age,
            self.individual_infection_counter,
            H_probability_matrix_csum,
            H_my_state,
            H_agents_in_state,
            H_state_total_counts,
            H_move_matrix_sum,
            H_cumsum_move,
            H_move_matrix_cumsum,
            self.nts,
            self.verbose,
            self.my_connections_type.array,
            self.coordinates,
            self.my_infection_weight,
            self.my_label,
            self.label_index,
            self.people_per_label,
        )

        out_time, out_state_counts, out_my_state = res

        track_memory("Arrays Conversion")

        self.time = np.array(out_time)
        self.state_counts = np.array(out_state_counts)
        self.my_state = np.array(out_my_state)
        # self.H_state_total_counts = np.array(out_H_state_total_counts)

    def make_dataframe(self):

        self.track_memory("Make DataFrame")
        self.df = df = simulation_utils.state_counts_to_df(self.time, self.state_counts)

        self.track_memory("Save CSV")
        utils.make_sure_folder_exist(self.filename)
        # save csv file
        df.to_csv(self.filename, index=False)
        return df

    def save_simulation_results(self, save_only_ID_0=False, time_elapsed=None):

        if save_only_ID_0 and self.ID != 0:
            return None

        utils.make_sure_folder_exist(self.filenames["network_network"], delete_file_if_exists=True)

        self.track_memory("Saving HDF5 File")
        with h5py.File(self.filenames["network_network"], "w") as f:  #
            f.create_dataset("coordinates", data=self.coordinates)
            f.create_dataset("coordinate_indices", data=self.coordinate_indices)
            f.create_dataset("my_state", data=self.my_state)
            f.create_dataset("my_number_of_contacts", data=self.my_number_of_contacts)
            f.create_dataset("my_age", data=self.my_age)
            f.create_dataset(
                "cfg_str", data=str(self.cfg)
            )  # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))

            if time_elapsed:
                f.create_dataset("time_elapsed", data=time_elapsed)

            if self.do_track_memory:
                memory_file = self.filenames["memory"]
                (
                    self.df_time_memory,
                    self.df_change_points,
                ) = simulation_utils.parse_memory_file(memory_file)
                df_time_memory_hdf5 = utils.dataframe_to_hdf5_format(
                    self.df_time_memory, cols_to_str=["ChangePoint"]
                )
                df_change_points_hdf5 = utils.dataframe_to_hdf5_format(
                    self.df_change_points, include_index=True
                )

                f.create_dataset("memory_file", data=Path(memory_file).read_text())
                f.create_dataset("df_time_memory", data=df_time_memory_hdf5)
                f.create_dataset("df_change_points", data=df_change_points_hdf5)

            # if do_include_awkward:
            #     if do_track_memory:
            #         track_memory('Awkward')
            #     g = awkward0.hdf5(f)
            #     g["my_connections"] = awkward0.fromiter(out_my_connections).astype(np.int32)
            #     g["my_rates"] = awkward0.fromiter(out_my_rates)

            for key, val in self.cfg.items():
                f.attrs[key] = val

        # self.track_memory('Finished')
        # if verbose:
        #     print(f"Run took in total: {t.elapsed:.1f}s.")

        if self.verbose:
            print("\n\n")
            print("coordinates", utils.get_size(self.coordinates))
            print("my_state", utils.get_size(self.my_state))
            print("my_number_of_contacts", utils.get_size(self.my_number_of_contacts))
            print("my_age", utils.get_size(self.my_age))

    def save_memory_figure(self, savefig=True):
        if self.do_track_memory:
            fig, ax = simulation_utils.plot_memory_comsumption(
                self.df_time_memory,
                self.df_change_points,
                min_TimeDiffRel=0.1,
                min_MemoryDiffRel=0.1,
                time_unit="s",
            )
            if savefig:
                fig.savefig(self.filenames["memory"].replace(".txt", ".pdf"))


#%%


def run_full_simulation(filename, verbose=False, force_rerun=False, only_initialize_network=False):

    with Timer() as t, warnings.catch_warnings():
        if not verbose:
            warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
            warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
            warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

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
filename = "Data/ABN/N_tot__58000__N_init__100__N_ages__10__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__rho__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__beta_scaling__1.0__age_mixing__1.0__algo__2/N_tot__58000__N_init__100__N_ages__1__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__rho__0.0__lambda_E__1.0__lambda_I__1.0__epsilon_rho__0.01__beta_scaling__1.0__age_mixing__1.0__algo__2__ID__000.csv"
# filename = filename.replace('ID__000', 'ID__001')
# filename = filename.replace('N_tot__58000', 'N_tot__10000')
filename = filename.replace("N_tot__58000", "N_tot__100000")


# if running just til file
if Path("").cwd().stem == "src":

    with Timer() as t:
        simulation = Simulation(filename, verbose)
        simulation.initialize_network(force_rerun=force_rerun)
        simulation.make_initial_infections()
        simulation.run_simulation()
        df = simulation.make_dataframe()
        display(df)
        # simulation.save_simulation_results(time_elapsed=t.elapsed)
        # simulation.save_memory_figure()

    # if False:
    #     my_number_of_contacts = simulation.my_number_of_contacts
    #     # counter_ages = simulation.counter_ages
    #     agents_in_age_group = simulation.agents_in_age_group
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     for i, agents in enumerate(agents_in_age_group):
    #         ax.hist(my_number_of_contacts[agents], 50, label=i, histtype='step', lw=2)
    #     ax.legend()

    # if False:

    #     import pyarrow as pa
    #     import pyarrow.parquet as pq

    #     x = List()
    #     x.append(np.arange(2))
    #     x.append(np.arange(3))
    #     y = ak.from_iter(x)
    #     ak.to_parquet(x, "x.parquet")
    #     ak.to_parquet(y, "y.parquet")

    #     arrow_x = ak.to_arrow(x)
    #     arrow_y = ak.to_arrow(y)

    #     table_x = pq.read_table('x.parquet')
    #     table_y = pq.read_table('y.parquet')

    #     table_x.to_pandas()
    #     table_y.to_pandas()

    #     table_x.to_pydict()

    # pa.Table.from_pydict({'x': arrow_x, 'y': })