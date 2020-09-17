import numpy as np
from numba import njit
from numba.typed import List
from pathlib import Path
import os

# path = Path("").cwd()
# if path.stem == "src":
#     os.chdir(path.parent)

# from src.utils import utils
# from src import simulation_utils

from src.simulation import simulation as sim_v2
from src.simulation import extra_functions as extra
from src.utils import utils

# from src import simulation_utils


@njit
def v1_initialize_connections_and_rates(N_tot, sigma_mu, beta, sigma_beta):
    my_connection_weight = np.ones(N_tot, dtype=np.float32)
    my_infection_weight = np.ones(N_tot, dtype=np.float32)
    for agent in range(N_tot):
        extra.set_connections_weight(my_connection_weight, agent, sigma_mu)
        extra.set_infection_weight(my_infection_weight, agent, sigma_beta, beta)
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

        do_stop = sim_v2.update_node_connections(
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
        do_stop = sim_v2.update_node_connections(
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
