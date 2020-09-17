import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
import h5py
from resource import getrusage, RUSAGE_SELF
import warnings
from importlib import reload
import os
from IPython.display import display
from contexttimer import Timer

# conda install -c numba/label/dev numba
import numba as nb
from numba import njit, prange, objmode, typeof
from numba.typed import List, Dict

from numba.core.errors import (
    NumbaTypeSafetyWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning,
)

import awkward as awkward0  # conda install awkward0, conda install -c conda-forge pyarrow
import awkward1 as ak  # pip install awkward1

while True:
    path = Path("").cwd()
    if "src" in str(path):
        os.chdir(path.parent)
    else:
        break


from src.utils import utils
from src.simulation import nb_simulation as nb

np.set_printoptions(linewidth=200)


class Simulation:
    def __init__(self, filename, verbose=False):

        self.verbose = verbose
        self._Filename = Filename = utils.Filename(filename)

        self.cfg = Filename.simulation_parameters
        self.ID = Filename.ID

        self.filenames = {}
        self.filename = self.filenames["filename"] = Filename.filename
        self.filenames["network_initialisation"] = Filename.filename_network_initialisation
        self.filenames["network_network"] = Filename.filename_network

        utils.set_numba_random_seed(self.ID)

    def _initialize_network(self):

        cfg = self.cfg
        self.coordinates, self.coordinate_indices = utils.load_coordinates(
            self._Filename.coordinates_filename, cfg.N_tot, self.ID
        )

        if self.verbose:
            print(f"INITIALIZE VERSION {cfg.version} NETWORK")

        if cfg.version >= 2:

            (
                people_in_household,
                age_distribution_per_people_in_household,
            ) = utils.load_household_data(self._Filename.household_data_filenames)
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
            ) = nb.place_and_connect_families(
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
            matrix_work = np.ones((N_ages, N_ages))
            matrix_work = matrix_work * counter_ages * counter_ages.reshape((-1, 1))
            matrix_work = matrix_work / matrix_work.sum()

            matrix_other = np.ones((N_ages, N_ages))
            matrix_other = matrix_other * counter_ages * counter_ages.reshape((-1, 1))
            matrix_other = matrix_other / matrix_other.sum()

            work_other_ratio = 0.5  # 20% work, 80% other

            if self.verbose:
                print("Connecting work and others, currently slow, please wait")
            nb.connect_work_and_others(
                cfg.N_tot,
                N_ages,
                mu_counter,
                cfg.mu,
                work_other_ratio,
                matrix_work,
                matrix_other,
                cfg.rho,
                cfg.epsilon_rho,
                self.coordinates,
                agents_in_age_group,
                my_connections,
                my_connections_type,
                my_number_of_contacts,
            )

        else:

            my_connections = utils.initialize_nested_lists(cfg.N_tot, dtype=np.uint32)  # initialize_list_set

            if self.verbose:
                print("MAKE RATES AND CONNECTIONS")
            (
                my_connection_weight,
                my_infection_weight,
                my_number_of_contacts,
                my_connections_type,
            ) = nb.v1_initialize_connections_and_rates(cfg.N_tot, cfg.sigma_mu, cfg.beta, cfg.sigma_beta)
            if self.verbose:
                print("CONNECT NODES")
            nb.v1_connect_nodes(
                cfg.N_tot,
                cfg.mu,
                cfg.rho,
                cfg.epsilon_rho,
                cfg.algo,
                my_connection_weight,
                my_connections,
                my_connections_type,
                my_number_of_contacts,
                self.coordinates,
            )
            counter_ages = np.array([cfg.N_tot], dtype=np.uint16)
            agents_in_age_group = List()
            agents_in_age_group.append(np.arange(cfg.N_tot, dtype=np.uint32))
            my_age = np.zeros(cfg.N_tot, dtype=np.uint8)

        self.counter_ages = counter_ages
        self.agents_in_age_group = agents_in_age_group
        self.my_number_of_contacts = my_number_of_contacts
        self.my_infection_weight = my_infection_weight

        return (
            my_connections,
            my_connections_type,
            my_number_of_contacts,
            my_infection_weight,
            my_age,
            agents_in_age_group,
        )

    def _save_network_initalization(
        self,
        my_age,
        agents_in_age_group,
        my_number_of_contacts,
        my_infection_weight,
        my_connections,
        my_connections_type,
        time_elapsed,
    ):
        utils.make_sure_folder_exist(self.filenames["network_initialisation"])
        with h5py.File(self.filenames["network_initialisation"], "w") as f:  #
            f.create_dataset("cfg_str", data=str(self.cfg))  # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("my_age", data=my_age)
            f.create_dataset("my_number_of_contacts", data=my_number_of_contacts)
            f.create_dataset("my_infection_weight", data=my_infection_weight)
            awkward0.hdf5(f)["my_connections"] = ak.to_awkward0(my_connections)
            awkward0.hdf5(f)["my_connections_type"] = ak.to_awkward0(my_connections_type)
            awkward0.hdf5(f)["agents_in_age_group"] = ak.to_awkward0(agents_in_age_group)
            for key, val in self.cfg.items():
                f.attrs[key] = val
            f.create_dataset("time_elapsed", data=time_elapsed)

    def _load_network_initalization(self):
        with h5py.File(self.filenames["network_initialisation"], "r") as f:
            my_age = f["my_age"][()]
            my_number_of_contacts = f["my_number_of_contacts"][()]
            my_infection_weight = f["my_infection_weight"][()]
            my_connections = awkward0.hdf5(f)["my_connections"]
            my_connections_type = awkward0.hdf5(f)["my_connections_type"]
            agents_in_age_group = awkward0.hdf5(f)["agents_in_age_group"]
        self.coordinates, self.coordinate_indices = utils.load_coordinates(
            self._Filename.coordinates_filename, self.cfg.N_tot, self.ID
        )
        return (
            my_age,
            ak.from_awkward0(agents_in_age_group),
            my_number_of_contacts,
            my_infection_weight,
            ak.from_awkward0(my_connections),
            ak.from_awkward0(my_connections_type),
        )

    def initialize_network(self, force_rerun=False, save_initial_network=True):
        utils.set_numba_random_seed(self.ID)

        OSError_flag = False

        filename_network_init = self.filenames["network_initialisation"]

        # try to load file (except if forced to rerun)
        if not force_rerun:
            try:
                (
                    my_age,
                    agents_in_age_group,
                    my_number_of_contacts,
                    my_infection_weight,
                    my_connections,
                    my_connections_type,
                ) = self._load_network_initalization()
                if self.verbose:
                    print(f"{filename_network_init} exists, continue with loading it")
            except OSError as e:
                if self.verbose:
                    if utils.file_exists(filename_network_init):
                        print(f"{filename_network_init} does not exist, continue to create it")
                    else:
                        print(f"{filename_network_init} had OSError, create a new one")
                OSError_flag = True

        # if ran into OSError above or forced to rerun:
        if OSError_flag or force_rerun:

            if self.verbose and not OSError_flag:
                print(f"{filename_network_init} does not exist, creating it")

            with Timer() as t:
                (
                    my_connections,
                    my_connections_type,
                    my_number_of_contacts,
                    my_infection_weight,
                    my_age,
                    agents_in_age_group,
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
                        my_connections=ak.to_awkward0(my_connections),
                        my_connections_type=ak.to_awkward0(my_connections_type),
                        time_elapsed=t.elapsed,
                    )
                except OSError as e:
                    print(f"\nSkipped saving network initialization for {self.filenames['network_initialisation']}")
                    print(e)

        self.my_age = my_age
        self.agents_in_age_group = agents_in_age_group
        self.N_ages = len(self.agents_in_age_group)
        self.my_connections = utils.MutableArray(my_connections)
        self.my_connections_type = utils.MutableArray(my_connections_type)
        self.my_number_of_contacts = my_number_of_contacts
        self.my_infection_weight = my_infection_weight

    def make_initial_infections(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("INITIAL INFECTIONS")

        cfg = self.cfg

        np.random.seed(self.ID)

        self.nts = 0.1  # Time step (0.1 - ten times a day)
        self.N_states = 9  # number of states
        self.N_infectious_states = 4  # This means the 5'th state
        self.initial_ages_exposed = np.arange(self.N_ages)  # means that all ages are exposed

        self.my_rates = utils.initialize_my_rates(self.my_infection_weight, self.my_number_of_contacts)

        self.my_state = np.full(cfg.N_tot, -1, dtype=np.int8)
        self.state_total_counts = np.zeros(self.N_states, dtype=np.uint32)
        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.g_cumulative_sum_of_state_changes = np.zeros(self.N_states, dtype=np.float64)
        self.g_cumulative_sum_infection_rates = np.zeros(self.N_states, dtype=np.float64)
        self.my_sum_of_rates = np.zeros(cfg.N_tot, dtype=np.float64)

        self.SIR_transition_rates = utils.initialize_SIR_transition_rates(self.N_states, self.N_infectious_states, cfg)

        self.g_total_sum_of_state_changes = nb.make_initial_infections(
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
            cfg.make_random_initial_infections,
            code_version=cfg.version,
        )

    def run_simulation(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("RUN SIMULATION")

        cfg = self.cfg

        self.individual_infection_counter = np.zeros(cfg.N_tot, dtype=np.uint16)

        res = nb.run_simulation(
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
            self.nts,
            self.verbose,
            self.my_connections_type.array,
        )

        out_time, out_state_counts, out_my_state = res

        self.time = np.array(out_time)
        self.state_counts = np.array(out_state_counts)
        self.my_state = np.array(out_my_state)

    def make_dataframe(self):
        #  Make DataFrame
        self.df = df = utils.state_counts_to_df(self.time, self.state_counts)

        # Save CSV
        utils.make_sure_folder_exist(self.filename)
        # save csv file
        df.to_csv(self.filename, index=False)
        return df

    def save_simulation_results(self, save_only_ID_0=False, time_elapsed=None):

        if save_only_ID_0 and self.ID != 0:
            return None

        utils.make_sure_folder_exist(self.filenames["network_network"], delete_file_if_exists=True)

        # Saving HDF5 File
        with h5py.File(self.filenames["network_network"], "w") as f:  #
            f.create_dataset("coordinate_indices", data=self.coordinate_indices)
            f.create_dataset("my_state", data=self.my_state)
            f.create_dataset("my_number_of_contacts", data=self.my_number_of_contacts)
            f.create_dataset("my_age", data=self.my_age)
            f.create_dataset("cfg_str", data=str(self.cfg))  # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))

            if time_elapsed:
                f.create_dataset("time_elapsed", data=time_elapsed)

            for key, val in self.cfg.items():
                f.attrs[key] = val

        if self.verbose:
            print("\n\n")
            print("coordinates", utils.get_size(self.coordinates))
            print("my_state", utils.get_size(self.my_state))
            print("my_number_of_contacts", utils.get_size(self.my_number_of_contacts))
            print("my_age", utils.get_size(self.my_age))


#%%


def run_full_simulation(
    filename,
    verbose=False,
    force_rerun=False,
    only_initialize_network=False,
    save_initial_network=True,
):

    with Timer() as t, warnings.catch_warnings():
        # if not verbose:
        # warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
        warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
        # warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        simulation = Simulation(filename, verbose)
        simulation.initialize_network(force_rerun=force_rerun, save_initial_network=save_initial_network)
        if only_initialize_network:
            return None

        simulation.make_initial_infections()
        simulation.run_simulation()
        simulation.make_dataframe()
        simulation.save_simulation_results(time_elapsed=t.elapsed)

        if verbose and simulation.ID == 0:
            print(f"\n\n{simulation.cfg}\n")

    if verbose:
        print("\n\nFinished!!!")


if utils.is_ipython and True:

    reload(utils)

    verbose = True
    force_rerun = True
    filename = "Data/ABM/v__1.0__N_tot__58000__N_init__100__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__algo__2__make_random_initial_infections__1/v__1.0__N_tot__58000__N_init__100__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__algo__2__make_random_initial_infections__1__ID__000.csv"
    # filename = filename.replace('ID__000', 'ID__001')

    simulation = Simulation(filename, verbose)
    simulation.initialize_network(force_rerun=force_rerun)
    # simulation.make_initial_infections()
    # simulation.run_simulation()
    # df = simulation.make_dataframe()
    # display(df)
