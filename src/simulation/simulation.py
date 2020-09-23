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

debugging = False
while True:
    path = Path("").cwd()
    if "src" in str(path):
        os.chdir(path.parent)
        debugging = True
    else:
        break

from src.utils import utils
from src.simulation import nb_simulation

np.set_printoptions(linewidth=200)


class Simulation:
    def __init__(self, filename, verbose=False):

        self.verbose = verbose
        self._Filename = Filename = utils.Filename(filename)

        self.cfg = Filename.simulation_parameters
        self.ID = Filename.ID
        self.N_tot = self.cfg.N_tot

        self.filenames = {}
        self.filename = self.filenames["filename"] = Filename.filename
        self.filenames["network_initialisation"] = Filename.filename_network_initialisation
        self.filenames["network_network"] = Filename.filename_network

        self.my = nb_simulation.initialize_My(self.cfg)

        utils.set_numba_random_seed(self.ID)

    def _initialize_network(self):

        cfg = self.cfg

        self.df_coordinates = utils.load_df_coordinates(self.N_tot, self.ID)
        coordinates_raw = utils.df_coordinates_to_coordinates(self.df_coordinates)

        if self.verbose:
            print(f"INITIALIZE VERSION {cfg.version} NETWORK")

        if cfg.version >= 2:

            (
                people_in_household,
                age_distribution_per_people_in_household,
            ) = utils.load_household_data(self._Filename.household_data_filenames)
            N_ages = age_distribution_per_people_in_household.shape[1]

            if self.verbose:
                print("Connect Families")

            (
                mu_counter,
                counter_ages,
                agents_in_age_group,
            ) = nb_simulation.place_and_connect_families(
                self.my,
                people_in_household,
                age_distribution_per_people_in_household,
                coordinates_raw,
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

            nb_simulation.connect_work_and_others(
                self.my,
                N_ages,
                mu_counter,
                work_other_ratio,
                matrix_work,
                matrix_other,
                agents_in_age_group,
            )

        else:

            if self.verbose:
                print("SETUP WEIGHTS AND COORDINATES")
            nb_simulation.v1_initialize_my(self.my, coordinates_raw)

            if self.verbose:
                print("CONNECT NODES")
            nb_simulation.v1_connect_nodes(self.my)

            # counter_ages = np.array([cfg.N_tot], dtype=np.uint16)
            agents_in_age_group = List()
            agents_in_age_group.append(np.arange(cfg.N_tot, dtype=np.uint32))

        self.agents_in_age_group = agents_in_age_group
        self.N_ages = len(self.agents_in_age_group)
        return None

    #     )

    def initialize_network(self, force_rerun=False, save_initial_network=True):
        utils.set_numba_random_seed(self.ID)

        with Timer() as t:
            self._initialize_network()

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

        self.state_total_counts = np.zeros(self.N_states, dtype=np.uint32)
        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.g = nb_simulation.Gillespie(self.my, self.N_states)

        self.SIR_transition_rates = utils.initialize_SIR_transition_rates(
            self.N_states, self.N_infectious_states, cfg
        )

        nb_simulation.make_initial_infections(
            self.my,
            self.g,
            self.state_total_counts,
            self.agents_in_state,
            self.SIR_transition_rates,
            self.agents_in_age_group,
            self.initial_ages_exposed,
            self.N_infectious_states,
        )

    def run_simulation(self):
        utils.set_numba_random_seed(self.ID)

        if self.verbose:
            print("RUN SIMULATION")

        N_daily_tests = 20_000  # 20000  # TODO make Par?
        labels = self.df_coordinates["idx"].values
        if self.my.cfg.version >= 2:
            interventions_to_apply = List([1, 4, 6])
        else:
            interventions_to_apply = None

        # 1: Lockdown (jobs and schools)
        # 2: Cover (with face masks)
        # 3: Tracking (infected and their connections)
        # 4: Test people with symptoms
        # 5: Isolate
        # 6: Random Testing
        # 0: Do nothing

        self.intervention = nb_simulation.Intervention(
            N_tot=self.cfg.N_tot,
            N_daily_tests=N_daily_tests,
            labels=labels,
            interventions_to_apply=interventions_to_apply,
            verbose=self.verbose,
        )

        res = nb_simulation.run_simulation(
            self.my,
            self.g,
            self.intervention,
            self.state_total_counts,
            self.agents_in_state,
            self.N_states,
            self.SIR_transition_rates,
            self.N_infectious_states,
            self.nts,
            self.verbose,
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


#%%


def run_full_simulation(
    filename,
    verbose=False,
    force_rerun=False,
    only_initialize_network=False,
    save_initial_network=True,
):

    with Timer() as t, warnings.catch_warnings():
        if not verbose:
            # ignore warning about run_algo
            warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
            warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
            # warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        simulation = Simulation(filename, verbose)
        simulation.initialize_network(
            force_rerun=force_rerun, save_initial_network=save_initial_network
        )
        if only_initialize_network:
            return None

        simulation.make_initial_infections()
        simulation.run_simulation()
        simulation.make_dataframe()
        # print("NOT SAVING SIM RESULTS")
        # if False:
        # simulation.save_simulation_results(time_elapsed=t.elapsed)

        if verbose and simulation.ID == 0:
            print(f"\n\n{simulation.cfg}\n")

    if verbose:
        print("\n\nFinished!!!")


if utils.is_ipython and debugging:

    verbose = True
    force_rerun = True

    filename = "Data/ABM/v__1.0__N_tot__58000__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__algo__2__N_init__100__lambda_E__1.0__lambda_I__1.0__make_random_initial_infections__0__N_connect_retries__0/v__1.0__N_tot__58000__rho__0.0__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__algo__2__N_init__100__lambda_E__1.0__lambda_I__1.0__make_random_initial_infections__0__N_connect_retries__0__ID__000.csv"
    filename = filename.replace("v__1.0", "v__2.0")
    # filename = filename.replace("rho__0.0__", "rho__0.1__")

    # reload(nb_simulation)

    simulation = Simulation(filename, verbose)
    simulation.initialize_network(force_rerun=force_rerun)
    simulation.make_initial_infections()
    simulation.run_simulation()
    df = simulation.make_dataframe()
    display(df)

    my = simulation.my
    cfg = simulation.cfg
    df_coordinates = simulation.df_coordinates
    intervention = simulation.intervention
    g = simulation.g
