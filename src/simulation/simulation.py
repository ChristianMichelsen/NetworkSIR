from re import X
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
import yaml

# conda install -c numba/label/dev numba
import numba as nb
from numba import njit, prange, objmode, typeof
from numba.typed import List, Dict
import uuid
import datetime
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

# print(Path("").cwd())
from src.utils import utils
from src.simulation import nb_simulation
from src.simulation import nb_load_jitclass
from src import file_loaders


hdf5_kwargs = dict(track_order=True)
np.set_printoptions(linewidth=200)


class Simulation:
    def __init__(self, cfg, verbose=False):

        self.verbose = verbose

        self.cfg = utils.DotDict(cfg)
        self.N_tot = self.cfg.N_tot

        # unique code that identifies this simulation
        self.hash = utils.cfg_to_hash(self.cfg)
        self.my = nb_simulation.initialize_My(self.cfg)
        utils.set_numba_random_seed(self.cfg.ID)

        if self.cfg.version == 1:
            if self.cfg.do_interventions:
                raise AssertionError("interventions not yet implemented for version 1")

    def _initialize_network(self):

        self.df_coordinates = utils.load_df_coordinates(self.N_tot, self.cfg.ID)
        coordinates_raw = utils.df_coordinates_to_coordinates(self.df_coordinates)

        if self.verbose:
            print(f"\nINITIALIZE VERSION {self.cfg.version} NETWORK")

        if self.cfg.version >= 2:

            (
                people_in_household,
                age_distribution_per_people_in_household,
            ) = utils.load_household_data()
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

            # work_other_ratio = 0.5  # 20% work, 80% other

            if self.verbose:
                print("Connecting work and others, currently slow, please wait")

            nb_simulation.connect_work_and_others(
                self.my,
                N_ages,
                mu_counter,
                # work_other_ratio,
                matrix_work,
                matrix_other,
                agents_in_age_group,
                verbose=self.verbose,
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
            agents_in_age_group.append(np.arange(self.cfg.N_tot, dtype=np.uint32))

        self.agents_in_age_group = agents_in_age_group
        self.N_ages = len(self.agents_in_age_group)
        return None

    def _save_initialized_network(self, filename):
        if self.verbose:
            print(f"Saving initialized network to {filename}", flush=True)
        utils.make_sure_folder_exist(filename)
        my_hdf5ready = nb_load_jitclass.jitclass_to_hdf5_ready_dict(self.my)

        with h5py.File(filename, "w", **hdf5_kwargs) as f:
            group_my = f.create_group("my")
            nb_load_jitclass.save_jitclass_hdf5ready(group_my, my_hdf5ready)
            utils.NestedArray(self.agents_in_age_group).add_to_hdf5_file(f, "agents_in_age_group")
            f.create_dataset("N_ages", data=self.N_ages)
            self._add_cfg_to_hdf5_file(f)

    def _load_initialized_network(self, filename):
        if self.verbose:
            print(f"Loading previously initialized network, please wait", flush=True)
        with h5py.File(filename, "r") as f:
            self.agents_in_age_group = utils.NestedArray.from_hdf5(
                f, "agents_in_age_group"
            ).to_nested_numba_lists()
            self.N_ages = f["N_ages"][()]
            my_hdf5ready = nb_load_jitclass.load_jitclass_to_dict(f["my"])
            self.my = nb_load_jitclass.load_My_from_dict(my_hdf5ready, self.cfg)
        self.df_coordinates = utils.load_df_coordinates(self.N_tot, self.cfg.ID)

    def initialize_network(
        self, force_rerun=False, save_initial_network=True, force_load_initial_network=False
    ):
        utils.set_numba_random_seed(self.cfg.ID)

        filename = "Data/initialized_network/"
        filename += f"initialized_network__{self.hash}__ID__{self.cfg.ID}.hdf5"

        cfg_network_initialized = utils.get_cfg_network_initialized(self.cfg)

        if force_load_initial_network:
            initialize_network = False
            if self.verbose:
                print("Force loading initialized network")
        elif len(utils.query_cfg(cfg_network_initialized)) == 0:
            initialize_network = True
            if self.verbose:
                print("Initializing network since it does not exist in database")
        elif not utils.file_exists(filename):
            initialize_network = True
            if self.verbose:
                print("Initializing network since the hdf5-file does not exist")
        elif force_rerun:
            initialize_network = True
            if self.verbose:
                print("Initializing network since it was forced to")
        else:
            initialize_network = False

        # Initalizing network and (optionally) saving it
        if initialize_network:
            self._initialize_network()
            if save_initial_network:
                self._save_initialized_network(filename)

        # Loading initialized network
        else:
            self._load_initialized_network(filename)

    def make_initial_infections(self):
        utils.set_numba_random_seed(self.cfg.ID)

        if self.verbose:
            print("\nINITIAL INFECTIONS")

        np.random.seed(self.cfg.ID)

        self.nts = 0.1  # Time step (0.1 - ten times a day)
        self.N_states = 9  # number of states
        self.N_infectious_states = 4  # This means the 5'th state
        self.initial_ages_exposed = np.arange(self.N_ages)  # means that all ages are exposed

        self.state_total_counts = np.zeros(self.N_states, dtype=np.uint32)
        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.g = nb_simulation.Gillespie(self.my, self.N_states)

        self.SIR_transition_rates = utils.initialize_SIR_transition_rates(
            self.N_states, self.N_infectious_states, self.cfg
        )
        if self.cfg.make_initial_infections_at_kommune:
            infected_per_kommune_ints, kommune_names, my_kommune = file_loaders.load_kommune_data(
                self.df_coordinates
            )

            nb_simulation.make_initial_infections_from_kommune_data(
                self.my,
                self.g,
                self.state_total_counts,
                self.agents_in_state,
                self.SIR_transition_rates,
                self.agents_in_age_group,
                self.initial_ages_exposed,
                # self.N_infectious_states,
                self.N_states,
                infected_per_kommune_ints,
                kommune_names,
                my_kommune,
                self.verbose,
            )

        else:
            nb_simulation.make_initial_infections(
                self.my,
                self.g,
                self.state_total_counts,
                self.agents_in_state,
                self.SIR_transition_rates,
                self.agents_in_age_group,
                self.initial_ages_exposed,
                # self.N_infectious_states,
                self.N_states,
            )

    def run_simulation(self, verbose_interventions=None):
        utils.set_numba_random_seed(self.cfg.ID)

        if self.verbose:
            print("\nRUN SIMULATION")

        labels = self.df_coordinates["idx"].values

        if verbose_interventions is None:
            verbose_interventions = self.verbose

        self.intervention = nb_simulation.Intervention(
            self.my.cfg,
            labels=labels,
            verbose=verbose_interventions,
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

        out_time, out_state_counts, out_my_state, intervention = res
        self.out_time = out_time
        self.my_state = np.array(out_my_state)
        self.df = utils.state_counts_to_df(np.array(out_time), np.array(out_state_counts))
        self.intervention = intervention
        return self.df

    def _get_filename(self, name="ABM", filetype="hdf5"):
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"Data/{name}/{self.hash}/{name}_{date}_{self.hash}_ID__{self.cfg.ID}.{filetype}"
        return filename

    def _save_cfg(self):
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        filename_cfg = f"Data/cfgs/cfg_{date}_{self.hash}.yaml"
        self.cfg.dump_to_file(filename_cfg, exclude="ID")
        return None

    def _add_cfg_to_hdf5_file(self, f, cfg=None):
        if cfg is None:
            cfg = self.cfg
        utils.add_cfg_to_hdf5_file(f, cfg)

    def _save_dataframe(self, save_csv=False, save_hdf5=True):

        # Save CSV
        if save_csv:
            filename_csv = self._get_filename(name="ABM", filetype="csv")
            utils.make_sure_folder_exist(filename_csv)
            self.df.to_csv(filename_csv, index=False)

        if save_hdf5:
            filename_hdf5 = self._get_filename(name="ABM", filetype="hdf5")
            utils.make_sure_folder_exist(filename_hdf5)
            with h5py.File(filename_hdf5, "w", **hdf5_kwargs) as f:  #
                f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))
                self._add_cfg_to_hdf5_file(f)

        return None

    def _save_simulation_results(self, save_only_ID_0=False, time_elapsed=None):

        if save_only_ID_0 and self.cfg.ID != 0:
            return None

        filename_hdf5 = self._get_filename(name="network", filetype="hdf5")
        utils.make_sure_folder_exist(filename_hdf5)

        with h5py.File(filename_hdf5, "w", **hdf5_kwargs) as f:  #
            f.create_dataset("my_state", data=self.my_state)
            f.create_dataset("my_corona_type", data=self.my.corona_type)
            f.create_dataset("my_number_of_contacts", data=self.my.number_of_contacts)
            f.create_dataset("day_found_infected", data=self.intervention.day_found_infected)
            # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("cfg_str", data=str(self.cfg))
            f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))
            f.create_dataset(
                "df_coordinates",
                data=utils.dataframe_to_hdf5_format(self.df_coordinates, cols_to_str="kommune"),
            )

            if time_elapsed:
                f.create_dataset("time_elapsed", data=time_elapsed)

            self._add_cfg_to_hdf5_file(f)

        return None

    def save(self, save_csv=False, save_hdf5=True, save_only_ID_0=False, time_elapsed=None):
        self._save_cfg()
        self._save_dataframe(save_csv=save_csv, save_hdf5=save_hdf5)
        self._save_simulation_results(save_only_ID_0=save_only_ID_0, time_elapsed=time_elapsed)


#%%


def run_single_simulation(
    cfg,
    verbose=False,
    force_rerun=False,
    only_initialize_network=False,
    save_initial_network=True,
    save_csv=False,
):

    with Timer() as t, warnings.catch_warnings():
        if not verbose:
            # ignore warning about run_algo
            warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
            warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
            # warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        simulation = Simulation(cfg, verbose)
        simulation.initialize_network(
            force_rerun=force_rerun, save_initial_network=save_initial_network
        )
        if only_initialize_network:
            return None

        simulation.make_initial_infections()
        simulation.run_simulation()
        simulation.save(time_elapsed=t.elapsed, save_hdf5=True, save_csv=save_csv)
    return cfg


from tinydb import TinyDB, Query
from tqdm import tqdm
from functools import partial
from p_tqdm import p_umap, p_uimap


def update_database(db_cfg, q, cfg):
    cfg["hash"] = utils.cfg_to_hash(cfg)
    cfg.pop("ID")
    if not db_cfg.contains(q.hash == cfg.hash):
        db_cfg.insert(cfg)


def run_simulations(
    d_simulation_parameters,
    N_runs=2,
    num_cores_max=None,
    N_tot_max=False,
    verbose=True,
    force_rerun=False,
    dry_run=False,
    **kwargs,
):

    db_cfg = utils.get_db_cfg()
    q = Query()

    cfgs_all = utils.generate_cfgs(d_simulation_parameters, N_runs, N_tot_max, verbose=verbose)
    if len(cfgs_all) == 0:
        N_files = 0
        return N_files

    db_counts = np.array([db_cfg.count(q.hash == utils.cfg_to_hash(cfg)) for cfg in cfgs_all])
    assert np.max(db_counts) <= 1

    # keep only cfgs that are not in the database already
    if force_rerun:
        cfgs = cfgs_all
    else:
        cfgs = [cfg for (cfg, count) in zip(cfgs_all, db_counts) if count == 0]

    N_files = len(cfgs)

    num_cores = utils.get_num_cores_N_tot(d_simulation_parameters, num_cores_max)

    if isinstance(d_simulation_parameters, dict):
        s_simulation_parameters = str(d_simulation_parameters)
    elif isinstance(d_simulation_parameters, list):
        s_simulation_parameters = f"{len(d_simulation_parameters)} MCMC runs, see cfg/simulation_parameters.yaml for more info"
    else:
        raise AssertionError("d_simulation_parameters neither list nor dict")

    print(
        f"\n\n" f"Generating {N_files:3d} network-based simulations",
        f"with {num_cores} cores",
        f"based on {s_simulation_parameters}.",
        "Please wait. \n",
        flush=True,
    )

    if dry_run or N_files == 0:
        return N_files

    # kwargs = {}
    if num_cores == 1:
        for cfg in tqdm(cfgs):
            run_single_simulation(cfg, verbose=verbose, **kwargs)
            update_database(db_cfg, q, cfg)
    else:
        # print("run simulation", flush=True)
        f_single_simulation = partial(run_single_simulation, verbose=False, **kwargs)
        for cfg in p_uimap(f_single_simulation, cfgs, num_cpus=num_cores):
            update_database(db_cfg, q, cfg)

    return N_files


# if utils.is_ipython and debugging:
if debugging:

    verbose = True
    force_rerun = False

    d_simulation_parameters = {
        "N_tot": 58_000,
        # "N_init": [100, 1000],
    }

    cfg = utils.DotDict(
        {
            "version": 2.1,
            "N_tot": 58_000,
            "rho": 0.1,
            "epsilon_rho": 0.04,
            "mu": 20.0,
            "sigma_mu": 0.0,
            "beta": 0.012,
            "sigma_beta": 0.0,
            "N_init": 2000,
            "lambda_E": 1.0,
            "lambda_I": 1.0,
            # other
            "make_random_initial_infections": False,
            "weighted_random_initial_infections": True,
            "make_initial_infections_at_kommune": False,
            "day_max": 0.0,
            "clustering_connection_retries": 0,
            "work_other_ratio": 0.5,
            "N_contacts_max": 200,
            # English type
            "N_init_English": 100,
            "beta_UK_multiplier": 1.7,
            # contacts
            "N_events": 0,
            "event_size_max": 50,
            "event_size_mean": 5.0,
            "event_beta_scaling": 5.0,
            "event_weekend_multiplier": 2.0,
            # lockdown / intervention
            "do_interventions": False,
            "interventions_to_apply": [3, 4, 5, 6],
            "f_daily_tests": 0.01,
            "test_delay_in_clicks": [0, 0, 25],
            "results_delay_in_clicks": [10, 10, 10],
            "chance_of_finding_infected": [0.0, 0.15, 0.15, 0.15, 0.0],
            "days_looking_back": 7.0,
            "masking_rate_reduction": [[0.0, 0.0, 0.3], [0.0, 0.0, 0.8]],
            "lockdown_rate_reduction": [[0.0, 1.0, 0.6], [0.0, 0.6, 0.6]],
            "isolation_rate_reduction": [0.2, 1.0, 1.0],
            "tracking_rates": [1.0, 0.8, 0.0],
            "tracking_delay": 1.0,
            "ID": 0,
        }
    )

    if __name__ == "__main__" and True:
        with Timer() as t:
            simulation = Simulation(cfg, verbose)
            simulation.initialize_network(
                force_rerun=force_rerun,
                save_initial_network=True,
                # force_load_initial_network=True,
            )
            simulation.make_initial_infections()
            df = simulation.run_simulation(verbose_interventions=False)
        display(df)
        print(f"Time taken: {t.elapsed:.1f}")
        # simulation.save(time_elapsed=t.elapsed, save_hdf5=True, save_csv=True)

        # run_simulations(d_simulation_parameters)

        my = simulation.my
        df_coordinates = simulation.df_coordinates
        intervention = simulation.intervention
        g = simulation.g

        # simulation.my_state
        # simulation.out_time
        # np.all(simulation.my_state[-2] == simulation.my_state[-1])

        # N_tot": 58_000,
        # "N_contacts_max": 100,
        # "work_other_ratio": 0.5,

        #%%

        # cfg = utils.DotDict(
        #     {
        #         "version": 2.1,
        #         "N_tot": 580000,
        #         "rho": 0.1,
        #         "epsilon_rho": 0.04,
        #         "mu": 29.5714,
        #         "sigma_mu": 0.0,
        #         "beta": 0.0141,
        #         "sigma_beta": 0.0,
        #         "N_init": 2000,
        #         "lambda_E": 1.0,
        #         "lambda_I": 1.0,
        #         "make_random_initial_infections": True,
        #         "make_initial_infections_at_kommune": False,
        #         "day_max": 35.0,
        #         "clustering_connection_retries": 0,
        #         "work_other_ratio": 0.2697,
        #         "N_contacts_max": 0,
        #         "N_events": 930,
        #         "event_size_max": 24,
        #         "event_size_mean": 6.1444,
        #         "event_beta_scaling": 5.0,
        #         "event_weekend_multiplier": 2.0,
        #         "do_interventions": False,
        #         "interventions_to_apply": [1, 4, 6],
        #         "f_daily_tests": 0.01,
        #         "test_delay_in_clicks": [0, 0, 25],
        #         "results_delay_in_clicks": [5, 10, 5],
        #         "chance_of_finding_infected": [0.0, 0.15, 0.15, 0.15, 0.0],
        #         "days_looking_back": 7.0,
        #         "masking_rate_reduction": [[0.0, 0.0, 0.3], [0.0, 0.0, 0.8]],
        #         "lockdown_rate_reduction": [[0.0, 1.0, 0.6], [0.0, 0.6, 0.6]],
        #         "isolation_rate_reduction": [0.2, 1.0, 1.0],
        #         "tracking_rates": [1.0, 0.8, 0.0],
        #         "ID": 0,
        #     }
        # )

# %%
