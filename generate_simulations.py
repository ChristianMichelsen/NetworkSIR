import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from importlib import reload
from src.utils import utils
from src.simulation import simulation
from functools import partial
import yaml
from contexttimer import Timer


N_tot_max = False

num_cores_max = 40
N_runs = 3

dry_run = False
force_rerun = False
verbose = True

#%%


if utils.is_local_computer():

    all_simulation_parameters = [
        {
            "N_tot": 580_000,
            # "N_tot": [58_000],
            # "make_random_initial_infections": True,
            # "weighted_random_initial_infections": True,
            # "test_delay_in_clicks": [0, 0, 25],
            # "results_delay_in_clicks": [[20, 20, 20]],
            # "tracking_delay": [0, 5, 10, 15, 20, 25, 30],
            # "weighted_random_initial_infections": True,
            # "do_interventions": True,
            # "interventions_to_apply": [[3, 4, 5, 6]],
            # "results_delay_in_clicks": [20, 20, 20],
            # "tracking_delay": 15
            # "N_contacts_max": 100,
            # "work_other_ratio": 0.5,
            "N_init": [4000],
            # "N_init": [1000],
            "N_init_UK": [50],
            "work_other_ratio": 0.95,  # "algo 1"
            # "rho": 0.1,
            # "beta": [0.004],
            "beta": [0.010],
            # "beta": [0.016, 0.018],
            "beta_UK_multiplier": [1.7],
            # "outbreak_position_UK": ["københavn", "nordjylland"],
            "outbreak_position_UK": ["københavn"],
            "N_daily_vaccinations": [0, int(10_000 / 5.8e6 * 580_000)],
            # "make_initial_infections_at_kommune": True,
            # "N_events": 1000,
            # "mu": 20,
            # "day_max": 150,
            # "event_size_max": 50,
        },
    ]

else:
    all_simulation_parameters = utils.get_simulation_parameters()


# all_simulation_parameters = utils.get_simulation_parameters()
# x = x

#%%

N_runs = 2 if utils.is_local_computer() else N_runs

N_files_total = 0


# if __name__ == "__main__":

with Timer() as t:

    if dry_run:
        print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

    if force_rerun:
        print("Notice: forced rerun is set to True")

    for d_simulation_parameters in all_simulation_parameters:
        # break

        N_files = simulation.run_simulations(
            d_simulation_parameters,
            N_runs=N_runs,
            num_cores_max=num_cores_max,
            N_tot_max=N_tot_max,
            verbose=verbose,
            force_rerun=force_rerun,
            dry_run=dry_run,
            save_csv=True,
            save_initial_network=False,
        )

        N_files_total += N_files

print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")

# %%
