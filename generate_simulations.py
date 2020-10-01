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


N_tot_max = 1_000_000
num_cores_max = 30
N_runs = 10
dry_run = False
force_rerun = False
verbose = True

#%%


if utils.is_local_computer():

    all_simulation_parameters = [
        {
            "N_tot": 58_000,
            "rho": 0,
            "N_events": [0, 100],
            "event_size_max": [10, 100],
        },
    ]

else:
    yaml_filename = "cfg/simulation_parameters.yaml"
    all_simulation_parameters = utils.load_yaml(yaml_filename)["all_simulation_parameters"]

#%%

N_runs = 2 if utils.is_local_computer() else N_runs

N_files_total = 0


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
            verbose=verbose,
            force_rerun=force_rerun,
            dry_run=dry_run,
        )

        N_files_total += N_files

print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")

# %%
