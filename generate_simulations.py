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
            "version": [1, 2],
            # "make_random_initial_infections": [0, 1],
            "N_connect_retries": [0, 1],
        },
    ]

else:
    yaml_filename = "cfg/simulation_parameters.yaml"
    all_simulation_parameters = utils.load_yaml(yaml_filename)["all_simulation_parameters"]

#%%

N_runs = 2 if utils.is_local_computer() else N_runs

N_files_total = 0

if __name__ == "__main__":

    with Timer() as t:

        if dry_run:
            print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

        for d_simulation_parameters in all_simulation_parameters:
            # break

            N_files = len(cfgs)
            N_files_total += N_files

            if dry_run:
                continue

            num_cores = utils.get_num_cores_N_tot(d_simulation_parameters, num_cores_max)
            print(
                f"Generating {N_files:3d} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.\n",
                flush=True,
            )

            if num_cores == 1:
                for cfg in tqdm(cfgs):
                    simulation.run_full_simulation(cfg, verbose=verbose, force_rerun=force_rerun)

            else:
                with mp.Pool(num_cores) as p:
                    kwargs = dict(verbose=False, force_rerun=force_rerun)
                    f = partial(simulation.run_full_simulation, **kwargs)
                    list(tqdm(p.imap_unordered(f, cfgs), total=N_files))

    print(
        f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}"
    )
    print("Finished simulating!")

# %%
