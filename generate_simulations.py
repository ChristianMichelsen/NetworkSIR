import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from importlib import reload
from src import simulation_v1 as simulation  # from src import simulation_v1 as simulation
from src import rc_params
from src import utils
from src import simulation_utils
from functools import partial
import yaml

num_cores_max = 30
N_loops = 10
dry_run = False
force_rerun = True
verbose = False

rc_params.set_rc_params()


#%%


if utils.is_local_computer():

    all_simulation_parameters = [
        {"beta": [0.01, 0.01 / 2, 0.01 * 2], "N_tot": 58_000, "rho": 0},
    ]

else:

    filename = "cfg_runs.yaml"
    with open(filename) as file:
        all_simulation_parameters = yaml.safe_load(file)["all_simulation_parameters"]


#%%

N_loops = 2 if utils.is_local_computer() else N_loops

N_files_total = 0
if __name__ == "__main__":

    if dry_run:
        print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

    for d_simulation_parameters in all_simulation_parameters:
        filenames = simulation_utils.generate_filenames(d_simulation_parameters, N_loops, force_rerun=force_rerun)

        N_files = len(filenames)
        N_files_total += N_files

        # make sure path exists
        if len(filenames) > 0:
            num_cores = simulation_utils.get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max)
            print(
                f"\nGenerating {N_files:3d} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.",
                flush=True,
            )

            if dry_run:
                continue

            if num_cores == 1:
                for filename in tqdm(filenames):
                    simulation.run_full_simulation(filename, verbose=verbose, force_rerun=force_rerun)

            else:
                with mp.Pool(num_cores) as p:
                    kwargs = dict(verbose=verbose, force_rerun=force_rerun)
                    f = partial(simulation.run_full_simulation, **kwargs)
                    list(tqdm(p.imap_unordered(f, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print(f"\nIn total: {N_files_total} files generated")
    print("Finished simulating!")

# %%
