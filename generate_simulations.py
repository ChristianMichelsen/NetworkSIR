import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from importlib import reload
from src import utils
from src import simulation_utils
from src import simulation  # from src import simulation_v1 as simulation
from functools import partial
import yaml


N_tot_max = 1_000_000
num_cores_max = 30
N_loops = 10
dry_run = True
force_rerun = False
verbose = True

#%%


if utils.is_local_computer():

    all_simulation_parameters = [
        # {"beta": [0.01, 0.01 / 2, 0.01 * 2], "N_tot": 58_000, "rho": 0, "version": [1, 2]},
        {"N_tot": 58_000, "rho": 0, "version": [1, 2]},
    ]

else:
    yaml_filename = "cfg/simulation_parameters.yaml"
    all_simulation_parameters = utils.load_yaml(yaml_filename)["all_simulation_parameters"]


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
        if len(filenames) == 0:
            print("No files to generate, everything already generated.")
            continue

        if dry_run:
            continue

        proposed_N_max = simulation_utils.extract_N_tot_max(d_simulation_parameters)
        if N_tot_max and proposed_N_max > N_tot_max:
            print(
                f"Skipping since N_tot={utils.human_format(proposed_N_max)} > N_tot_max={utils.human_format(N_tot_max)}"
            )
            continue

        num_cores = simulation_utils.get_num_cores_N_tot_specific(d_simulation_parameters, num_cores_max)
        print(
            f"\nGenerating {N_files:3d} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.",
            flush=True,
        )

        if num_cores == 1:
            for filename in tqdm(filenames):
                simulation.run_full_simulation(filename, verbose=verbose, force_rerun=force_rerun)

        else:
            with mp.Pool(num_cores) as p:
                kwargs = dict(verbose=False, force_rerun=force_rerun)
                f = partial(simulation.run_full_simulation, **kwargs)
                list(tqdm(p.imap_unordered(f, filenames), total=N_files))

    print(f"\nIn total: {N_files_total} files generated")
    print("Finished simulating!")

# %%
