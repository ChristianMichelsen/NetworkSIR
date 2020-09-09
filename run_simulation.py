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

num_cores_max = 30
N_loops = 10
dry_run = False
force_overwrite = False
verbose = True  # only for 1 core

rc_params.set_rc_params()

#%%

mus = [10, 20, 25, 30, 40, 50, 60, 80, 100]
betas = [0.005, 0.01, 0.02, 0.05, 0.1]
epsilon_rhos = [
    0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.9,
    0.9,
    0.95,
    0.99,
    1.0,
]
rhos = [0, 0.005, 0.010, 0.015, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


if utils.is_local_computer():
    all_sim_pars = [
        {"N_tot": [58_000], "mu": [20, 40, 60],},
    ]


else:

    all_sim_pars = [
        {"mu": mus},
        {"beta": betas},
        {"epsilon_rho": epsilon_rhos, "rho": [0.1], "algo": [2, 1],},
        {"N_tot": [580_000], "rho": rhos},
        {"N_tot": [580_000], "rho": rhos, "beta": [0.01 / 2]},
        {"sigma_beta": [0, 0.25, 0.5, 0.75, 1], "sigma_mu": [0, 1], "rho": [0, 0.1],},
        {"sigma_beta": [0, 1], "sigma_mu": [0, 0.25, 0.5, 0.75, 1], "rho": [0, 0.1],},
        {"N_init": [1, 5, 50, 500, 1_000, 5_000],},
        {"beta": [0.01 * 2, 0.01 * 4], "mu": [40 / 2, 40 / 4], "sigma_mu": [0, 1], "sigma_beta": [0, 1], "rho": [0, 0.1],},
        {"lambda_E": [0.5, 1, 2, 4],},
        {"lambda_I": [0.5, 1, 2, 4],},
        {"N_tot": [100_000, 200_000, 500_000, 580_000],},
        {"N_tot": [1_000_000]},
        {"N_tot": [2_000_000]},
        {"N_tot": [3_000_000]},
        {"N_tot": [4_000_000]},
        {"N_tot": [5_000_000]},
        {"N_tot": [5_800_000]},
        {"N_tot": [5_800_000], "rho": rhos},
        {"N_tot": [5_800_000], "rho": rhos, "beta": [0.01 / 2]},
    ]


#%%

N_loops = 2 if utils.is_local_computer() else N_loops

N_files_total = 0
if __name__ == "__main__":

    if dry_run:
        print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

    for d_sim_pars in all_sim_pars:
        filenames = simulation_utils.generate_filenames(d_sim_pars, N_loops, force_overwrite=force_overwrite)

        N_files = len(filenames)
        N_files_total += N_files

        # make sure path exists
        if len(filenames) > 0:
            num_cores = simulation_utils.get_num_cores_N_tot_specific(d_sim_pars, num_cores_max)
            print(
                f"\nGenerating {N_files:3d} network-based simulations with {num_cores} cores based on {d_sim_pars}, please wait.",
                flush=True,
            )

            if dry_run:
                continue

            if num_cores == 1:
                for filename in tqdm(filenames):
                    simulation.run_full_simulation(filename, verbose=verbose, force_rerun=False)

            else:
                with mp.Pool(num_cores) as p:
                    kwargs = dict(verbose=False, force_rerun=False)
                    f = partial(simulation.run_full_simulation, **kwargs)
                    list(tqdm(p.imap_unordered(f, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print(f"\nIn total: {N_files_total} files generated")
    print("Finished simulating!")

# %%
