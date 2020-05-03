import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 38
N_loops = 100


if __name__ == '__main__':

    all_sim_pars = [
                    {}, # d_sim_par 1

                    { 
                        'sigma': [0.0, 0.25, 0.5, 0.75, 1.0], 
                        'gamma': [0.0, 0.25, 0.5, 0.75, 1.0],
                        },
    
                    { 
                        'alpha': [1, 2, 3, 4, 5, 6, 7, 8]
                        },

                    {
                        'beta': [0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02],
                        },

                    {
                        'mu': [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0],
                        },

                    {
                        'Mrate1': [0.5, 1, 2, 4],
                        },

                    {
                        'Mrate2': [0.5, 1, 2, 4],
                        },

                    {
                        'N0': [10_000, 50_000, 100_000, 500_000, 1_000_000],
                        },

                    ]

    filenames = []
    for d_simulation_parameters in all_sim_pars:
        N_loops = 1 if extra_funcs.is_local_computer() else N_loops
        filenames.extend(extra_funcs.generate_filenames(d_simulation_parameters, N_loops))
    N_files = len(filenames)

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max


    # make sure path exists
    if len(filenames) > 0:
        filename = filenames[0]
        print(f"Generating {N_files} network-based simulations with {num_cores} cores, please wait.", flush=True)

        if num_cores == 1:
            for filename in tqdm(filenames):
                extra_funcs.single_run_and_save(filename)

        else:
            with mp.Pool(num_cores) as p:
                list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
    else:
        print("No files to generate, everything already generated.")

    print("Finished simulating!")
