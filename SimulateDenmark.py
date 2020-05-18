import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmark_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 10
N_loops = 13
force_SK_P1_UK = False
dry_run = False

#%%

all_sim_pars = [

                {   'BB': [1, 0],
                }, 

                {
                    'N0': [100_000, 200_000, 500_000],
                },

                { 
                    'sigma': [0.0, 0.5, 1.0], 
                    'gamma': [0.0, 0.5, 1.0],
                },

                {   'N0': [500_000],
                    'Ninit': [1, 5, 50, 500, 5_000],
                }, 

                {   'N0': [100_000],
                    'Ninit': [1, 10, 100, 1_000],
                }, 

                {
                    'beta': [0.005, 0.01, 0.02, 0.05, 0.1],
                },

                {
                    'beta': [0.01*2, 0.01*4],
                    'mu': [20/2, 20/4],
                    'gamma': [0, 1],
                    'sigma': [0, 1],
                },

                {
                    'beta': [0.01*2, 0.01*4],
                    'Mrate2': [1*2, 1*4],
                    'gamma': [0, 1],
                    'sigma': [0, 1],
                },
                

                {
                    # 'mu': [5, 10, 15, 20, 30, 40, 60, 80],
                    'mu': [5, 10, 20, 40, 80],
                },

                {
                    'Mrate1': [0.5, 1, 2, 4],
                },

                {
                    'Mrate2': [0.5, 1, 2, 4],
                },

                { 
                    'alpha': [0, 1, 2, 4, 6, 8, 10, 15, 20],
                    # 'BB': [1],
                },

                {
                    'N0': [1_000_000, 2_000_000],
                },

                {
                    'N0': [5_000_000],
                },

                ]


# x=x

#%%


# x=x


N_files_all = 0
# reload(extra_funcs)
if __name__ == '__main__':

    # num_cores = extra_funcs.get_num_cores(num_cores_max)
    N_loops = 1 if extra_funcs.is_local_computer() else N_loops

    all_filenames = extra_funcs.get_filenames_iter(all_sim_pars, force_SK_P1_UK, N_loops)

    for filenames, d_simulation_parameters in zip(*all_filenames):
        N_files = len(filenames)
        N_files_all += N_files

        # make sure path exists
        if len(filenames) > 0:
            # filename = filenames[0]

            num_cores = extra_funcs.get_num_cores_N0_specific(d_simulation_parameters, num_cores_max)
            
            print(f"Generating {N_files:3d} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.", flush=True)

            if dry_run:
                continue

            # if 'N0' in d_simulation_parameters.keys():
                # break

            # break

            if num_cores == 1:
                for filename in tqdm(filenames):
                    extra_funcs.single_run_and_save(filename)

            else:
                with mp.Pool(num_cores) as p:
                    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print(f"Total: {N_files_all}")
    print("Finished simulating!")
