import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmark_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 8
N_loops = 10
dry_run = True
force_overwrite = False

#%%

all_sim_pars = [

                { 
                    'rho': [0, 1, 2, 5, 10, 15, 20],
                    'connect_algo': [1, 2],
                },

                {
                    'epsilon_rho': [0, 0.5, 1, 2, 5],
                    'rho': [20],
                },

                {   'connect_algo': [1, 2],
                }, 

                {
                    'N_tot': [100_000, 200_000, 500_000],
                },

                { 
                    'sigma_beta': [0.0, 1.0], 
                    'sigma_mu': [0.0, 1.0],
                },

                {   'N_tot': [500_000],
                    'N_init': [1, 5, 50, 500, 1_000, 5_000],
                }, 

                {   'N_tot': [100_000],
                    'N_init': [1, 10, 100, 1_000],
                }, 

                {
                    'beta': [0.005, 0.01, 0.02, 0.05, 0.1],
                },

                {
                    'beta': [0.01*2, 0.01*4],
                    'mu': [20/2, 20/4],
                    'sigma_mu': [0, 1],
                    'sigma_beta': [0, 1],
                },

                {
                    'beta': [0.01*2, 0.01*4],
                    'lambda_I': [1*2, 1*4],
                    'sigma_mu': [0, 1],
                    'sigma_beta': [0, 1],
                },

                {
                    'mu': [5, 10, 20, 40, 80],
                },

                {
                    'lambda_E': [0.5, 1, 2, 4],
                },

                {
                    'lambda_I': [0.5, 1, 2, 4],
                },

                # {
                #     'N_tot': [1_000_000, 2_000_000],
                # },

                # {
                #     'N_tot': [5_000_000],
                # },

                ]


#%%

N_loops = 2 if extra_funcs.is_local_computer() else N_loops

N_files_all = 0
reload(extra_funcs)
if __name__ == '__main__':

    for d_sim_pars in all_sim_pars:
        filenames = extra_funcs.generate_filenames(d_sim_pars, N_loops, force_overwrite=force_overwrite)

        N_files = len(filenames)
        N_files_all += N_files

        # make sure path exists
        if len(filenames) > 0:
            num_cores = extra_funcs.get_num_cores_N_tot_specific(d_sim_pars, num_cores_max)
            print(f"\nGenerating {N_files:3d} network-based simulations with {num_cores} cores based on {d_sim_pars}, please wait.", flush=True)

            if dry_run:
                continue

            if num_cores == 1:
                for filename in tqdm(filenames):
                    extra_funcs.single_run_and_save(filename)

            else:
                with mp.Pool(num_cores) as p:
                    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print(f"\nIn total: {N_files_all} files generated")
    print("Finished simulating!")
