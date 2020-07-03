import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmarkAgeHospitalization_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 30
N_loops = 10
dry_run = True
force_overwrite = False
verbose = True # only for 1 core

if dry_run:
    print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

#%%

all_sim_pars = [

                # {
                #     'N_tot': [100_000],
                # },

                {
                    'sigma_beta': [0, 1],
                    'sigma_mu': [0, 1],
                    'rho': [0, 100],
                },

                {
                    'sigma_beta': [0, 0.25, 0.5, 0.75, 1],
                    'sigma_mu': [0, 1],
                    'rho': [0, 100],
                },

                {
                    'N_tot': [580_000],
                    'epsilon_rho': [0],
                    'rho': [100], 
                    'N_init': [100, 1000],
                    'algo': [2],
                    'beta_scaling': [1, 25, 50, 75, 100]
                },


                {
                    'N_tot': [5_800_000],
                    'epsilon_rho': [0],
                    'rho': [100],  # 300
                    'N_init': [100, 1000],
                    'algo': [2],
                    'beta_scaling': [1, 25, 50, 75, 100]
                },


                {
                    'epsilon_rho': [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.95, 0.99, 1.0],
                    'rho': [100],
                    'algo': [2, 1],
                },

                {   
                    'algo': [2, 1],
                }, 

                {
                    'N_tot': [100_000, 200_000, 500_000, 580_000],
                    'algo': [2, 1],
                },

                { 
                    'sigma_beta': [0.0, 1.0], 
                    'sigma_mu': [0.0, 1.0],
                    'algo': [2, 1],
                },


                { 
                    'rho': [0, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500],
                    'algo': [2, 1],
                },

                {   'N_init': [1, 5, 50, 500, 1_000, 5_000],
                    'algo': [2, 1],
                }, 


                {
                    'beta': [0.005, 0.01, 0.02, 0.05, 0.1],
                },

                {
                    'beta': [0.01*2, 0.01*4],
                    'mu': [40/2, 40/4],
                    'sigma_mu': [0, 1],
                    'sigma_beta': [0, 1],
                    'algo': [2, 1],
                },

                {
                    'beta': [0.01*2, 0.01*4],
                    'lambda_I': [1*2, 1*4],
                    'sigma_mu': [0, 1],
                    'sigma_beta': [0, 1],
                },

                {
                    'mu': [10, 20, 25, 30, 40, 50, 60, 80, 100],
                },

                {
                    'lambda_E': [0.5, 1, 2, 4],
                },

                {
                    'lambda_I': [0.5, 1, 2, 4],
                },


                {
                    'sigma_mu': [0.0, 0.25, 0.5, 0.75, 1.0],
                },

                {
                    'sigma_beta': [0.0, 0.25, 0.5, 0.75, 1.0],
                },


                { 
                    'sigma_beta': [0.0, 0.25, 0.5, 0.75, 1.0], 
                    'rho': [0, 100],
                    'algo': [2, 1],
                },

                { 
                    'sigma_mu': [0.0, 1.0], 
                    'rho': [0, 100],
                    'algo': [2, 1],
                },

                {
                    'epsilon_rho': [0, 0.005, 0.01, 0.02, 0.05],
                    'rho': [100],
                    'algo': [2, 1],
                },


                {
                    'N_tot': [5_800_000],
                    'rho': [0, 25, 50, 100, 150, 200, 300],
                    'algo': [2],
                },


                {
                    'N_tot': [5_800_000],
                    'sigma_beta': [0.0, 1.0],
                    'rho': [0, 100],
                    '   ': [0],
                    'algo': [2],
                    'beta': [0.01],
                },


                {
                    'N_tot': [580_000],
                    'sigma_beta': [0.0, 1.0],
                    'rho': [0, 100],
                    'algo': [2],
                    'beta': [0.01],
                },

                {
                    'N_tot': [5_800_000],
                    'sigma_beta': [0.0, 1.0],
                    'rho': [0, 100],
                    'algo': [2],
                    'beta': [0.01],
                },

                {
                    'N_tot': [1_000_000],
                },

                {
                    'N_tot': [2_000_000],
                },

                {
                    'N_tot': [5_000_000],
                },

                {
                    'N_tot': [5_800_000],
                },


                # {
                #     'N_ages': [1, 3, 10],
                #     'age_mixing': [0, 0.25, 0.5, 0.75, 1],
                #     'sigma_mu': [0, 0.5, 1],
                # },


                # {
                #     'N_init': [1_000],
                #     'N_ages': [10],
                #     'age_mixing': [0.5],
                # },


                ]


#%%

N_loops = 2 if extra_funcs.is_local_computer() else N_loops

N_files_total = 0
# reload(extra_funcs)
if __name__ == '__main__':

    for d_sim_pars in all_sim_pars:
        filenames = extra_funcs.generate_filenames(d_sim_pars, N_loops, force_overwrite=force_overwrite)

        N_files = len(filenames)
        N_files_total += N_files

        # make sure path exists
        if len(filenames) > 0:
            num_cores = extra_funcs.get_num_cores_N_tot_specific(d_sim_pars, num_cores_max)
            print(f"\nGenerating {N_files:3d} network-based simulations with {num_cores} cores based on {d_sim_pars}, please wait.", flush=True)

            if dry_run:
                continue

            if num_cores == 1:
                for filename in tqdm(filenames):
                    extra_funcs.single_run_and_save(filename, verbose=verbose)

            else:
                with mp.Pool(num_cores) as p:
                    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print(f"\nIn total: {N_files_total} files generated")
    print("Finished simulating!")
