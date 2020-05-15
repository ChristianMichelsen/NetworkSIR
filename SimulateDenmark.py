import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmark_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 8
N_loops = 100
force_SK_P1_UK = False

#%%

all_sim_pars = [

                {   'BB': [0, 1],
                }, 

                {
                    'N0': [10_000, 20_000, 50_000, 100_000, 200_000, 500_000],
                },

                # {
                #     'N0': [1_000_000],
                # },

                { 
                    'sigma': [0.0, 0.5, 1.0], 
                    'gamma': [0.0, 0.5, 1.0],
                    # 'BB': [0, 1],
                },

                {   'N0': [50_000, 500_000],
                    'Ninit': [10, 100, 1000],
                    # 'BB': [0, 1],
                }, 

                {
                    'beta': [0.005, 0.01, 0.02, 0.05, 0.1],
                    # 'BB': [0, 1],
                },

                {
                    'mu': [5, 10, 15, 20, 30, 40, 60, 80],
                    # 'BB': [0, 1],
                },

                {
                    'Mrate1': [0.5, 1, 2, 4],
                    # 'BB': [0, 1],
                },

                {
                    'Mrate2': [0.5, 1, 2, 4],
                    # 'BB': [0, 1],
                },

                { 
                    'alpha': [0, 1, 2, 4, 6, 8, 10, 15, 20],
                    'BB': [0, 1],
                },

                ]


#%%

# import time
# for N0 in [10_000, 100_000, 1_000_000]:
#     print(N0)
#     filename = f'Data/NetworkSimulation/N0_{N0}_mu_20.0_alpha_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_100/N0_{N0}_mu_20.0_alpha_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_100_ID_000.csv'
#     start_time = time.time()
#     extra_funcs.single_run_and_save(filename)
#     print("--- %s seconds ---" % (time.time() - start_time))

# x=x

# reload(extra_funcs)
if __name__ == '__main__':

    # num_cores = extra_funcs.get_num_cores(num_cores_max)
    N_loops = 1 if extra_funcs.is_local_computer() else N_loops

    all_filenames = extra_funcs.get_filenames_iter(all_sim_pars, force_SK_P1_UK, N_loops)

    for filenames, d_simulation_parameters in zip(*all_filenames):
        N_files = len(filenames)

        # make sure path exists
        if len(filenames) > 0:
            # filename = filenames[0]
            
            if 'N0' in d_simulation_parameters.keys() and max(d_simulation_parameters['N0']) > 600_000:
                num_cores = 1
            else:
                num_cores = extra_funcs.get_num_cores(num_cores_max)

            print(f"Generating {N_files:3d} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.", flush=True)

            # if 'N0' in d_simulation_parameters.keys():
                # break

            if num_cores == 1:
                for filename in tqdm(filenames):
                    extra_funcs.single_run_and_save(filename)

            else:
                with mp.Pool(num_cores) as p:
                    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print("Finished simulating!")


