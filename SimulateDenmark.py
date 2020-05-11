import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmark_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 8
N_loops = 100
N_Denmark = extra_funcs.N_Denmark # 535_806

# x=x

all_sim_pars = [

                {   'BB': [0, 1],
                }, 

                { 
                    'sigma': [0.0, 0.5, 1.0], 
                    'gamma': [0.0, 0.5, 1.0],
                    'BB': [0, 1],
                },

                {   'N0': [N_Denmark, N_Denmark//10],
                    'Ninit': [10, 100, 1000],
                    'BB': [0, 1],
                }, 


                { 
                    'alpha': [1, 2, 4, 6, 8, 15, 20],
                    'BB': [0, 1],
                },

                {
                    'beta': [0.005, 0.01, 0.02, 0.05, 0.1],
                    'BB': [0, 1],
                },

                {
                    'mu': [5, 10, 20, 40, 80],
                    'BB': [0, 1],
                },

                {
                    'Mrate1': [0.5, 1, 2, 4],
                    'BB': [0, 1],
                },

                {
                    'Mrate2': [0.5, 1, 2, 4],
                    'BB': [0, 1],
                },

                {
                    'N0': [10_000, 50_000, 100_000, 500_000],
                    'BB': [0, 1],
                },

                ]


#%%

# filename = 'Data/NetworkSimulation/N0_535806_mu_20.0_alpha_20.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_100/N0_535806_mu_20.0_alpha_20.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_100_ID_000.csv'

# extra_funcs.single_run_and_save(filename)

# x=x

# reload(extra_funcs)

if __name__ == '__main__':

    for d_simulation_parameters in all_sim_pars:

        N_loops = 1 if extra_funcs.is_local_computer() else N_loops
        # N_loops = 100 if extra_funcs.is_local_computer() else N_loops
        filenames = extra_funcs.generate_filenames(d_simulation_parameters, N_loops, force_SK_P1_UK=False)
        N_files = len(filenames)
        # filename = filenames[0]

        print(d_simulation_parameters, N_files)
        # continue

        # if 'N0' in d_simulation_parameters.keys() and np.max(d_simulation_parameters['N0']) > 100_000:
        #     num_cores = extra_funcs.get_num_cores(10)
        # else:
        num_cores = extra_funcs.get_num_cores(num_cores_max)

        # make sure path exists
        if len(filenames) > 0:
            filename = filenames[0]
            print(f"Generating {N_files} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.", flush=True)

            if num_cores == 1:
                for filename in tqdm(filenames):
                    extra_funcs.single_run_and_save(filename)

            else:
                with mp.Pool(num_cores) as p:
                    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print("Finished simulating!")


