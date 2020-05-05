import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 30
N_loops = 100


all_sim_pars = [

                # {'N0': [50]},

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

                # {
                #     'N0': [51_000],
                #     },

                ]


#%%

# filename = 'Data/NetworkSimulation/N0_51000_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_5/N0_51000_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_5_ID_000.csv'

if __name__ == '__main__':

    for d_simulation_parameters in all_sim_pars:

        N_loops = 1 if extra_funcs.is_local_computer() else N_loops
        filenames = extra_funcs.generate_filenames(d_simulation_parameters, N_loops, force_SK_P1_UK=True)
        N_files = len(filenames)


        if 'N0' in d_simulation_parameters.keys() and np.max(list(d_simulation_parameters.values())) > 100_000:
            num_cores = extra_funcs.get_num_cores(10)
        else:
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



        # filenames.extend(extra_funcs.generate_filenames(d_simulation_parameters, N_loops))
    # N_files = len(filenames)

# %%
