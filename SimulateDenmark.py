import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmark_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 10
N_loops = 100
force_SK_P1_UK = False

#%%

all_sim_pars = [

                {   'BB': [0, 1],
                }, 

                {
                    'N0': [50_000, 100_000, 200_000, 500_000],
                },

                { 
                    'sigma': [0.0, 0.5, 1.0], 
                    'gamma': [0.0, 0.5, 1.0],
                },

                {   'N0': [50_000, 500_000],
                    'Ninit': [10, 100, 1000],
                }, 

                {
                    'beta': [0.005, 0.01, 0.02, 0.05, 0.1],
                },

                {
                    'mu': [5, 10, 15, 20, 30, 40, 60, 80],
                },

                {
                    'Mrate1': [0.5, 1, 2, 4],
                },

                {
                    'Mrate2': [0.5, 1, 2, 4],
                },

                { 
                    'alpha': [0, 1, 2, 4, 6, 8, 10, 15, 20],
                    'BB': [0, 1],
                },

                {
                    'N0': [1_000_000, 2_000_000, 5_000_000],
                },

                ]


# x=x

#%%

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
            
            if isinstance(d_simulation_parameters, dict) and 'N0' in d_simulation_parameters.keys() and max(d_simulation_parameters['N0']) > 600_000:
                num_cores = 1
            elif isinstance(d_simulation_parameters, str):
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




    # filename = 'Data/NetworkSimulation/N0_2000000_mu_20.0_alpha_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_100/N0_2000000_mu_20.0_alpha_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_100_ID_000.csv'


    # cfg = extra_funcs.filename_to_dict(filename)
    # ID = extra_funcs.filename_to_ID(filename)

    # P1 = np.load('Data/GPS_coordinates.npy')
    # if cfg.N0 > len(P1):
    #     raise AssertionError("N0 cannot be larger than P1 (number of houses in DK)")

    # np.random.seed(ID)
    # index = np.arange(len(P1))
    # index_subset = np.random.choice(index, cfg.N0, replace=False)
    # P1 = P1[index_subset]

    # res = extra_funcs.single_run_numba(**cfg, ID=ID, P1=P1, verbose=True)
    # out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK, SIRfile_AK_initial, SIRfile_Rate_initial = res

    # header = ['Time', 
    #         'E1', 'E2', 'E3', 'E4', 
    #         'I1', 'I2', 'I3', 'I4', 
    #         'R',
    #         ]

    # import pandas as pd
    # df_raw = pd.DataFrame(out_single_run, columns=header).convert_dtypes()

    # df_raw.to_csv('test_2_000_000.csv', index=False) 
