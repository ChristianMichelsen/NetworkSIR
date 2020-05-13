import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateDenmark_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 8
N_loops = 100
N_Denmark = extra_funcs.N_Denmark # 535_806
force_SK_P1_UK = True

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


                { 
                    'alpha': [1, 2, 4, 6, 8, 15, 20],
                    'BB': [0, 1],
                },

                ]


#%%

# x=x

def get_filenames_iter(all_sim_pars, force_SK_P1_UK, N_loops):
    all_filenames = []
    if not force_SK_P1_UK:
        d_sim_pars = []
        for d_simulation_parameters in all_sim_pars:
            filenames = extra_funcs.generate_filenames(d_simulation_parameters, N_loops, force_SK_P1_UK=force_SK_P1_UK)
            all_filenames.append(filenames)
            d_sim_pars.append(d_simulation_parameters)
    else:
        for d_simulation_parameters in all_sim_pars:
            all_filenames.extend(extra_funcs.generate_filenames(d_simulation_parameters, N_loops, force_SK_P1_UK=force_SK_P1_UK))
        all_filenames = [all_filenames]
        d_sim_pars = ['all configurations']
    return all_filenames, d_sim_pars

# reload(extra_funcs)
if __name__ == '__main__':

    num_cores = extra_funcs.get_num_cores(num_cores_max)
    N_loops = 1 if extra_funcs.is_local_computer() else N_loops

    all_filenames = get_filenames_iter(all_sim_pars, force_SK_P1_UK, N_loops)

    for filenames, d_simulation_parameters in zip(*all_filenames):
        N_files = len(filenames)
        # filenames = extra_funcs.generate_filenames(d_simulation_parameters, N_loops, force_SK_P1_UK=force_SK_P1_UK)
        # filename = filenames[0]

        # print(d_simulation_parameters, N_files)
        # continue

        # make sure path exists
        if len(filenames) > 0:
            filename = filenames[0]
            print(f"Generating {N_files} network-based simulations with {num_cores} cores based on {d_simulation_parameters}, please wait.", flush=True)
            # print(f"Generating {N_files} network-based simulations with {num_cores} cores, please wait.", flush=True)

            if num_cores == 1:
                for filename in tqdm(filenames):
                    extra_funcs.single_run_and_save(filename)

            else:
                with mp.Pool(num_cores) as p:
                    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
        else:
            print("No files to generate, everything already generated.")

    print("Finished simulating!")


