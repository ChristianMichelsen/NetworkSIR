import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 38
N_loops = 10


if __name__ == '__main__':

    d_sim_par1 = {}

    d_sim_par2 = {
            'sigma': [0.0, 0.5, 1.0], 
            'gamma': [0.0, 0.5, 1.0],
        }
    
    d_sim_par3 = {
            'alpha': [1, 2, 4, 8]
            # 'psi':   [0.15, 0.3, 0.45], 
        }

    all_pars = [d_sim_par1, d_sim_par2, d_sim_par3]
    filenames = []
    for d_simulation_parameters in all_pars:
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
        with mp.Pool(num_cores) as p:
            list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
    else:
        print("No files to generate, everything already generated.")

    print("Finished simulating!")
