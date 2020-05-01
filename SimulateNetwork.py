import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from itertools import product
from importlib import reload

num_cores_max = 38

def is_local_computer(N_local_cores=8):
    import platform
    if mp.cpu_count() <= N_local_cores and platform.system() == 'Darwin':
        return True
    else:
        return False

def generate_filenames(d, N_loops=10, force_overwrite=False):
    filenames = []
    dict_in = dict(
                    N0 = 10_000 if is_local_computer() else 50_000,
                    mu = 20.0,  # Average number connections
                    alpha = 0.0, # Spatial parameter
                    psi = 0.0, # cluster effect
                    beta = 0.01, # Mean rate
                    sigma = 0.0, # Spread in rate
                    Mrate1 = 1.0, # E->I
                    Mrate2 = 1.0, # I->R
                    gamma = 0.0, # Parameter for skewed connection shape
                    nts = 0.1, 
                    Nstates = 9,
                )

    dict_in['Ninit'] = int(dict_in['N0'] * 0.1 / 1000) # Initial Infected, 1 permille

    nameval_to_str = [[f'{name}_{x}' for x in lst] for (name, lst) in d.items()]
    all_combinations = list(product(*nameval_to_str))

    for combination in all_combinations:
        for s in combination:
            name, val = s.split('_')
            val = float(val)
            dict_in[name] = val
        
        for ID in range(N_loops):
            filename = extra_funcs.dict_to_filename_with_dir(dict_in, ID)
            if not Path(filename).exists() or force_overwrite:
                filenames.append(filename)
        
    return filenames



if __name__ == '__main__':

    d = {'alpha': [0.15, 0.3, 0.45], 
        'gamma': [0.0, 0.5, 1.0],
        }

    N_loops = 1 if is_local_computer() else 10
    filenames = generate_filenames(d, N_loops)
    # filenames = filenames[:20]
    N_files = len(filenames)
    # extra_funcs.single_run_and_save(filenames[0])

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


# SK, P1, UK
