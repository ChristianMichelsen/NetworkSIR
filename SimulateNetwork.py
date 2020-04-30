import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path
from importlib import reload

num_cores_max = 38

def is_local_computer(N_local_cores=8):
    import platform
    if mp.cpu_count() <= N_local_cores and platform.system() == 'Darwin':
        return True
    else:
        return False

def generate_filenames(N_loops=10, force_overwrite=False):
    filenames = []
    dict_in = dict(
                    N0 = 10_000 if is_local_computer() else 500_000,
                    mu = 20.0,  # Average number connections
                    alpha = 0.0, # Spatial parameter
                    psi = 0.0, # cluster effect
                    beta = 1.0, # Mean rate
                    sigma = 0.0, # Spread in rate
                    Ninit = 10, # Initial Infected
                    Mrate1 = 1.0, # E->I
                    Mrate2 = 1.0, # I->R
                    gamma = 0.0, # Parameter for skewed connection shape
                    nts = 0.1, 
                    Nstates = 9,
                )

    # for gamma in [0.15, 0.3, 0.45]:
    #     dict_in['gamma'] = gamma

    #     for sigma in [0.15, 0.3, 0.45]:
    #         dict_in['sigma'] = sigma

    for alpha in [1, 2, 4, 8]:
        dict_in['alpha'] = alpha

        for psi in [0, 1, 4]:
            dict_in['psi'] = psi

            for ID in range(N_loops):
                filename = extra_funcs.dict_to_filename(dict_in, ID)
                if not Path(filename).exists() or force_overwrite:
                    filenames.append(filename)
        
    return filenames


if __name__ == '__main__':

    N_loops = 1 if is_local_computer() else 100
    filenames = generate_filenames(N_loops)
    # filenames = filenames[:20]
    N_files = len(filenames)
    # extra_funcs.single_run_and_save(filenames[0])

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max
    
    # filename = filenames[0]
    # make sure path exists
    if len(filenames) > 0:
        Path(filenames[0]).parent.mkdir(parents=True, exist_ok=True)

        print(f"Generating {N_files} network-based simulations, please wait.", flush=True)
        with mp.Pool(num_cores) as p:
            list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))
    else:
        print("No files to generate, everything already generated.")

    print("Finished simulating!")


# SK, P1, UK
