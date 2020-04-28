from tqdm import tqdm
import multiprocessing as mp
import SimulateNetwork_extra_funcs as extra_funcs
from pathlib import Path

def generate_filenames(N_loops=10, force_overwrite=False):
    filenames = []
    dict_in = dict(
                    N0 = 50_000,
                    mu = 20.0,  # Average number connections
                    alpha = 1.0, # Spatial parameter
                    psi = 2.0, # cluster effect
                    beta = 1.0, # Mean rate
                    sigma = 0.8, # Spread in rate
                    Ninit = 10, # Initial Infected
                    Mrate1 = 1.0, # E->I
                    Mrate2 = 1.0, # I->R
                    gamma = 0.047, # Parameter for skewed connection shape
                    delta = 0.05, # Minimum probability to connect
                    nts = 0.1, 
                    Nstates = 9,
                )

    for psi in [1, 2, 4]:
        dict_in['psi'] = psi
        alphas = [1, 2, 4, 8]
        if psi == 0:
            alphas = alphas + [0]

        for alpha in alphas:
            dict_in['alpha'] = alpha

            for ID in range(N_loops):
                filename = extra_funcs.dict_to_filename(dict_in, ID)
                if not Path(filename).exists() or force_overwrite:
                    filenames.append(filename)
        
    return filenames


if __name__ == '__main__':

    filenames = generate_filenames(N_loops=1000)
    N_files = len(filenames)
    # extra_funcs.single_run_and_save(filenames[0])

    num_cores = mp.cpu_count() - 1
    num_cores_max = 30
    if num_cores >= num_cores_max:
        num_cores = num_cores_max

    print(f"Generating {N_files} network-based simulations, please wait.", flush=True)
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))

    print("Finished simulating!")


