from tqdm import tqdm
import multiprocessing as mp
import NewSpeedImprove_extra_funcs as extra_funcs


def generate_filenames(N_loops=10):
    filenames = []
    dict_in = dict(
                    N0 = 50_000,
                    mu = 20.0,  # Average number connections
                    alpha = 0.0, # Spatial parameter
                    beta = 1.0, # Mean rate
                    sigma = 0.8, # Spread in rate
                    Ninit = 10, # Initial Infected
                    # Mrate1 = 1.2, # E->I
                    # Mrate2 = 1.2, # I->R
                    gamma = 0.0, # Parameter for skewed connection shape
                    delta = 0.05, # Minimum probability to connect
                    nts = 0.1, 
                    Nstates = 9,
                )

    for Mrate1 in [0.5, 1, 2]:
        dict_in['Mrate1'] = Mrate1

        for Mrate2 in [1]:
            dict_in['Mrate2'] = Mrate2

            for ID in range(N_loops):
                filename = extra_funcs.dict_to_filename(dict_in, ID)
                filenames.append(filename)
        
    return filenames


filenames = generate_filenames(N_loops=1000)
N_files = len(filenames)
# extra_funcs.single_run_and_save(filenames[0])

num_cores = mp.cpu_count() - 1

print(f"Generating {N_files} network-based simulations, please wait.", flush=True)
with mp.Pool(num_cores) as p:
    list(tqdm(p.imap_unordered(extra_funcs.single_run_and_save, filenames), total=N_files))

print("Finished simulating!")