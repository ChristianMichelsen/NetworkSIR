# mprof run --include-children python numba_numpy_list_test.py



# conda install -c anaconda memory_profiler
from numba import njit
import numpy as np
from numba.typed import List
import time

@njit
def get_size_gb(x):
    return x.size * x.itemsize / 10**9

N_test = 1_000_000_000

@njit
def get_np(N_test):
    return np.arange(N_test)

def f_np():
    x_np = get_np(N_test)
    print('waiting 5 seconds', flush=True)
    time.sleep(5)
    

@njit
def get_nb(N_test):
    x_nb = List()
    for i in range(N_test):
        x_nb.append(i)
    return x_nb

def f_nb():
    x_nb = get_nb(N_test)
    print('waiting 5 seconds', flush=True)
    time.sleep(5)


def main():

    _ = get_nb(10)

    f_np()
    print('waiting 1 seconds', flush=True)
    time.sleep(1)

    f_nb()
    print('waiting 1 seconds', flush=True)
    time.sleep(1)

main()

