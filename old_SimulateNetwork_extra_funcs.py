import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import joblib
import multiprocessing as mp
from itertools import product

# conda install awkward
# conda install -c conda-forge pyarrow
import awkward

def is_local_computer(N_local_cores=8):
    import platform
    if mp.cpu_count() <= N_local_cores and platform.system() == 'Darwin':
        return True
    else:
        return False

def generate_filenames(d, N_loops=10, force_overwrite=False, force_SK_P1_UK=False):
    filenames = []
    cfg = dict(
                    N0 = 50_000 if is_local_computer() else 50_000,
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
                    BB = 1,
                )

    nameval_to_str = [[f'{name}_{x}' for x in lst] for (name, lst) in d.items()]
    all_combinations = list(product(*nameval_to_str))

    for combination in all_combinations:
        for s in combination:
            name, val = s.split('_')
            val = float(val) if name != 'N0' else int(val)
            cfg[name] = val


        cfg['Ninit'] = max(1, int(cfg['N0'] * 0.1 / 1000)) # Initial Infected, 1 permille        
        for ID in range(N_loops):
            filename = dict_to_filename_with_dir(cfg, ID)

            if ID == 0 and force_SK_P1_UK:
                filenames.append(filename)

            elif not Path(filename).exists() or force_overwrite:
                filenames.append(filename)
        
    return filenames


@njit
def deep_copy_1D_int(X):
    outer = np.zeros(len(X), np.int_)
    for ix in range(len(X)):
        outer[ix] = X[ix]
    return outer

@njit
def deep_copy_2D_jagged_int(X, min_val=-1):
    outer = []
    n, m = X.shape
    for ix in range(n):
        inner = []
        for jx in range(m):
            if X[ix, jx] > min_val:
                inner.append(int(X[ix, jx]))
        outer.append(inner)
    return outer


@njit
def deep_copy_2D_jagged(X, min_val=-1):
    outer = []
    n, m = X.shape
    for ix in range(n):
        inner = []
        for jx in range(m):
            if X[ix, jx] > min_val:
                inner.append(X[ix, jx])
        outer.append(inner)
    return outer


@njit
def deep_copy_2D_int(X):
    n, m = X.shape
    outer = np.zeros((n, m), np.int_)
    for ix in range(n):
        for jx in range(m):
            outer[ix, jx] = X[ix, jx]
    return outer

# @njit
# def setdiff1d(X, Y):
#     x_set = set(X)
#     y_set = set(Y)
#     return x_set - y_set

@njit
def create_P1(N0):
    xx = 0.0 
    yy = 0.0 
    psi_epsilon = 1e-2
    psi = 0

    D0 = 0.01 
    D = D0*100

    dt = 0.01 

    tnext = (1/np.random.random())**(1/(psi+psi_epsilon))-1
    # rD = np.sqrt(2*D0*dt*N0) / 10
    rD = 1
    
    RT = 0

    P1 = np.zeros((N0, 2))

    for i in range(N0):
        RT += dt
        acc = 0;
        while acc == 0:
            if (RT > tnext):
                dx = np.sqrt(2*D*dt)*np.random.normal()
                dy = np.sqrt(2*D*dt)*np.random.normal()
            else:
                dx = np.sqrt(2*D0*dt)*np.random.normal()
                dy = np.sqrt(2*D0*dt)*np.random.normal()
            r = np.sqrt( (xx + dx)**2 + (yy + dy)**2)
            if (r < rD):
                acc = 1
                xx += dx
                yy += dy
                if (RT > tnext):    
                    ra = (1/np.random.random())**(1/(psi+psi_epsilon))-1  
                    tnext = RT + ra

        P1[i, 0] = xx
        P1[i, 1] = yy
    return P1



@njit
def single_run_numba(N0, mu, alpha, psi, beta, sigma, Ninit, Mrate1, Mrate2, gamma, nts, Nstates, BB, P1):

    # N0 = 1000
    # mu = 20.0  # Average number connections
    # alpha = 0.0 # Spatial parameter
    # psi = 0.0 # cluster effect
    # beta = 0.01 # Mean rate
    # sigma = 0.0 # Spread in rate
    # Mrate1 = 1.0 # E->I
    # Mrate2 = 1.0 # I->R
    # gamma = 0.0 # Parameter for skewed connection shape
    # nts = 0.1 
    # Nstates = 9
    # BB = 1
    # Ninit = max(1, int(N0 * 0.1 / 1000))
    ID = 0

    np.random.seed(ID+1)

    NRe = N0


    N_AK_MAX = 1000
    # For generating Network
    AK = -1*np.ones((N0, N_AK_MAX), np.int_)
    UK = np.zeros(N0, np.int_)
    UKRef = np.zeros(N0, np.int_)
    # DK = np.zeros(N0, np.int_)
    Prob = np.ones(N0)
    Sig = np.ones(N0)
    WMa = np.ones(N0)
    SK = -1*np.ones(N0, np.int_)
    AKRef = -1*np.ones((N0, N_AK_MAX), np.int_)
    Rate = -1*np.ones((N0, N_AK_MAX))
    SAK = -1*np.ones((Nstates, N0), np.int_)
    S = np.zeros(Nstates, np.int_)
    Par = np.zeros(Nstates)
    csMov = np.zeros(Nstates)
    csInf = np.zeros(Nstates)
    InfRat = np.zeros(N0)

    Ninfectious = 4 # This means the 5'th state
    # For simulating Actual Disease
    NExp = 0 
    NInf = 0

    Par[:4] = Mrate1
    Par[4:8] = Mrate2

    # Here we initialize the system
    xx = 0.0 
    yy = 0.0 
    psi_epsilon = 1e-2
    tnext = (1/np.random.random())**(1/(psi+psi_epsilon))-1


    # D0 = 0.01 
    # D = D0*100


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RATES AND CONNECTIONS # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # dt = 0.01 
    # RT = 0
    # rD = np.sqrt(2*D0*dt*N0) / 10
    rD = 1

    for i in range(N0):
        ra = np.random.rand()
        if (ra < gamma):
            Prob[i] = 0.1 -np.log( np.random.rand())/1.0
        else:
            Prob[i] = 1.1;
        ra = np.random.rand()
        if (ra < sigma):
            rat = -np.log(np.random.rand())*beta
            Sig[i] = rat;
        else:
            Sig[i] = beta;

    PT = np.sum(Prob)
    PC = np.cumsum(Prob);
    PP = PC/PT


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # SPATIAL # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    print(P1)

    # print(P1.shape)
    # print(P1.dtype)
    
    # P1 = create_P1(N0)
    # P1 = np.random.normal(size=(N0, 2))

    # print("\n")
    # print(P1.shape)
    # print(P1.dtype)
    # if P1 is None:

    #     P1 = np.zeros((N0, 2))

    #     for i in range(NRe):
    #         RT += dt
    #         acc = 0;
    #         while acc == 0:
    #             if (RT > tnext):
    #                 dx = np.sqrt(2*D*dt)*np.random.normal()
    #                 dy = np.sqrt(2*D*dt)*np.random.normal()
    #             else:
    #                 dx = np.sqrt(2*D0*dt)*np.random.normal()
    #                 dy = np.sqrt(2*D0*dt)*np.random.normal()
    #             r = np.sqrt( (xx + dx)**2 + (yy + dy)**2)
    #             if (r < rD):
    #                 acc = 1
    #                 xx += dx
    #                 yy += dy
    #                 if (RT > tnext):    
    #                     ra = (1/np.random.random())**(1/(psi+psi_epsilon))-1  
    #                     tnext = RT + ra

    #         P1[i, 0] = xx
    #         P1[i, 1] = yy



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # CONNECT NODES # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    print("Connecting Nodes")

    if (BB == 0):
        for c in range(int(mu*NRe)):
            accra = 0
            while accra == 0:
                ra1 = np.random.rand()
                ra2 = np.random.rand()            
                id1 = np.searchsorted(PP,ra1);
                id2 = np.searchsorted(PP,ra2);
                acc = 1
                for i1 in range(UK[id1]):         #  Make sure no element is present twice
                    if AK[id1, i1] == id2:
                        acc = 0         
                if (UK[id1] < N_AK_MAX) and (UK[id2] < N_AK_MAX) and (id1 != id2) and (acc == 1):
                    
                    r = np.sqrt((P1[id1, 0] - P1[id2, 0])**2 + (P1[id1, 1] - P1[id2, 1])**2)


                    ra = np.random.rand()
                    if np.exp(-alpha*r/rD) > ra:
                        rat = -np.log(np.random.rand())*beta
                        Rate[id1, UK[id1]] = Sig[id1]
                        Rate[id2, UK[id2]] = Sig[id1]

                        AK[id1, UK[id1]] = id2	        
                        AKRef[id1, UK[id1]] = id2                        
                        AK[id2, UK[id2]] = id1 	
                        AKRef[id2, UK[id2]] = id1

                        UK[id1] += 1 
                        UK[id2] += 1
                        UKRef[id1] += 1 
                        UKRef[id2] += 1
                        accra = 1                    
    else:
        for c in range(int(mu*NRe)):
            ra1 = np.random.rand()
            id1 = np.searchsorted(PP, ra1) 
            accra = 0
            while accra == 0:
                ra2 = np.random.rand()          
                id2 = np.searchsorted(PP, ra2)
                acc = 1
                #  Make sure no element is present twice
                for i1 in range(UK[id1]):               
                    if AK[id1, i1] == id2:
                        acc = 0
                if (UK[id1] < N_AK_MAX) and (UK[id2] < N_AK_MAX) and (id1 != id2) and (acc == 1):
                    r = np.sqrt((P1[id1, 0] - P1[id2, 0])**2 + (P1[id1, 1] - P1[id2, 1])**2)
                    ra = np.random.rand()
                    if np.exp(-alpha*r/rD) > ra:
                        rat = -np.log(np.random.rand())*beta
                        Rate[id1, UK[id1]] = Sig[id1]
                        Rate[id2, UK[id2]] = Sig[id1]

                        AK[id1, UK[id1]] = id2
                        AKRef[id1, UK[id1]] = id2
                        AK[id2, UK[id2]] = id1
                        AKRef[id2, UK[id2]] = id1

                        UK[id1] += 1
                        UK[id2] += 1
                        UKRef[id1] += 1
                        UKRef[id2] += 1
                        accra = 1


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    on = 1  
    Tot = 0  
    TotMov = 0 
    TotInf = 0  
    click = 0 
    c = 0  
    Csum = 0 
    RT = 0 

    ##  Now make initial infectious
    for iin in range(Ninit):
        idx = iin*10
        SK[idx] = 0  
        SAK[0, S[0]] = idx
        S[0] += 1  
        # DK[idx] = 1  
        TotMov += Par[0]
        csMov[:] += Par[0]
        for i1 in range(UKRef[idx]):
            Af = AKRef[idx, i1]
            for i2 in range(UK[Af]):
                if AK[Af, i2] == idx:
                    for i3 in range(i2, UK[Af]):
                        AK[Af, i3] = AK[Af, i3+1] 
                        Rate[Af, i3] = Rate[Af, i3+1]
                    UK[Af] -= 1 
                    break 


    #   #############/

    SIRfile = []
    SIRfile_SK = []
    SIRfile_UK = []
    # SIRfile_AK = []
    # SIRfile_AK.append(deep_copy_2D_int(AK))
    SIRfile_AK = deep_copy_2D_jagged_int(AK)
    SIRfile_Rate = deep_copy_2D_jagged(Rate)
    

    # SIRfile_Rate = []
    SK_UK_counter = 0


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # RUN SIMULATION  # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


    # Run the simulation ################################
    while on == 1:
        
        c += 1 
        Tot = TotMov + TotInf
        ra1 = np.random.rand()   
        dt = - np.log(ra1)/Tot    
        RT = RT + dt
        Csum = 0 
        ra1 = np.random.rand()
        #######/ Here we move infected between states

        AC = 0 
        if TotMov/Tot > ra1:
            x = csMov/Tot
            i1 = np.searchsorted(x, ra1)
            Csum = csMov[i1]/Tot
            for i2 in range(S[i1]):
                Csum += Par[i1]/Tot
                if Csum > ra1:
                    idx = SAK[i1, i2]
                    AC = 1
                    break                
            
            # We have chosen idx to move -> here we move it
            SAK[i1+1, S[i1+1]] = idx
            for j in range(i2, S[i1]):
                SAK[i1, j] = SAK[i1, j+1] 

            SK[idx] += 1
            S[i1] -= 1 
            S[i1+1] += 1      
            TotMov -= Par[i1] 
            TotMov += Par[i1+1]     
            csMov[i1] -= Par[i1]
            csMov[i1+1:Nstates] += (Par[i1+1]-Par[i1])
            csInf[i1] -= InfRat[idx]

            if SK[idx] == Ninfectious: # Moves TO infectious State from non-infectious
                for i1 in range(UK[idx]): # Loop over row idx	  
                    if SK[AK[idx, i1]] < 0:
                        TotInf += Rate[idx, i1]
                        InfRat[idx] += Rate[idx, i1]
                        csInf[SK[idx]:Nstates] += Rate[idx, i1]
            if SK[idx] == Nstates-1: # If this moves to Recovered state
                for i1 in range(UK[idx]): # Loop over row idx
                    TotInf -= Rate[idx, i1] 
                    InfRat[idx] -= Rate[idx, i1]
                    csInf[SK[idx]:Nstates] -= Rate[idx, i1]


        # Here we infect new states
        else:
            x = TotMov/Tot + csInf/Tot
            i1 = np.searchsorted(x, ra1)
            Csum = TotMov/Tot + csInf[i1]/Tot
            for i2 in range(S[i1]):
                idy = SAK[i1, i2]
                for i3 in range(UK[idy]): 
                    Csum += Rate[idy][i3]/Tot
                    if Csum > ra1:
                        idx = AK[idy, i3]	      
                        SK[idx] = 0 
                        # NrDInf += 1
                        SAK[0, S[0]] = idx	      
                        S[0] += 1
                        TotMov += Par[0]	      
                        csMov[:] += Par[0]
                        AC = 1
                        break                    
                if AC == 1:
                    break
            # Here we update infection lists      
            for i1 in range(UKRef[idx]):
                acc = 0
                Af = AKRef[idx, i1]
                for i2 in range(UK[Af]):
                    if AK[Af, i2] == idx:
                        if (SK[Af] >= Ninfectious) and (SK[Af] < Nstates-1):	      
                            TotInf -= Rate[Af, i2]
                            InfRat[Af] -= Rate[Af, i2]
                            csInf[SK[Af]:Nstates] -= Rate[Af, i2]
                        for i3 in range(i2, UK[Af]):
                            AK[Af, i3] = AK[Af, i3+1]
                            Rate[Af, i3] = Rate[Af, i3+1]
                        UK[Af] -= 1 
                        break

        ################

        if nts*click < RT:
            SIRfile_tmp = np.zeros(Nstates + 1)
            icount = 0
            SIRfile_tmp[icount] = RT
            for s in S:
                icount += 1
                SIRfile_tmp[icount] = s #<< "\t"
            SIRfile.append(SIRfile_tmp)
            SK_UK_counter += 1

            if SK_UK_counter >= 10:
                SK_UK_counter = 0

                # deepcopy
                SIRfile_SK.append(deep_copy_1D_int(SK))
                SIRfile_UK.append(deep_copy_1D_int(UK))

            click += 1 

    
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # BUG CHECK  # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        if c > 10000000: 
            on = 0
        
        if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001): 
            on = 0 
            # print("Equilibrium")
        
        if S[Nstates-1] > N0-10:      
            # print("2/3 through")
            on = 0

        # Check for bugs
        if AC == 0: 
            print("No Chosen rate", csMov)
            on = 0
        
        if (TotMov < 0) and (TotMov > -0.001):
            TotMov = 0 
            
        if (TotInf < 0) and (TotInf > -0.001):
            TotInf = 0 
            
        if (TotMov < 0) or (TotInf < 0): 
            print("\nNegative Problem", TotMov, TotInf)
            print(alpha, beta, gamma)
            on = 0 
    
    return SIRfile, SIRfile_SK, P1, SIRfile_UK, SIRfile_AK, SIRfile_Rate




def dict_to_filename_with_dir(cfg, ID):
    filename = Path('Data') / 'NetworkSimulation' 
    file_string = ''
    for key, val in cfg.items():
        file_string += f"{key}_{val}_"
    file_string = file_string[:-1] # remove trailing _
    filename = filename / file_string
    file_string += f"_ID_{ID:03d}.csv"
    filename = filename / file_string
    return str(filename)


def filename_to_dict(filename, normal_string=False, SK_P1_UK=False):
    cfg = {}
    if normal_string:
        keyvals = filename.split('_')
    elif SK_P1_UK:
        keyvals = filename.split('/')[-1].split('_')[:-2]
    else:
        # keyvals = filename.split('/')[2].split('_')
        keyvals = filename.split('/')[2].split('_')

    keyvals_chunks = [keyvals[i:i + 2] for i in range(0, len(keyvals), 2)]
    ints = ['N0', 'Ninit', 'Nstates', 'BB']
    for key, val in keyvals_chunks:
        if not key == 'ID':
            if key in ints:
                cfg[key] = int(val)
            else:
                cfg[key] = float(val)
    return cfg



def single_run_and_save(filename):


    # TODO
    filename = 'Data/NetworkSimulation/N0_10000_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_1/N0_10000_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_1_ID_000.csv'




    cfg = filename_to_dict(filename)
    ID = filename_to_ID(filename)

    # N0 = 1000
    # cfg['N0'] = N0

    P1 = np.load('Data/GPS_coordinates.npy')

    # df_P1 = pd.read_csv('Data/GPS_coordinates.csv')
    # P1 = df_P1.values
    # P1 = P1[:cfg['N0']]
    # P1 = np.random.normal(size=(N0, 2))

    # P1 = create_P1(N0)

    # res = single_run_numba(**cfg, ID=ID)


    res = single_run_numba(cfg['N0'], 
                            cfg['mu'], 
                            cfg['alpha'], 
                            cfg['psi'], 
                            cfg['beta'], 
                            cfg['sigma'], 
                            cfg['Ninit'], 
                            cfg['Mrate1'], 
                            cfg['Mrate2'], 
                            cfg['gamma'], 
                            cfg['nts'], 
                            cfg['Nstates'], 
                            cfg['BB'], 
                            P1)

    # res = single_run_numba(N0, cfg['mu'], cfg['alpha'], cfg['psi'], cfg['beta'], cfg['sigma'], cfg['Ninit'], cfg['Mrate1'], cfg['Mrate2'], cfg['gamma'], cfg['nts'], cfg['Nstates'], cfg['BB'], ID=0)


    if len(res) == 1:
        out_single_run = res
    elif len(res) == 4:
        out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK = res
    elif len(res) == 5:
        out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK, SIRfile_AK_initial = res
    elif len(res) == 6:
        out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK, SIRfile_AK_initial, SIRfile_Rate_initial = res
    else:
        raise AssertionError('Wrong result from single_run_numba')


    header = ['Time', 
            'E1', 'E2', 'E3', 'E4', 
            'I1', 'I2', 'I3', 'I4', 
            'R',
            ]
    df = pd.DataFrame(out_single_run, columns=header)

    # make sure parent folder exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # save csv file
    # df.to_csv(filename, index=False)
    print("savefile as test.csv")
    df.to_csv('test.csv', index=False)


    import extra_funcs
    import matplotlib.pyplot as plt

    df = extra_funcs.pandas_load_file('test.csv', return_only_df=True)

    fig, ax = plt.subplots()
    ax.plot(df['Time'].values, df['I'].values, '-')
    ax.set(title=f'{N0=}')









    # save SK, P1, and UK, once for each set of parameters
    if ID == 0:
        SIRfile_SK = np.array(SIRfile_SK, dtype=int)
        SIRfile_P1 = np.array(SIRfile_P1)
        SIRfile_UK = np.array(SIRfile_UK, dtype=int)
        
        filename_SK_P1_UK = str(Path('Data_SK_P1_UK') / Path(filename).stem) + '.SK_P1_UK.joblib'

        Path(filename_SK_P1_UK).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump([SIRfile_SK, SIRfile_P1, SIRfile_UK], filename_SK_P1_UK)
        # pickle.dump([SIRfile_SK, SIRfile_P1, SIRfile_UK], open(filename_SK_P1_UK.replace('joblib', 'pickle'), "wb"))

        SIRfile_AK_initial = awkward.fromiter(SIRfile_AK_initial).astype(np.int32)
        filename_AK = filename_SK_P1_UK.replace('SK_P1_UK.joblib', 'AK_initial.parquet')
        awkward.toparquet(filename_AK, SIRfile_AK_initial)

        SIRfile_Rate_initial = awkward.fromiter(SIRfile_Rate_initial)
        filename_Rate = filename_AK.replace('AK_initial.parquet', 'Rate_initial.parquet')
        awkward.toparquet(filename_Rate, SIRfile_Rate_initial)

    return None


def filename_to_ID(filename):
    return int(filename.split('ID_')[1].strip('.csv'))


class DotDict(dict):
    """
    Class that allows a dict to indexed using dot-notation.
    Example:
    >>> dotdict = DotDict({'first_name': 'Christian', 'last_name': 'Michelsen'})
    >>> dotdict.last_name
    'Michelsen'
    """

    def __getattr__(self, item):
        if item in self:
            return self.get(item)
        raise KeyError(f"'{item}' not in dict")

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
            return
        raise KeyError(
            "Only allowed to change existing keys with dot notation. Use brackets instead."
        )


def filename_to_dotdict(filename, normal_string=False, SK_P1_UK=False):
    return DotDict(filename_to_dict(filename, normal_string=normal_string, SK_P1_UK=SK_P1_UK))


def get_num_cores(num_cores_max):
    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max
    return num_cores


def convert_df(df_raw):

    for state in ['E', 'I']:
        df_raw[state] = sum([df_raw[col] for col in df_raw.columns if state in col and len(col) == 2])

    # only keep relevant columns
    df = df_raw[['Time', 'E', 'I', 'R']].copy()
    return df




#%%

# # TODO
# filename = 'Data/NetworkSimulation/N0_10000_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_1/N0_10000_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_1_ID_000.csv'

# cfg = filename_to_dict(filename)
# ID = filename_to_ID(filename)

# N0 = 10000
# cfg['N0'] = N0
# cfg['beta'] *= 10
# cfg['alpha'] = 20
# cfg['Ninit'] = 100

# df_P1 = pd.read_csv('Data/GPS_coordinates.csv')
# P1 = df_P1.values
# P1 = P1[:cfg['N0']]
# # P1 = np.random.normal(size=(N0, 2))

# # P1 = create_P1(N0)

# # res = single_run_numba(**cfg, ID=ID)

# res = single_run_numba(cfg['N0'], 
#                        cfg['mu'], 
#                        cfg['alpha'], 
#                        cfg['psi'], 
#                        cfg['beta'], 
#                        cfg['sigma'], 
#                        cfg['Ninit'], 
#                        cfg['Mrate1'], 
#                        cfg['Mrate2'], 
#                        cfg['gamma'], 
#                        cfg['nts'], 
#                        cfg['Nstates'], 
#                        cfg['BB'], 
#                        P1)


# if len(res) == 1:
#     out_single_run = res
# elif len(res) == 4:
#     out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK = res
# elif len(res) == 5:
#     out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK, SIRfile_AK_initial = res
# elif len(res) == 6:
#     out_single_run, SIRfile_SK, SIRfile_P1, SIRfile_UK, SIRfile_AK_initial, SIRfile_Rate_initial = res
# else:
#     raise AssertionError('Wrong result from single_run_numba')


# header = ['Time', 
#         'E1', 'E2', 'E3', 'E4', 
#         'I1', 'I2', 'I3', 'I4', 
#         'R',
#         ]
# df_raw = pd.DataFrame(out_single_run, columns=header)

# import matplotlib.pyplot as plt

# df = convert_df(df_raw)

# fig, ax = plt.subplots()
# ax.plot(df['Time'].values, df['I'].values, '-')
# ax.set(title=f'{N0=}')

    
    

# %%
