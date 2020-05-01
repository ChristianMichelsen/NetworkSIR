import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import joblib
import multiprocessing as mp
from itertools import product


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

    nameval_to_str = [[f'{name}_{x}' for x in lst] for (name, lst) in d.items()]
    all_combinations = list(product(*nameval_to_str))

    for combination in all_combinations:
        for s in combination:
            name, val = s.split('_')
            val = float(val)
            dict_in[name] = val


        dict_in['Ninit'] = int(dict_in['N0'] * 0.1 / 1000) # Initial Infected, 1 permille        
        for ID in range(N_loops):
            filename = dict_to_filename_with_dir(dict_in, ID)
            if not Path(filename).exists() or force_overwrite:
                filenames.append(filename)
        
    return filenames


@njit
def single_run_numba(N0, mu, alpha, psi, beta, sigma, Ninit, Mrate1, Mrate2, gamma, nts, Nstates):

    # N0 = 10_000 
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
    # Ninit = int(N0 * 0.1 / 1000)

    NRe = N0

    # For generating Network

    P1 = np.zeros((N0, 2))
    AK = -1*np.ones((N0, 1000), np.int_)
    UK = np.zeros(N0, np.int_)
    UKRef = np.zeros(N0, np.int_)
    DK = np.zeros(N0, np.int_)
    Prob = np.ones(N0)
    SK = -1*np.ones(N0, np.int_)
    AKRef = -1*np.ones((N0, 1000), np.int_)
    Rate = -1*np.ones((N0, 1000))
    SAK = -1*np.ones((Nstates, N0), np.int_)
    S = np.zeros(Nstates, np.int_)
    Par = np.zeros(Nstates)
    csMov = np.zeros(Nstates)
    csInf = np.zeros(Nstates)
    InfRat = np.zeros(N0)

    # Ninit = int(N0 * 0.1 / 1000)

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


    D0 = 0.01 
    D = D0*100

    dt = 0.01 
    RT = 0
    rD = np.sqrt(2*D0*dt*N0) / 10
    for i in range(N0):
        ra = np.random.rand()
        if (ra < gamma):
            Prob[i] = 0.1 -np.log( np.random.rand())/1.0
        else:
            Prob[i] = 1.1;

    PT = np.sum(Prob)
    PC = np.cumsum(Prob);
    PP = PC/PT


    for i in range(NRe):
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


    # Here we construct and connect network #############################
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
            if (UK[id1] < 1000) and (UK[id2] < 1000) and (id1 != id2) and (acc == 1):
                r = np.sqrt((P1[id1, 0] - P1[id2, 0])**2 + (P1[id1, 1] - P1[id2, 1])**2)
                ra = np.random.rand()
                if np.exp(-alpha*r/rD) > ra:
                    ran1 = np.random.rand()

                    AK[id1, UK[id1]] = id2	        
                    AKRef[id1, UK[id1]] = id2
                    ra1 = np.random.rand()
                    if (ra1 < sigma):
                        Rate[id1, UK[id1]] = beta
                        Rate[id2, UK[id2]] = beta
                    else:
                        rat = -np.log(np.random.rand())/beta
                        Rate[id1, UK[id1]] = rat
                        Rate[id2, UK[id2]] = rat

                    AK[id2, UK[id2]] = id1 	
                    AKRef[id2, UK[id2]] = id1

                    UK[id1] += 1 
                    UK[id2] += 1
                    UKRef[id1] += 1 
                    UKRef[id2] += 1
                    accra = 1


    #   ###############  ####### ########  ########  ############## 

    idx = 0   
    on = 1  
    AC = 0
    Tot = 0  
    TotMov = 0 
    TotInf = 0  
    NRecov = 0
    cinf = 0 
    cupd = 0 
    clickN = 0
    click = 0 
    NrDInf = Ninit
    c = 0  
    Csum = 0 
    RT = 0 
    # dt 
    # exI,exM
    
    
    #   cout << "START!!!    " << endl
    #   clock_t begin1 = clock()

    ##  Now make initial infectious
    for iin in range(Ninit):
        idx = iin*10    
        SK[idx] = 0  
        SAK[0, S[0]] = idx
        S[0] += 1  
        DK[idx] = 1  
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

    #   #   cout << "Here " << endl

    SIRfile = []
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

                # If it has moven to infectious state we update rates

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
                DK[idx] = 2
                NRecov += 1


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
                        NrDInf += 1
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
            click += 1 

            SIRfile_tmp = np.zeros(Nstates + 1)
            icount = 0
            SIRfile_tmp[icount] = RT
            for s in S:
                icount += 1
                SIRfile_tmp[icount] = s #<< "\t"
            # SIRfile_tmp[icount+1] = NrDInf
            SIRfile.append(SIRfile_tmp)

        # Criteria to stop
        #     if (ssum < TotInf: on = 0 cout << " Higher rates than expected " << endl}
        #     if (ssum > TotInf: on = 0 cout << " Not all rates added " << ssum << " " << TotInf << " " << c << endl for (i = 0 i < 9 i++:cout << S[i] << endl}}

        # if exM > TotMov+0.1:
        #     on = 0 
        #     # cout << "Move problem " << endl
        #     print("Move problem")
        
        if c > 10000000: 
            on = 0 
        
        if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001): 
            on = 0 
            # cout << "Equilibrium " << endl
            # print("Equilibrium")
        
        if S[Nstates-1] > N0-10:      
            # cout << "2/3 through " << endl 
            # print("2/3 through")
            on = 0

        # Check for bugs
        if AC == 0: 
            # cout << "No Chosen rate " << Csum << " " << c << endl 
            print("No Chosen rate", csMov)
            on = 0
        
        if (TotMov < 0) and (TotMov > -0.001):
            TotMov = 0 
            
        if (TotInf < 0) and (TotInf > -0.001):
            TotInf = 0 
            
        if (TotMov < 0) or (TotInf < 0): 
            # cout << "Negative Problem " << " " << TotMov << " " << TotInf << endl 
            print("\nNegative Problem", TotMov, TotInf)
            print(alpha, beta, gamma)
            on = 0  
    
    return SIRfile






@njit
def single_run_numba_SK_P1_UK(N0, mu, alpha, psi, beta, sigma, Ninit, Mrate1, Mrate2, gamma, nts, Nstates):

    NRe = N0

    # For generating Network

    P1 = np.zeros((N0, 2))
    AK = -1*np.ones((N0, 1000), np.int_)
    UK = np.zeros(N0, np.int_)
    UKRef = np.zeros(N0, np.int_)
    DK = np.zeros(N0, np.int_)
    Prob = np.ones(N0)
    SK = -1*np.ones(N0, np.int_)
    AKRef = -1*np.ones((N0, 1000), np.int_)
    Rate = -1*np.ones((N0, 1000))
    SAK = -1*np.ones((Nstates, N0), np.int_)
    S = np.zeros(Nstates, np.int_)
    Par = np.zeros(Nstates)
    csMov = np.zeros(Nstates)
    csInf = np.zeros(Nstates)
    InfRat = np.zeros(N0)

    # Ninit = int(N0 * 0.1 / 1000)


    Ninfectious = 4 # This means the 5'th state
    # For simulating Actual Disease
    NExp = 0 
    NInf = 0

    Par[:4] = Mrate1
    Par[4:8] = Mrate2

    # Here we initialize the system
    # psi = 2.0
    # alpha = 1.0
    xx = 0.0 
    yy = 0.0 
    psi_epsilon = 1e-2
    tnext = (1/np.random.random())**(1/(psi+psi_epsilon))-1
    rD = 1.0;
    # D0 = 0.01 
    # D = D0*100
    dt = 0.01 
    RT = 0


    D0 = 0.01 
    D = D0*100

    dt = 0.01 
    RT = 0
    rD = np.sqrt(2*D0*dt*N0) / 10
    for i in range(N0):
        ra = np.random.rand()
        if (ra < gamma):
            Prob[i] = 0.1 -np.log( np.random.rand())/1.0
        else:
            Prob[i] = 1.1;

    PT = np.sum(Prob)
    PC = np.cumsum(Prob);
    PP = PC/PT


    for i in range(NRe):
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


    # Here we construct and connect network #############################
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
            if (UK[id1] < 1000) and (UK[id2] < 1000) and (id1 != id2) and (acc == 1):
                r = np.sqrt((P1[id1, 0] - P1[id2, 0])**2 + (P1[id1][1] - P1[id2][1])**2)
                ra = np.random.rand()
                if np.exp(-alpha*r/rD) > ra:
                    ran1 = np.random.rand()

                    AK[id1, UK[id1]] = id2	        
                    AKRef[id1, UK[id1]] = id2
                    Rate[id1, UK[id1]] = beta + sigma*(-1+2*ran1)

                    AK[id2, UK[id2]] = id1 	
                    AKRef[id2, UK[id2]] = id1
                    Rate[id2, UK[id2]] = beta + sigma*(-1+2*ran1)

                    UK[id1] += 1 
                    UK[id2] += 1
                    UKRef[id1] += 1 
                    UKRef[id2] += 1
                    accra = 1


    #   ###############  ####### ########  ########  ############## 

    idx = 0   
    on = 1  
    AC = 0
    Tot = 0  
    TotMov = 0 
    TotInf = 0  
    NRecov = 0
    cinf = 0 
    cupd = 0 
    clickN = 0
    click = 0 
    NrDInf = Ninit
    c = 0  
    Csum = 0 
    RT = 0 
    # dt 
    # exI,exM
    
    
    #   cout << "START!!!    " << endl
    #   clock_t begin1 = clock()

    ##  Now make initial infectious
    for iin in range(Ninit):
        idx = iin*10    
        SK[idx] = 0  
        SAK[0, S[0]] = idx
        S[0] += 1  
        DK[idx] = 1  
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

    #   #   cout << "Here " << endl

    # SK, P1, UK
    SIRfile_SK = []
    # SIRfile_P1 = [] 
    SIRfile_UK = []


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

                # If it has moven to infectious state we update rates

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
                DK[idx] = 2
                NRecov += 1


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
                        NrDInf += 1
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
            click += 1 

            # deepcopy
            SK_tmp = np.zeros(len(SK))
            for ix in range(len(SK)):
                SK_tmp[ix] = SK[ix]
            SIRfile_SK.append(SK_tmp)

            UK_tmp = np.zeros(len(UK))
            for ix in range(len(SK)):
                UK_tmp[ix] = UK[ix]
            SIRfile_UK.append(UK_tmp)

        # Criteria to stop
        #     if (ssum < TotInf: on = 0 cout << " Higher rates than expected " << endl}
        #     if (ssum > TotInf: on = 0 cout << " Not all rates added " << ssum << " " << TotInf << " " << c << endl for (i = 0 i < 9 i++:cout << S[i] << endl}}

        # if exM > TotMov+0.1:
        #     on = 0 
        #     # cout << "Move problem " << endl
        #     print("Move problem")
        
        if c > 10000000: 
            on = 0 
        
        if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001): 
            on = 0 
            # cout << "Equilibrium " << endl
            # print("Equilibrium")
        
        if S[Nstates-1] > N0-10:      
            # cout << "2/3 through " << endl 
            # print("2/3 through")
            on = 0

        # Check for bugs
        if AC == 0: 
            # cout << "No Chosen rate " << Csum << " " << c << endl 
            print("No Chosen rate", csMov)
            on = 0
        
        if (TotMov < 0) and (TotMov > -0.001):
            TotMov = 0 
            
        if (TotInf < 0) and (TotInf > -0.001):
            TotInf = 0 
            
        if (TotMov < 0) or (TotInf < 0): 
            # cout << "Negative Problem " << " " << TotMov << " " << TotInf << endl 
            print("Negative Problem", TotMov, TotInf)
            on = 0  
    
    return SIRfile_SK, P1, SIRfile_UK



# from extra_funcs import human_format

def dict_to_filename_with_dir(dict_in, ID):
    filename = Path('Data') / 'NetworkSimulation' 
    file_string = ''
    for key, val in dict_in.items():
        file_string += f"{key}_{val}_"
    file_string = file_string[:-1] # remove trailing _
    filename = filename / file_string
    file_string += f"_ID_{ID:03d}.csv"
    filename = filename / file_string
    return str(filename)

# def dict_to_filename(dict_in, ID):
#     filename = Path('Data') 
#     file_string = 'NetworkSimulation'
#     for key, val in dict_in.items():
#         file_string += f"_{key}_{val}"
#     file_string += f"_ID_{ID:03d}.csv"
#     filename = filename / file_string
#     return str(filename)

def filename_to_dict(filename, normal_string=False):
    dict_in = {}
    if normal_string:
        raise AssertionError('AssertionError')
        keyvals = filename.split('_')
    else:
        keyvals = filename.split('/')[2].split('_')
        # keyvals = filename.split('.csv')[0].split('_')[1:]
    keyvals_chunks = [keyvals[i:i + 2] for i in range(0, len(keyvals), 2)]
    ints = ['N0', 'Ninit', 'Nstates']
    for key, val in keyvals_chunks:
        if not key == 'ID':
            if key in ints:
                dict_in[key] = int(val)
            else:
                dict_in[key] = float(val)
    return dict_in

def single_run_and_save(filename):

    dict_in = filename_to_dict(filename)
    out_single_run = single_run_numba(**dict_in)

    header = ['Time', 
            'E1', 'E2', 'E3', 'E4', 
            'I1', 'I2', 'I3', 'I4', 
            'R',
            # 'NrDInf',
            ]
    df = pd.DataFrame(out_single_run, columns=header)

    # make sure parent folder exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # save csv file
    df.to_csv(filename, index=False)

    # save SK, P1, and UK, once for each set of parameters
    ID = filename_to_ID(filename)
    if ID == 0:
        SIRfile_SK, SIRfile_P1, SIRfile_UK = single_run_numba_SK_P1_UK(**dict_in)
        SIRfile_SK = np.array(SIRfile_SK, dtype=int)
        SIRfile_P1 = np.array(SIRfile_P1)
        SIRfile_UK = np.array(SIRfile_UK, dtype=int)
        
        filename_SK_P1_UK = (Path('Data_SK_P1_UK') / Path(filename).stem).with_suffix('.SK_P1_SK.joblib')

        Path(filename_SK_P1_UK).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump([SIRfile_SK, SIRfile_P1, SIRfile_UK], str(filename_SK_P1_UK))

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


def filename_to_dotdict(filename, normal_string=False):
    return DotDict(filename_to_dict(filename, normal_string))