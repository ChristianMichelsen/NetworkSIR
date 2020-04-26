import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

@njit
def single_run_numba(N0, mu, alpha, psi, beta, sigma, Ninit, Mrate1, Mrate2, gamma, delta, nts, Nstates):

    NRe = N0

    # For generating Network

    P1 = np.zeros((N0, 2))
    AK = -1*np.ones((N0, 200), np.int_)
    UK = np.zeros(N0, np.int_)
    UKRef = np.zeros(N0, np.int_)
    DK = np.zeros(N0, np.int_)
    Prob = np.ones(N0)
    SK = -1*np.ones(N0, np.int_)
    AKRef = -1*np.ones((N0, 200), np.int_)
    Rate = -1*np.ones((N0, 200))
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
    # for c in range(N0):
    #     while True:
    #         P1[c, :] = np.random.uniform(-1, 1, 2)
    #         if np.sqrt(P1[c, 0]**2 + P1[c, 1]**2) < 1.0:
    #             break


    # Here we initialize the system
    # psi = 2.0
    # alpha = 1.0
    x = 0.0 
    y = 0.0 
    tnext = (1/np.random.random())**(1/psi)-1
    D0 = 0.01 
    D = D0*1000 
    dt = 0.01 
    RT = 0
    DS = D0

    for i in range(NRe):
        RT += dt
        if (RT > tnext):
            x += np.sqrt(2*D*dt)*np.random.normal()
            y += np.sqrt(2*D*dt)*np.random.normal()
            ra = (1/np.random.random())**(1/psi)-1  
            DS = D
            tnext = RT + ra       
        else:
            x += np.sqrt(2*D0*dt)*np.random.normal()
            y += np.sqrt(2*D0*dt)*np.random.normal()    

        P1[i, 1] = x
        P1[i, 2] = y


    # initialize Prob
    # gamma = 0.47; 
    for i in range(N0):
        Prob[i] = 0.5 - gamma + 2*gamma*np.random.random()

    
    # Here we construct and connect network #############################
    for c in range(int(mu*NRe)):
        accra = 0

        while accra == 0:

            ra1 = np.random.rand()
            ra2 = np.random.rand()
            
            id1 = int(NRe*ra1)     
            id2 = int(NRe*ra2)

            ra1 = np.random.rand()
            ra2 = np.random.rand()
            
            if (ra1 < Prob[id1]) and (ra2 < Prob[id2]) and (UK[id1] < 200) and (UK[id2] < 200):
                accra = 1

        #  Make sure no element is present twice
        acc = 1
        for i1 in range(UK[id1]): 
            if AK[id1, i1] == id2:
                acc = 0
            
        
        #  Assign contacts and rates
        if (id1 != id2) and (acc == 1): 
            r = np.sqrt((P1[id1, 0] - P1[id2, 0])**2 + (P1[id1][1] - P1[id2][1])**2)
            ra = np.random.rand()
            if np.exp(-alpha*r/DS) > ra:
                ran1 = np.random.rand()
                ran2 = np.random.rand()

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
            # c += 1 


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
    NR0Inf = Ninit
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


    # if include_tqdm:
    #     pbar = tqdm()
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
                        NR0Inf += 1
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

            SIRfile_tmp = np.zeros(Nstates + 2)
            icount = 0
            SIRfile_tmp[icount] = RT
            for s in S:
                icount += 1
                SIRfile_tmp[icount] = s #<< "\t"
            SIRfile_tmp[icount+1] = NR0Inf
            SIRfile.append(SIRfile_tmp)

        # Criteria to stop
        #     if (ssum < TotInf: on = 0 cout << " Higher rates than expected " << endl}
        #     if (ssum > TotInf: on = 0 cout << " Not all rates added " << ssum << " " << TotInf << " " << c << endl for (i = 0 i < 9 i++:cout << S[i] << endl}}

        # TODO: Uncommented
        # if exM > TotMov+0.1:
        #     on = 0 
        #     # cout << "Move problem " << endl
        #     print("Move problem")
        
        if c > 10000000: 
            on = 0 
        
        if (TotInf + TotMov < 0.0001) and (TotMov + TotInf > -0.00001): 
            on = 0 
            # cout << "Equilibrium " << endl
            print("Equilibrium")
        
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
    
    return SIRfile




def dict_to_filename(dict_in, ID):
    filename = Path('Data') 
    file_string = 'NetworkSimulation'
    for key, val in dict_in.items():
        file_string += f"_{key}_{val}"
    file_string += f"_ID_{ID}.csv"
    filename = filename / file_string
    return str(filename)

def filename_to_dict(filename, normal_string=False):
    dict_in = {}
    if normal_string:
        keyvals = filename.split('_')
    else:
        keyvals = filename.split('.csv')[0].split('_')[1:]
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
            'NR0Inf',
            ]
    df = pd.DataFrame(out_single_run, columns=header)
    df.to_csv(filename, index=False)
    return None




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