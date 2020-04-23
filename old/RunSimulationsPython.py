import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from numba import njit, prange
import time


test = 0
testN = 0

N0 = 10_000
# N0 = 500_000# 2378.19

mu = 20.0  # Average number connections
alpha = 1.0*test # Spatial parameter
beta = 1.0 # Mean rate
sigma = 0.8 # Spread in rate
Ninit = 10 # Initial Infected
Mrate1 = 0.5 # S->E
Mrate2 = 4.5 # E->I
Mrate3 = 4.5 # I->R
gamma = 0.0 # Parameter for skewed connection shape
delta = 0.05 # Minimum probability to connect

NRe = N0



@njit
def single_run(N0):

    # For generating Network

    P1 = np.zeros((N0, 2))
    AK = -1*np.ones((N0, 200), np.int_)
    UK = np.zeros(N0, np.int_)
    UKRef = np.zeros(N0, np.int_)
    DK = np.zeros(N0, np.int_)
    Prob = np.ones(N0, np.int_)
    SK = -1*np.ones(N0, np.int_)
    AKRef = -1*np.ones((N0, 200), np.int_)
    Rate = -1*np.ones((N0, 200))
    SAK = -1*np.ones((13, N0), np.int_)
    S = np.zeros(13, np.int_)
    Par = np.zeros(13)


    Nstates = 13
    Ninfectious = 4 # This means the 5'th state
    # For simulating Actual Disease
    NExp = 0 
    NInf = 0

    Par[:4] = Mrate1
    Par[4:8] = Mrate2
    Par[8:12] = Mrate3

    # Here we initialize the system
    for c in range(N0):
        while True:
            P1[c, :] = np.random.uniform(-1, 1, 2)
            if np.sqrt(P1[c, 0]**2 + P1[c, 1]**2) < 1.0:
                break
    
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
            if np.exp(-alpha*r) > ra:
                ran1 = np.random.rand()
                ran2 = np.random.rand()

            # nran1 = np.cos(2*np.pi*ran2)*np.sqrt(-2*np.log(ran1)) 
            # nran2 = np.sin(2*np.pi*ran2)*np.sqrt(-2*np.log(ran1))

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
    nts = 0.1 
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
        for i1 in range(UKRef[idx]):
            Af = AKRef[idx, i1]
            for i2 in range(UK[Af]):
                if AK[Af, i2] == idx:
                    # for (i3 = i2 i3 < UK[Af] i3++:
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
            for i1 in range(Nstates-1):
                for i2 in range(S[i1]):
                    Csum += Par[i1]/Tot
                    if Csum > ra1:
                        idx = SAK[i1, i2]
                        cin1 = i1 
                        cjn1 = i2
                        AC = 1
                        break
                
                if AC == 1:
                    break
            
            # We have chosen idx to move -> here we move it
            SAK[cin1+1, S[cin1+1]] = idx
            for j in range(cjn1, S[cin1]):
                SAK[cin1, j] = SAK[cin1, j+1] 

            SK[idx] += 1
            S[cin1] -= 1 
            S[cin1+1] += 1      
            TotMov -= Par[cin1] 
            TotMov += Par[cin1+1]     

            # If it has moven to infectious state we update rates
            if SK[idx] == Ninfectious: # Moves TO infectious State from non-infectious
                for i1 in range(UK[idx]): # Loop over row idx	  
                    if SK[AK[idx, i1]] < 0:
                        TotInf += Rate[idx, i1]
            

            if SK[idx] == Nstates-1: # If this moves to Recovered state
                for i1 in range(UK[idx]): # Loop over row idx
                    TotInf -= Rate[idx, i1] 
                DK[idx] = 2
                NRecov += 1


        # Here we infect new states
        else:
            Csum = TotMov/Tot
            for i1 in range(Ninfectious, Nstates-1):
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
                            AC = 1
                            break
                    
                    if AC == 1:
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
                        for i3 in range(i2, UK[Af]):
                            AK[Af, i3] = AK[Af, i3+1]
                            Rate[Af, i3] = Rate[Af, i3+1]
                        UK[Af] -= 1 
                        break

        ################

        if nts*click < RT:
            click += 1 

            SIRfile_tmp = np.zeros(15)
            icount = 0
            SIRfile_tmp[icount] = RT
            for s in S:
                icount += 1
                SIRfile_tmp[icount] = s #<< "\t"
            SIRfile_tmp[icount+1] = NR0Inf
            SIRfile.append(SIRfile_tmp)

        # Criteria to stop
        #     if (ssum < TotInf: on = 0 cout << " Higher rates than expected " << endl}
        #     if (ssum > TotInf: on = 0 cout << " Not all rates added " << ssum << " " << TotInf << " " << c << endl for (i = 0 i < 13 i++:cout << S[i] << endl}}

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
        
        if S[12] > N0-10:      
            # cout << "2/3 through " << endl 
            print("2/3 through")
            on = 0

        # Check for bugs
        if AC == 0: 
            # cout << "No Chosen rate " << Csum << " " << c << endl 
            print("No Chosen rate", Csum, c)
            on = 0
        
        if (TotMov < 0) and (TotMov > -0.001):
            TotMov = 0 
            
        if (TotInf < 0) and (TotInf > -0.001):
            TotInf = 0 
            
        if (TotMov < 0) or (TotInf < 0): 
            # cout << "Negative Problem " << " " << TotMov << " " << TotInf << endl 
            print("Negative Problem", TotMov, TotInf)
            on = 0  


        # if include_tqdm:
        #     pbar.update(1)

    # if include_tqdm:
    #     pbar.close()


    
    return SIRfile



header = ['Time', 
        'S1', 'S2', 'S3', 'S4', 
        'E1', 'E2', 'E3', 'E4', 
        'I1', 'I2', 'I3', 'I4', 
        'R',
        'NR0Inf',
        ]




@njit(parallel=True) # 
def multiple_loops(N_loops):
    SIRfiles = []
    for i in prange(N_loops):
        SIRfile = single_run(N0)
        SIRfiles.append(SIRfile)
    return SIRfiles

N_loops = 4

start = time.time()
SIRfiles = multiple_loops(N_loops)
end = time.time()
print(f"\nElapsed (with compilation) = {end - start:.2f}")



for testN in range(N_loops):

    SIRfilename = f"Data/SIRResult2_Mu{int(mu)}_N{int(NRe/1000)}_In{Ninit}_alpha{int(alpha)}_beta{int(beta)}_sigmaF{int(sigma*100)}_test{testN}.csv"

    df = pd.DataFrame(np.array(SIRfiles[testN]), columns=header)
    df.to_csv(SIRfilename, index=False)


# if False:

#     fig = go.Figure()

#     for col in header[1:]:

#         fig.add_trace(go.Scatter(   
#                             x=df['Time'], 
#                             y=df[col],
#                             mode='lines',
#                             name=col,
#                             # visible=True if col in show_cols else 'legendonly',
#                             ),
#                     )

#     fig.update_yaxes(rangemode="tozero")

#     # Edit the layout
#     fig.update_layout(title='Counts',
#                     xaxis_title='Time',
#                     yaxis_title='Antal',
#                     height=600, width=800,
#                     )

#     fig.show()