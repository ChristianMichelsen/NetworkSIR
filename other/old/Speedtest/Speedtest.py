import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from numba import njit, prange
import time
import joblib

from ExternalFunctions import Chi2Regression
from iminuit import Minuit

#%%

test = 0
testN = 0


N0 = 500
# N0 = 500_000# 2378.19

mu = 20.0  # Average number connections
alpha = 1.0*test # Spatial parameter
beta = 1.0 # Mean rate
sigma = 0.8 # Spread in rate
Ninit = 10 # Initial Infected
Mrate1 = 1.2 # E->I
Mrate2 = 1.2 # I->R
gamma = 0.0 # Parameter for skewed connection shape
delta = 0.05 # Minimum probability to connect
nts = 0.1 
Nstates = 9


@njit
def single_run(N0):

    # print(N0)
    # print(type(N0))

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
            x = csMov/Tot;
            i1 = np.searchsorted(x, ra1)
            Csum = csMov[i1]/Tot;
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
            x = TotMov/Tot + csInf/Tot;
            i1 = np.searchsorted(x, ra1)
            Csum = TotMov/Tot + csInf[i1]/Tot;
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
        
        if S[8] > N0-10:      
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


single_run(100)

#%%

N0s = np.linspace(100, 500_000, 10)

do_run = False

if do_run:

    # xmin = 2
    # xmax = 5
    # multiplier = 3

    single_run(100)

    # N0s = np.logspace(xmin, xmax, (xmax-xmin)*multiplier+1)
    
    times = np.zeros_like(N0s)
    for i, N0 in enumerate(N0s):
        start = time.time()
        single_run(int(N0))
        end = time.time()
        times[i] = end - start
        print(i, times[i])

    joblib.dump(times, 'times.joblib')

else:
    times = joblib.load('times.joblib')




#%%


#%%

def quadratic(N, a, b, c):
    return c + b*N + a*N**2

# def quadratic(N, a):
#     return a*N**2


# ----------------------------------------------------------------- #
# Plot the data and fit it with an exponential fit:
# ----------------------------------------------------------------- #

N0_min = 0
mask = N0s > N0_min
chi2_exp = Chi2Regression(quadratic, N0s[mask], times[mask], np.sqrt(times[mask]))
minuit_exp = Minuit(chi2_exp, pedantic=False, print_level=0, a=1, b=0, c=0)
# minuit_exp = Minuit(chi2_exp, pedantic=False, print_level=0, a=1)
minuit_exp.migrad()
if (not minuit_exp.get_fmin().is_valid) :
    print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

print(minuit_exp.parameters)
print(minuit_exp.np_values())
print(minuit_exp.np_errors())

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=N0s, y=times, name='Timings'))

x = np.linspace(0, max(N0s), 1000)
y = quadratic(x, *minuit_exp.args)

ypos = 0.95
delta = -0.05
for N in [1e6, 5e6]:
    time = quadratic(N, *minuit_exp.args) / 60 / 60
    fit_string = f"f( N = {int(N/1e6)}e6 ) = {time:.2f} hours"
    # a = ({minuit_exp.values['a']*1e9:.2f} ± {minuit_exp.errors['a']*1e9:.2f}) * 10^-9"
    fig.add_annotation(
                x=0.07,
                y=ypos,
                xref="paper",
                yref="paper",
                text=fit_string,
                showarrow=False,
    )
    ypos += delta

fig.add_trace(go.Scatter(x=x, y=y, name=f"Fit"))

k_scale=1
# Edit the layout
fig.update_layout(title=f'Timings',
                   xaxis_title='N',
                   yaxis_title='Time',
                   height=600*k_scale, width=800*k_scale,
                   )

fig.show()
fig.write_html(f"Figures/timings.html")


# %%
