import matplotlib.pyplot as plt
import matplotlib as mpl

def set_rc_params():
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['figure.dpi'] = 300
    mpl.rc('axes', edgecolor='k', linewidth=2)
