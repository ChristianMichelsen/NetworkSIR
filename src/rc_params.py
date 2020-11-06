import matplotlib.pyplot as plt
import matplotlib as mpl


def set_rc_params(dpi=300):
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["figure.dpi"] = dpi
    mpl.rc("axes", edgecolor="k", linewidth=2)
