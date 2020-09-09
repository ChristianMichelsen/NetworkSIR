import numpy as np
import pandas as pd
from tqdm import tqdm
import mpl_scatter_density
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

#%%

Boligsiden_data = "Data/Boligsiden_DW_NBI_2019_09_03.csv"
df_boligsiden = pd.read_csv(
    Boligsiden_data, usecols=["Sag_GisX_WGS84", "Sag_GisY_WGS84"]
)
df_boligsiden = df_boligsiden.dropna()
df_boligsiden.columns = ["x", "y"]
df_boligsiden = df_boligsiden.query("(8.05 < x < 20) and (54 < y < 58)")

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
# ax.scatter_density(df_boligsiden['x'], df_boligsiden['y'], color='k', dpi=50)


#%%

P1_org = df_boligsiden.values
N = len(P1_org)

#%%

dpi = 100

pdf_name = f"Figures/GPS_coordinates_generated_bw_comparison_dpi_{dpi}.pdf"
with PdfPages(pdf_name) as pdf:

    for bw in tqdm([0.1, 0.01, 0.001, 0.0005, 0.0001]):

        fig = plt.figure(figsize=(16, 20))

        ax00 = fig.add_subplot(2, 2, 1, projection="scatter_density")
        ax00.scatter_density(P1_org[:, 0], P1_org[:, 1], color="k", dpi=dpi)
        ax00.set(title="Original Data")

        ax01 = fig.add_subplot(2, 2, 3, projection="scatter_density")
        ax01.scatter_density(P1_org[:, 0], P1_org[:, 1], color="k", dpi=dpi / 2)
        ax01.set_xlim(12.4, 12.7)
        ax01.set_ylim(55.6, 55.8)
        ax01.set(title="Original Data: Zoom")

        #%%
        kde = KernelDensity(kernel="gaussian", bandwidth=bw, metric="haversine").fit(
            P1_org
        )  #
        P1_gen = kde.sample(N, random_state=42)

        # fig = plt.figure(figsize=(10, 13))
        ax10 = fig.add_subplot(2, 2, 2, projection="scatter_density")
        ax10.scatter_density(P1_gen[:, 0], P1_gen[:, 1], color="k", dpi=dpi)
        ax10.set(title="Generated Data")

        ax11 = fig.add_subplot(2, 2, 4, projection="scatter_density")
        ax11.scatter_density(P1_gen[:, 0], P1_gen[:, 1], color="k", dpi=dpi / 2)
        ax11.set_xlim(12.4, 12.7)
        ax11.set_ylim(55.6, 55.8)
        ax11.set(title="Generated Data: Zoom")

        fig.suptitle(f"bw={bw}", fontsize=30)
        pdf.savefig(fig, dpi=600)

        plt.close("all")

#%%

bw = 0.0005
N_out = 10_000_000

kde_out = KernelDensity(kernel="gaussian", bandwidth=bw, metric="haversine").fit(
    P1_org
)  #
P1_out = kde_out.sample(N_out, random_state=42)

GPS_filename_out = "Data/GPS_coordinates"
np.save(GPS_filename_out + ".npy", P1_out)
np.savetxt(GPS_filename_out + ".csv", P1_out)
