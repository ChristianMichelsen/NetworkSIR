import numpy as np
import pandas as pd
from tqdm import tqdm
import mpl_scatter_density
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import geopandas as gpd  # conda install -c conda-forge geopandas
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping as _polygon_to_array
from numba import njit, prange, set_num_threads
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import utils

# from src import simulation_utils

#%%

# make_plot = False
save_coordinates = True
N_out = 10_000_000
shapefile_size = "large"
# shapefile_size = "small"

num_cores = utils.get_num_cores(num_cores_max=30, subtract_cores=1)
set_num_threads(num_cores)

print(f"Generating {N_out} GPS coordinates, please wait.")


#%%

Boligsiden_data = "Data/Boligsiden_DW_NBI_2019_09_03.csv"
print("Loading Boligsiden data")
df_boligsiden = pd.read_csv(Boligsiden_data, usecols=["Sag_GisX_WGS84", "Sag_GisY_WGS84"])
df_boligsiden = df_boligsiden.dropna()
df_boligsiden.columns = ["x", "y"]
df_boligsiden = df_boligsiden.query("(8.05 < x < 20) and (54 < y < 58)")

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
# ax.scatter_density(df_boligsiden['x'], df_boligsiden['y'], color='k', dpi=50)

#%%

bw = 0.0005
N_tmp = int(N_out * 1.2)


print("Fitting 2D KDE to data, please wait")
kde_out = KernelDensity(kernel="gaussian", bandwidth=bw, metric="haversine").fit(df_boligsiden.values)  #
coordinates_out = kde_out.sample(N_tmp, random_state=42)

#%%


def polygon_to_array(polygon):
    return np.array(_polygon_to_array(polygon)["coordinates"][0])[:, :2]


@njit
def polygon_contains_point(point, polygon):
    # https://stackoverflow.com/a/48760556
    x, y = point
    n = len(polygon)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@njit(parallel=True)
def polygon_contains_points(points, polygon):
    N = len(points)
    contains = np.zeros(N, dtype=np.bool_)
    for i in prange(N):
        contains[i] = polygon_contains_point(points[i], polygon)
    return contains


#%%

kommuner, name_to_idx, idx_to_name = utils.load_kommune_shapefiles(shapefile_size, verbose=True)

N = len(coordinates_out)
np_points = coordinates_out[:N]
df_points = pd.DataFrame(np_points, columns=["Longitude", "Lattitude"])
indices = np.full(N, fill_value=-1, dtype=np.int16)

print("Computing which kommune each coordinate is part of", flush=True)
for _, kommune in tqdm(kommuner[["geometry", "idx"]].iterrows(), total=len(kommuner)):
    # break
    kommune_polygon = kommune["geometry"]  # x = pnts.within(kommune_polygon)
    polygon = polygon_to_array(kommune_polygon)
    contains = polygon_contains_points(np_points, polygon)
    indices[contains] = kommune["idx"]

df_points["idx"] = indices
df_points = df_points.iloc[indices != -1]
df_points["kommune"] = df_points["idx"].apply(lambda x: idx_to_name[x])

assert len(df_points) > N_out
df_points = df_points.iloc[:N_out].reset_index(drop=True)


#%%


GPS_filename_out = "Data/GPS_coordinates"

if save_coordinates:
    print(f"Saving coordinates as {GPS_filename_out}")
    df_points.to_feather(GPS_filename_out + ".feather")
    np.save(GPS_filename_out + ".npy", utils.df_coordinates_to_coordinates(df_points))
    df_points.to_hdf(GPS_filename_out + ".hdf5", key="coordinates")
else:
    print("Note: is not saving any files")


#%%


# if make_plot:

#     dpi = 100

#     pdf_name = f"Figures/GPS_coordinates_generated_bw_comparison_dpi_{dpi}.pdf"
#     with PdfPages(pdf_name) as pdf:

#         for bw in tqdm([0.1, 0.01, 0.001, 0.0005, 0.0001]):

#             fig = plt.figure(figsize=(16, 20))

#             ax00 = fig.add_subplot(2, 2, 1, projection="scatter_density")
#             ax00.scatter_density(coordinates_org[:, 0], coordinates_org[:, 1], color="k", dpi=dpi)
#             ax00.set(title="Original Data")

#             ax01 = fig.add_subplot(2, 2, 3, projection="scatter_density")
#             ax01.scatter_density(coordinates_org[:, 0], coordinates_org[:, 1], color="k", dpi=dpi / 2)
#             ax01.set_xlim(12.4, 12.7)
#             ax01.set_ylim(55.6, 55.8)
#             ax01.set(title="Original Data: Zoom")

#             #%%
#             kde = KernelDensity(kernel="gaussian", bandwidth=bw, metric="haversine").fit(coordinates_org)  #
#             coordinates_generated = kde.sample(N, random_state=42)

#             # fig = plt.figure(figsize=(10, 13))
#             ax10 = fig.add_subplot(2, 2, 2, projection="scatter_density")
#             ax10.scatter_density(coordinates_generated[:, 0], coordinates_generated[:, 1], color="k", dpi=dpi)
#             ax10.set(title="Generated Data")

#             ax11 = fig.add_subplot(2, 2, 4, projection="scatter_density")
#             ax11.scatter_density(coordinates_generated[:, 0], coordinates_generated[:, 1], color="k", dpi=dpi / 2)
#             ax11.set_xlim(12.4, 12.7)
#             ax11.set_ylim(55.6, 55.8)
#             ax11.set(title="Generated Data: Zoom")

#             fig.suptitle(f"bw={bw}", fontsize=30)
#             pdf.savefig(fig, dpi=600)

#             plt.close("all")

#     #%%
