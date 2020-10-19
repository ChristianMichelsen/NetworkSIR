import matplotlib as mpl

# mpl.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src import rc_params
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd
from matplotlib.ticker import PercentFormatter, EngFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
from collections import defaultdict
import warnings
from matplotlib.animation import FuncAnimation
from mpl_scatter_density import ScatterDensityArtist
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from collections import Counter, defaultdict
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from src.utils import utils
from src import file_loaders
import h5py
from functools import partial
from p_tqdm import p_umap, p_uimap

mpl.rc("axes", edgecolor="k", linewidth=2)

#%%


def get_inverse_mapping(mapping):
    inv_mapping = defaultdict(list)
    for key, val in mapping.items():
        inv_mapping[val].append(key)
    return dict(inv_mapping)


#%%
def unique_counter(x, mapping=None):
    vals, counts = np.unique(x, return_counts=True)
    d = {val: count for val, count in zip(vals, counts)}

    if mapping is None:
        return d

    d2 = Counter()
    for key, val in d.items():
        d2[mapping[key]] += val
    d2 = dict(d2)

    for key in set(mapping.values()):
        if not key in d2:
            d2[key] = 0

    return d2


def compute_df_daily_counts(my_state, N_days, verbose=False):
    daily_counts = {}
    days = range(N_days)
    if verbose:
        days = tqdm(days, desc="Creating df_counts")
    for day in days:
        daily_counts[day] = unique_counter(my_state[day], mapping=mapping)
    df_counts = pd.DataFrame(daily_counts).T
    return df_counts


def compute_R_eff(df_counts, cfg):
    I = df_counts["I"].values
    R = df_counts["R"].values
    S = (cfg.N_tot - df_counts[["I", "R"]].sum(axis=1)).values
    with np.errstate(divide="ignore", invalid="ignore"):
        R_eff = -(S[1:] - S[:-1]) / (R[1:] - R[:-1])
    R_eff[np.isinf(R_eff)] = np.nan
    return R_eff


def get_mask(my_state, inverse_mapping, day, state):
    return np.isin(my_state[day], inverse_mapping[state])


def interpolate_R_eff(R_eff):
    N = len(R_eff)
    x = np.arange(N)
    y = R_eff
    f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    x_interpolated = np.linspace(0, N - 1, 10_000)
    y_interpolated = f(x_interpolated)
    df_R_eff = pd.DataFrame({"t": x_interpolated, "R_eff": y_interpolated})
    return df_R_eff


def remove_spines(ax, spines=None):
    if spines is None:
        spines = ["top", "right"]
    for spine in spines:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "right"]:
        if spine not in spines:
            ax.yaxis.set_ticks_position(spine)

    for spine in ["top", "bottom"]:
        if spine not in spines:
            ax.xaxis.set_ticks_position(spine)


def f_norm(vmax):
    return ImageNormalize(vmin=0.0, vmax=vmax, stretch=LogStretch())


def compute_filename_out(filename):
    hash_ = file_loaders.filename_to_hash(filename)
    ID = int(filename.split("ID__")[1].split(".hdf5")[0])
    filename_out = f"./Figures/animations/animation_{hash_}_ID_{ID}.mp4"
    return filename_out


#%%

mapping = {-1: "S", 0: "I", 1: "I", 2: "I", 3: "I", 4: "I", 5: "I", 6: "I", 7: "I", 8: "R"}
inverse_mapping = get_inverse_mapping(mapping)

states = ["S", "I", "R"]
state_names = {
    "S": "Susceptable",
    "I": r"Infected $\&$ Exposed",
    "R": "Recovered",
}

d_colors = {
    "S": "#7F7F7F",
    "I": "#D62728",
    "R": "#2CA02C",
}


# create the new map
cmap = mpl.colors.ListedColormap([d_colors["R"], d_colors["I"]])
bounds = [0, 0.5, 1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
scatter_kwargs = dict(cmap=cmap, norm=norm, edgecolor="none")


#%%


def animate_single_network(
    filename, verbose=False, force_rerun=False, N_day_max=None, dpi=50, frames=None, fps=10
):

    cfg = file_loaders.filename_to_cfg(filename)
    if cfg is None:
        print(f"Couldnt find a proper cfg for {filename}, skipping for now")
        return

    filename_out = compute_filename_out(filename)

    if utils.file_exists(filename_out) and not force_rerun:
        return

    utils.make_sure_folder_exist(filename_out)

    with h5py.File(filename, "r") as f:
        if verbose:
            print(f"Loading {filename}")
        df_coordinates = pd.DataFrame(f["df_coordinates"][()])  # .drop("index", axis=1)
        my_state = f["my_state"][()]
        # self.coordinate_indices = f["coordinate_indices"][()]
        # df_raw = pd.DataFrame(f["df"][()])
        # my_number_of_contacts = f["my_number_of_contacts"][()]

    coordinates = utils.df_coordinates_to_coordinates(df_coordinates)

    if N_day_max is None:
        N_days = len(my_state)
    else:
        N_days = min(len(my_state), N_day_max)

    df_counts = compute_df_daily_counts(my_state, N_days, verbose)
    R_eff = compute_R_eff(df_counts, cfg)

    if frames is None:
        frames = np.arange(N_days)

    title = utils.dict_to_title(cfg)

    fig, ax = plt.subplots(figsize=(8.5 * 1.4, 10 * 1.4))

    ax.set(xlim=(7.9, 13.3), ylim=(54.5, 58.2), xlabel="Longitude")
    ax.set_ylabel("Latitude", rotation=90)  # fontsize=20, labelpad=20
    ax.set_title(title, pad=20, fontsize=18)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    time_text = ax.text(0.03, 0.03, "", transform=ax.transAxes, fontsize=38)

    plot_kwargs = {}
    plot_kwargs["S"] = dict(alpha=0.5, norm=f_norm(1000))
    plot_kwargs["I"] = dict(alpha=1.0, norm=f_norm(50))
    plot_kwargs["R"] = dict(alpha=0.6, norm=f_norm(1000))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")

        d_scatter = {}
        for state in states:
            scatter = ScatterDensityArtist(
                ax,
                [],
                [],
                color=d_colors[state],
                dpi=dpi,
                label=state,
                **plot_kwargs[state],
            )
            ax.add_artist(scatter)
            d_scatter[state] = scatter

    kw_args_circle = dict(xdata=[0], ydata=[0], marker="o", color="w", markersize=16)
    circles = [
        Line2D(
            label=state_names[state],
            markerfacecolor=d_colors[state],
            **kw_args_circle,
        )
        for state in states
    ]
    ax.legend(handles=circles, fontsize=30, frameon=False, loc=(0, 0.82))

    I_max_rel = df_counts["I"].max() / cfg["N_tot"]

    # These are in unitless percentage of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.63, 0.75, 0.39 * 0.6, 0.08]
    ax_Infected = fig.add_axes([left, bottom, width, height])
    ax_Infected.set(xlim=(0, N_days * 1.1), ylim=(0, I_max_rel * 1.1))

    decimals = max(int(-np.log10(I_max_rel)) - 1, 0)  # max important, otherwise decimals=-1
    ax_Infected.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))
    ax_Infected.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    remove_spines(ax_Infected)

    (line_I_1,) = ax_Infected.plot([], [], "-", color=d_colors["I"], lw=3)
    (line_I_2,) = ax_Infected.plot([], [], "o", color=d_colors["I"])

    ax_Infected.text(0.01, 1.18, "Fraction Infected", transform=ax_Infected.transAxes, fontsize=30)

    R_eff_max = np.nanmax(R_eff)

    ax_R_eff = fig.add_axes([left, bottom - height * 2, width, height])
    ax_R_eff.set(ylim=(0, R_eff_max * 1.1), xlim=(0, N_days * 1.1))
    ax_R_eff.set_xlabel(r"Time [days]", fontsize=30)

    ax_R_eff.axhline(1, ls="--", color="k", lw=1)  # x = 0
    scatter_R_eff = ax_R_eff.scatter([], [], s=10, c=[], **scatter_kwargs)  # , )
    scatter_R_eff_today = ax_R_eff.scatter([], [], s=100, c=[], **scatter_kwargs)
    ax_R_eff.text(
        0.4, 1.18, r"$\mathcal{R}_\mathrm{eff}$", transform=ax_R_eff.transAxes, fontsize=30
    )
    ax_R_eff.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax_R_eff.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    remove_spines(ax_R_eff)

    def update_scatter(scatter, day, state):
        mask = np.isin(my_state[day], inverse_mapping[state])
        x, y = coordinates[mask].T
        scatter.set_xy(x, y)

    def init():
        for scatter in d_scatter.values():
            scatter.set_xy([], [])
        time_text.set_text("")
        line_I_1.set_data([], [])
        line_I_2.set_data([], [])
        scatter_R_eff.set_offsets([])
        scatter_R_eff_today.set_offsets([])
        return (
            *d_scatter.values(),
            time_text,
            line_I_1,
            line_I_2,
            scatter_R_eff,
            scatter_R_eff_today,
        )

    def animate(day):

        day += 1

        for state, scatter in d_scatter.items():
            try:
                update_scatter(scatter, day, state)
            except IndexError:
                pass
        time_text.set_text(f"Day = {day:3d}")

        I_up_to_today = df_counts["I"].iloc[: day + 1] / cfg["N_tot"]

        x_I = I_up_to_today.index.values
        y_I = I_up_to_today.values
        line_I_1.set_data(x_I, y_I)
        line_I_2.set_data(x_I[-1], y_I[-1])

        R_eff_up_to_today = interpolate_R_eff(R_eff[: day + 1])
        data = np.c_[R_eff_up_to_today["t"], R_eff_up_to_today["R_eff"]]
        scatter_R_eff.set_offsets(data)  # Set coordinates
        z_colors = (R_eff_up_to_today["R_eff"] > 1) / 1
        scatter_R_eff.set_array(z_colors)  # Set colors

        R_eff_today = R_eff_up_to_today.iloc[-1]
        z_today = np.array([R_eff_today["R_eff"] > 1])
        data_today = np.c_[R_eff_today["t"], R_eff_today["R_eff"]]
        scatter_R_eff_today.set_offsets(data_today)  # Set coordinates
        scatter_R_eff_today.set_array(z_today)  # Set colors

        return *d_scatter.values(), time_text, line_I_1, line_I_2, scatter_R_eff

    if verbose:
        frames = tqdm(frames, desc="Creating animation")

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        interval=1,
        blit=True,
    )

    # fig.tight_layout()
    anim.save(filename_out, fps=fps, extra_args=["-vcodec", "libx264"], dpi=100)


filename = "Data/network/95a0789cf3/network_2020-10-12_95a0789cf3_ID__0.hdf5"
# animate_single_network(
#     filename,
#     verbose=True,
#     force_rerun=True,
#     N_day_max=None,
#     dpi=50,
#     frames=None,
#     fps=10,
# )


def try_animate_single_network(filename, **kwargs):
    try:
        animate_single_network(filename, **kwargs)
    except Exception as e:
        print(f"Got error 0 at {filename}, skipping for now")
        print(e)


def animate_all_networks(
    base_dir="./Data/network",
    num_cores=1,
    ID=None,
    verbose=False,
    force_rerun=False,
    **kwargs,
):
    filenames = [str(file) for file in Path(base_dir).rglob(f"*.hdf5")]
    if ID is not None:
        filenames = [filename for filename in filenames if f"ID__{ID}." in filename]

    if not force_rerun:
        filenames = [
            filename
            for filename in filenames
            if not utils.file_exists(compute_filename_out(filename))
        ]

    if len(filenames) == 0:
        return None

    num_cores = utils.get_num_cores(num_cores)

    print(f"Creating {len(filenames)} animations using {num_cores} cores")

    if "N_day_max" in kwargs and kwargs["N_day_max"] is not None:
        print(f"Note, running only for {kwargs['N_day_max']} days")

    desc = "Animating"

    # kwargs = {}
    if num_cores == 1:
        for filename in tqdm(filenames, desc=desc):
            try_animate_single_network(filename, verbose=False, **kwargs)
    else:
        p_umap(
            partial(try_animate_single_network, verbose=False, **kwargs),
            filenames,
            num_cpus=num_cores,
            desc=desc,
        )


# animate_all_networks(base_dir="./Data/network", num_cores=None, N_day_max=None, ID=0)