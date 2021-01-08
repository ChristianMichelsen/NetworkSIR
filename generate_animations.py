import numpy as np
from numba import njit
from scipy import interpolate
import pandas as pd
from pathlib import Path
from scipy.stats import uniform as sp_uniform
import joblib
from tqdm import tqdm
import multiprocessing as mp
from p_tqdm import p_umap
import os
from functools import partial
import awkward
from importlib import reload
import h5py
from src import rc_params
from src.utils import utils
from src import animation_utils
from src import file_loaders

rc_params.set_rc_params(dpi=50)  #
num_cores_max = 40

#%%

# pip install mpl-scatter-density
import mpl_scatter_density
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# conda install astropy
# Make the norm object to define the image stretch
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter, PercentFormatter, MaxNLocator
from scipy.interpolate import interp1d
import matplotlib as mpl


#%%
import shutil
import subprocess
import warnings
from scipy import signal


from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
import matplotlib.font_manager as fm

fontprops = fm.FontProperties(size=24)

longitudes_per_50km = 0.8392
# compas_rose_img = plt.imread('Figures/CompasRose/CompasRose.png')


def add_spines(ax, exclude=None):
    if exclude is None:
        exclude = []
    spines = ["left", "bottom"]
    for spine in spines:
        if not spine in exclude:
            ax.spines[spine].set_color("k")
            ax.spines[spine].set_linewidth(2)
    ax.tick_params(axis="x", pad=10)


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


def convert_df_byte_cols(df):
    for col in df.select_dtypes([np.object]):
        df[col] = df[col].str.decode("utf-8")
    return df


from pathos.helpers import cpu_count
from pathos.multiprocessing import ProcessPool as Pool

# from pathos.threading import ThreadPool as Pool
from collections.abc import Sized


class AnimationBase:
    def __init__(
        self,
        filename,
        animation_type="animation",
        do_tqdm=False,
        verbose=False,
        N_max=None,
    ):

        self.filename = filename
        self.animation_type = animation_type
        self.do_tqdm = do_tqdm
        self.verbose = verbose
        self._load_hdf5_file()

        self.N_max = N_max
        if self.is_valid_file:
            if N_max is None:
                self.N_days = len(self.my_state)
            else:
                # if N_max < 12:
                # print(f"N_max has to be 12 or larger (choosing 12 instead of {N_max} for now).")
                # N_max = 12
                self.N_days = N_max
        # self._Filename = utils.Filename(filename)
        self.cfg = file_loaders.filename_to_cfg(filename)
        self.__name__ = "AnimationBase"

    def __len__(self):
        return self.N_days

    def _load_data(self):

        with h5py.File(self.filename, "r") as f:

            if self.verbose:
                print("Loading hdf5-file")

            # self.coordinate_indices = f["coordinate_indices"][()]
            self.df_raw = pd.DataFrame(f["df"][()])
            # self.df_coordinates = pd.DataFrame(f["df_coordinates"][()])  # .drop("index", axis=1)
            self.coordinates = f["coordinates"][()]
            self.my_state = f["my_state"][()]
            self.my_number_of_contacts = f["my_number_of_contacts"][()]
            self.my_corona_type = f["my_corona_type"][()]

            # g = awkward.hdf5(f)
            # g["my_connections"]
            # g["my_rates"]

        # self.df_coordinates = utils.load_coordinates_from_indices(self.coordinate_indices)
        # self.coordinates = utils.df_coordinates_to_coordinates(self.df_coordinates)

    def _load_hdf5_file(self):
        try:
            self._load_data()
            self._is_valid_file = True
        except OSError:
            print(f"\n\n\n!!! Error at {self.filename} !!! \n\n\n")
            self._is_valid_file = False
            return None

    @property
    def is_valid_file(self):
        if not self._is_valid_file:
            if self.verbose:
                print(f"Still error at {self.filename}")
            return False
        else:
            return True

    def __repr__(self):
        s = f"{self.__name__}(filename='{self.filename}', animation_type='{self.animation_type}', do_tqdm={self.do_tqdm}, verbose={self.verbose}, N_max={self.N_max})"
        return s

    # @abstractmethod
    # def _plot_i_day(self, i_day, **kwargs):
    #     pass

    def _make_animation(
        self, remove_frames=True, force_rerun=False, make_gif=True, optimize_gif=True, **kwargs
    ):

        name = f"{self.animation_type}_" + self._get_sim_pars_str() + ".gif"
        gif_name = str(Path(f"Figures/{self.animation_type}") / name)
        video_name = gif_name.replace("gif", "mp4")

        if not Path(video_name).exists() or force_rerun:
            if self.verbose and not self.do_tqdm:
                print("\nMake individual frames", flush=True)
            try:
                self._initialize_data()
            except AttributeError:
                pass
            except KeyError:
                print(f"Got KeyError at {{self.filename}}")
                pass
            except:
                raise
            self._make_png_files(force_rerun, **kwargs)

            if self.verbose:
                print("\nMake video", flush=True)
            self._make_video_file(video_name)

            if make_gif:
                if self.verbose:
                    print("\nMake GIF", flush=True)
                self._make_gif_file(gif_name)
                if optimize_gif:
                    self._optimize_gif(gif_name)

            if remove_frames:
                if self.verbose:
                    print("\nDelete temporary frames", flush=True)
                self._remove_tmp_frames()
        else:
            if self.verbose:
                print(f"{self.animation_type} already exists.")

    def make_animation(
        self, remove_frames=True, force_rerun=False, make_gif=True, optimize_gif=True, **kwargs
    ):

        if not self.is_valid_file:
            return None

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                self._make_animation(
                    remove_frames=remove_frames,
                    force_rerun=force_rerun,
                    make_gif=make_gif,
                    optimize_gif=optimize_gif,
                    **kwargs,
                )

        except OSError as e:
            print(f"\n\nOSError at {self.filename}")
            print(e)
            print("\n")

        except ValueError as e:
            print(f"\n\nValueError at {self.filename}")
            print(e)
            print("\n")

    def _get_sim_pars_str(self):
        return Path(self.filename).stem.replace(".animation", "")

    def _get_png_name(self, i_day):
        sim_pars_str = self._get_sim_pars_str()
        return f"Figures/{self.animation_type}/tmp_{sim_pars_str}/{self.animation_type}_{sim_pars_str}_frame_{i_day:06d}.png"

    def _make_single_frame(self, i_day, force_rerun=False, **kwargs):

        # print(os.getpid())
        # print(i_day)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            dpi = kwargs.get("dpi", 50)
            png_name = self._get_png_name(i_day)
            if not Path(png_name).exists() or force_rerun:
                fig, _ = self._plot_i_day(i_day, **kwargs)
                Path(png_name).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_name, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
                plt.close(fig)
                plt.close("all")
                del fig

    def _make_png_files(self, force_rerun=False, **kwargs):

        # n_jobs = kwargs.pop("n_jobs", 1)
        do_tqdm = kwargs.pop("do_tqdm", self.do_tqdm)

        it = range(self.N_days)

        # make_single_frame = partial(self._make_single_frame(do_tqdm=do_tqdm, force_rerun=force_rerun, **kwargs))
        # make_single_frame = lambda i_day: self._make_single_frame(
        # i_day=i_day, do_tqdm=do_tqdm, force_rerun=force_rerun, **kwargs
        # )

        # if n_jobs == 1:

        if do_tqdm:
            it = tqdm(it, desc="Make individual frames")

        for i_day in it:
            self._make_single_frame(
                i_day=i_day,
                force_rerun=force_rerun,
                **kwargs,
            )
            # make_single_frame(i_day)

        # else:
        #     from pathos.pools import ProcessPool
        #     pool = ProcessPool(nodes=n_jobs)
        #     pool.map(make_single_frame, it)
        #     # p_umap(make_single_frame, it, num_cpus=n_jobs, do_tqdm=do_tqdm)

        # p_umap(make_single_frame, it, num_cpus=n_jobs)
        # with mp.Pool(n_jobs) as p:
        # iterator = p_uimap(make_single_frame, it)
        # for result in iterator:
        #     print(result) # prints '1a', '2b', '3c' in any order
        # list(tqdm(p.imap_unordered(make_single_frame, it), total=self.N_days))
        return None

    def _make_gif_file(self, gif_name):
        png_name = self._get_png_name(i_day=1)
        files_in = png_name.replace("000001", "*")
        subprocess.call(f"convert -delay 10 -loop 1 {files_in} {gif_name}", shell=True)
        subprocess.call(
            f"convert {gif_name} \( +clone -set delay 300 \) +swap +delete {gif_name}",
            shell=True,
        )
        return None

    def _make_video_file(self, video_name):
        png_name = self._get_png_name(i_day=1)
        files_in = png_name.replace("000001", "%06d")
        fps = 10
        subprocess.call(
            f"ffmpeg -loglevel warning -r {fps} -i {files_in} -vcodec mpeg4 -y -vb 40M {video_name}",
            shell=True,
        )
        return None

    def _remove_tmp_frames(self):
        png_name = self._get_png_name(i_day=1)
        shutil.rmtree(Path(png_name).parent)  # Path(png_name).parent.unlink() # delete file

    def _optimize_gif(self, gif_name):
        # pip install pygifsicle
        from pygifsicle import optimize

        if self.verbose:
            print("Optimize gif")
        optimize(gif_name, colors=100)


#%%


from collections import Counter, defaultdict


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

    # df = pd.DataFrame(counts, columns=['counts'])
    # df['vals'] = vals
    # df['states'] = df['vals'].replace(mapping)
    # d = df.groupby('states')['counts'].sum().to_dict()
    # for key in set(mapping.values()):
    #     if not key in d:
    #         d[key] = 0
    # return d


def get_inverse_mapping(mapping):
    inv_mapping = defaultdict(list)
    for key, val in mapping.items():
        inv_mapping[val].append(key)
    return dict(inv_mapping)


class AnimateSIR(AnimationBase):
    def __init__(
        self,
        filename,
        do_tqdm=False,
        verbose=False,
        N_max=None,
        df_counts=None,
        split_corona_types=False,
    ):
        super().__init__(
            filename,
            animation_type="animation",
            do_tqdm=do_tqdm,
            verbose=verbose,
            N_max=N_max,
        )
        self.mapping = {
            -1: "S",
            #  0: 'E', 1: 'E', 2:'E', 3: 'E',
            0: "I",
            1: "I",
            2: "I",
            3: "I",
            4: "I",
            5: "I",
            6: "I",
            7: "I",
            8: "R",
        }
        self.inverse_mapping = get_inverse_mapping(self.mapping)

        self.df_counts = df_counts
        self.__name__ = "AnimateSIR"
        self.split_corona_types = split_corona_types
        self._initialize_plot()

    def _initialize_plot(self):

        # self.colors = ["#7F7F7F", "#D62728", "#2CA02C"]
        self.d_colors = {
            "S": "#7F7F7F",
            "I": "#D62728",
            "I_UK": "#135DD8",
            "R": "#2CA02C",
        }  # orangy red: #D66727, normal red: #D62728

        factor = self.cfg["N_tot"] / 580_000

        self.norm_1000 = ImageNormalize(vmin=0.0, vmax=1000 * factor, stretch=LogStretch())
        self.norm_100 = ImageNormalize(vmin=0.0, vmax=100 * factor, stretch=LogStretch())
        self.norm_10 = ImageNormalize(vmin=0.0, vmax=10 * factor, stretch=LogStretch())
        self.f_norm = lambda x: ImageNormalize(vmin=0.0, vmax=x * factor, stretch=LogStretch())

        # self.states = ['S', 'E', 'I', 'R']
        if self.split_corona_types:
            self.states = ["S", "I", "I_UK", "R"]
        else:
            self.states = ["S", "I", "R"]

        self.state_names = {
            "S": "Susceptable",
            "I": r"Infected $\&$ Exposed",
            "I_UK": r"I $\&$ E UK",
            "R": "Recovered",
        }

        # create the new map
        cmap = mpl.colors.ListedColormap([self.d_colors["R"], self.d_colors["I"]])
        bounds = [0, 0.5, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        self._scatter_kwargs = dict(cmap=cmap, norm=norm, edgecolor="none")

        self._geo_plot_kwargs = {}
        self._geo_plot_kwargs["S"] = dict(alpha=0.2, norm=self.norm_1000)
        self._geo_plot_kwargs["R"] = dict(alpha=0.3, norm=self.norm_100)
        self._geo_plot_kwargs["I"] = dict(norm=self.norm_10)
        self._geo_plot_kwargs["I_UK"] = dict(norm=self.norm_10)

    def _initialize_data(self):

        if not self.is_valid_file:
            return None

        # calc counts and R_eff and N_tot
        if self.df_counts is None:
            self.df_counts = self._compute_df_counts()
        self.R_eff = self._compute_R_eff()
        self.R_eff_smooth = self._smoothen(
            self.R_eff,
            method=None,
            # method="savgol", window_length=11, polyorder=3
        )
        assert self.cfg["N_tot"] == self.df_counts.iloc[0].sum()

    def _compute_df_counts(self):
        counts_i_day = {}
        it = range(self.N_days)
        if self.do_tqdm:
            it = tqdm(it, desc="Creating df_counts")
        for i_day in it:
            counts_i_day[i_day] = unique_counter(self.my_state[i_day], mapping=self.mapping)
        df_counts = pd.DataFrame(counts_i_day).T
        return df_counts

    def _compute_R_eff(self):
        df_counts = self.df_counts
        I = df_counts["I"].values
        R = df_counts["R"].values
        S = (self.cfg["N_tot"] - df_counts[["I", "R"]].sum(axis=1)).values
        R_eff = -(S[1:] - S[:-1]) / (R[1:] - R[:-1])
        # R_eff = (I[1:] - I[:-1]) / (R[1:] - R[:-1]) + 1
        return R_eff

    def _smoothen(self, x, method="none", **kwargs):  # window_length=11, polyorder=3
        if method is None or method.lower() == "none":
            return x
        elif "savgol" in method:
            return signal.savgol_filter(
                x, **kwargs
            )  # window size used for filtering, # order of fitted polynomial
        elif any([s in method for s in ["moving", "rolling", "average"]]):
            return pd.Series(x).rolling(**kwargs).mean().values
        else:
            raise AssertionError(f"Got wrong type of method for _smoothen(), got {method}")

    def _interpolate_R_eff(self, R_eff):
        N = len(R_eff)
        x = np.arange(N)
        y = R_eff
        f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
        x_interpolated = np.linspace(0, N - 1, 10_000)
        y_interpolated = f(x_interpolated)
        df_R_eff = pd.DataFrame({"t": x_interpolated, "R_eff": y_interpolated})
        return df_R_eff

    def _get_mask(self, i_day, state, split_corona_types=False):
        # mask = np.isin(self.my_state[i_day], self.inverse_mapping[state])

        if split_corona_types and "I" in state:
            mask = np.isin(self.my_state[i_day], self.inverse_mapping["I"])
            if state == "I_UK":
                return mask & (self.my_corona_type == 1)
            elif state == "I":
                return mask & (self.my_corona_type == 0)
            else:
                pass
        else:
            return np.isin(self.my_state[i_day], self.inverse_mapping[state])

    def _plot_i_day(self, i_day, dpi=50, include_Bornholm=True):

        # Main plot
        k_scale = 1.7
        fig = plt.figure(figsize=(10 * k_scale, 12 * k_scale))
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

        for state in self.states:
            if state in self.df_counts.columns and self.df_counts.loc[i_day, state] == 0:
                continue

            mask = self._get_mask(i_day, state, self.split_corona_types)
            if mask.sum() == 0:
                continue

            ax.scatter_density(
                *self.coordinates[mask].T,
                color=self.d_colors[state],
                dpi=dpi,
                **self._geo_plot_kwargs[state],
            )

        if include_Bornholm:
            ax.set(xlim=(7.9, 15.3), ylim=(54.5, 58.2))
        else:
            ax.set(xlim=(7.9, 13.3), ylim=(54.5, 58.2))
        # ax.set(xlim=(9.2, 11.3), ylim=(57.1, 58), xlabel='Longitude') # NORDJYLLAND
        ax.set(xlabel="Longitude")
        ax.set_ylabel("Latitude", rotation=90)  # fontsize=20, labelpad=20

        kw_args_circle = dict(xdata=[0], ydata=[0], marker="o", color="w", markersize=18)
        circles = [
            Line2D(
                label=self.state_names[state],
                markerfacecolor=self.d_colors[state],
                **kw_args_circle,
            )
            for state in self.states
        ]
        ax.legend(handles=circles, loc="upper left", fontsize=34, frameon=False)

        s_legend = [
            utils.human_format(self.df_counts.loc[i_day, state], digits=1)
            for state in self.states
            if "UK" not in state
        ]
        delta_s = 0.0261
        for i, s in enumerate(s_legend):
            ax.text(
                0.41,
                0.9698 - i * delta_s,
                s,
                fontsize=34,
                transform=ax.transAxes,
                ha="right",
            )

        # # left, bottom, width, height
        # legend_background_box = [(0.023, 0.91), 0.398, 0.085]
        # ax.add_patch(
        #     mpatches.Rectangle(
        #         *legend_background_box,
        #         facecolor="white",
        #         edgecolor="white",
        #         transform=ax.transAxes,
        #     )
        # )

        # self.cfg = utils.Filename(self.filename)._Filename.simulation_parameters
        title = utils.dict_to_title(self.cfg)
        title += "\n\n" + "Simulation of COVID-19 epidemic with no intervention"
        ax.set_title(title, pad=40, fontsize=32)

        # secondary plots:

        # These are in unitless percentage of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.58, 0.75, 0.38 * 0.8, 0.08 * 0.8]

        background_box = [(0.49, 0.60), 0.49, 0.35]
        ax.add_patch(
            mpatches.Rectangle(
                *background_box,
                facecolor="white",
                edgecolor="white",
                transform=ax.transAxes,
            )
        )

        i_day_max = i_day + max(3, i_day * 0.1)

        # delta_width = 0 * width / 100
        ax2 = fig.add_axes([left, bottom, width, height])
        I_up_to_today = self.df_counts["I"].iloc[: i_day + 1] / self.cfg["N_tot"]
        ax2.plot(
            I_up_to_today.index,
            I_up_to_today,
            "-",
            color=self.d_colors["I"],
            lw=2,
        )
        ax2.plot(
            I_up_to_today.index[-1],
            I_up_to_today.iloc[-1],
            "o",
            color=self.d_colors["I"],
        )
        I_max = np.max(I_up_to_today)
        ax2.set(
            xlabel=r"$t \,\, \mathrm{(days)}$",
            ylim=(0, I_max * 1.2),
            xlim=(0, i_day_max),
        )
        decimals = max(int(-np.log10(I_max)) - 1, 0)  # max important, otherwise decimals=-1
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))
        ax2.text(
            0,
            1.18,
            "Infected",
            fontsize=22,
            transform=ax2.transAxes,
            rotation=0,
            ha="center",
        )
        ax2.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        add_spines(ax2)

        ax3 = fig.add_axes([left, bottom - height * 1.9, width, height])

        if i_day > 0:
            R_eff_up_to_today = self._interpolate_R_eff(self.R_eff_smooth[: i_day + 1])
            z = (R_eff_up_to_today["R_eff"] > 1) / 1
            ax3.scatter(
                R_eff_up_to_today["t"],
                R_eff_up_to_today["R_eff"],
                s=10,
                c=z,
                **self._scatter_kwargs,
            )
            R_eff_today = R_eff_up_to_today.iloc[-1]
            z_today = R_eff_today["R_eff"] > 1
            ax3.scatter(
                R_eff_today["t"],
                R_eff_today["R_eff"],
                s=100,
                c=z_today,
                **self._scatter_kwargs,
            )

        R_eff_max = 4
        ax3.axhline(1, ls="--", color="k", lw=1)  # x = 0
        ax3.set(
            xlabel=r"$t \,\, \mathrm{(days)}$",
            ylim=(0, R_eff_max * 1.1),
            xlim=(0, i_day_max),
        )
        ax3.text(
            0,
            1.18,
            r"$\mathcal{R}_\mathrm{eff}$",
            fontsize=26,
            transform=ax3.transAxes,
            rotation=0,
            ha="center",
        )
        ax3.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        add_spines(ax3)

        # add_compas_rose = False
        # if add_compas_rose:
        #     ax4 = fig.add_axes([0.13, 0.68, 0.1, 0.1])
        #     ax4.imshow(compas_rose_img, alpha=0.9)
        #     ax4.axis('off')  # clear x-axis and y-axis

        ax.text(
            0.70,
            0.97,
            f"Day: {i_day}",
            fontsize=34,
            transform=ax.transAxes,
            backgroundcolor="white",
        )
        # ax.text(0.012, 0.012, f"Simulation of COVID-19 epidemic with no intervention.", fontsize=24, transform=ax.transAxes, backgroundcolor='white')
        ax.text(
            0.99,
            0.01,
            f"Niels Bohr Institute\narXiv: 2007.XXXXX",
            ha="right",
            fontsize=20,
            transform=ax.transAxes,
            backgroundcolor="white",
        )

        scalebar = AnchoredSizeBar(
            ax.transData,
            longitudes_per_50km,
            "50 km",
            # longitudes_per_50km/5, '10 km', # NORDJYLLAND
            loc="upper left",
            sep=10,
            color="black",
            frameon=False,
            size_vertical=0.003,
            fontproperties=fontprops,
            bbox_to_anchor=Bbox.from_bounds(8, 57.4, 0, 0),
            # bbox_to_anchor=Bbox.from_bounds(9.3, 57.89, 0, 0), # NORDJYLLAND
            bbox_transform=ax.transData,
        )

        ax.add_artist(scalebar)

        plt.close("all")
        return fig, (ax, ax2, ax3)


#%%


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


class Animate_my_number_of_contacts(AnimationBase):
    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None):
        super().__init__(
            filename,
            animation_type="my_number_of_contacts",
            do_tqdm=do_tqdm,
            verbose=verbose,
            N_max=N_max,
        )
        self.__name__ = "Animate_my_number_of_contacts"

    def _plot_i_day(self, i_day):

        my_state_day = self.my_state[i_day]
        my_number_of_contacts_day0 = self.my_number_of_contacts[0]

        my_number_of_contacts_max = len(my_number_of_contacts_day0)

        weighted_quantile(
            np.arange(my_number_of_contacts_max),
            99,
            sample_weight=my_number_of_contacts_day0,
        )

        range_max = np.percentile(my_number_of_contacts_day0, 99.9)
        N_bins = int(range_max)

        fig, ax = plt.subplots()
        counts, edges, _ = ax.hist(
            my_number_of_contacts_day0[my_state_day == -1],
            range=(0, range_max),
            bins=N_bins,
            label="S",
            histtype="step",
            lw=2,
        )
        ax.hist(
            my_number_of_contacts_day0[my_state_day != -1],
            range=(0, range_max),
            bins=N_bins,
            label="EIR",
            histtype="step",
            lw=2,
        )

        mean_N = np.mean(my_number_of_contacts_day0[my_state_day == -1])
        ax.axvline(mean_N, label="Mean S", lw=1.5, alpha=0.8, ls="--")
        ax.hist(
            my_number_of_contacts_day0,
            range=(0, range_max),
            bins=N_bins,
            label="Total",
            color="gray",
            alpha=0.8,
            histtype="step",
            lw=1,
        )

        title = utils.dict_to_title(self.cfg)
        ax.text(
            -0.1,
            -0.13,
            f"Day: {i_day}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=30,
        )
        ax.set(
            xlabel="# of connections",
            ylabel="Counts",
            title=title,
            xlim=(0, range_max - 1),
            ylim=(10, None),
        )
        ax.set_yscale("log")
        ax.legend(fontsize=20)

        s_mean = r"$\mu_S = " + f"{mean_N:.1f}" + r"$"
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        weight = 0.95
        log_middle = 10 ** np.average(np.log10(ax.get_ylim()), weights=[1 - weight, weight])
        ax.text(
            mean_N + 5,
            log_middle,
            s_mean,
            ha="left",
            va="center",
            fontdict=dict(size=30, color=colors[0]),
            bbox=dict(ec=colors[0], fc="white", alpha=0.9),
        )

        return fig, ax


#%%


@njit
def hist2d_numba(data_2D, bins, ranges):
    H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)
    for t in range(data_2D.shape[0]):
        i = (data_2D[t, 0] - ranges[0, 0]) * delta[0]
        j = (data_2D[t, 1] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1
    return H


@njit
def get_ranges(x):
    return np.array(([x[:, 0].min(), x[:, 0].max()], [x[:, 1].min(), x[:, 1].max()]))


def histogram2d(data_2D, bins=None, ranges=None):
    if bins is None:
        print("No binning provided, using (100, 100) as default")
        bins = np.array((100, 100))
    if isinstance(bins, int):
        bins = np.array([bins, bins])
    elif isinstance(bins, list) or isinstance(bins, tuple):
        bins = np.array(bins)
    if ranges is None:
        ranges = get_ranges(data_2D)
        ranges[:, 0] *= 0.99
        ranges[:, 1] *= 1.01
    return hist2d_numba(data_2D, bins=bins, ranges=ranges)


def compute_N_box_index(coordinates, N_bins_x, N_bins_y, threshold=0.8, verbose=False):

    counts = histogram2d(coordinates, bins=(N_bins_x, N_bins_y))
    counts_1d = counts.flatten()

    counts_1d_nonzero = counts_1d[counts_1d > 0]
    counts_sorted = np.sort(counts_1d_nonzero)[::-1]

    # threshold = 0.8
    cumsum = np.cumsum(counts_sorted) / counts_sorted.sum()
    index = np.argmax(cumsum > threshold) + 1

    if verbose:
        print(f"{len(coordinates)=}")
        print(f"{len(counts_1d)=}")
        print(f"{len(counts_1d_nonzero)=}")
        print(f"{index=}")
        print(f"{index / len(counts_1d_nonzero)=}")
    return index, counts_1d


def compute_spatial_correlation_day(coordinates, my_state_day, N_bins_x, N_bins_y, verbose=False):

    counts_1d_all = histogram2d(coordinates, bins=(N_bins_x, N_bins_y)).flatten()
    counts_1d_I = histogram2d(
        coordinates[(-1 < my_state_day) & (my_state_day < 8)], bins=(N_bins_x, N_bins_y)
    ).flatten()

    counts_1d_nonzero_all = counts_1d_all[counts_1d_all > 0]
    counts_1d_nonzero_I = counts_1d_I[counts_1d_all > 0]

    f = counts_1d_nonzero_I / counts_1d_nonzero_I
    return np.corrcoef(f)[0, 1]


from functools import lru_cache


class InfectionHomogeneityIndex(AnimationBase):
    def __init__(self, filename):
        super().__init__(filename, animation_type="InfectionHomogeneityIndex")
        self.__name__ = "InfectionHomogeneityIndex"

    @lru_cache
    def _get_N_bins_xy(self):

        coordinates = self.coordinates

        lon_min = coordinates[:, 0].min()
        lon_max = coordinates[:, 0].max()
        lon_mid = np.mean([lon_min, lon_max])

        lat_min = coordinates[:, 1].min()
        lat_max = coordinates[:, 1].max()
        lat_mid = np.mean([lat_min, lat_max])

        N_bins_x = int(utils.haversine(lon_min, lat_mid, lon_max, lat_mid)) + 1
        N_bins_y = int(utils.haversine(lon_mid, lat_min, lon_mid, lat_max)) + 1

        return N_bins_x, N_bins_y

    def _compute_IHI(self, threshold):

        N_bins_x, N_bins_y = self._get_N_bins_xy()

        N = len(self.my_state)
        x = np.arange(N - 1)
        IHI = np.zeros(len(x))
        N_box_all, counts_1d_all = compute_N_box_index(
            self.coordinates, N_bins_x, N_bins_y, threshold=threshold
        )
        for i_day in x:
            my_state_day = self.my_state[i_day]
            coordinates_infected = self.coordinates[(-1 < my_state_day) & (my_state_day < 8)]
            N_box_infected, counts_1d_infected = compute_N_box_index(
                coordinates_infected, N_bins_x, N_bins_y, threshold=threshold
            )
            ratio_N_box = N_box_infected / N_box_all
            IHI[i_day] = ratio_N_box
        return IHI

    def _make_plot(self, verbose=False, savefig=True, force_rerun=False):

        pdf_name = str(Path("Figures/IHI") / "IHI_") + Path(self.filename).stem + ".pdf"
        if not Path(pdf_name).exists() or force_rerun:

            self.IHI_thresholds = {}
            for threshold in np.linspace(0.1, 0.9, 9):
                self.IHI_thresholds[threshold] = self._compute_IHI(threshold)
            self.IHI_thresholds = pd.DataFrame(self.IHI_thresholds)
            df = self.IHI_thresholds

            fig, ax = plt.subplots()
            for col in df:
                ax.plot(df.index, df[col], label=f"Threshold = {col:.1f}")
            ax.legend()
            title = utils.dict_to_title(self.cfg)
            ax.set(
                xlabel="Day",
                ylabel="Infection Homogeneity Index ",
                title=title,
                ylim=(0, 1),
            )
            if savefig:
                Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(pdf_name, dpi=600, bbox_inches="tight", pad_inches=0.3)
            return fig, ax

        else:
            if verbose:
                print(f"{pdf_name} already exists, skipping for now.")

    def make_plot(self, verbose=False, savefig=True, force_rerun=False):
        if self.is_valid_file:
            return self._make_plot(verbose=verbose, savefig=savefig, force_rerun=force_rerun)


# %%


#%%


@njit
def fraction_recovered(states_array):
    return (states_array == 8).mean()


@njit
def fraction_infected_or_exposed(states_array):
    I_or_E = np.arange(8)
    return np.isin(states_array, I_or_E).mean()


@njit
def nb_compute_daily_kommune_fraction(my_state, i_day, my_kommune, shapefile_kommune, agg_func):
    N = len(shapefile_kommune)
    N_kommuner = shapefile_kommune.max() + 1
    fracs = np.full(N, fill_value=-1, dtype=np.float32)
    for idx in range(N_kommuner):
        my_mask = my_kommune == idx
        if my_mask.sum() == 0:
            continue
        fracs[shapefile_kommune == idx] = agg_func(my_state[i_day][my_mask])
    return fracs


def compute_daily_kommune_fraction_recovered(df_kommuner, df_coordinates, my_state, i_day):
    my_kommune = df_coordinates["idx"].values
    shapefile_kommune = df_kommuner["idx"].values
    fracs = nb_compute_daily_kommune_fraction(
        my_state, i_day, my_kommune, shapefile_kommune, agg_func=fraction_recovered
    )
    fracs[fracs == -1] = np.nan
    df_kommuner["frac_R"] = fracs
    return df_kommuner


#%%

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter


class KommuneMapAnimation(AnimationBase):
    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None, shapefile_size="small"):
        super().__init__(
            filename,
            animation_type="kommune_animation",
            do_tqdm=do_tqdm,
            verbose=verbose,
            N_max=N_max,
        )

        self.shapefile_size = shapefile_size
        self.__name__ = "KommuneMapAnimation"
        self._load_kommune_data()

    def _load_kommune_data(self):

        df_kommuner, name_to_idx, idx_to_name = utils.load_kommune_shapefiles(
            self.shapefile_size, verbose=False
        )
        self.df_kommuner = df_kommuner[["KOMNAVN", "idx", "geometry"]]

    def _plot_i_day(self, i_day, normalize_legend=True):

        df_kommuner = compute_daily_kommune_fraction_recovered(
            self.df_kommuner, self.df_coordinates, self.my_state, i_day
        )

        vmin, vmax = 0, 1

        missing_kwds = {
            "color": "lightgrey",
            # "edgecolor": "crimson",
            "hatch": "///",
        }

        # Main plot
        k_scale = 1.7
        fig, ax = plt.subplots(figsize=(13 * k_scale, 13 * k_scale))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=-4)

        df_kommuner.plot(
            column="frac_R",
            ax=ax,
            legend=True,
            cax=cax,
            legend_kwds={
                "label": "Fraction Recovered",
                # "orientation": "horizontal",
            },
            missing_kwds=missing_kwds,
            vmin=vmin if normalize_legend else None,
            vmax=vmax if normalize_legend else None,
        )

        cax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        ax.set(xlim=(7.9, 15.3), ylim=(54.5, 58.2), xlabel="Longitude")
        ax.set_ylabel("Latitude", rotation=90)  # fontsize=20, labelpad=20

        title = utils.dict_to_title(self.cfg)
        title += "\n\n" + "Simulation of COVID-19 epidemic with no intervention"
        ax.set_title(title, pad=40, fontsize=32)

        ax.text(
            0.03,  # 0.70,
            0.95,  # 0.97,
            f"Day: {i_day}",
            fontsize=34,
            transform=ax.transAxes,
            backgroundcolor="white",
        )

        ax.text(
            0.98,
            0.02,
            f"Niels Bohr Institute\narXiv: 2007.XXXXX",
            ha="right",
            fontsize=20,
            transform=ax.transAxes,
            backgroundcolor="white",
        )

        scalebar = AnchoredSizeBar(
            ax.transData,
            longitudes_per_50km,
            "50 km",
            loc="lower left",
            sep=10,
            color="black",
            frameon=False,
            size_vertical=0.003,
            fontproperties=fontprops,
            bbox_to_anchor=Bbox.from_bounds(8, 54.5, 0, 0),
            bbox_transform=ax.transData,
        )

        ax.add_artist(scalebar)

        plt.close("all")
        return fig, ax


# animation = KommuneMapAnimation(filenames[0])
# fig, ax = animation._plot_i_day(50)
# fig

#%%


def animate_file(
    filename,
    do_tqdm=False,
    verbose=False,
    dpi=50,
    remove_frames=True,
    force_rerun=False,
    make_gif=True,
    optimize_gif=True,
    animate_kommuner=False,
    N_max=None,
    split_corona_types=False,
    **kwargs,
):

    animation = AnimateSIR(
        filename,
        do_tqdm=do_tqdm,
        verbose=verbose,
        N_max=N_max,
        split_corona_types=split_corona_types,
    )
    animation.make_animation(
        remove_frames=remove_frames,
        force_rerun=force_rerun,
        make_gif=make_gif,
        optimize_gif=optimize_gif,
        dpi=dpi,
        **kwargs,
    )

    # if animate_kommuner:
    #     kommune_animation = KommuneMapAnimation(filename, do_tqdm=do_tqdm, verbose=verbose)
    #     kommune_animation.make_animation(
    #         remove_frames=remove_frames,
    #         force_rerun=force_rerun,
    #         make_gif=False,
    #         normalize_legend=False,  # TODO: Set to True normally
    #         # **kwargs,
    #     )


#%%

reload(utils)
reload(animation_utils)
num_cores = utils.get_num_cores(num_cores_max)


network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
# print("Only keeping animations with ID_0 for now")
filenames = [filename for filename in network_files.iter_all_files() if "ID__0" in filename]
# filenames = [filename for filename in network_files.iter_all_files()]


N_files = len(filenames)
# filename = filenames[1]

if N_files <= 1:
    num_cores = 1

kwargs = dict(
    do_tqdm=True,
    verbose=True,
    force_rerun=False,
    # N_max=200,
    split_corona_types=True,
    include_Bornholm=False,
)


#%%

# filename = filenames[0]

# for filename in filenames:

#     animation = AnimateSIR(filename, do_tqdm=True, verbose=True, N_max=100, split_corona_types=True)

#     animation.make_animation(
#         remove_frames=True,
#         force_rerun=True,
#         make_gif=False,
#         optimize_gif=False,
#         dpi=50,
#         include_Bornholm=False,
#     )

# x = x

#%%

import sys

if __name__ == "__main__":

    if utils.is_ipython:
        print("Not running animations for now")

    else:

        if len(filenames) == 1:
            animate_file(filenames[0], **kwargs)

        else:
            if num_cores == 1:
                for filename in tqdm(filenames):
                    animate_file(filename, **kwargs)

            else:
                print(
                    f"Generating {N_files} animations using {num_cores} cores, please wait",
                    flush=True,
                )
                kwargs["do_tqdm"] = False
                kwargs["verbose"] = False

                p_umap(partial(animate_file, **kwargs), filenames, num_cpus=num_cores)

                # with mp.Pool(num_cores) as p:
                #     list(
                #         tqdm(
                #             p.imap_unordered(partial(animate_file, **kwargs), filenames),
                #             total=N_files,
                #         )
                #     )

        print("\n\nFinished generating animations!")


# from pympler.asizeof import asizeof
# from pympler import summary
# from pympler import muppy
# from pympler import tracker


# def get_size(obj):
#     "returns size i MiB"
#     return asizeof(obj) / 2 ** 20


# def print_size(animation, min_size=None):
#     print(f"animation = {get_size(animation):.1f} MiB")
#     for key, val in animation.__dict__.items():
#         if min_size is None or get_size(val) > min_size:
#             print(f"{key} = {get_size(val):.1f} MiB")


if False:

    # tr = tracker.SummaryTracker()
    # tr.print_diff()

    filename = "Data/network/958bc1a031/network_2020-10-12_958bc1a031_ID__0.hdf5"
    animation = AnimateSIR(filename, do_tqdm=True, verbose=True)

    # import cartopy.crs as ccrs
    # projected = gv.operation.project(points, projection=ccrs.GOOGLE_MERCATOR)

    # tr.print_diff()

    # # print_size(animation)
    # all_objects = muppy.get_objects()
    # summary1 = summary.summarize(all_objects)
    # summary.print_(summary1)

    animation._initialize_data()

    # tr.print_diff()

    # i_day = 100
    # animation._make_single_frame(i_day=i_day)
    # print_size(animation)
    # animation._make_png_files()

    for i_day in tqdm(range(30)):
        # animation._make_single_frame(i_day=i_day, force_rerun=True)
        _ = animation._plot_i_day(i_day, dpi=50)
        # if (i_day % 10) == 0:
        # print(f"\ni_day: {i_day}")
        # tr.print_diff()

    # summary2 = summary.summarize(muppy.get_objects())
    # diff = summary.get_diff(summary1, summary2)
    # summary.print_(diff)


#%%
