import numpy as np
from numba import njit
from scipy import interpolate
import pandas as pd
from pathlib import Path
from scipy.stats import uniform as sp_uniform
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tqdm import tqdm
import multiprocessing as mp
import awkward
import extra_funcs
from importlib import reload
import rc_params
rc_params.set_rc_params()

num_cores_max = 1

#%%

def get_animation_filenames():
    filenames = Path('Data_animation').glob(f'*.joblib')
    return [str(file) for file in sorted(filenames)]

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


def add_spines(ax, exclude=None):
    if exclude is None:
        exclude = []
    spines = ['left', 'bottom']
    for spine in spines:
        if not spine in exclude:
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(2)
    ax.tick_params(axis='x', pad=10)

from abc import ABC, abstractmethod
class AnimationBase(ABC):

    def __init__(self, file_in, animation_type='animation', do_tqdm=False, verbose=False, N_max=None):
        
        if isinstance(file_in, str):
            filename = file_in
            self._non_copy_constructor(filename, animation_type, do_tqdm, verbose, N_max)

        elif isinstance(file_in, AnimationBase):
            if verbose:
                print("Copying class")
            self._copy_constructor(file_in, animation_type, do_tqdm, verbose, N_max)
        else:
            raise AssertionError(f'Got wrong type of input to AnimationBase, got {type(file_in)}')

        self.__name__ = 'AnimationBase'

    def _non_copy_constructor(self, filename, animation_type, do_tqdm, verbose, N_max):
        self.filename = filename
        self.animation_type = animation_type
        self.do_tqdm = do_tqdm
        self.verbose = verbose
        if verbose:
            print(f"Loading: \n{self.filename}")
        self.which_state, self.coordinates, self.N_connections = joblib.load(filename, mmap_mode='r')
        if N_max is None:
            self.N = len(self.which_state)
        else:
            self.N = N_max
            self.N_truth = len(self.which_state)
        self.cfg = extra_funcs.filename_to_dotdict(filename, animation=True)


    def _copy_constructor(self, file_in, animation_type, do_tqdm, verbose, N_max):
        self.filename = file_in.filename
        self.animation_type = animation_type
        self.do_tqdm = do_tqdm
        self.verbose = verbose
        self.which_state = file_in.which_state
        self.coordinates = file_in.coordinates
        self.N_connections = file_in.N_connections
        if N_max is None:
            self.N = len(self.which_state)
        else:
            self.N = N_max
            self.N_truth = len(self.which_state)
        self.cfg = file_in.cfg

    def __repr__(self):
        s = f"{self.__name__}(filename='{self.filename}', animation_type='{self.animation_type}', do_tqdm={self.do_tqdm}, verbose={self.verbose}, N={self.N})"
        return s

        
    @abstractmethod
    def _plot_i_day(self, i_day, **kwargs):
        pass

    def make_animation(self, remove_frames=True, force_rerun=False, optimize_gif=True, **kwargs):
        name = f'{self.animation_type}_' + self._get_sim_pars_str() + '.gif'
        gifname = str(Path(f'Figures/{self.animation_type}') / name)

        if not Path(gifname).exists() or force_rerun:
            if self.verbose and not self.do_tqdm:
                print("\nMake individual frames", flush=True)
            try:
                self._initialize_plot_and_df_counts()
            except AttributeError:
                pass
            except:
                raise
            self._make_png_files(self.do_tqdm, force_rerun, **kwargs)

            if self.verbose:
                print("\nMake GIF", flush=True)
            self._make_gif_file(gifname)
            if optimize_gif:
                self._optimize_gif(gifname)
            
            if self.verbose:
                print("\nMake video", flush=True)
            self._make_video_file(gifname)

            if remove_frames:
                if self.verbose:
                    print("\nDelete temporary frames", flush=True)
                self._remove_tmp_frames()
        else:
            print(f'{self.animation_type} already exists.')
        return None

    def _get_sim_pars_str(self):
        return Path(self.filename).stem.replace('.animation', '')

    def _get_png_name(self, i_day):
        sim_pars_str = self._get_sim_pars_str()
        return f"Figures/{self.animation_type}/tmp_{sim_pars_str}/{self.animation_type}_{sim_pars_str}_frame_{i_day:06d}.png"

    def _make_png_files(self, do_tqdm, force_rerun=False, **kwargs):
        if 'dpi' in kwargs:
            dpi = kwargs['dpi']
        else:
            dpi = 50

        it = range(self.N)
        if do_tqdm:
            it = tqdm(it, desc='Make individual frames')
        for i_day in it:
            png_name = self._get_png_name(i_day)
            if not Path(png_name).exists() or force_rerun:
                fig, _ = self._plot_i_day(i_day, **kwargs)
                Path(png_name).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_name, dpi=dpi, bbox_inches='tight', pad_inches=0.3)
                plt.close(fig)
                plt.close('all')
        return None

    def _make_gif_file(self, gifname):
        png_name = self._get_png_name(i_day=1)
        files_in = png_name.replace("000001", "*")
        subprocess.call(f"convert -delay 10 -loop 1 {files_in} {gifname}", shell=True)
        subprocess.call(f"convert {gifname} \( +clone -set delay 300 \) +swap +delete {gifname}", shell=True) 
        return None
    
    def _make_video_file(self, gifname):
        png_name = self._get_png_name(i_day=1)
        files_in = png_name.replace("000001", "%06d")
        video_name = gifname.replace('gif', 'mp4')
        fps = 10
        subprocess.call(f"ffmpeg -r {fps} -i {files_in} -vcodec mpeg4 -y -vb 40M {video_name}", shell=True)
        return None

    def _remove_tmp_frames(self):
        png_name = self._get_png_name(i_day=1)
        shutil.rmtree(Path(png_name).parent) # Path(png_name).parent.unlink() # delete file

    def _optimize_gif(self, gifname):   
        # pip install pygifsicle
        from pygifsicle import optimize
        if self.verbose:
            print("Optimize gif")
        optimize(gifname, colors=100)



class AnimateSIR(AnimationBase):

    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None, df_counts=None):
        super().__init__(filename, animation_type='animation', do_tqdm=do_tqdm, verbose=verbose, N_max=N_max)
        self.mapping = {-1: 'S', 
                        #  0: 'E', 1: 'E', 2:'E', 3: 'E',
                         0: 'I', 1: 'I', 2:'I', 3: 'I',
                         4: 'I', 5: 'I', 6:'I', 7: 'I',
                         8: 'R',
                        }
        self.N_connections_max = self._get_N_connections_max()
        self.df_counts = df_counts
        self.__name__ = 'AnimateSIR'

    def _get_df(self, i_day):
        df = pd.DataFrame(self.coordinates, columns=['x', 'y'])
        df['which_state_num'] = self.which_state[i_day]
        df['N_connections_num'] = self.N_connections[i_day]
        df["which_state"] = df['which_state_num'].replace(self.mapping).astype('category')
        return df


    def _get_N_connections_max(self):
        return self._get_df(0)['N_connections_num'].max()


    def _initialize_plot_and_df_counts(self):

        self.colors = ['#7F7F7F', '#D62728', '#2CA02C']
        self.d_colors = {'S': '#7F7F7F', 'I': '#D62728', 'R': '#2CA02C'}
        
        factor = self.N / 500_000

        self.norm_1000 = ImageNormalize(vmin=0., vmax=1000*factor, stretch=LogStretch())
        self.norm_100 = ImageNormalize(vmin=0., vmax=100*factor, stretch=LogStretch())
        self.norm_10 = ImageNormalize(vmin=0., vmax=10*factor, stretch=LogStretch())

        # self.states = ['S', 'E', 'I', 'R']
        self.states = ['S', 'I', 'R']
        self.state_names = {'S': 'Susceptable', 'I': 'Infected & Exposed', 'R': 'Recovered'}

        # create the new map
        cmap = mpl.colors.ListedColormap([self.d_colors['R'], self.d_colors['I']])
        bounds = [0, 0.5, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        self._scatter_kwargs = dict(cmap=cmap, norm=norm, edgecolor='none')


        # calc counts and R_eff and N_tot
        if self.df_counts is None:
            self.df_counts = self._compute_df_counts()
        self.R_t = self._calc_infection_rate_R_t()
        self.N_tot = self.df_counts.iloc[0].sum()
        return None


    def _compute_df_counts(self):
        counts_i_day = {}
        it = range(self.N)
        if self.do_tqdm:
            it = tqdm(it, desc="Creating df_counts")
        for i_day in it:
            df = self._get_df(i_day)
            dfs = {s: df.query("which_state == @s") for s in self.states}
            counts_i_day[i_day] = {key: len(val) for key, val in dfs.items()}
        df_counts = pd.DataFrame(counts_i_day).T
        return df_counts


    def _interpolate_R_t(self, R_t):
        N = len(R_t)
        x = np.arange(N)
        y = R_t 
        f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
        x_interpolated = np.linspace(0, N-1, 10_000)
        y_interpolated = f(x_interpolated)
        df_R_t = pd.DataFrame({'t': x_interpolated, 'R_t': y_interpolated})
        return df_R_t


    def _calc_infection_rate_R_t(self, time_delay=1, laplace_factor=0):
        df_counts = self.df_counts
        I = df_counts['I']
        R = df_counts['R']
        N = len(df_counts)
        R_t = np.zeros(N)
        R_t[:time_delay] = np.nan
        for i in range(time_delay, N):
            num = I.iloc[i] - I.iloc[i-time_delay] + laplace_factor
            den = R.iloc[i] - R.iloc[i-time_delay] + laplace_factor
            if den != 0:
                R_t[i] = num / den + 1
            else:
                R_t[i] = np.nan
        # df_R_t = pd.Series(R_t)
        # plt.plot(R_t)
        # df_R_t.rolling(window=10).median().plot()
        return R_t


    def _plot_i_day(self, i_day, dpi):

        df = self._get_df(i_day)
        dfs = {s: df.query("which_state == @s") for s in self.states}

        # Main plot
        k_scale = 1.8
        fig = plt.figure(figsize=(10*k_scale, 13*k_scale))
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

        if len(dfs['S']) > 0:
            ax.scatter_density(dfs['S']['x'], dfs['S']['y'], color=self.d_colors['S'], alpha=0.2, norm=self.norm_1000, dpi=dpi)
        if len(dfs['R']) > 0:
            ax.scatter_density(dfs['R']['x'], dfs['R']['y'], color=self.d_colors['R'], alpha=0.3, norm=self.norm_100, dpi=dpi)
        if len(dfs['I']) > 0:
            ax.scatter_density(dfs['I']['x'], dfs['I']['y'], color=self.d_colors['I'], norm=self.norm_10, dpi=dpi)
        ax.set(xlim=(8, 13.7), ylim=(54.52, 58.2))

        kw_args_circle = dict(xdata=[0], ydata=[0], marker='o', color='w', markersize=16)
        circles = [Line2D(label=self.state_names[state], markerfacecolor=self.d_colors[state], **kw_args_circle) for state in self.states]
        ax.legend(handles=circles, loc='upper left', fontsize=20)

        cfg = extra_funcs.filename_to_dotdict(self.filename, animation=True)
        title = extra_funcs.dict_to_title(cfg)
        ax.set_title(title, pad=50, fontsize=22)

        # secondary plots:

        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.62, 0.76, 0.3*0.95, 0.08*0.9]

        background_box = [(0.51, 0.61), 0.47, 0.36]
        ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='lightgray', transform=ax.transAxes))

        i_day_max = i_day + max(3, i_day*0.1)

        # delta_width = 0 * width / 100
        ax2 = fig.add_axes([left, bottom, width, height])
        I_up_to_today = self.df_counts['I'].iloc[:i_day+1] / self.N_tot
        ax2.plot(I_up_to_today.index, I_up_to_today, '-', color=self.d_colors['I'])
        ax2.plot(I_up_to_today.index[-1], I_up_to_today.iloc[-1], 'o', color=self.d_colors['I'])
        I_max = np.max(I_up_to_today)
        ax2.set(xlabel='t', ylim=(0, I_max*1.2), xlim=(0, i_day_max))
        decimals = max(int(-np.log10(I_max)) - 1, 0) # max important, otherwise decimals=-1
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.text(-0.28, 0.2, 'Infected', fontsize=20, transform=ax2.transAxes, rotation=90)
        add_spines(ax2)

        ax3 = fig.add_axes([left, bottom-height*1.8, width, height])
        if i_day > 0:
            R_t_up_to_today = self._interpolate_R_t(self.R_t[:i_day+1])
            z = (R_t_up_to_today['R_t'] > 1) / 1
            ax3.scatter(R_t_up_to_today['t'], R_t_up_to_today['R_t'], s=10, c=z, **self._scatter_kwargs)
            R_t_today = R_t_up_to_today.iloc[-1]
            z_today = (R_t_today['R_t'] > 1)
            ax3.scatter(R_t_today['t'], R_t_today['R_t'], s=100, c=z_today, **self._scatter_kwargs)
        R_t_max = 3
        ax3.axhline(1, ls='--', color='k', lw=1) # x = 0
        # ax3.axhline(0, color='k', lw=2) # x = 0
        ax3.set(xlabel='t', ylim=(0, R_t_max), xlim=(0, i_day_max))
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.text(-0.25, 0.4, r'$\mathregular{R}_\mathregular{eff}$', fontsize=24, transform=ax3.transAxes, rotation=90)
        add_spines(ax3)

        ax.text(0.02, 0.02, f"Day: {i_day}", fontsize=24, transform=ax.transAxes)

        plt.close('all')
        return fig, (ax, ax2, ax3)



#%%

if False:

    animation = AnimateSIR(filename, verbose=True, do_tqdm=True, N_max=20)

    dpi=50
    remove_frames=True
    force_rerun=False
    optimize_gif=True

    animation.make_animation(
                                    remove_frames=remove_frames, 
                                    force_rerun=True, 
                                    optimize_gif=optimize_gif,
                                    dpi=dpi
                                    )


#%%


class Animate_N_connections(AnimationBase):

    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None):
        super().__init__(filename, animation_type='N_connections', do_tqdm=do_tqdm, verbose=verbose, N_max=N_max)
        self.__name__ = 'Animate_N_connections'


    def _plot_i_day(self, i_day):

        which_state_day = self.which_state[i_day]
        N_connections_day0 = self.N_connections[0]

        range_max = np.percentile(N_connections_day0, 99.9)
        N_bins = int(range_max)

        fig, ax = plt.subplots()
        counts, edges, _ = ax.hist(N_connections_day0[which_state_day == -1], range=(0, range_max), bins=N_bins, label='S', histtype='step', lw=2)
        ax.hist(N_connections_day0[which_state_day != -1], range=(0, range_max), bins=N_bins, label='EIR', histtype='step', lw=2)

        mean_N = np.mean(N_connections_day0[which_state_day == -1])
        ax.axvline(mean_N, label='Mean S', lw=1.5, alpha=0.8, ls='--')
        ax.hist(N_connections_day0, range=(0, range_max), bins=N_bins, label='Total', color='gray', alpha=0.8, histtype='step', lw=1)
        
        title = extra_funcs.dict_to_title(self.cfg)
        ax.text(-0.1, -0.13, f"Day: {i_day}", ha='left', va='top', transform=ax.transAxes, fontdict=dict(size=30))
        ax.set(xlabel='# of connections', ylabel='Counts', title=title, xlim=(0, range_max-1), ylim=(10, None))
        ax.set_yscale('log')
        ax.legend(fontsize=20)

        s_mean = r"$\mu_S = " + f'{mean_N:.1f}'+ r'$'
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        weight = 0.95
        log_middle = 10**np.average(np.log10(ax.get_ylim()), weights=[1-weight, weight])
        ax.text(mean_N+5, log_middle, s_mean, ha='left', va='center', fontdict=dict(size=30, color=colors[0]), bbox=dict(ec=colors[0], fc='white', alpha=0.9))

        return fig, ax

if False:
    animation_N = Animate_N_connections(filename, do_tqdm=True, verbose=True)

    animation_N.make_animation(remove_frames=remove_frames, 
                            force_rerun=True, 
                            optimize_gif=optimize_gif)



#%%

from SimulateDenmarkAgeHospitalization_extra_funcs import haversine
from extra_funcs import human_format

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
    return np.array(([x[:, 0].min(), x[:, 0].max()], 
                        [x[:, 1].min(), x[:, 1].max()]))

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


def compute_N_box_index(coordinates, N_bins_x, N_bins_y, verbose=False):

    counts = histogram2d(coordinates, bins=(N_bins_x, N_bins_y))
    counts_1d = counts.flatten()

    counts_1d_nonzero = counts_1d[counts_1d > 0]
    counts_sorted =  np.sort(counts_1d_nonzero)[::-1]
    
    threshold = 0.8
    cumsum = np.cumsum(counts_sorted) / counts_sorted.sum()
    index = np.argmax(cumsum > threshold) + 1
    if verbose:
        print(len(coordinates))
        print(len(counts_1d))
        print(len(counts_1d_nonzero))
        print(index)
        print(index / len(counts_1d_nonzero))
    return index, counts_1d


def get_N_bins_xy(coordinates):

    lon_min = coordinates[:, 0].min()
    lon_max = coordinates[:, 0].max()
    lon_mid = np.mean([lon_min, lon_max])

    lat_min = coordinates[:, 1].min()
    lat_max = coordinates[:, 1].max()
    lat_mid = np.mean([lat_min, lat_max])

    N_bins_x = int(haversine(lon_min, lat_mid, lon_max, lat_mid)) + 1
    N_bins_y = int(haversine(lon_mid, lat_min, lon_mid, lat_max)) + 1

    return N_bins_x, N_bins_y

def plot_IHI(file_in, verbose=True, savefig=True):

    if isinstance(file_in, str):
        animation = AnimateSIR(file_in, verbose=verbose)
    elif isinstance(file_in, AnimateSIR):
        animation = file_in
    else:
        raise AssertionError(f'Got wrong type of input to plot_IHI, got {type(file_in)}')
    
    filename = animation.filename
    coordinates = animation.coordinates
    out_which_state = animation.which_state

    N_bins_x, N_bins_y = get_N_bins_xy(coordinates)
    N_box_all, counts_1d_all = compute_N_box_index(coordinates, N_bins_x, N_bins_y)
    out_which_state = np.array(out_which_state, dtype=np.int8)

    x = np.arange(0, len(out_which_state)-1, 1)
    IHI = np.zeros(len(x))
    for i, i_day in enumerate(x):
        which_state_day = out_which_state[i_day]
        coordinates_infected = coordinates[(-1 < which_state_day) & (which_state_day < 8)]
        N_box_infected, counts_1d_infected = compute_N_box_index(coordinates_infected, N_bins_x, N_bins_y)
        ratio_N_box = N_box_infected / N_box_all
        IHI[i] = ratio_N_box

    fig, ax = plt.subplots()
    ax.plot(x, IHI)
    title = extra_funcs.dict_to_title(animation.cfg)
    ax.set(xlabel='Day', ylabel='Infection Homogeneity Index ', title=title)
    if savefig:
        pdf_name = str(Path('Figures/IHI') / 'IHI_') + Path(filename).stem + '.pdf'
        Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_name, dpi=600, bbox_inches='tight', pad_inches=0.3)


# %%

def animate_file(filename, do_tqdm=False, verbose=False, dpi=50, remove_frames=True, force_rerun=False, optimize_gif=True, make_IHI_plot=True, make_N_connections_animation=True):
    animation = AnimateSIR(filename, do_tqdm=do_tqdm, verbose=verbose)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Attempting to set identical")
        animation.make_animation(remove_frames=remove_frames, 
                                 force_rerun=force_rerun, 
                                 optimize_gif=optimize_gif,
                                 dpi=dpi,
                                 )
    if make_IHI_plot:
        if verbose:
            print(f"Making IHI plot")
        plot_IHI(animation, verbose=False, savefig=True)
    if make_N_connections_animation:
        animation_N_connections = Animate_N_connections(animation, do_tqdm=do_tqdm, verbose=verbose)
        animation_N_connections.make_animation(remove_frames=remove_frames, 
                                               force_rerun=force_rerun, 
                                               optimize_gif=optimize_gif)

    return None

def get_num_cores(num_cores_max, subtract_cores=1):
    num_cores = mp.cpu_count() - subtract_cores
    if num_cores >= num_cores_max:
        num_cores = num_cores_max
    return num_cores


#%%

num_cores = get_num_cores(num_cores_max)

filenames = get_animation_filenames()
filename = filenames[-1]
N_files = len(filenames)


if False:
    animation = AnimateSIR(filename, do_tqdm=True, verbose=True)
    fig, ax = plt.subplots()
    ax.hist(animation.N_connections[0], 100, range=(0, 200));
    ax.set(xlabel='N_connections', ylabel='Counts')
    # fig.savefig('N_connections_rho_300_algo_1.pdf')


if False:
    animate_file(filename, do_tqdm=True, verbose=True, force_rerun=False, make_IHI_plot=True, make_N_connections_animation=True)


if False: # XXX

    for filename in tqdm(filenames, desc='Making IHI plots'):
        if 'epsilon_rho__0.0__' in filename:
            plot_IHI(filename, verbose=False, savefig=True)

#%%

animation = AnimateSIR(filename, do_tqdm=True, verbose=True)
animation._initialize_plot_and_df_counts()

#%%


#%%

if __name__ == '__main__':

    if False:

        if num_cores == 1:

            for filename in tqdm(filenames):
                animate_file(filename, do_tqdm=True, verbose=True, force_rerun=False)

        else:
            print(f"Generating frames using {num_cores} cores, please wait", flush=True)
            with mp.Pool(num_cores) as p:
                list(tqdm(p.imap_unordered(animate_file, filenames), total=N_files))



#%%
