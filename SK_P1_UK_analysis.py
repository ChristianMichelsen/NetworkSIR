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

num_cores_max = 15

#%%

def get_SK_P1_UK_filenames():
    filenames = Path('Data_SK_P1_UK').glob(f'*.joblib')
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


from copy import copy
class AnimateSIR:

    def __init__(self, filename, do_tqdm=False, verbose=False, df_counts=None, **R_t_kwargs):
        self.filename = filename
        self.do_tqdm = do_tqdm
        self.verbose = verbose
        if verbose:
            print(f"Loading: \n{filename}")
        self.SK, self.P1, self.UK = joblib.load(filename)
        # filename_AK = filename.replace('SK_P1_UK.joblib', 'AK_initial.parquet')
        # self.AK = awkward.fromparquet(filename_AK)
        # filename_Rate = filename_AK.replace('AK_initial.parquet', 'Rate_initial.parquet')
        # self.Rate = awkward.fromparquet(filename_Rate)
        self.N = len(self.SK)
        self.mapping = {-1: 'S', 
                        #  0: 'E', 1: 'E', 2:'E', 3: 'E',
                         0: 'I', 1: 'I', 2:'I', 3: 'I',
                         4: 'I', 5: 'I', 6:'I', 7: 'I',
                         8: 'R',
                        }
        self.UK_max = self._get_UK_max()
        self._init_plot()
        self.df_counts = df_counts if df_counts is not None else self._compute_df_counts()
        self.R_t = self._calc_infection_rate_R_t(**R_t_kwargs)
        self.N0 = self.df_counts.iloc[0].sum()

    def _get_df(self, i_day):
        df = pd.DataFrame(self.P1, columns=['x', 'y'])
        df['SK_num'] = self.SK[i_day]
        df['UK_num'] = self.UK[i_day]
        df["SK"] = df['SK_num'].replace(self.mapping).astype('category')
        return df
    
    def _get_UK_max(self):
        return self._get_df(0)['UK_num'].max()

    def _init_plot(self):

        # self.colors = ['#7F7F7F', '#1F77B4', '#D62728', '#2CA02C']
        self.colors = ['#7F7F7F', '#D62728', '#2CA02C']
        self.d_colors = {'S': '#7F7F7F', 'I': '#D62728', 'R': '#2CA02C'}
        
        self.norm_1000 = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
        self.norm_100 = ImageNormalize(vmin=0., vmax=100, stretch=LogStretch())
        self.norm_10 = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())

        # self.states = ['S', 'E', 'I', 'R']
        self.states = ['S', 'I', 'R']
        self.state_names = {'S': 'Susceptable', 'I': 'Infected', 'R': 'Recovered'}

        # create the new map
        cmap = mpl.colors.ListedColormap([self.d_colors['R'], self.d_colors['I']])
        bounds = [0, 0.5, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        self._scatter_kwargs = dict(cmap=cmap, norm=norm, edgecolor='none')
        return None

    def _compute_df_counts(self):
        counts_i_day = {}
        it = range(self.N)
        if self.do_tqdm:
            it = tqdm(it, desc="Creating df_counts")
        for i_day in it:
            df = self._get_df(i_day)
            dfs = {s: df.query("SK == @s") for s in self.states}
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


    def _plot_df_i_day(self, i_day, dpi):

        df = self._get_df(i_day)
        dfs = {s: df.query("SK == @s") for s in self.states}

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

        cfg = extra_funcs.filename_to_dotdict(self.filename, SK_P1_UK=True)
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
        I_up_to_today = self.df_counts['I'].iloc[:i_day+1] / self.N0
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
        ax3.text(-0.25, 0.4, r'$\mathregular{R_\infty}$', fontsize=24, transform=ax3.transAxes, rotation=90)
        add_spines(ax3)

        ax.text(0.02, 0.02, f"Day: {i_day}", fontsize=24, transform=ax.transAxes)

        plt.close('all')
        return fig, (ax, ax2, ax3)


    def make_animation(self, dpi=50, remove_frames=True, force_rerun=False, optimize_gif=True):
        filename = self.filename
        name = 'animation_' + Path(filename).stem + '.gif'
        gifname = str(Path('Figures_SK_P1_UK') / name)

        if not Path(gifname).exists() or force_rerun:
            if self.verbose and not self.do_tqdm:
                print("\nMake individual frames", flush=True)
            self._make_png_files(dpi, self.do_tqdm, force_rerun)

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
        return None

    def _get_sim_pars_str(self):
        return Path(self.filename).stem.replace('.SK_P1_UK', '')

    def _get_png_name(self, i_day):
        sim_pars_str = self._get_sim_pars_str()
        return f"Figures_SK_P1_UK/tmp_{sim_pars_str}/animation_{sim_pars_str}_frame_{i_day:06d}.png"

    def _make_png_files(self, dpi, do_tqdm, force_rerun=False):
        it = range(self.N)
        if do_tqdm:
            it = tqdm(it, desc='Make individual frames')
        for i_day in it:
            png_name = self._get_png_name(i_day)
            if not Path(png_name).exists() or force_rerun:
                fig, _ = self._plot_df_i_day(i_day, dpi)
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


# %%

def animate_file(filename, do_tqdm=False, verbose=False, dpi=50, remove_frames=True, force_rerun=False, optimize_gif=True):
    animation = AnimateSIR(filename, do_tqdm=do_tqdm, verbose=verbose)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Attempting to set identical")
        animation.make_animation(dpi=dpi, 
                                 remove_frames=remove_frames, 
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

filenames = get_SK_P1_UK_filenames()
filename = filenames[1]
N_files = len(filenames)           

# animate_file(filename, do_tqdm=True, verbose=True, force_rerun=True)

#%%

# for filename in tqdm(filenames):
#     animate_file(filename, do_tqdm=True, verbose=True, force_rerun=True)

if __name__ == '__main__':
    print(f"Generating frames using {num_cores} cores, please wait", flush=True)
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(animate_file, filenames), total=N_files))
