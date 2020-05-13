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


#%%
import shutil
# import imageio # conda install imageio
import subprocess
import warnings

from copy import copy
class AnimateSIR:

    def __init__(self, filename, verbose=False):
        self.filename = filename
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

        self.norm_1000 = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
        self.norm_100 = ImageNormalize(vmin=0., vmax=100, stretch=LogStretch())
        self.norm_10 = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())

        # self.states = ['S', 'E', 'I', 'R']
        self.states = ['S', 'I', 'R']
        return None

    def _plot_df_i_day(self, i_day, dpi):

        df = self._get_df(i_day)
        dfs = {s: df.query("SK == @s") for s in self.states}

        # Main plot
        k_scale = 1.8
        fig = plt.figure(figsize=(10*k_scale, 13*k_scale))
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

        if len(dfs['S']) > 0:
            ax.scatter_density(dfs['S']['x'], dfs['S']['y'], color=self.colors[0], alpha=0.2, norm=self.norm_1000, dpi=dpi)
        if len(dfs['R']) > 0:
            ax.scatter_density(dfs['R']['x'], dfs['R']['y'], color=self.colors[2], alpha=0.3, norm=self.norm_100, dpi=dpi)
        # if len(dfs['E']) > 0:
            # ax.scatter_density(dfs['E']['x'], dfs['E']['y'], color=self.colors[1], norm=self.norm_10, dpi=dpi)
        if len(dfs['I']) > 0:
            ax.scatter_density(dfs['I']['x'], dfs['I']['y'], color=self.colors[1], norm=self.norm_10, dpi=dpi)
        ax.set(xlim=(8, 13.7), ylim=(54.52, 58.2))

        kw_args_circle = dict(xdata=[0], ydata=[0], marker='o', color='w', markersize=12)
        circles = [Line2D(label=state, markerfacecolor=color, **kw_args_circle) for color, state in zip(self.colors, self.states)]
        ax.legend(handles=circles, loc='upper left')

        cfg = extra_funcs.filename_to_dotdict(self.filename, SK_P1_UK=True)
        title = extra_funcs.dict_to_title(cfg)
        ax.set_title(title, pad=50, fontsize=20)

        # secondary plots:

        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.67, 0.76, 0.25, 0.1]

        background_box = [(0.6, 0.577), 0.38, 0.42]
        ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='lightgray', transform=ax.transAxes))

        delta_width = 6 * width / 100
        ax2 = fig.add_axes([left+delta_width, bottom, width-delta_width, height])
        ax2.hist(df['UK_num'], range=(0, self.UK_max), bins=self.UK_max, color=self.colors[0])
        ax2.set(xlabel='# Connections', ylabel='Counts')

        ax3 = fig.add_axes([left, bottom-height*1.6, width, height])
        x, y = np.unique(df['SK_num'], return_counts=True)
        ax3.bar(x, y, color=self.colors[0])
        ax3.xaxis.set_ticks(np.arange(-1, 9))
        ax3.set(xlabel='State', ylabel='Counts')
        ax3.set_yscale('log')

        ax.text(0.02, 0.02, f"Day: {i_day}", fontsize=24, transform=ax.transAxes)

        plt.close('all')
        return fig, ax


    def make_animation(self, dpi=50, do_tqdm=False, remove_frames=True, force_rerun=False, optimize_gif=True):
        filename = self.filename
        name = 'animation_' + Path(filename).stem + '.gif'
        gifname = str(Path('Figures_SK_P1_UK') / name)

        if not Path(gifname).exists() or force_rerun:
            if self.verbose:
                print("Make individual frames", flush=True)
            self._make_png_files(dpi, do_tqdm)

            if self.verbose:
                print("Make GIF", flush=True)
            self._make_gif_file(gifname)
            if optimize_gif:
                self._optimize_gif(gifname)
            
            if self.verbose:
                print("Make video", flush=True)
            self._make_video_file(gifname)

            if remove_frames:
                if self.verbose:
                    print("Delete temporary frames", flush=True)
                self._remove_tmp_frames()
        return None

    def _get_sim_pars_str(self):
        return Path(self.filename).stem.replace('.SK_P1_UK', '')

    def _get_png_name(self, i_day):
        sim_pars_str = self._get_sim_pars_str()
        return f"Figures_SK_P1_UK/tmp_{sim_pars_str}/animation_{sim_pars_str}_frame_{i_day:06d}.png"

    def _make_png_files(self, dpi, do_tqdm):
        it = range(self.N)
        if do_tqdm:
            it = tqdm(it, desc='Make individual frames')
        for i_day in it:
            fig, ax = self._plot_df_i_day(i_day, dpi)
            png_name = self._get_png_name(i_day)
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

def animate_file(filename, do_tqdm=False):
    animation = AnimateSIR(filename, verbose=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Attempting to set identical")
        animation.make_animation(dpi=50, do_tqdm=do_tqdm)
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

animate_file(filename, do_tqdm=True)

# for filename in tqdm(filenames):
#     animate_file(filename)

if __name__ == '__main__':
    print(f"Generating frames using {num_cores} cores, please wait", flush=True)
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(animate_file, filenames), total=N_files))
