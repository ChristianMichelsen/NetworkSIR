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
from functools import partial
import extra_funcs
from importlib import reload
import h5py
import rc_params
rc_params.set_rc_params(fig_dpi=50) # 

num_cores_max = 1

#%%

def get_animation_filenames():
    filenames = Path('Data_animation').glob(f'*.animation.hdf5')
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
from scipy import signal


from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=24)

longitudes_per_50km = 0.8392
compas_rose_img = plt.imread('Figures/CompasRose/CompasRose.png')


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
# class AnimationBase(ABC):
class AnimationBase():

    def __init__(self, filename, animation_type='animation', do_tqdm=False, verbose=False, N_max=None, load_into_memory=False):

        self.filename = filename
        self.animation_type = animation_type
        self.do_tqdm = do_tqdm
        self.verbose = verbose
        self.load_into_memory = load_into_memory
        # if verbose:
            # print(f"Loading: \n{self.filename}")
        self._load_hdf5_file()

        self.N_max = N_max
        if self._is_valid_file:
            if N_max is None:
                self.N_days = len(self.which_state)
            else:
                if N_max < 12:
                    print(f"N_max has to be 12 or larger (choosing 12 instead of {N_max} for now).")
                    N_max = 12
                self.N_days = N_max
        
        self.cfg = extra_funcs.filename_to_dotdict(filename, animation=True)
        self.__name__ = 'AnimationBase'
        
    def _load_hdf5_file(self):
        try:
            f = h5py.File(self.filename, "r")
            self._is_valid_file = True
        except OSError:
            print(f"\n\n\n!!! Error at {self.filename} !!! \n\n\n")
            self._is_valid_file = False
            return None

        self.f = f
        self.f_is_open = True
        self.coordinates = f["coordinates"][()]
        self.df_raw = pd.DataFrame(f["df"][()])
        self.which_state = f["which_state"]
        self.N_connections = f["N_connections"]
        if self.load_into_memory:
            self.which_state = self.which_state[()]
            self.N_connections = self.N_connections[()]
        # g = awkward.hdf5(f)
        # g["which_connections"] 
        # g["individual_rates"] 

    @property
    def is_valid_file(self):
        if not self._is_valid_file:
            if self.verbose:
                print(f"Still error at {self.filename}")
            return False
        else:
            return True

    def __enter__(self):
        if not self.is_valid_file:
            return None

        if not self.f_is_open:
            print(f"Reloading {self.filename}")
            self._load_hdf5_file()
        return self

    def __exit__(self, type, value, traceback):
        if not self.is_valid_file:
            return None

        self.f.close()
        self.f_is_open = False

    def __repr__(self):
        s = f"{self.__name__}(filename='{self.filename}', animation_type='{self.animation_type}', do_tqdm={self.do_tqdm}, verbose={self.verbose}, N_max={self.N_max})"
        return s

    # @abstractmethod
    # def _plot_i_day(self, i_day, **kwargs):
    #     pass

    def _make_animation(self, remove_frames=True, force_rerun=False, optimize_gif=True, **kwargs):

        name = f'{self.animation_type}_' + self._get_sim_pars_str() + '.gif'
        gifname = str(Path(f'Figures/{self.animation_type}') / name)


        if not Path(gifname).exists() or force_rerun:
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
            if self.verbose:
                print(f'{self.animation_type} already exists.')


    def make_animation(self, remove_frames=True, force_rerun=False, optimize_gif=True, **kwargs):
        
        if not self.is_valid_file:
            return None

        try:
            self._make_animation(remove_frames=remove_frames, force_rerun=force_rerun, optimize_gif=optimize_gif, **kwargs)

        except OSError as e:
                print(f"\n\n\nOSError at {filename} \n\n\n")
                print(e)
        

        except ValueError as e:
                print(f"\n\n\nValueError at {filename} \n\n\n")
                print(e)



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

        it = range(self.N_days)
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
        subprocess.call(f"ffmpeg -loglevel warning -r {fps} -i {files_in} -vcodec mpeg4 -y -vb 40M {video_name}", shell=True)
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

    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None, load_into_memory=False, df_counts=None):
        super().__init__(filename, animation_type='animation', do_tqdm=do_tqdm, verbose=verbose, N_max=N_max, load_into_memory=load_into_memory)
        self.mapping = {-1: 'S', 
                        #  0: 'E', 1: 'E', 2:'E', 3: 'E',
                         0: 'I', 1: 'I', 2:'I', 3: 'I',
                         4: 'I', 5: 'I', 6:'I', 7: 'I',
                         8: 'R',
                        }
        # self.N_connections_max = self._get_N_connections_max()
        self.df_counts = df_counts
        self.__name__ = 'AnimateSIR'
        self.dfs_all = {}
        self._initialize_plot()
        
    def __getstate__(self):
        d_out = dict(filename=self.filename, 
                     do_tqdm=self.do_tqdm, 
                     verbose=self.verbose, 
                     N_max=self.N_max, 
                     load_into_memory=self.load_into_memory, 
                     df_counts=self.df_counts, 
                     dfs_all=self.dfs_all, 
                     d_colors=self.d_colors,
                     )
        return d_out
    

    def __setstate__(self, d_in):
        self.__init__(
                      filename=d_in['filename'], 
                      do_tqdm=d_in['do_tqdm'], 
                      verbose=d_in['verbose'], 
                      N_max=d_in['N_max'], 
                      load_into_memory=d_in['load_into_memory'],
                      df_counts=d_in['df_counts'],
        )
        self.dfs_all = d_in['dfs_all']
        self.d_colors = d_in['d_colors']
        


    def _get_df(self, i_day):
        df = pd.DataFrame(self.coordinates, columns=['x', 'y'])
        df['which_state_num'] = self.which_state[i_day]
        df['N_connections_num'] = self.N_connections[i_day]
        df["which_state"] = df['which_state_num'].replace(self.mapping).astype('category')
        return df


    # def _get_N_connections_max(self):
    #     return self._get_df(0)['N_connections_num'].max()


    def _initialize_plot(self):

        self.colors = ['#7F7F7F', '#D62728', '#2CA02C']
        self.d_colors = {'S': '#7F7F7F', 'I': '#D62728', 'R': '#2CA02C'} # orangy red: #D66727, normal red: #D62728
        
        factor = self.cfg['N_tot'] / 580_000

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


    def _initialize_data(self):

        if not self.is_valid_file:
            return None

        # calc counts and R_eff and N_tot
        if self.df_counts is None:
            self.df_counts = self._compute_df_counts()
        self.R_eff = self._compute_R_eff()
        self.R_eff_smooth = self._smoothen(self.R_eff, method='savgol', window_length=11, polyorder=3)
        # self.N_tot = self.df_counts.iloc[0].sum()
        assert self.cfg['N_tot'] == self.df_counts.iloc[0].sum()

        it = range(self.N_days)
        if self.do_tqdm:
            it = tqdm(it, desc="dfs_all")
        for i_day in it:
        # print("XXX")
        # for i_day in tqdm(range(self.N_days), desc='dfs_all'):
            df = self._get_df(i_day)
            self.dfs_all[i_day] = {s: df.query("which_state == @s") for s in self.states}


    def _compute_df_counts(self):
        counts_i_day = {}
        it = range(self.N_days)
        if self.do_tqdm:
            it = tqdm(it, desc="Creating df_counts")
        for i_day in it:
            df = self._get_df(i_day)
            dfs = {s: df.query("which_state == @s") for s in self.states}
            counts_i_day[i_day] = {key: len(val) for key, val in dfs.items()}
        df_counts = pd.DataFrame(counts_i_day).T
        return df_counts

    def _compute_R_eff(self):
        df_counts =  self.df_counts
        I = df_counts['I'].values
        R = df_counts['R'].values
        R_eff = (I[1:] - I[:-1]) / (R[1:] - R[:-1]) + 1
        return R_eff
    
    def _smoothen(self, x, method='savgol', **kwargs): # window_length=11, polyorder=3
        if 'savgol' in method:
            return signal.savgol_filter(x, **kwargs)  # window size used for filtering, # order of fitted polynomial
        elif any([s in method for s in ['moving', 'rolling', 'average']]):
            return pd.Series(x).rolling(**kwargs).mean().values
        else:
            raise AssertionError(f"Got wrong type of method for _smoothen(), got {method}")

    def _interpolate_R_eff(self, R_eff):
        N = len(R_eff)
        x = np.arange(N)
        y = R_eff 
        f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
        x_interpolated = np.linspace(0, N-1, 10_000)
        y_interpolated = f(x_interpolated)
        df_R_eff = pd.DataFrame({'t': x_interpolated, 'R_eff': y_interpolated})
        return df_R_eff

    def _plot_i_day(self, i_day, dpi=50):

        # df = self._get_df(i_day)
        # dfs = {s: df.query("which_state == @s") for s in self.states}
        dfs = self.dfs_all[i_day]

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
        ax.set(xlim=(7.9, 13.7), ylim=(54.4, 58.2), xlabel='Longitude')
        ax.set_ylabel('Latitude', rotation=90) # fontsize=20, labelpad=20


        kw_args_circle = dict(xdata=[0], ydata=[0], marker='o', color='w', markersize=18)
        circles = [Line2D(label=self.state_names[state], 
                          markerfacecolor=self.d_colors[state], **kw_args_circle)
                          for state in self.states]
        ax.legend(handles=circles, loc='upper left', fontsize=24, frameon=False)

        # string = "\n".join([human_format(len(dfs[state]), decimals=1) for state in self.states])
        s_legend = [human_format(len(dfs[state]), decimals=1) for state in self.states]
        delta_s = 0.0261
        for i, s in enumerate(s_legend):
            ax.text(0.41, 0.9698-i*delta_s, s, fontsize=24, transform=ax.transAxes, ha='right')
        
        # left, bottom, width, height
        legend_background_box = [(0.023, 0.91), 0.398, 0.085] 
        ax.add_patch(mpatches.Rectangle(*legend_background_box, facecolor='white', edgecolor='white', transform=ax.transAxes))


        cfg = extra_funcs.filename_to_dotdict(self.filename, animation=True)
        title = extra_funcs.dict_to_title(cfg)
        ax.set_title(title, pad=50, fontsize=28)

        # secondary plots:

        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.56, 0.75, 0.39*0.8, 0.08*0.8]

        background_box = [(0.49, 0.60), 0.49, 0.35]
        ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='white', transform=ax.transAxes))

        i_day_max = i_day + max(3, i_day*0.1)

        # delta_width = 0 * width / 100
        ax2 = fig.add_axes([left, bottom, width, height])
        I_up_to_today = self.df_counts['I'].iloc[:i_day+1] / self.cfg['N_tot'] 
        ax2.plot(I_up_to_today.index, I_up_to_today, '-', color=self.d_colors['I'])
        ax2.plot(I_up_to_today.index[-1], I_up_to_today.iloc[-1], 'o', color=self.d_colors['I'])
        I_max = np.max(I_up_to_today)
        ax2.set(xlabel=r'$t \,\, \mathrm{(days)}$', ylim=(0, I_max*1.2), xlim=(0, i_day_max))
        decimals = max(int(-np.log10(I_max)) - 1, 0) # max important, otherwise decimals=-1
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=decimals))
        ax2.text(0, 1.18, 'Infected', fontsize=22, transform=ax2.transAxes, rotation=0, ha="center")
        ax2.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        add_spines(ax2)

        ax3 = fig.add_axes([left, bottom-height*1.9, width, height])

        if i_day > 0:
            R_eff_up_to_today = self._interpolate_R_eff(self.R_eff_smooth[:i_day+1])
            z = (R_eff_up_to_today['R_eff'] > 1) / 1
            ax3.scatter(R_eff_up_to_today['t'], R_eff_up_to_today['R_eff'], s=10, c=z, **self._scatter_kwargs)
            R_eff_today = R_eff_up_to_today.iloc[-1]
            z_today = (R_eff_today['R_eff'] > 1)
            ax3.scatter(R_eff_today['t'], R_eff_today['R_eff'], s=100, c=z_today, **self._scatter_kwargs)
        
        R_eff_max = 4
        ax3.axhline(1, ls='--', color='k', lw=1) # x = 0
        ax3.set(xlabel=r'$t \,\, \mathrm{(days)}$', ylim=(0, R_eff_max*1.1), xlim=(0, i_day_max))
        ax3.text(0, 1.18, r'$\mathcal{R}_\mathregular{eff}$', fontsize=26, transform=ax3.transAxes, rotation=0, ha='center')
        ax3.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        add_spines(ax3)

        add_compas_rose = False
        if add_compas_rose:
            ax4 = fig.add_axes([0.13, 0.68, 0.1, 0.1])
            ax4.imshow(compas_rose_img, alpha=0.9)
            ax4.axis('off')  # clear x-axis and y-axis

        ax.text(0.70, 0.97, f"Day: {i_day}", fontsize=34, transform=ax.transAxes, backgroundcolor='white')
        ax.text(0.012, 0.012, f"Simulation of COVID-19 epidemic with no intervention.", fontsize=24, transform=ax.transAxes, backgroundcolor='white')
        ax.text(0.99, 0.01, f"Niels Bohr Institute\narXiv: 2006.XXXXX", ha='right', fontsize=18, transform=ax.transAxes, backgroundcolor='white')

        scalebar = AnchoredSizeBar(ax.transData,
                                longitudes_per_50km, '50 km', 
                                loc='upper left', 
                                sep=10,
                                color='black',
                                frameon=False,
                                size_vertical=0.003,  
                                fontproperties=fontprops,
                                bbox_to_anchor=Bbox.from_bounds(8, 57.8, 0, 0),
                                bbox_transform=ax.transData
                                )

        ax.add_artist(scalebar)


        plt.close('all')
        return fig, (ax, ax2, ax3)



#%%


class Animate_N_connections(AnimationBase):

    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None, load_into_memory=False):
        super().__init__(filename, animation_type='N_connections', do_tqdm=do_tqdm, verbose=verbose, N_max=N_max, load_into_memory=load_into_memory)
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
        ax.text(-0.1, -0.13, f"Day: {i_day}", ha='left', va='top', transform=ax.transAxes, fontsize=30)
        ax.set(xlabel='# of connections', ylabel='Counts', title=title, xlim=(0, range_max-1), ylim=(10, None))
        ax.set_yscale('log')
        ax.legend(fontsize=20)

        s_mean = r"$\mu_S = " + f'{mean_N:.1f}'+ r'$'
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        weight = 0.95
        log_middle = 10**np.average(np.log10(ax.get_ylim()), weights=[1-weight, weight])
        ax.text(mean_N+5, log_middle, s_mean, ha='left', va='center', fontdict=dict(size=30, color=colors[0]), bbox=dict(ec=colors[0], fc='white', alpha=0.9))

        return fig, ax

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


def compute_N_box_index(coordinates, N_bins_x, N_bins_y, threshold=0.8, verbose=False):

    counts = histogram2d(coordinates, bins=(N_bins_x, N_bins_y))
    counts_1d = counts.flatten()

    counts_1d_nonzero = counts_1d[counts_1d > 0]
    counts_sorted =  np.sort(counts_1d_nonzero)[::-1]
    
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


def compute_spatial_correlation_day(coordinates, which_state_day, N_bins_x, N_bins_y, verbose=False):

    counts_1d_all = histogram2d(coordinates, bins=(N_bins_x, N_bins_y)).flatten()
    counts_1d_I = histogram2d(coordinates[(-1 < which_state_day) & (which_state_day < 8)], bins=(N_bins_x, N_bins_y)).flatten()

    counts_1d_nonzero_all = counts_1d_all[counts_1d_all > 0]
    counts_1d_nonzero_I = counts_1d_I[counts_1d_all > 0]

    f = counts_1d_nonzero_I / counts_1d_nonzero_I
    return np.corrcoef(f)[0, 1]




from functools import lru_cache

class InfectionHomogeneityIndex(AnimationBase):

    def __init__(self, filename):
        super().__init__(filename, animation_type='InfectionHomogeneityIndex')
        self.__name__ = 'InfectionHomogeneityIndex'


    @lru_cache
    def _get_N_bins_xy(self):

        coordinates = self.coordinates

        lon_min = coordinates[:, 0].min()
        lon_max = coordinates[:, 0].max()
        lon_mid = np.mean([lon_min, lon_max])

        lat_min = coordinates[:, 1].min()
        lat_max = coordinates[:, 1].max()
        lat_mid = np.mean([lat_min, lat_max])

        N_bins_x = int(haversine(lon_min, lat_mid, lon_max, lat_mid)) + 1
        N_bins_y = int(haversine(lon_mid, lat_min, lon_mid, lat_max)) + 1

        return N_bins_x, N_bins_y


    def _compute_IHI(self, threshold):

        N_bins_x, N_bins_y = self._get_N_bins_xy()

        N = len(self.which_state)
        x = np.arange(N-1)
        IHI = np.zeros(len(x))
        N_box_all, counts_1d_all = compute_N_box_index(self.coordinates, N_bins_x, N_bins_y, threshold=threshold)
        for i_day in x:
            which_state_day = self.which_state[i_day]
            coordinates_infected = self.coordinates[(-1 < which_state_day) & (which_state_day < 8)]
            N_box_infected, counts_1d_infected = compute_N_box_index(coordinates_infected, N_bins_x, N_bins_y, threshold=threshold)
            ratio_N_box = N_box_infected / N_box_all
            IHI[i_day] = ratio_N_box
        return IHI


    def _make_plot(self, verbose=False, savefig=True, force_rerun=False):

        pdf_name = str(Path('Figures/IHI') / 'IHI_') + Path(self.filename).stem + '.pdf'
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
            title = extra_funcs.dict_to_title(self.cfg)
            ax.set(xlabel='Day', ylabel='Infection Homogeneity Index ', title=title, ylim=(0, 1))
            if savefig:
                Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(pdf_name, dpi=600, bbox_inches='tight', pad_inches=0.3)
            return fig, ax
        
        else:
            if verbose:
                print(f"{pdf_name} already exists, skipping for now.")


    def make_plot(self, verbose=False, savefig=True, force_rerun=False):
        if self.is_valid_file:
            return self._make_plot(verbose=verbose, savefig=savefig, force_rerun=force_rerun)


# # def m(x, w):
#     """Weighted Mean"""
#     # return np.sum(x * w) / np.sum(w)
#     # return np.average(x, weights=w)

# def cov(x, y, w):
#     """Weighted Covariance"""
#     mx = np.average(x, weights=w)
#     my = np.average(y, weights=w)

#     # return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)
#     return np.average((x - mx) * (y - my), weights=w)

# def corr(x, y, w):
#     """Weighted Correlation"""
#     return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))




if False:
    IHI = InfectionHomogeneityIndex(filename)
    IHI.make_plot(verbose=True, savefig=False, force_rerun=True)


    threshold = 0.1

    N_bins_x, N_bins_y = IHI._get_N_bins_xy()

    N = len(IHI.which_state)
    x = np.arange(N-1)
    ihi = np.zeros(len(x))

    N_box_all, counts_1d_all = compute_N_box_index(IHI.coordinates, N_bins_x, N_bins_y, threshold=threshold, verbose=True)

    i_day = 50
    which_state_day = IHI.which_state[i_day]
    coordinates_infected = IHI.coordinates[(-1 < which_state_day) & (which_state_day < 8)]
    N_box_infected, counts_1d_infected = compute_N_box_index(coordinates_infected, N_bins_x, N_bins_y, threshold=threshold, verbose=True)
    ratio_N_box = N_box_infected / N_box_all

# %%

def animate_file(filename, do_tqdm=False, verbose=False, dpi=50, remove_frames=True, force_rerun=False, optimize_gif=True, load_into_memory=False,
                make_geo_animation=True, 
                make_IHI_plot=True, 
                make_N_connections_animation=True, 
                ):


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Attempting to set identical")
        warnings.filterwarnings("ignore", message="invalid value encountered in")
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


        if make_geo_animation:
            animation = AnimateSIR(filename, do_tqdm=do_tqdm, verbose=verbose, load_into_memory=load_into_memory)
            with animation:
                animation.make_animation(remove_frames=remove_frames, 
                                        force_rerun=force_rerun, 
                                        optimize_gif=optimize_gif,
                                        dpi=dpi,
                                        )
            

        if make_IHI_plot:
            IHI = InfectionHomogeneityIndex(filename)
            with IHI:
                IHI.make_plot(verbose=False, savefig=True, force_rerun=force_rerun)
                plt.close('all')
            
        if make_N_connections_animation:
            animation_N_connections = Animate_N_connections(filename, do_tqdm=do_tqdm, verbose=verbose, load_into_memory=load_into_memory)
            with animation_N_connections:
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
filename = filenames[1]
N_files = len(filenames)

x=x


#%%



#%%

kwargs = dict(do_tqdm=False, 
              verbose=False, 
              force_rerun=False,
              make_geo_animation=True,
              make_N_connections_animation=True, 
              make_IHI_plot=True)


if __name__ == '__main__' and True:

    if num_cores == 1:

        for filename in tqdm(filenames):
            animate_file(filename, **kwargs)

    else:
        print(f"Generating {N_files} animations using {num_cores} cores, please wait", flush=True)
        kwargs['do_tqdm'] = False
        kwargs['verbose'] = False
        with mp.Pool(num_cores) as p:
            list(tqdm(p.imap_unordered(partial(animate_file, **kwargs), filenames), total=N_files))

x=x

#%%
        

#%%


def foo(i_day, animation):
# def foo(i_day):
    png_name = f"test_{i_day}.png"
    fig, _ = animation._plot_i_day(i_day, dpi=50)
    fig.savefig(png_name, dpi=50, bbox_inches='tight', pad_inches=0.3) 
    plt.close(fig)
    plt.close('all')


if __name__ == '__main__' and True:

    i_day = 12

    animation = AnimateSIR(filename, do_tqdm=True, verbose=True, N_max=i_day)
    animation._initialize_data()

    for i_day in tqdm(range(12)):
        foo(i_day, animation)

        # self.coordinates = f["coordinates"][()]
        # self.df_raw = pd.DataFrame(f["df"][()])
        # self.which_state = f["which_state"]
        # self.N_connections

    # with mp.Pool(6) as p:
    #     list(tqdm(p.imap_unordered(partial(foo, animation=animation), range(12)), total=12))
    #     # list(tqdm(p.imap_unordered(foo, range(12)), total=12))



#%%

if False:

    import pickle

    pickle.dump(animation, open('animation_test.pickle', 'wb'))
    z = pickle.load(open('animation_test.pickle', 'rb'))




#     d_out = animation.__dict__.copy()

#     del d_out['f']  # remove filehandle entry
#     del d_out['which_state']  # remove filehandle entry
#     del d_out['N_connections']  # remove filehandle entry

# # # Example
# class SomeClass:
#     def __init__(self, filename):
#         self.name = filename
#         self.file = open(filename)
    

# obj = SomeClass('animation_test.pickle')


#%%

class SomeClass:
    def __init__(self, filename):
        self.filename = filename
        f = h5py.File(self.filename, "r")
        self.f = f
        self.coordinates = f["coordinates"][()]
        self.df_raw = pd.DataFrame(f["df"][()])
        self.which_state = f["which_state"]
        self.N_connections = f["N_connections"]
    
    def __getstate__(self):
        d_out = self.__dict__.copy()  # get attribute dictionary
        del d_out['f']  # remove filehandle entry
        del d_out['which_state']  # remove filehandle entry
        del d_out['N_connections']  # remove filehandle entry

        d_out = {'filename': self.filename}
        return d_out
    
    def __setstate__(self, dict):
        filename = dict['filename']
        self.__init__(filename)



x = SomeClass(filename)

#%%

if False:

    pickle.dump(x, open('x.pickle', 'wb'))

    y = pickle.load(open('x.pickle', 'rb'))
    y.f
    y.df_raw

# %%



#%%

animation = AnimateSIR(filename, do_tqdm=True, verbose=True, N_max=12, load_into_memory=True)
animation._initialize_data()


coordinates = animation.coordinates
which_state = animation.which_state
which_state = which_state.reshape((*which_state.shape, -1))
N_connections = animation.N_connections
N_connections = N_connections.reshape((*N_connections.shape, -1))


df_counts = animation.df_counts
R_eff = animation.R_eff
R_eff_smooth = animation.R_eff_smooth



# import xarray as xr
# coordinates_3D = np.tile(coordinates, (len(which_state),1,1))
# data = np.c_[coordinates_3D, which_state, N_connections]
# locs = ['lon', 'lat', 'which_state', 'N_connections']
# times = np.arange(len(data))
# foo = xr.DataArray(data, coords=[times, locs], dims=['time', 'space'])
# airtemps = xr.tutorial.open_dataset('air_temperature')

# from SimulateDenmarkAgeHospitalization_extra_funcs import haversine_scipy


# import dask_distance
# import dask.array as da

X = IHI.coordinates
X = X[:10_000]

# # x = dask_distance.cdist(X, X, haversine_scipy)
# y = dask_distance.pdist(X, haversine_scipy)
# y.compute()

# h, bins = da.histogram(x, bins=100, range=[0, 1_000_000])
# h.compute()
# np.asarray(h)


# y.persist()




from scipy.spatial import cKDTree as KDTree


# t1 = KDTree(X)
# # we need a distance to not look beyond, if you have real knowledge use it, otherwise guess
# # maxD = np.linalg.norm(l1[0] - l2[0]) # this could be closest but anyhting further is certainly not
# # get a sparce matrix of all the distances

# ans = t1.sparse_distance_matrix(t2, 1000)


#%%
