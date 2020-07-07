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

# from p_tqdm import p_uimap, p_umap
from functools import partial

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


def convert_df_byte_cols(df):
    for col in df.select_dtypes([np.object]):
        df[col] = df[col].str.decode('utf-8') 
    return df



from pathos.helpers import cpu_count
from pathos.multiprocessing import ProcessPool as Pool
from collections.abc import Sized

def _parallel(ordered, function, *iterables, **kwargs):
    """Returns a generator for a parallel map with a progress bar.
    Arguments:
        ordered(bool): True for an ordered map, false for an unordered map.
        function(Callable): The function to apply to each element of the given Iterables.
        iterables(Tuple[Iterable]): One or more Iterables containing the data to be mapped.
    Returns:
        A generator which will apply the function to each element of the given Iterables
        in parallel in order with a progress bar.
    """

    # Extract num_cpus
    num_cpus = kwargs.pop('num_cpus', None)
    do_tqdm = kwargs.pop('do_tqdm', True)

    # Determine num_cpus
    if num_cpus is None:
        num_cpus = cpu_count()
    elif type(num_cpus) == float:
        num_cpus = int(round(num_cpus * cpu_count()))

    # Determine length of tqdm (equal to length of shortest iterable)
    length = min(len(iterable) for iterable in iterables if isinstance(iterable, Sized))

    # Create parallel generator
    map_type = 'imap' if ordered else 'uimap'
    pool = Pool(num_cpus)
    map_func = getattr(pool, map_type)

    # create iterable
    items = map_func(function, *iterables)

    # add progress bar
    if do_tqdm:
        items = tqdm(items, total=length, **kwargs)

    for item in items:
        yield item

    pool.clear()


def p_umap(function, *iterables, **kwargs):
    """Performs a parallel unordered map with a progress bar."""

    ordered = False
    generator = _parallel(ordered, function, *iterables, **kwargs)
    result = list(generator)

    return result




class AnimationBase():

    def __init__(self, filename, animation_type='animation', do_tqdm=False, verbose=False, N_max=None):

        self.filename = filename
        self.animation_type = animation_type
        self.do_tqdm = do_tqdm
        self.verbose = verbose
        self._load_hdf5_file()

        self.N_max = N_max
        if self.is_valid_file:
            if N_max is None:
                self.N_days = len(self.which_state)
            else:
                if N_max < 12:
                    print(f"N_max has to be 12 or larger (choosing 12 instead of {N_max} for now).")
                    N_max = 12
                self.N_days = N_max
        self.cfg = dict(extra_funcs.filename_to_dotdict(filename, animation=True))
        self.__name__ = 'AnimationBase'
        
    def __len__(self):
        return self.N_days
    
    def _load_data(self):

        with h5py.File(self.filename, 'r') as f:

            self.coordinates = f["coordinates"][()]
            self.df_raw = pd.DataFrame(f["df"][()]).drop('index', axis=1)
            self.ages = f["ages"][()]
            self.which_state = f["which_state"][()]
            self.N_connections = f["N_connections"][()]

            if 'df_time_memory' in f.keys():
                self.df_time_memory = convert_df_byte_cols(pd.DataFrame(f["df_time_memory"][()])
                                                           .rename(columns={"index": "Time"}))

            if 'df_change_points' in f.keys():
                self.df_change_points = convert_df_byte_cols(pd.DataFrame(f["df_change_points"][()])
                                                             .rename(columns={"index": "ChangePoint"}))

        # g = awkward.hdf5(f)
        # g["which_connections"] 
        # g["individual_rates"] 

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
            self._make_png_files(force_rerun, **kwargs)

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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
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

    def _make_single_frame(self, i_day, do_tqdm, force_rerun, **kwargs):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            dpi = kwargs.get('dpi', 50)
            png_name = self._get_png_name(i_day)
            if not Path(png_name).exists() or force_rerun:
                fig, _ = self._plot_i_day(i_day, **kwargs)
                Path(png_name).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_name, dpi=dpi, bbox_inches='tight', pad_inches=0.3) 
                plt.close(fig)
                plt.close('all')

    def _make_png_files(self, force_rerun, **kwargs):
        
        n_jobs = kwargs.pop('n_jobs', 1)
        do_tqdm = kwargs.pop('do_tqdm', self.do_tqdm)

        it = range(self.N_days)
        
        # make_single_frame = partial(self._make_single_frame(do_tqdm=do_tqdm, force_rerun=force_rerun, **kwargs))
        make_single_frame = lambda i_day: self._make_single_frame(i_day=i_day, do_tqdm=do_tqdm, force_rerun=force_rerun, **kwargs)

        if n_jobs == 1:
            
            if do_tqdm:
                it = tqdm(it, desc='Make individual frames')
            
            for i_day in it:
                make_single_frame(i_day)
        
        else:
            # with mp.Pool(n_jobs) as p:
            p_umap(make_single_frame, it, num_cpus=n_jobs, do_tqdm=do_tqdm)
            # iterator = p_uimap(make_single_frame, it)
            # for result in iterator:
            #     print(result) # prints '1a', '2b', '3c' in any order
                # list(tqdm(p.imap_unordered(make_single_frame, it), total=self.N_days))
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

    def __init__(self, filename, do_tqdm=False, verbose=False, N_max=None, df_counts=None):
        super().__init__(filename, animation_type='animation', do_tqdm=do_tqdm, verbose=verbose, N_max=N_max)
        self.mapping = {-1: 'S', 
                        #  0: 'E', 1: 'E', 2:'E', 3: 'E',
                         0: 'I', 1: 'I', 2:'I', 3: 'I',
                         4: 'I', 5: 'I', 6:'I', 7: 'I',
                         8: 'R',
                        }
        self.inverse_mapping = get_inverse_mapping(self.mapping)
        
        self.df_counts = df_counts
        self.__name__ = 'AnimateSIR'
        self.dfs_all = {}
        self._initialize_plot()
        
    # def __getstate__(self):
    #     d_out = dict(filename=self.filename, 
    #                  do_tqdm=self.do_tqdm, 
    #                  verbose=self.verbose, 
    #                  N_max=self.N_max, 
    #                  load_into_memory=self.load_into_memory, 
    #                  df_counts=self.df_counts, 
    #                  dfs_all=self.dfs_all, 
    #                  d_colors=self.d_colors,
    #                  )
    #     return d_out
    

    # def __setstate__(self, d_in):
    #     self.__init__(
    #                   filename=d_in['filename'], 
    #                   do_tqdm=d_in['do_tqdm'], 
    #                   verbose=d_in['verbose'], 
    #                   N_max=d_in['N_max'], 
    #                   load_into_memory=d_in['load_into_memory'],
    #                   df_counts=d_in['df_counts'],
    #     )
    #     self.dfs_all = d_in['dfs_all']
    #     self.d_colors = d_in['d_colors']
        


    # def _get_df(self, i_day):
    #     df = pd.DataFrame(self.coordinates, columns=['x', 'y'])
    #     df['which_state_num'] = self.which_state[i_day]
    #     df["which_state"] = df['which_state_num'].replace(self.mapping).astype('category')
    #     return df


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

        self._geo_plot_kwargs = {}
        self._geo_plot_kwargs['S'] = dict(alpha=0.2, norm=self.norm_1000)
        self._geo_plot_kwargs['R'] = dict(alpha=0.3, norm=self.norm_100)
        self._geo_plot_kwargs['I'] = dict(norm=self.norm_10)


    def _initialize_data(self):

        if not self.is_valid_file:
            return None

        # calc counts and R_eff and N_tot
        if self.df_counts is None:
            self.df_counts = self._compute_df_counts()
        self.R_eff = self._compute_R_eff()
        self.R_eff_smooth = self._smoothen(self.R_eff, method='savgol', window_length=11, polyorder=3)
        assert self.cfg['N_tot'] == self.df_counts.iloc[0].sum()


    def _compute_df_counts(self):
        counts_i_day = {}
        it = range(self.N_days)
        if self.do_tqdm:
            it = tqdm(it, desc="Creating df_counts")
        for i_day in it:
            counts_i_day[i_day] = unique_counter(self.which_state[i_day], mapping=self.mapping)
            # df = self._get_df(i_day)
            # dfs = {s: df.query("which_state == @s") for s in self.states}
            # counts_i_day[i_day] = {key: len(val) for key, val in dfs.items()}
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

    def _get_mask(self, i_day, state):
        return np.isin(self.which_state[i_day], self.inverse_mapping[state])

    def _plot_i_day(self, i_day, dpi=50):

        # df = self._get_df(i_day)
        # dfs = {s: df.query("which_state == @s") for s in self.states}
        # dfs = self.dfs_all[i_day]

        # Main plot
        k_scale = 1.7
        fig = plt.figure(figsize=(13*k_scale, 13*k_scale))
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

        for state in self.states:
            if self.df_counts.loc[i_day, state] > 0:
                ax.scatter_density(*self.coordinates[self._get_mask(i_day, state)].T, color=self.d_colors[state], dpi=dpi, **self._geo_plot_kwargs[state])
        # if len(dfs['I']) > 0:
        #     ax.scatter_density(dfs['I']['x'], dfs['I']['y'], color=self.d_colors['I'], , dpi=dpi)
        
        # ax.set(xlim=(7.9, 13.7), ylim=(54.4, 58.2), xlabel='Longitude')
        ax.set(xlim=(7.9, 15.3), ylim=(54.5, 58.2), xlabel='Longitude')
        ax.set_ylabel('Latitude', rotation=90) # fontsize=20, labelpad=20


        kw_args_circle = dict(xdata=[0], ydata=[0], marker='o', color='w', markersize=18)
        circles = [Line2D(label=self.state_names[state], 
                          markerfacecolor=self.d_colors[state], **kw_args_circle)
                          for state in self.states]
        ax.legend(handles=circles, loc='upper left', fontsize=24, frameon=False)

        # string = "\n".join([human_format(len(dfs[state]), decimals=1) for state in self.states])
        s_legend = [human_format(self.df_counts.loc[i_day, state], decimals=1) for state in self.states]
        delta_s = 0.0261
        for i, s in enumerate(s_legend):
            ax.text(0.41, 0.9698-i*delta_s, s, fontsize=24, transform=ax.transAxes, ha='right')
        
        # left, bottom, width, height
        legend_background_box = [(0.023, 0.91), 0.398, 0.085] 
        ax.add_patch(mpatches.Rectangle(*legend_background_box, facecolor='white', edgecolor='white', transform=ax.transAxes))


        cfg = extra_funcs.filename_to_dotdict(self.filename, animation=True)
        title = extra_funcs.dict_to_title(cfg)
        title += "\n\n" + "Simulation of COVID-19 epidemic with no intervention"
        ax.set_title(title, pad=40, fontsize=32)

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
        # ax.text(0.012, 0.012, f"Simulation of COVID-19 epidemic with no intervention.", fontsize=24, transform=ax.transAxes, backgroundcolor='white')
        ax.text(0.99, 0.01, f"Niels Bohr Institute\narXiv: 2007.XXXXX", ha='right', fontsize=20, transform=ax.transAxes, backgroundcolor='white')

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





# %%

def animate_file(filename, do_tqdm=False, verbose=False, dpi=50, remove_frames=True, force_rerun=False, optimize_gif=True, make_geo_animation=True, make_IHI_plot=True, make_N_connections_animation=True):


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Attempting to set identical")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")



        if make_geo_animation:
            animation = AnimateSIR(filename, do_tqdm=do_tqdm, verbose=verbose)
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
            animation_N_connections = Animate_N_connections(filename, do_tqdm=do_tqdm, verbose=verbose)
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
filename = filenames[0]
N_files = len(filenames)

# x=x

# import dill

# test = AnimationBase(filename)
# test = AnimateSIR(filename, do_tqdm=True, verbose=True, N_max=50)
# dill.dumps(test)

# with open("test.dill", "wb") as dill_file:
#      dill.dump(test, dill_file)

# with open("test.dill", "rb") as dill_file:
#      test = dill.load(dill_file)

for filename in filenames:
    animation = AnimateSIR(filename, do_tqdm=True, verbose=True)
    if animation.cfg['N_tot'] < 1_000_000:
        animation.make_animation(remove_frames=True, 
                                force_rerun=True, 
                                optimize_gif=True,
                                dpi=50,
                                n_jobs=7,
                                )
# 
x=x



# def bar(i_day):
#     return i_day, unique_counter(animation.which_state[i_day], mapping=animation.mapping)

# num_cores = 6

# if __name__ == '__main__' and True:
#     with mp.Pool(num_cores) as p:
#         y = list(tqdm(p.imap_unordered(bar, range(len(animation))), total=len(animation)))
#         # x=x
#     y = {key: val for (key, val) in y}

    # print(y)

# animation._initialize_data()

# x=x
# 
# import dill
# dill.dumps(animation)


#%%

# if __name__ == '__main__' and True:
#     animation._make_png_files(do_tqdm=True, force_rerun=True, dpi=50, n_jobs=2)
#     x=x


    # processes

#%%


#%%

    # def _make_single_frame(self, i_day, do_tqdm, force_rerun, **kwargs):
    #     dpi = kwargs.get('dpi', 50)
    #     png_name = self._get_png_name(i_day)
    #     if not Path(png_name).exists() or force_rerun:
    #         fig, _ = self._plot_i_day(i_day, **kwargs)
    #         Path(png_name).parent.mkdir(parents=True, exist_ok=True)
    #         fig.savefig(png_name, dpi=dpi, bbox_inches='tight', pad_inches=0.3) 
    #         plt.close(fig)
    #         plt.close('all')

    # def _make_png_files(self, do_tqdm, force_rerun, **kwargs):
        
    #     n_jobs = kwargs.get('n_jobs', 1)

    #     it = range(self.N_days)
    #     if do_tqdm and (n_jobs == 1):
    #         it = tqdm(it, desc='Make individual frames')

    #     make_single_frame = partial(self._make_single_frame(do_tqdm=do_tqdm, force_rerun=force_rerun, **kwargs))

    #     if n_jobs == 1:
    #         for i_day in it:
    #             make_single_frame(i_day)
    #     else:
    #         with mp.Pool(num_cores) as p:
    #             list(tqdm(p.imap_unordered(make_single_frame, it), total=self.N_days)
    #     return None



# animation.df_counts

# fig, axes = animation._plot_i_day(0)
# fig

#%%

dpi = 50
num_cores = 6
force_rerun = True

def foo(i_day):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Attempting to set identical")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        png_name = animation._get_png_name(i_day)
        if not Path(png_name).exists() or force_rerun:
            fig, _ = animation._plot_i_day(i_day)
            Path(png_name).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_name, dpi=dpi, bbox_inches='tight', pad_inches=0.3) 
            plt.close(fig)
            plt.close('all')


if __name__ == '__main__' and True:
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(foo, range(len(animation))), total=len(animation)))
    # x=x

# # #%%



# # # Main plot
# # k_scale = 1.8
# # fig = plt.figure(figsize=(10*k_scale, 13*k_scale))
# # ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

# # state = 'S'
# # i_day = 0
# # dpi=50

# # if animation.df_counts.loc[i_day, state] > 0:
# #     ax.scatter_density(*animation.coordinates[animation._get_mask(i_day, state)].T, color=animation.d_colors[state], dpi=dpi, **animation._geo_plot_kwargs[state])


# #%%


# # kwargs = dict(do_tqdm=True, 
# #               verbose=True, 
# #               force_rerun=True,
# #               make_geo_animation=True,
# #               make_N_connections_animation=True, 
# #               make_IHI_plot=True)



# kwargs = dict(do_tqdm=True, 
#               verbose=True, 
#               load_into_memory=False,
#               force_rerun=True,
#               make_geo_animation=True,
#               make_N_connections_animation=True, 
#               make_IHI_plot=True)


# if __name__ == '__main__' and False:

#     if num_cores == 1:

#         for filename in tqdm(filenames):
#             animate_file(filename, **kwargs)

#     else:
#         print(f"Generating {N_files} animations using {num_cores} cores, please wait", flush=True)
#         kwargs['do_tqdm'] = False
#         kwargs['verbose'] = False
#         with mp.Pool(num_cores) as p:
#             list(tqdm(p.imap_unordered(partial(animate_file, **kwargs), filenames), total=N_files))

# x=x

# #%%
        

# #%%


# def foo(i_day, animation):
# # def foo(i_day):
#     png_name = f"test_{i_day}.png"
#     fig, _ = animation._plot_i_day(i_day, dpi=50)
#     fig.savefig(png_name, dpi=50, bbox_inches='tight', pad_inches=0.3) 
#     plt.close(fig)
#     plt.close('all')


# if __name__ == '__main__' and True:

#     i_day = 12

#     animation = AnimateSIR(filename, do_tqdm=True, verbose=True, N_max=i_day)
#     animation._initialize_data()

#     for i_day in tqdm(range(12)):
#         foo(i_day, animation)

#         # self.coordinates = f["coordinates"][()]
#         # self.df_raw = pd.DataFrame(f["df"][()])
#         # self.which_state = f["which_state"]
#         # self.N_connections

#     # with mp.Pool(6) as p:
#     #     list(tqdm(p.imap_unordered(partial(foo, animation=animation), range(12)), total=12))
#     #     # list(tqdm(p.imap_unordered(foo, range(12)), total=12))



# #%%

# if False:

#     import pickle

#     pickle.dump(animation, open('animation_test.pickle', 'wb'))
#     z = pickle.load(open('animation_test.pickle', 'rb'))


# #%%

# class SomeClass:
#     def __init__(self, filename):
#         self.filename = filename
#         f = h5py.File(self.filename, "r")
#         self.f = f
#         self.coordinates = f["coordinates"][()]
#         self.df_raw = pd.DataFrame(f["df"][()])
#         self.which_state = f["which_state"]
#         self.N_connections = f["N_connections"]
    
#     def __getstate__(self):
#         d_out = self.__dict__.copy()  # get attribute dictionary
#         del d_out['f']  # remove filehandle entry
#         del d_out['which_state']  # remove filehandle entry
#         del d_out['N_connections']  # remove filehandle entry

#         d_out = {'filename': self.filename}
#         return d_out
    
#     def __setstate__(self, dict):
#         filename = dict['filename']
#         self.__init__(filename)



# x = SomeClass(filename)

# #%%

# if False:

#     pickle.dump(x, open('x.pickle', 'wb'))

#     y = pickle.load(open('x.pickle', 'rb'))
#     y.f
#     y.df_raw

# #%%


# %%
