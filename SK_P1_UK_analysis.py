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
# import matplotlib.patches as patches

def plot_df(df, filename, dpi=75, UK_max=100):

    discrete_colors = ['#7F7F7F', '#1F77B4', '#D62728', '#2CA02C']

    norm_1000 = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
    norm_100 = ImageNormalize(vmin=0., vmax=100, stretch=LogStretch())
    norm_10 = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())

    states = ['S', 'E', 'I', 'R']

    dfs = {s: df.query("SK == @s") for s in ['S', 'E', 'I', 'R']}

    # Main plot

    fig = plt.figure(figsize=(15, 18))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

    if len(dfs['S']) > 0:
        ax.scatter_density(dfs['S']['x'], dfs['S']['y'], color=discrete_colors[0], alpha=0.2, norm=norm_1000, dpi=dpi)
    if len(dfs['R']) > 0:
        ax.scatter_density(dfs['R']['x'], dfs['R']['y'], color=discrete_colors[3], alpha=0.3, norm=norm_100, dpi=dpi)
    if len(dfs['E']) > 0:
        ax.scatter_density(dfs['E']['x'], dfs['E']['y'], color=discrete_colors[1], norm=norm_10, dpi=dpi)
    if len(dfs['I']) > 0:
        ax.scatter_density(dfs['I']['x'], dfs['I']['y'], color=discrete_colors[2], norm=norm_10, dpi=dpi)
    ax.set(xlim=(8, 13.7), ylim=(54.52, 58.2))

    circles = [Line2D([0], [0], marker='o', color='w', label=state, markerfacecolor=color, markersize=12) for color, state in zip(discrete_colors, states)]
    ax.legend(handles=circles, loc='upper left')

    cfg = extra_funcs.filename_to_dotdict(filename, SK_P1_UK=True)
    title = extra_funcs.dict_to_title(cfg)
    ax.set_title(title, pad=50, fontsize=20)

    # secondary plots:

    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.67, 0.76, 0.25, 0.1]

    background_box = [(0.6, 0.577), 0.38, 0.42]
    ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='lightgray', transform=ax.transAxes))

    delta_width = 6 * width / 100
    ax2 = fig.add_axes([left+delta_width, bottom, width-delta_width, height])
    ax2.hist(df['UK_num'], range=(0, UK_max), bins=UK_max, color=discrete_colors[0])
    ax2.set(xlabel='# Connections', ylabel='Counts')
    # ax2.set_yscale('log')

    ax3 = fig.add_axes([left, bottom-height*1.6, width, height])
    x, y = np.unique(df['SK_num'], return_counts=True)
    ax3.bar(x, y, color=discrete_colors[0])
    ax3.xaxis.set_ticks(np.arange(-1, 9))
    ax3.set(xlabel='State', ylabel='Counts')
    ax3.set_yscale('log')

    plt.close('all')

    return fig, ax



#%%


# def plot_dfs


from copy import copy
class SIRfile:

    def __init__(self, filename, i_day=None):
        self.filename = filename
        print(f"Loading: \n{filename}")
        self.SK, self.P1, self.UK = joblib.load(filename)
        filename_AK = filename.replace('SK_P1_UK.joblib', 'AK_initial.parquet')
        self.AK = awkward.fromparquet(filename_AK)
        filename_Rate = filename_AK.replace('AK_initial.parquet', 'Rate_initial.parquet')
        self.Rate = awkward.fromparquet(filename_Rate)

        self.N = len(self.SK)
        if i_day is not None:
            self.i_day = i_day

    def __call__(self, i_day):
        self.i_day = i_day
        return copy(self)

    def to_df(self, i_day=None):
        if i_day is None and self.i_day is None:
            raise AssertionError(f'Both i_day and self.i_day is None, have to be defined')
        if i_day is None:
            i_day = self.i_day

        mapping = {-1: 'S', 
                    0: 'E', 1: 'E', 2:'E', 3: 'E',
                    4: 'I', 5: 'I', 6:'I', 7: 'I',
                    8: 'R'}

        df = pd.DataFrame(self.P1, columns=['x', 'y'])
        df['SK_num'] = self.SK[i_day]
        df['UK_num'] = self.UK[i_day]
        df["SK"] = df['SK_num'].replace(mapping).astype('category')
        # dfs = {s: df.query("SK == @s") for s in ['S', 'E', 'I', 'R']}
        return df


def plot_SIRfile(SIR_object, UK_max=100):
    df = SIR_object.to_df()
    filename = SIR_object.filename
    fig, ax = plot_df(df, filename, UK_max=UK_max);
    
    i_day = SIR_object.i_day

    ax.text(0.02, 0.02, f"Day: {i_day}", fontsize=24, transform=ax.transAxes)

    figname = 'Figures_SK_P1_UK/animation_N'
    figname += SIR_object.filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib')
    figname += f'.{SIR_object.i_day:06d}.png'

    Path(figname).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname, dpi=75, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    plt.close('all')
    return None


import imageio # conda install imageio
def animate_SIR_file(filename, num_cores_max=20, do_tqdm=False, remove_frames=True, force_rerun=False):

    name = 'animation_' + Path(filename).stem + '.gif'
    gifname = str(Path('Figures_SK_P1_UK') /  name)

    if not Path(gifname).exists() or force_rerun:

        num_cores = mp.cpu_count() - 1
        if num_cores >= num_cores_max:
            num_cores = num_cores_max

        SIR_base = SIRfile(filename)
        N = SIR_base.N
        SIR_objects = [SIR_base(i) for i in range(N)]
        UK_max = SIR_objects[0].to_df()['UK_num'].max()

        # for SIR_object in tqdm(SIR_objects, desc='Creating individual frames'):
        for SIR_object in SIR_objects:
            plot_SIRfile(SIR_object, UK_max=UK_max);

        # print(f"Generating frames using {num_cores} cores, please wait", flush=True)
        # with mp.Pool(num_cores) as p:
        #     list(tqdm(p.imap_unordered(plot_SIRfile, SIR_objects), total=N))
        
        it_frames = sorted(Path(gifname).parent.rglob(f"{Path(gifname).stem}*.png"))
        if do_tqdm:
            it_frames = tqdm(it_frames, desc='Stitching frames to gif')

        with imageio.get_writer(gifname, mode='I', duration=0.1) as writer:
            
            for i, figname in enumerate(it_frames):
                image = imageio.imread(figname)
                writer.append_data(image)

                # if last frame add it N_last times           
                if i+1 == len(it_frames):
                    N_last = len(it_frames) // 2
                    for j in range(N_last):
                        writer.append_data(image)
                
                if remove_frames:
                    Path(figname).unlink() # delete file

        # pip install pygifsicle
        if False:
            from pygifsicle import optimize
            optimize(gifname, colors=256)


# %%

num_cores_max = 20
num_cores = mp.cpu_count() - 1
if num_cores >= num_cores_max:
    num_cores = num_cores_max


filenames = get_SK_P1_UK_filenames()
filename = filenames[1]
N_files = len(filenames)                

print("start", flush=True)

# for filename in tqdm(filenames):
#     animate_SIR_file(filenames[10], do_tqdm=True, remove_frames=True)

if __name__ == '__main__':
    print(f"Generating frames using {num_cores} cores, please wait", flush=True)
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(animate_SIR_file, filenames), total=N_files))


#%%

# from matplotlib import animation

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # animation function.  This is called sequentially
# def animate(i):
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     return line,

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)

# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()


#%%

# dpi=75 
# UK_max=100


# discrete_colors = ['#7F7F7F', '#1F77B4', '#D62728', '#2CA02C']

# norm_1000 = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
# norm_100 = ImageNormalize(vmin=0., vmax=100, stretch=LogStretch())
# norm_10 = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())

# states = ['S', 'E', 'I', 'R']


# fig = plt.figure(figsize=(15, 18))
# ax = fig.add_subplot(111, projection='scatter_density', xlim=(8, 13.7), ylim=(54.52, 58.2))

# circles = [Line2D([0], [0], marker='o', color='w', label=state, markerfacecolor=color, markersize=12) for color, state in zip(discrete_colors, states)]
# ax.legend(handles=circles, loc='upper left')

# cfg = extra_funcs.filename_to_dotdict(filename, SK_P1_UK=True)
# title = extra_funcs.dict_to_title(cfg)
# ax.set_title(title, pad=50, fontsize=20)


# # secondary plots:

# # These are in unitless percentages of the figure size. (0,0 is bottom left)
# left, bottom, width, height = [0.67, 0.76, 0.25, 0.1]

# background_box = [(0.6, 0.577), 0.38, 0.42]
# ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='lightgray', transform=ax.transAxes))

# delta_width = 6 * width / 100
# ax2 = fig.add_axes([left+delta_width, bottom, width-delta_width, height])
# ax2.set(xlabel='# Connections', ylabel='Counts')
# # ax2.set_yscale('log')

# ax3 = fig.add_axes([left, bottom-height*1.6, width, height])



# # particles holds the locations of the particles
# scatter_R = ax.scatter_density([1], [1], color=discrete_colors[0], alpha=0.2, norm=norm_1000, dpi=dpi)
# text_day = ax.text(0.02, 0.92, '', fontsize=24, transform=ax.transAxes)




#     dfs = {s: df.query("SK == @s") for s in ['S', 'E', 'I', 'R']}

#     # Main plot

#     fig = plt.figure(figsize=(15, 18))
#     ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

#     if len(dfs['S']) > 0:
#         ax.scatter_density(dfs['S']['x'], dfs['S']['y'], color=discrete_colors[0], alpha=0.2, norm=norm_1000, dpi=dpi)
#     if len(dfs['R']) > 0:
#         ax.scatter_density(dfs['R']['x'], dfs['R']['y'], color=discrete_colors[3], alpha=0.3, norm=norm_100, dpi=dpi)
#     if len(dfs['E']) > 0:
#         ax.scatter_density(dfs['E']['x'], dfs['E']['y'], color=discrete_colors[1], norm=norm_10, dpi=dpi)
#     if len(dfs['I']) > 0:
#         ax.scatter_density(dfs['I']['x'], dfs['I']['y'], color=discrete_colors[2], norm=norm_10, dpi=dpi)
#     ax.set(xlim=(8, 13.7), ylim=(54.52, 58.2))

#     circles = [Line2D([0], [0], marker='o', color='w', label=state, markerfacecolor=color, markersize=12) for color, state in zip(discrete_colors, states)]
#     ax.legend(handles=circles, loc='upper left')

#     cfg = extra_funcs.filename_to_dotdict(filename, SK_P1_UK=True)
#     title = extra_funcs.dict_to_title(cfg)
#     ax.set_title(title, pad=50, fontsize=20)

#     # secondary plots:

#     # These are in unitless percentages of the figure size. (0,0 is bottom left)
#     left, bottom, width, height = [0.67, 0.76, 0.25, 0.1]

#     background_box = [(0.6, 0.577), 0.38, 0.42]
#     ax.add_patch(mpatches.Rectangle(*background_box, facecolor='white', edgecolor='lightgray', transform=ax.transAxes))

#     delta_width = 6 * width / 100
#     ax2 = fig.add_axes([left+delta_width, bottom, width-delta_width, height])
#     ax2.hist(df['UK_num'], range=(0, UK_max), bins=UK_max, color=discrete_colors[0])
#     ax2.set(xlabel='# Connections', ylabel='Counts')
#     # ax2.set_yscale('log')

#     ax3 = fig.add_axes([left, bottom-height*1.6, width, height])
#     x, y = np.unique(df['SK_num'], return_counts=True)
#     ax3.bar(x, y, color=discrete_colors[0])
#     ax3.xaxis.set_ticks(np.arange(-1, 9))
#     ax3.set(xlabel='State', ylabel='Counts')
#     ax3.set_yscale('log')