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

#%%

def get_SK_P1_UK_filenames():
    filenames = Path('Data_SK_P1_UK').glob(f'*.joblib')
    return [str(file) for file in sorted(filenames)]

# import plotly.express as px
# def animate_single_file(filename, frac=0, Nbins=100, remove_frames=True, do_tqdm=False, plot_first_day=False, force_rerun=False):
    
#     name = 'animation_' + Path(filename).stem + '.gif'
#     gifname = str(Path('Figures_SK_P1_UK') /  name)

#     if not Path(gifname).exists() or force_rerun:

#         SIRfile_SK, SIRfile_P1, SIRfile_UK = joblib.load(filename)

#         fignames = []

#         # categories = 'S, E, I, R'.split(', ')
#         mapping = {-1: 'S', 
#                     0: 'E', 1: 'E', 2:'E', 3: 'E',
#                     4: 'I', 5: 'I', 6:'I', 7: 'I',
#                     8: 'R'}

#         # frac: fraction of total to save figures for
#         if frac == 0:
#             it = np.arange(0, len(SIRfile_SK))
#         else:
#             it = np.arange(0, len(SIRfile_SK), int(len(SIRfile_SK)*frac))
#         if do_tqdm:
#             it = tqdm(it, desc='Creating individual frames')

#         # i_day = 50
#         for i_day in it:

#             df = pd.DataFrame(SIRfile_P1, columns=['x', 'y'])
#             df['SK_num'] = SIRfile_SK[i_day]
#             df['UK_num'] = SIRfile_UK[i_day]
#             df["SK"] = df['SK_num'].replace(mapping).astype('category')

#             px_colors = px.colors.qualitative.D3
#             discrete_colors = [px_colors[7], px_colors[0],  px_colors[3], px_colors[2]]

#             fig_P1 = px.scatter(df, x="x", y="y", color="SK",
#                 #  color_discrete_sequence=discrete_colors,
#                 category_orders={"SK": ["S", "E", "I", "R"]},
#                 # size=1,
#                 color_discrete_map={
#                     'S': px_colors[7],
#                     'E': px_colors[0],
#                     'I': px_colors[3],
#                     'R': px_colors[2],
#                     },
#                 title="Explicit color mapping")
#             fig_P1.update_traces(marker=dict(size=2))
#             traces_P1 = fig_P1['data']

#             fig = make_subplots(rows=1, cols=3, 
#                                 subplot_titles=['SK', 'UK', 'P1'], 
#                                 column_widths=[0.3, 0.3, 0.4])

#             x, y = np.unique(df['SK_num'], return_counts=True)
#             fig.add_trace(go.Bar(x=x, y=y, showlegend=False), row=1, col=1)
#             fig.update_xaxes(title_text="SK", row=1, col=1)
#             fig.update_yaxes(title_text="Counts (log)", type="log", row=1, col=1)

#             fig.add_trace(go.Histogram(x=df['UK_num'], nbinsx=Nbins, showlegend=False), row=1, col=2)
#             fig.update_xaxes(title_text="UK", row=1, col=2)
#             fig.update_yaxes(title_text="Counts", row=1, col=2) # type="log"

#             for trace in traces_P1:
#                 fig.add_trace(trace, row=1, col=3)
#             fig.update_xaxes(title_text="x", range=[df['x'].min(), df['x'].max()], row=1, col=3)
#             fig.update_yaxes(title_text="y", range=[df['y'].min(), df['y'].max()], row=1, col=3) # type="log"

#             # Edit the layout
#             k = 1.5

#             cfg = extra_funcs.filename_to_dotdict(filename, SK_P1_UK=True)
#             title = extra_funcs.dict_to_title(cfg)

#             fig.update_layout(title=f'{title}, {i_day=}',
#                             height=450*k, width=1000*k,
#                             # itemsizing='constant',
#                             # showlegend=False,
#                             )

#             figname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + f'{i_day:06d}.png'
#             # figname = f"Figures_SK_P1_UK/.tmp_{filename}_{i_day}.png"
#             Path(figname).parent.mkdir(parents=True, exist_ok=True)

#             fig.write_image(figname)
#             fignames.append(figname)

#             if i_day == 0 and plot_first_day:
#                 fig.show()

#         import imageio # conda install imageio
#         with imageio.get_writer(gifname, mode='I', duration=0.1) as writer:
#             it_frames = fignames
#             if do_tqdm:
#                 it_frames = tqdm(it_frames, desc='Stitching frames to gif')
#             for i, figname in enumerate(it_frames):
#                 image = imageio.imread(figname)
#                 writer.append_data(image)

#                 # if last frame add it N_last times           
#                 if i+1 == len(it_frames):
#                     N_last = len(it_frames)
#                     for j in range(N_last):
#                         writer.append_data(image)
                
#                 if remove_frames:
#                     Path(figname).unlink() # delete file
        
#         # pip install pygifsicle
#         from pygifsicle import optimize
#         print("Optimizing gif")
#         optimize(gifname, colors=20)

#     else:
#         print(f"{name} already exists, skips creation")



#%%


filenames = get_SK_P1_UK_filenames()
filename = filenames[1]
N_files = len(filenames)

x=x

# print(filenames)
# animate_single_file(filename, frac=0, remove_frames=True, do_tqdm=True)
# x=x

#%%
def get_dfs():

    # import matplotlib.pyplot as plt
    SIRfile_SK, SIRfile_P1, SIRfile_UK = joblib.load(filename)
    i_day = 50


    # categories = 'S, E, I, R'.split(', ')
    mapping = {-1: 'S', 
                0: 'E', 1: 'E', 2:'E', 3: 'E',
                4: 'I', 5: 'I', 6:'I', 7: 'I',
                8: 'R'}


    df = pd.DataFrame(SIRfile_P1, columns=['x', 'y'])
    df['SK_num'] = SIRfile_SK[i_day]
    df['UK_num'] = SIRfile_UK[i_day]
    df["SK"] = df['SK_num'].replace(mapping).astype('category')


    dfs = {s: df.query("SK == @s") for s in ['S', 'E', 'I', 'R']}

    return dfs

dfs = get_dfs()

#%%

# fig, ax = plt.subplots()
# ax.scatter(dfs['S']['x'], dfs['S']['y'], s=2)



px_colors = px.colors.qualitative.D3
discrete_colors = [px_colors[7], px_colors[0],  px_colors[3], px_colors[2]]

# fig_P1 = px.scatter(df, x="x", y="y", color="SK",
#     #  color_discrete_sequence=discrete_colors,
#     category_orders={"SK": ["S", "E", "I", "R"]},
#     # size=1,
#     color_discrete_map={
#         'S': px_colors[7],
#         'E': px_colors[0],
#         'I': px_colors[3],
#         'R': px_colors[2],
#         },
#     title="Explicit color mapping")
# fig_P1.update_traces(marker=dict(size=2))
# traces_P1 = fig_P1['data']

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

    def to_dfs(self, i_day=None):
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

        dfs = {s: df.query("SK == @s") for s in ['S', 'E', 'I', 'R']}

        return dfs


   
def plot_SIRfile(SIR_object):
    df = SIR_object.to_df()
    fig = df_to_fig(df);
    i_day = SIR_object.i_day

    ax = fig.axes[0]
    ax.text(0.05, 0.95, f"{i_day=}", 
                    horizontalalignment='center',
                    verticalalignment='center', 
                    transform=ax.transAxes)

    figname = 'Figures_SK_P1_UK/animation_N'
    figname += SIR_object.filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib')
    figname += f'.{SIR_object.i_day:06d}.png'

    Path(figname).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname, dpi=75)
    plt.close(fig)
    plt.close('all')
    return None




def animate_SIR_file(filename, num_cores_max=20, do_tqdm=True, remove_frames=True):

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max

    SIR_base = SIRfile(filename)
    N = SIR_base.N
    SIR_objects = [SIR_base(i) for i in range(N)]
    # SIR_object = SIR_objects[100]

    for SIR_object in tqdm(SIR_objects, desc='Creating individual frames'):
        SIR_object_to_image(SIR_object);

    # with mp.Pool(num_cores) as p:
    #     list(tqdm(p.imap_unordered(SIR_object_to_image, SIR_objects), total=N))

    import imageio # conda install imageio
    gifname = 'Figures_SK_P1_UK/animation_N' 
    gifname += filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') 
    gifname += '.gif'

    it_frames = sorted(Path(gifname).parent.rglob(f"{Path(gifname).stem}*.png"))
    with imageio.get_writer(gifname, mode='I', duration=0.1) as writer:
        if do_tqdm:
            it_frames = tqdm(it_frames, desc='Stitching frames to gif')
        for i, figname in enumerate(it_frames):
            image = imageio.imread(figname)
            writer.append_data(image)

            # if last frame add it N_last times           
            if i+1 == len(it_frames):
                N_last = 100
                for j in range(N_last):
                    writer.append_data(image)
            
            if remove_frames:
                Path(figname).unlink() # delete file




# pip install mpl-scatter-density
import mpl_scatter_density
import matplotlib.pyplot as plt

# conda install astropy
# Make the norm object to define the image stretch
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

#%%

do_tqdm = True
remove_frames = True

x=x


norm_1000 = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
norm_100 = ImageNormalize(vmin=0., vmax=100, stretch=LogStretch())
norm_10 = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())
dpi = 50

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

states = ['S', 'E', 'I', 'R']

fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

if len(dfs['S']) > 0:
    ax.scatter_density(dfs['S']['x'], dfs['S']['y'], color=discrete_colors[0], alpha=0.2, norm=norm_1000, dpi=dpi)
if len(dfs['R']) > 0:
    ax.scatter_density(dfs['R']['x'], dfs['R']['y'], color=discrete_colors[3], alpha=0.3, norm=norm_100, dpi=dpi)
if len(dfs['E']) > 0:
    ax.scatter_density(dfs['E']['x'], dfs['E']['y'], color=discrete_colors[1], norm=norm_10, dpi=dpi)
if len(dfs['I']) > 0:
    ax.scatter_density(dfs['I']['x'], dfs['I']['y'], color=discrete_colors[2], norm=norm_10, dpi=dpi)

ax.set(xlim=(7.9, 12.8), ylim=(54.54, 57.8))

circles = [Line2D([0], [0], marker='o', color='w', label=state, markerfacecolor=color, markersize=10) for color, state in zip(discrete_colors, states)]
ax.legend(handles=circles)



figname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + f'{i_day:06d}.png'
Path(figname).parent.mkdir(parents=True, exist_ok=True)

fig.savefig(figname)
plt.close('all')



name = 'animation_' + Path(filename).stem + '.gif'
gifname = str(Path('Figures_SK_P1_UK') /  name)
fignames = Path('').glob(gifname.replace('SK_P1_UK.gif', '*.png'))
it_frames = sorted([str(f) for f in fignames])
if do_tqdm:
    it_frames = tqdm(it_frames, desc='Stitching frames to gif')

import imageio # conda install imageio
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
from pygifsicle import optimize
print("Optimizing gif")
optimize(gifname, colors=100)



#%%


# fig.savefig('gaussian.png')

#%%


# filename = filenames[0]
# filename = 'Data_SK_P1_UK/N0_535806_mu_20.0_alpha_6.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_0_Ninit_100_ID_000.SK_P1_UK.joblib'
# animate_single_file(filename, frac=0, remove_frames=True, do_tqdm=True)




# # for filename in tqdm(filenames):
# #     animate_single_file(filename, frac=0, remove_frames=True, do_tqdm=True)

# num_cores = mp.cpu_count() - 1
# num_cores_max = 15
# if num_cores >= num_cores_max:
#     num_cores = num_cores_max


# if __name__ == '__main__':

#     print(f"Animating {N_files} files using {num_cores} cores, please wait.", flush=True)
#     with mp.Pool(num_cores) as p:
#         list(tqdm(p.imap_unordered(animate_single_file, filenames), total=N_files))

#     print("Finished")


# # Do you want the application" orca.app to accept incoming network connections
# # https://github.com/plotly/orca/issues/269 

# # #%%

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# plt.style.use('matplotlibrc')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# color_key = {str(label): col for col, label in zip(colors, ['S', 'E', 'I', 'R'])}


# import datashader as ds
# import datashader.transfer_functions as tf

# def df_to_fig(df, plot_width=1000, plot_height=1000, figsize=(6, 6), legend_fontsize=12, frameon=False):

#     canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
#                        x_range=[df['x'].min(), df['x'].max()], 
#                        y_range=[df['y'].min(), df['y'].max()],
#                        x_axis_type='linear', y_axis_type='linear',
#                     )
#     aggc = canvas.points(df, 'x', 'y', ds.count_cat('SK'))

#     # agg_I_E=aggc.sel(SK=['I', 'E']).sum(dim='SK')

#     img_I = tf.shade(aggc.sel(SK='I'), name="I", how='log')
#     img_E = tf.shade(aggc.sel(SK='E'), name="E", how='log')

#     spread_I = tf.dynspread(img_I, threshold=0.9, max_px=1)
#     spread_E = tf.dynspread(img_E, threshold=0.9, max_px=1)

#     images = tf.Images(spread_I, spread_E)


#     # color_key = color_key_b_c_uds_g
#     # img = tf.shade(agg, color_key=color_key, how='log') # eq_hist
#     img = tf.shade(aggc, color_key=color_key, how='eq_hist') # eq_hist
#     spread = tf.dynspread(img, threshold=0.9, max_px=1)
#     pil = spread.to_pil()

#     fig, ax = plt.subplots(figsize=figsize)

#     ax.imshow(np.array(pil))
#     ax.set_xticks([], [])
#     ax.set_yticks([], [])

#     legend_elements = [Line2D([0], [0], marker='o', color='white', markerfacecolor=val, label=key) for key, val in color_key.items()]
#     ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.1, -0.05), fontsize=legend_fontsize, frameon=frameon)
#     return fig




# def SIRfiles_i_day_to_df(i_SIR_tuple):
#     i_day, (SIRfile_SK, SIRfile_P1, SIRfile_UK) = i_SIR_tuple

#     # categories = 'S, E, I, R'.split(', ')
#     mapping = {-1: 'S', 
#                 0: 'E', 1: 'E', 2:'E', 3: 'E',
#                 4: 'I', 5: 'I', 6:'I', 7: 'I',
#                 8: 'R'}

#     df = pd.DataFrame(SIRfile_P1, columns=['x', 'y'])
#     df['SK_num'] = SIRfile_SK[i_day]
#     df['UK_num'] = SIRfile_UK[i_day]
#     df["SK"] = df['SK_num'].replace(mapping).astype('category')
#     return df




# # if False:

# #     # conda install -c conda-forge holoviews
# #     from holoviews.operation.datashader import datashade
# #     import holoviews as hv

# #     hv.extension('bokeh')


# #     np.random.seed(1)
# #     n = 1000000 # Number of points
# #     f = filter_width = 5000 # momentum or smoothing parameter, for a moving average filter

# #     # filtered random walk
# #     xs = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f)/f).cumsum()
# #     ys = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f)/f).cumsum()

# #     # Add "mechanical" wobble on the x axis
# #     xs += 0.1*np.sin(0.1*np.array(range(n-1+f)))

# #     # Add "measurement" noise
# #     xs += np.random.normal(0, 0.005, size=n-1+f)
# #     ys += np.random.normal(0, 0.005, size=n-1+f)

# #     # Add a completely incorrect value
# #     xs[int(len(xs)/2)] = 100
# #     ys[int(len(xs)/2)] = 0

# #     # Create a dataframe
# #     df = pd.DataFrame(dict(x=xs,y=ys))


# #     filename = 'Data_SK_P1_UK/test.SK_P1_UK.joblib'
# #     SIR_base = SIRfile(filename, 10)
# #     df = SIR_base.to_df()

# #     opts = hv.opts.RGB(width=1000, height=1000)
# #     datashade(hv.Path(df, kdims=['x','y']), normalization='linear', aggregator=ds.any()).opts(opts)




# # #%%


# # import datashader as ds
# # import datashader.transfer_functions as tf
# # from datashader.layout import random_layout, circular_layout, forceatlas2_layout
# # from datashader.bundling import connect_edges, hammer_bundle

# # from itertools import chain

# # # %%

# # np.random.seed(0)
# # n=100
# # m=20000

# # nodes = pd.DataFrame(["node"+str(i) for i in range(n)], columns=['name'])
# # nodes.head()


# # #%%
# # edges = pd.DataFrame(np.random.randint(0,len(nodes), size=(m, 2)),
# #                      columns=['source', 'target'])
# # edges.tail()


# # # %%

# # randomloc = random_layout(nodes)
# # circular  = circular_layout(nodes, uniform=False)

# # # %%

# # cvsopts = dict(plot_height=400, plot_width=400)

# # def nodesplot(nodes, name=None, canvas=None, cat=None):
# #     canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
# #     aggregator=None if cat is None else ds.count_cat(cat)
# #     agg=canvas.points(nodes,'x','y',aggregator)
# #     return tf.spread(tf.shade(agg, cmap=["#FF3333"]), px=3, name=name)


# # tf.Images(nodesplot(randomloc,"Random layout"),
# #           nodesplot(circular, "Circular layout"))


# # #%%

# # %time forcedirected = forceatlas2_layout(nodes, edges)
# # tf.Images(nodesplot(forcedirected, "ForceAtlas2 layout"))

# # # %%

# # def edgesplot(edges, name=None, canvas=None):
# #     canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
# #     return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)
    
# # def graphplot(nodes, edges, name="", canvas=None, cat=None):
# #     if canvas is None:
# #         xr = nodes.x.min(), nodes.x.max()
# #         yr = nodes.y.min(), nodes.y.max()
# #         canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)
        
# #     np = nodesplot(nodes, name + " nodes", canvas, cat)
# #     ep = edgesplot(edges, name + " edges", canvas)
# #     return tf.stack(ep, np, how="over", name=name)


# # # %%

# # cd = circular
# # fd = forcedirected

# # # %%

# # cd = circular
# # fd = forcedirected

# # %time cd_d = graphplot(cd, connect_edges(cd,edges), "Circular layout")
# # %time fd_d = graphplot(fd, connect_edges(fd,edges), "Force-directed") 
# # %time cd_b = graphplot(cd, hammer_bundle(cd,edges), "Circular layout, bundled")
# # %time fd_b = graphplot(fd, hammer_bundle(fd,edges), "Force-directed, bundled") 

# # tf.Images(cd_d,fd_d,cd_b,fd_b).cols(2)

# # # %%

# # rd = randomloc

# # %time rd_d = graphplot(rd, connect_edges(rd,edges), "Random layout")
# # %time rd_b = graphplot(rd, hammer_bundle(rd,edges), "Random layout, bundled")

# # tf.Images(rd_d,rd_b)


# # # %%

# # import networkx as nx

# # filename = 'Data_SK_P1_UK/N0_50_mu_20.0_alpha_0.0_psi_0.0_beta_0.01_sigma_0.0_Mrate1_1.0_Mrate2_1.0_gamma_0.0_nts_0.1_Nstates_9_BB_1_Ninit_1_ID_000.SK_P1_UK.joblib'
# # SIR_base = SIRfile(filename, 0)
# # df = SIR_base.to_df()

# # #%%

# # import matplotlib.pyplot as plt

# # i_day = 1

# # G=nx.Graph()

# # for i, xy in enumerate(SIR_base.P1):
# #     G.add_node(i, pos=xy)

# # AK_day = SIR_base.AK[i_day]
# # Rate_day = SIR_base.Rate[i_day]
# # for i, _ in enumerate(AK_day):
# #     for j, _ in enumerate(AK_day[i]):
# #         G.add_edge(i, AK_day[i][j], weight=Rate_day[i][j])

# # pos = nx.get_node_attributes(G, 'pos')

# # nx.draw(G, pos, with_labels=True)


# # #%%


# # nodes = pd.DataFrame(SIR_base.P1, columns=['x', 'y'])
# # # nodes.set_index('id', inplace=True)

# # @njit
# # def AK_day_to_edges(AK_day):
# #     # edges = []
# #     # N = len(AK_day)
# #     # edges = np.zeros((N*N, 2), np.int_)
# #     edges = [np.int_(x) for x in range(0)]
# #     i = 0
# #     for source, _ in enumerate(AK_day):
# #         for target in AK_day[i]:
# #             if source < target:
# #                 edges[i, :] = source, target
# #             else:
# #                 edges[i, :] = target, source
# #             i+= 1
# #     return edges

# # edges = AK_day_to_edges(AK_day)

# # edges = pd.DataFrame(list(dict.fromkeys(edges)), columns=['source', 'target'])


# # direct = connect_edges(nodes, edges)
# # bundled_bw005 = hammer_bundle(nodes, edges)
# # bundled_bw030 = hammer_bundle(nodes, edges, initial_bandwidth=0.30)

# # graphplot(nodes, direct)
# # graphplot(nodes, bundled_bw005, "Bundled bw=0.05")
# # graphplot(nodes, bundled_bw030, "Bundled bw=0.30")

# # # %%

# # # G.add_nodes_from(pos.keys())

# # # for n, p in pos.iteritems():
# # #     G.node[n]['pos'] = p



# # #%%

# # # https://plotly.com/python/network-graphs/

# # edge_x = []
# # edge_y = []
# # for edge in G.edges():
# #     x0, y0 = G.nodes[edge[0]]['pos']
# #     x1, y1 = G.nodes[edge[1]]['pos']
# #     edge_x.append(x0)
# #     edge_x.append(x1)
# #     edge_x.append(None)
# #     edge_y.append(y0)
# #     edge_y.append(y1)
# #     edge_y.append(None)

# # edge_trace = go.Scatter(
# #     x=edge_x, y=edge_y,
# #     line=dict(width=0.5, color='#888'),
# #     hoverinfo='none',
# #     mode='lines')

# # node_x = []
# # node_y = []
# # for node in G.nodes():
# #     x, y = G.nodes[node]['pos']
# #     node_x.append(x)
# #     node_y.append(y)

# # node_trace = go.Scatter(
# #     x=node_x, y=node_y,
# #     mode='markers',
# #     hoverinfo='text',
# #     marker=dict(
# #         showscale=True,
# #         # colorscale options
# #         #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
# #         #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
# #         #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
# #         colorscale='YlGnBu',
# #         reversescale=True,
# #         color=[],
# #         size=10,
# #         colorbar=dict(
# #             thickness=15,
# #             title='Node Connections',
# #             xanchor='left',
# #             titleside='right'
# #         ),
# #         line_width=2))


# # # %%

# # node_adjacencies = []
# # node_text = []
# # for node, adjacencies in enumerate(G.adjacency()):
# #     node_adjacencies.append(len(adjacencies[1]))
# #     node_text.append('# of connections: '+str(len(adjacencies[1])))

# # node_trace.marker.color = node_adjacencies
# # node_trace.text = node_text


# # # %%

# # fig = go.Figure(data=[edge_trace, node_trace],
# #              layout=go.Layout(
# #                 title='<br>Network graph made with Python',
# #                 titlefont_size=16,
# #                 showlegend=False,
# #                 hovermode='closest',
# #                 margin=dict(b=20,l=5,r=5,t=40),
# #                 annotations=[ dict(
# #                     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
# #                     showarrow=False,
# #                     xref="paper", yref="paper",
# #                     x=0.005, y=-0.002 ) ],
# #                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
# #                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
# #                 )
# # fig.show()


# # # %%

# # def ng(graph,name):
# #     graph.name = name
# #     return graph

# # def nx_layout(graph):
# #     layout = nx.circular_layout(graph)
# #     data = [[node]+layout[node].tolist() for node in graph.nodes]

# #     nodes = pd.DataFrame(data, columns=['id', 'x', 'y'])
# #     nodes.set_index('id', inplace=True)

# #     edges = pd.DataFrame(list(graph.edges), columns=['source', 'target'])
# #     return nodes, edges

# # def nx_plot(graph, name=""):
# #     print(graph.name, len(graph.edges))
# #     nodes, edges = nx_layout(graph)
    
# #     direct = connect_edges(nodes, edges)
# #     bundled_bw005 = hammer_bundle(nodes, edges)
# #     bundled_bw030 = hammer_bundle(nodes, edges, initial_bandwidth=0.30)

# #     return [graphplot(nodes, direct,         graph.name),
# #             graphplot(nodes, bundled_bw005, "Bundled bw=0.05"),
# #             graphplot(nodes, bundled_bw030, "Bundled bw=0.30")]


# # # %%

# # plots = nx_plot(G, name="")

# # tf.Images(*chain.from_iterable(plots))


# # # %%


# # %%
