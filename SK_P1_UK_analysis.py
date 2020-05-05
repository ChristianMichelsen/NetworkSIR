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


def get_SK_P1_UK_filenames():
    filenames = Path('Data_SK_P1_UK').glob(f'*.joblib')
    return [str(file) for file in sorted(filenames)]

import plotly.express as px
def animate_single_file(filename, frac=0, Nbins=100, remove_frames=True, do_tqdm=False, plot_first_day=False):
    
    SIRfile_SK, SIRfile_P1, SIRfile_UK = joblib.load(filename)

    fignames = []

    # categories = 'S, E, I, R'.split(', ')
    mapping = {-1: 'S', 
                0: 'E', 1: 'E', 2:'E', 3: 'E',
                4: 'I', 5: 'I', 6:'I', 7: 'I',
                8: 'R'}

    # frac: fraction of total to save figures for
    if frac == 0:
        it = np.arange(0, len(SIRfile_SK))
    else:
        it = np.arange(0, len(SIRfile_SK), int(len(SIRfile_SK)*frac))
    if do_tqdm:
        it = tqdm(it, desc='Creating individual frames')

    # i_day = 50
    for i_day in it:
    # for i_day, _ in enumerate(it):

        df = pd.DataFrame(SIRfile_P1, columns=['x', 'y'])
        df['SK_num'] = SIRfile_SK[i_day]
        df['UK_num'] = SIRfile_UK[i_day]
        df["SK"] = df['SK_num'].replace(mapping).astype('category')


        px_colors = px.colors.qualitative.D3
        discrete_colors = [px_colors[7], px_colors[0],  px_colors[3], px_colors[2]]

        fig_P1 = px.scatter(df, x="x", y="y", color="SK",
            #  color_discrete_sequence=discrete_colors,
            category_orders={"SK": ["S", "E", "I", "R"]},
            # size=1,
            color_discrete_map={
                'S': px_colors[7],
                'E': px_colors[0],
                'I': px_colors[3],
                'R': px_colors[2],
                },
             title="Explicit color mapping")
        fig_P1.update_traces(marker=dict(size=2))
        traces_P1 = fig_P1['data']

        fig = make_subplots(rows=1, cols=3, 
                            subplot_titles=['SK', 'UK', 'P1'], 
                            column_widths=[0.3, 0.3, 0.4])

        x, y = np.unique(df['SK_num'], return_counts=True)
        fig.add_trace(go.Bar(x=x, y=y, showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text="SK", row=1, col=1)
        fig.update_yaxes(title_text="Counts (log)", type="log", row=1, col=1)

        fig.add_trace(go.Histogram(x=df['UK_num'], nbinsx=Nbins, showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="UK", row=1, col=2)
        fig.update_yaxes(title_text="Counts", row=1, col=2) # type="log"

        for trace in traces_P1:
            fig.add_trace(trace, row=1, col=3)
        fig.update_xaxes(title_text="x", range=[df['x'].min(), df['x'].max()], row=1, col=3)
        fig.update_yaxes(title_text="y", range=[df['y'].min(), df['y'].max()], row=1, col=3) # type="log"

        # Edit the layout
        k = 1.2

        cfg = extra_funcs.filename_to_dotdict(filename, SK_P1_UK=True)
        title = extra_funcs.dict_to_title(cfg)

        fig.update_layout(title=f'{title}, {i_day=}',
                        height=400*k, width=1000*k,
                        itemsizing='constant',
                        # showlegend=False,
                        )

        figname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + f'{i_day:6d}.png'
        # figname = f"Figures_SK_P1_UK/.tmp_{filename}_{i_day}.png"
        Path(figname).parent.mkdir(parents=True, exist_ok=True)

        fig.write_image(figname)
        fignames.append(figname)

        if i_day == 0 and plot_first_day:
            fig.show()


    import imageio # conda install imageio
    gifname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + '.gif'
    with imageio.get_writer(gifname, mode='I', duration=0.1) as writer:
        it_frames = fignames
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




filenames = get_SK_P1_UK_filenames()
filename = filenames[0]
N_files = len(filenames)

x=x

animate_single_file(filename, frac=0, remove_frames=True, do_tqdm=True)

x=x

# filename = filenames[0]
# animate_single_file(filename, frac=0, remove_frames=True, do_tqdm=True)
for filename in tqdm(filenames):
    animate_single_file(filename, frac=0, remove_frames=True, do_tqdm=False)

x=x

num_cores = mp.cpu_count() - 1
num_cores_max = 15
if num_cores >= num_cores_max:
    num_cores = num_cores_max


if __name__ == '__main__':

    print(f"Animating {N_files} files using {num_cores} cores, please wait.", flush=True)
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(animate_single_file, filenames), total=N_files))

    print("Finished")


# Do you want the application" orca.app to accept incoming network connections
# https://github.com/plotly/orca/issues/269 



from copy import copy
class SIRfile:

    def __init__(self, filename, i=None):
        self.filename = filename
        print("Loading filename")
        self.SK, self.P1, self.UK = joblib.load(filename)
        filename_AK = filename.replace('SK_P1_UK.joblib', 'AK.parquet')
        self.AK = awkward.fromparquet(filename_AK)

        self.N = len(self.SK)
        if i:
            self.i = i

    def __call__(self, i):
        self.i = i
        return copy(self)

    def to_df(self, i=None):
        if i is None:
            i = self.i

        # categories = 'S, E, I, R'.split(', ')
        mapping = {-1: 'S', 
                    0: 'E', 1: 'E', 2:'E', 3: 'E',
                    4: 'I', 5: 'I', 6:'I', 7: 'I',
                    8: 'R'}

        df = pd.DataFrame(self.P1, columns=['x', 'y'])
        df['SK_num'] = self.SK[i]
        df['UK_num'] = self.UK[i]
        df["SK"] = df['SK_num'].replace(mapping).astype('category')
        # self.df = df
        return df


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('matplotlibrc')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_key = {str(label): col for col, label in zip(colors, ['S', 'E', 'I', 'R'])}


import datashader as ds
import datashader.transfer_functions as tf

def df_to_fig(df, plot_width=600, plot_height=600, figsize=(6, 6), legend_fontsize=12, frameon=False):

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                       x_range=[df['x'].min(), df['x'].max()], 
                       y_range=[df['y'].min(), df['y'].max()],
                       x_axis_type='linear', y_axis_type='linear',
                    )
    agg = canvas.points(df, 'x', 'y', ds.count_cat('SK'))

    # color_key = color_key_b_c_uds_g
    # img = tf.shade(agg, color_key=color_key, how='log') # eq_hist
    img = tf.shade(agg, color_key=color_key, how='eq_hist') # eq_hist
    spread = tf.dynspread(img, threshold=0.9, max_px=1)
    pil = spread.to_pil()

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(np.array(pil))
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    legend_elements = [Line2D([0], [0], marker='o', color='white', markerfacecolor=val, label=key) for key, val in color_key.items()]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.1, -0.05), fontsize=legend_fontsize, frameon=frameon)
    return fig

   
def SIR_object_to_image(SIR_object):
    df = SIR_object.to_df()
    fig = df_to_fig(df);
    i_day = SIR_object.i

    ax = fig.axes[0]
    ax.text(0.05, 0.95, f"{i_day=}", 
                    horizontalalignment='center',
                    verticalalignment='center', 
                    transform=ax.transAxes)

    figname = 'Figures_SK_P1_UK/animation_N'
    figname += SIR_object.filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib')
    figname += f'.{SIR_object.i:06d}.png'

    Path(figname).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname, dpi=75)
    return None


def animate_SIR_file(filename, num_cores_max=20, do_tqdm=True, remove_frames=True):

    num_cores = mp.cpu_count() - 1
    if num_cores >= num_cores_max:
        num_cores = num_cores_max

    SIR_base = SIRfile(filename)
    N = SIR_base.N
    SIR_objects = [SIR_base(i) for i in range(N)]
    # SIR_object = SIR_objects[200]

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




def SIRfiles_i_day_to_df(i_SIR_tuple):
    i_day, (SIRfile_SK, SIRfile_P1, SIRfile_UK) = i_SIR_tuple

    # categories = 'S, E, I, R'.split(', ')
    mapping = {-1: 'S', 
                0: 'E', 1: 'E', 2:'E', 3: 'E',
                4: 'I', 5: 'I', 6:'I', 7: 'I',
                8: 'R'}

    df = pd.DataFrame(SIRfile_P1, columns=['x', 'y'])
    df['SK_num'] = SIRfile_SK[i_day]
    df['UK_num'] = SIRfile_UK[i_day]
    df["SK"] = df['SK_num'].replace(mapping).astype('category')
    return df




if False:

    # conda install -c conda-forge holoviews
    from holoviews.operation.datashader import datashade
    import holoviews as hv

    hv.extension('bokeh')


    np.random.seed(1)
    n = 1000000 # Number of points
    f = filter_width = 5000 # momentum or smoothing parameter, for a moving average filter

    # filtered random walk
    xs = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f)/f).cumsum()
    ys = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f)/f).cumsum()

    # Add "mechanical" wobble on the x axis
    xs += 0.1*np.sin(0.1*np.array(range(n-1+f)))

    # Add "measurement" noise
    xs += np.random.normal(0, 0.005, size=n-1+f)
    ys += np.random.normal(0, 0.005, size=n-1+f)

    # Add a completely incorrect value
    xs[int(len(xs)/2)] = 100
    ys[int(len(xs)/2)] = 0

    # Create a dataframe
    df = pd.DataFrame(dict(x=xs,y=ys))


    filename = 'Data_SK_P1_UK/test.SK_P1_UK.joblib'
    SIR_base = SIRfile(filename, 10)
    df = SIR_base.to_df()

    opts = hv.opts.RGB(width=1000, height=1000)
    datashade(hv.Path(df, kdims=['x','y']), normalization='linear', aggregator=ds.any()).opts(opts)




#%%


import datashader as ds
import datashader.transfer_functions as tf
from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle

from itertools import chain

# %%

np.random.seed(0)
n=100
m=20000

nodes = pd.DataFrame(["node"+str(i) for i in range(n)], columns=['name'])
nodes.head()


#%%
edges = pd.DataFrame(np.random.randint(0,len(nodes), size=(m, 2)),
                     columns=['source', 'target'])
edges.tail()


# %%

randomloc = random_layout(nodes)
circular  = circular_layout(nodes, uniform=False)

# %%

cvsopts = dict(plot_height=400, plot_width=400)

def nodesplot(nodes, name=None, canvas=None, cat=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    aggregator=None if cat is None else ds.count_cat(cat)
    agg=canvas.points(nodes,'x','y',aggregator)
    return tf.spread(tf.shade(agg, cmap=["#FF3333"]), px=3, name=name)


tf.Images(nodesplot(randomloc,"Random layout"),
          nodesplot(circular, "Circular layout"))


#%%

%time forcedirected = forceatlas2_layout(nodes, edges)
tf.Images(nodesplot(forcedirected, "ForceAtlas2 layout"))

# %%

def edgesplot(edges, name=None, canvas=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)
    
def graphplot(nodes, edges, name="", canvas=None, cat=None):
    if canvas is None:
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)
        
    np = nodesplot(nodes, name + " nodes", canvas, cat)
    ep = edgesplot(edges, name + " edges", canvas)
    return tf.stack(ep, np, how="over", name=name)


# %%

cd = circular
fd = forcedirected

# %%

cd = circular
fd = forcedirected

%time cd_d = graphplot(cd, connect_edges(cd,edges), "Circular layout")
%time fd_d = graphplot(fd, connect_edges(fd,edges), "Force-directed") 
%time cd_b = graphplot(cd, hammer_bundle(cd,edges), "Circular layout, bundled")
%time fd_b = graphplot(fd, hammer_bundle(fd,edges), "Force-directed, bundled") 

tf.Images(cd_d,fd_d,cd_b,fd_b).cols(2)

# %%

rd = randomloc

%time rd_d = graphplot(rd, connect_edges(rd,edges), "Random layout")
%time rd_b = graphplot(rd, hammer_bundle(rd,edges), "Random layout, bundled")

tf.Images(rd_d,rd_b)


# %%

import networkx as nx

filename = 'Data_SK_P1_UK/test.SK_P1_UK.joblib'
SIR_base = SIRfile(filename, 10)
df = SIR_base.to_df()

#%%

import matplotlib.pyplot as plt

i_day = 1

G=nx.Graph()

for i, xy in enumerate(SIR_base.P1):
    G.add_node(i, pos=xy)


for i, ak in enumerate(SIR_base.AK[i_day]):
    for j in ak:
        G.add_edge(i, j)

pos = nx.get_node_attributes(G, 'pos')

nx.draw(G, pos, with_labels=True)



# %%

# G.add_nodes_from(pos.keys())

# for n, p in pos.iteritems():
#     G.node[n]['pos'] = p



