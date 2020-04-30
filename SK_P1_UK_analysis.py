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


def get_SK_P1_UK_filenames():
    filenames = Path('Data_SK_P1_UK').glob(f'*.joblib')
    return [str(file) for file in sorted(filenames)]


filenames = get_SK_P1_UK_filenames()
filename = filenames[0]
N_files = len(filenames)

def animate_single_file(filename, remove_frames=True, do_tqdm=False, plot_first_day=False):
    SIRfile_SK, SIRfile_P1, SIRfile_UK = joblib.load(filename)
    fignames = []

    # categories = 'S, E, I, R'.split(', ')
    mapping = {-1: 'S', 
                0: 'E', 1: 'E', 2:'E', 3: 'E',
                4: 'I', 5: 'I', 6:'I', 7: 'I',
                8: 'R'}

    it = SIRfile_SK
    if do_tqdm:
        it = tqdm(it, desc='Creating individual frames')
    for i_day, _ in enumerate(it):

        df = pd.DataFrame(SIRfile_P1[i_day], columns=['x', 'y'])
        df['SK_num'] = SIRfile_SK[i_day]
        df['UK_num'] = SIRfile_UK[i_day]
        df["SK"] = df['SK_num'].replace(mapping)
        # df.sort_values('SK_num', ascending=True, inplace=True)        



        import plotly.express as px

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
                # 3: "magenta",
                # 4: 'black',
                },
             title="Explicit color mapping")
        fig_P1.update_traces(marker=dict(size=2))
        traces_P1 = fig_P1['data']

        # fig = px.scatter(df, x="x", y="y", color="SK")


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

        # fig.add_trace(go.Scattergl(x=df['x'], y=df['y'], 
        #                          name=f'Fit',
        #                          mode='markers',
        #                          marker=dict(
        #                                     size=1.5,
        #                                     color=df['SK_num'],
        #                                     showscale=True,
        #                                     ),
        #                                     ), 
        #                         row=1, col=3)
        for trace in traces_P1:
            fig.add_trace(trace, row=1, col=3)
        fig.update_xaxes(title_text="x", range=[-1, 1], row=1, col=3)
        fig.update_yaxes(title_text="y", range=[-1, 1], row=1, col=3) # type="log"

        # Edit the layout
        k = 1.2
        fig.update_layout(title=f'SK P1 UK, {i_day=}',
                        height=400*k, width=1000*k,
                        # showlegend=False,
                        )

        # fig.update_yaxes(rangemode="tozero")
        # fig.show()
        figname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + f'{i_day}.png'
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


Nbins = 100

num_cores = mp.cpu_count() - 1
num_cores_max = 15
if num_cores >= num_cores_max:
    num_cores = num_cores_max

# x=x

animate_single_file(filenames[2], remove_frames=True, do_tqdm=True)

x=x

if __name__ == '__main__':

    print(f"Animating {N_files} files using {num_cores} cores, please wait.", flush=True)
    with mp.Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(animate_single_file, filenames), total=N_files))

    print("Finished")


# Do you want the application" orca.app to accept incoming network connections
# https://github.com/plotly/orca/issues/269 