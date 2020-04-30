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


def animate_single_file(filename, remove_frames=True, do_tqdm=False):
    SIRfile_SK, SIRfile_P1, SIRfile_UK = joblib.load(filename)
    fignames = []

    it = SIRfile_SK
    if do_tqdm:
        it = tqdm(it, desc='Creating individual frames')
    for i_day, _ in enumerate(it):
        SIRfile_SK_i = SIRfile_SK[i_day]
        SIRfile_UK_i = SIRfile_UK[i_day]
        SIRfile_P1_i = SIRfile_P1[i_day]

        fig = make_subplots(rows=1, cols=3, 
                            subplot_titles=['SK', 'UK', 'P1'], 
                            column_widths=[0.3, 0.3, 0.4])

        x, y = np.unique(SIRfile_SK_i, return_counts=True)
        fig.add_trace(go.Bar(x=x, y=y), row=1, col=1)
        fig.update_xaxes(title_text="SK", row=1, col=1)
        fig.update_yaxes(title_text="Counts (log)", type="log", row=1, col=1)

        x, y = np.unique(SIRfile_UK_i, return_counts=True)
        fig.add_trace(go.Histogram(x=SIRfile_UK_i, nbinsx=Nbins), row=1, col=2)
        fig.update_xaxes(title_text="UK", row=1, col=2)
        fig.update_yaxes(title_text="Counts", row=1, col=2) # type="log"

        fig.add_trace(go.Scatter(x=SIRfile_P1_i[:, 0], y=SIRfile_P1_i[:, 1], name=f'Fit'), 
                                row=1, col=3)
        fig.update_xaxes(title_text="x", range=[-1, 1], row=1, col=3)
        fig.update_yaxes(title_text="y", range=[-1, 1], row=1, col=3) # type="log"

        # Edit the layout
        k = 1.2
        fig.update_layout(title=f'SK P1 UK, {i_day=}',
                        height=400*k, width=1000*k,
                        showlegend=False,
                        )

        # fig.update_yaxes(rangemode="tozero")
        # fig.show()
        figname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + f'{i_day}.png'
        # figname = f"Figures_SK_P1_UK/.tmp_{filename}_{i_day}.png"
        Path(figname).parent.mkdir(parents=True, exist_ok=True)

        fig.write_image(figname)
        fignames.append(figname)


    import imageio # conda install imageio
    gifname = 'Figures_SK_P1_UK/animation_N' + filename.strip('Data_SK_P1_UK/NetworkSimulation_').strip('.joblib') + '.gif'
    with imageio.get_writer(gifname, mode='I', duration=0.1) as writer:
        it_frames = fignames
        if do_tqdm:
            it_frames = tqdm(it_frames, desc='Stitching frames to gif')
        for figname in it_frames:
            image = imageio.imread(figname)
            writer.append_data(image)
            if remove_frames:
                Path(figname).unlink() # delete file
    

Nbins = 100

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