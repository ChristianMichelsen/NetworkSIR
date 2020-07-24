import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src import rc_params
rc_params.set_rc_params()
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

try:
    from src import utils
    from src import simulation_utils
    from src import file_loaders
    from src import SIR
except ImportError:
    import utils
    import simulation_utils
    import file_loaders
    import SIR



def make_SIR_curves(abn_files, variable='I', force_overwrite=False):

    d_ylabel = {'I': 'Infected', 'R': 'Recovered'}
    d_label_loc = {'I': 'upper right', 'R': 'lower right'}

    pdf_name = "test.pdf"

    pdf_name = Path(f"Figures/ABN_{variable}.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_overwrite:
        print(f"{pdf_name} already exists")
        return None


    with PdfPages(pdf_name) as pdf:

        for ABN_parameter in tqdm(abn_files.keys):

            cfg = utils.string_to_dict(ABN_parameter)

            fig, ax = plt.subplots() # figsize=(20, 10)

            Tmax = 0
            lw = 0.1 * 10 / np.sqrt(len(abn_files[ABN_parameter]))


            # file = abn_files[ABN_parameter][0]
            for i, file in enumerate(abn_files[ABN_parameter]):
                try:
                    df = file_loaders.pandas_load_file(file)
                except EmptyDataError as e:
                    print(f"Skipping {filename_ID} because empty file")
                    continue
                label = 'Simulations' if i == 0 else None
                ax.plot(df['time'].values, df[variable].values, lw=lw, c='k', label=label)
                if df['time'].max() > Tmax:
                    Tmax = df['time'].max()

            Tmax = max(Tmax, 50)

            while True:
                ts = 0.1
                df_fit = SIR.integrate(cfg, Tmax, dt=0.01, ts=ts)

                if df_fit[variable].iloc[-1] - df_fit[variable].iloc[-1-int(1/ts)]

            ax.plot(df_fit['time'], df_fit[variable], lw=2.5, color='red', label='SIR')
            leg = ax.legend(loc=d_label_loc[variable])
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

            try:
                title = utils.dict_to_title(cfg, len(abn_files[ABN_parameter]))
            except KeyError as e:
                print(cfg)
                raise e

            ax.set(title=title, xlabel='Time', ylim=(0, None), ylabel=d_ylabel[variable])

            ax.set_rasterized(True)
            ax.set_rasterization_zorder(0)

            pdf.savefig(fig, dpi=100)
            plt.close('all')


