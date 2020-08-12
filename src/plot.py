import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src import rc_params
rc_params.set_rc_params()
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd

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

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/ABN_{variable}.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_overwrite:
        print(f"{pdf_name} already exists")
        return None


    with PdfPages(pdf_name) as pdf:


        for ABN_parameter in tqdm(abn_files.keys):
            # break

            cfg = utils.string_to_dict(ABN_parameter)

            fig, ax = plt.subplots() # figsize=(20, 10)

            Tmax = 0
            lw = 0.1 * 10 / np.sqrt(len(abn_files[ABN_parameter]))


            # file, i = abn_files[ABN_parameter][0], 0
            for i, file in enumerate(abn_files[ABN_parameter]):
                try:
                    df = file_loaders.pandas_load_file(file)
                except EmptyDataError as e:
                    print(f"Skipping {filename_ID} because empty file")
                    continue
                label = 'Simulations' if i == 0 else None
                ax.plot(df['time'].values, df[variable].values.astype(int), lw=lw, c='k', label=label)
                if df['time'].max() > Tmax:
                    Tmax = df['time'].max()

            Tmax = max(Tmax, 50)


            # checks that the curve has flattened out
            while True:
                ts = 0.1
                df_fit = SIR.integrate(cfg, Tmax, dt=0.01, ts=ts)
                Tmax *= 1.5
                delta_1_day = (df_fit[variable].iloc[-1] - df_fit[variable].iloc[-1-int(1/ts)])
                delta_rel = delta_1_day / cfg.N_tot
                if delta_rel < 1e-5:
                    break

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


# %%


def SDOM(x):
    "standard deviation of the mean"
    return np.std(x) / np.sqrt(len(x))


def compute_ABN_mSEIR_proportions(filenames):
    "Compute the fraction (z) between ABN and mSEIR for I_max and R_inf "

    I_max_ABN = []
    R_inf_ABN = []
    for filename in filenames:
        try:
            df = file_loaders.pandas_load_file(filename)
        except EmptyDataError:
            print(f"Empty file error at {filename}")
            continue
        I_max_ABN.append(df['I'].max())
        R_inf_ABN.append(df['R'].iloc[-1])
    I_max_ABN = np.array(I_max_ABN)
    R_inf_ABN = np.array(R_inf_ABN)

    Tmax = max(df['time'].max()*1.2, 300)
    cfg = utils.string_to_dict(filename)
    df_SIR = SIR.integrate(cfg, Tmax, dt=0.01, ts=0.1)

    z_rel_I = I_max_ABN / df_SIR['I'].max()
    z_rel_R = R_inf_ABN / df_SIR['R'].iloc[-1]

    return z_rel_I, z_rel_R, cfg



def get_1D_scan_results(scan_parameter, non_default_parameters):
    "Compute the fraction between ABN and mSEIR for all simulations related to the scan_parameter"

    simulation_parameters_1D_scan = simulation_utils.get_simulation_parameters_1D_scan(scan_parameter, non_default_parameters)
    N_simulation_parameters = len(simulation_parameters_1D_scan)
    if N_simulation_parameters == 0:
        return None

    base_dir = Path('Data') / 'ABN'

    x = np.zeros(N_simulation_parameters)
    y_I = np.zeros(N_simulation_parameters)
    y_R = np.zeros(N_simulation_parameters)
    sy_I = np.zeros(N_simulation_parameters)
    sy_R = np.zeros(N_simulation_parameters)
    n = np.zeros(N_simulation_parameters)

    # ABN_parameter = simulation_parameters_1D_scan[0]
    for i, ABN_parameter in enumerate(tqdm(simulation_parameters_1D_scan, desc=scan_parameter)):
        filenames = [str(filename) for filename in base_dir.rglob('*.csv') if f"{ABN_parameter}/" in str(filename)]

        z_rel_I, z_rel_R, cfg = compute_ABN_mSEIR_proportions(filenames)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_I[i] = SDOM(z_rel_I)
        sy_R[i] = SDOM(z_rel_R)
        n[i] = len(z_rel_I)

    return x, y_I, y_R, sy_I, sy_R, n, cfg


def extract_limits(ylim):
    """ deals with both limits of the form (0, 1) and [(0, 1), (0.5, 1.5)] """
    if isinstance(ylim, (tuple, list)):
        if isinstance(ylim[0], (float, int)):
            ylim0 = ylim1 = ylim
        elif isinstance(ylim[0], (tuple, list)):
            ylim0, ylim1 = ylim
    else:
        ylim0 = ylim1 = (None, None)

    return ylim0, ylim1


from pandas.errors import EmptyDataError
def plot_1D_scan(scan_parameter, do_log=False, ylim=None, non_default_parameters=None):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_results(scan_parameter, non_default_parameters)
    if not res:
        return None
    x, y_I, y_R, sy_I, sy_R, n, cfg = res


    d_par_pretty = utils.get_d_translate()
    title = utils.dict_to_title(cfg, exclude=[scan_parameter, 'ID'])
    xlabel = r"$" + d_par_pretty[scan_parameter] + r"$"

    normed_by = 'fit'

    ylim0, ylim1 = extract_limits(ylim)


    # n>1 datapoints
    mask = (n > 1)

    factor = 0.8
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16*factor, 9*factor)) #
    fig.suptitle(title, fontsize=28*factor)

    ax0.errorbar(x[mask], y_I[mask], sy_I[mask], fmt='.', color='black', ecolor='black', elinewidth=1, capsize=10)
    ax0.errorbar(x[~mask], y_I[~mask], sy_I[~mask], fmt='.', color='grey', ecolor='grey', elinewidth=1, capsize=10)
    ax0.set(xlabel=xlabel, ylim=ylim0)

    ax0.set_ylabel(r'$I_\mathrm{max}^\mathrm{ABN} \, / \,\, I_\mathrm{max}^\mathrm{'+normed_by+'}$')

    ax1.errorbar(x[mask], y_R[mask], sy_R[mask], fmt='.', color='black', ecolor='black', elinewidth=1, capsize=10)
    ax1.errorbar(x[~mask], y_R[~mask], sy_R[~mask], fmt='.', color='grey', ecolor='grey', elinewidth=1, capsize=10)
    ax1.set(xlabel=xlabel)
    ylabel = r'$R_\infty^\mathrm{ABN} \, / \,\, R_\infty^\mathrm{'+normed_by+'}$'
    ax1.set(ylabel=ylabel, ylim=ylim1)

    if do_log:
        ax0.set_xscale('log')
        ax1.set_xscale('log')
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, wspace=0.45)

    figname_pdf = f"Figures/1D_scan/1D_scan_{scan_parameter}"
    for key, val in non_default_parameters.items():
        figname_pdf += f"_{key}_{val}"
    figname_pdf += f'_normed_by_{normed_by}.pdf'

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100) # bbox_inches='tight', pad_inches=0.3
    plt.close('all')
