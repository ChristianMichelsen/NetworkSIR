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

# def filename_to_dotdict(filename):
#     return utils.string_to_dict(Path(filename).stem)


def get_filenames_different_than_default(parameter, **kwargs):

    base_dir = Path('Data') / 'ABN'
    all_sim_pars = sorted([str(x.name) for x in base_dir.glob('*') if '.DS' not in str(x.name)])

    all_sim_pars_as_dict = {s: utils.string_to_dict(s) for s in all_sim_pars}

    df_sim_pars = pd.DataFrame.from_dict(all_sim_pars_as_dict, orient='index')

    default_pars = simulation_utils.get_cfg_default()
    for key, val in kwargs.items():
        default_pars[key] = val

    if isinstance(parameter, str):
        parameter = [parameter]

    query = ''
    for key, val in default_pars.items():
        if not key in parameter:
            query += f"{key} == {val} & "
    query = query[:-3]

    df_different_than_default = df_sim_pars.query(query).sort_values(parameter)
    return list(df_different_than_default.index)




def SDOM(x):
    # standard deviation of the mean
    return np.std(x) / np.sqrt(len(x))


def foo_no_fit(filenames):

    I_max_ABN = []
    R_inf_ABN = []

    # filename = filenames[0]
    for filename in filenames:
        cfg = utils.string_to_dict(filename)
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
    # df_SIR = ODE_integrate_cfg_to_df(cfg, Tmax, dt=0.01, ts=0.1)
    df_SIR = SIR.integrate(cfg, Tmax, dt=0.01, ts=0.1)

    z_rel_I = I_max_ABN / df_SIR['I'].max()
    z_rel_R = R_inf_ABN / df_SIR['R'].iloc[-1]

    return z_rel_I, z_rel_R, cfg


def foo_with_fit(filenames, fit_objects_all):

    cfg = utils.string_to_dict(filenames[0])
    sim_par = dict_to_str(cfg)

    z_rel_I = []
    z_rel_R = []

    # filename = filenames[0]
    for fit_object in fit_objects_all[sim_par].values():

        z_rel_I.append( fit_object.I_max_ABN / fit_object.I_max_fit )
        z_rel_R.append( fit_object.R_inf_ABN / fit_object.R_inf_fit )

    z_rel_I = np.array(z_rel_I)
    z_rel_R = np.array(z_rel_R)

    return z_rel_I, z_rel_R, cfg


def foo(parameter, fit_objects_all, **kwargs):

    # kwargs = {}
    default_files_as_function_of_parameter = get_filenames_different_than_default(parameter, **kwargs)
    if len(default_files_as_function_of_parameter) == 0:
        return None

    base_dir = Path('Data') / 'ABN'

    x = []
    y_I = []
    y_R = []
    sy_I = []
    sy_R = []
    n = []

    # sim_par = default_files_as_function_of_parameter[0]
    for sim_par in tqdm(default_files_as_function_of_parameter, desc=parameter):
        filenames = [str(filename) for filename in base_dir.rglob('*.csv') if f"{sim_par}/" in str(filename)]

        if fit_objects_all is None:
            z_rel_I, z_rel_R, cfg = foo_no_fit(filenames)
        else:
            z_rel_I, z_rel_R, cfg = foo_with_fit(filenames, fit_objects_all)

        x.append(cfg[parameter])
        y_I.append(np.mean(z_rel_I))
        y_R.append(np.mean(z_rel_R))
        sy_I.append(SDOM(z_rel_I))
        sy_R.append(SDOM(z_rel_R))
        n.append(len(z_rel_I)) # not len(filenames) in case any empty files

    x = np.array(x)
    y_I = np.array(y_I)
    y_R = np.array(y_R)
    sy_I = np.array(sy_I)
    sy_R = np.array(sy_R)
    n = np.array(n)

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
def plot_1D_scan(parameter, fit_objects_all=None, do_log=False, ylim=None, **kwargs):

    # kwargs = {}
    res = foo(parameter, fit_objects_all, **kwargs)
    if res is None:
        return None
    x, y_I, y_R, sy_I, sy_R, n, cfg = res


    d_par_pretty = utils.get_d_translate()
    title = utils.dict_to_title(cfg, exclude=[parameter, 'ID'])
    xlabel = r"$" + d_par_pretty[parameter] + r"$"

    if fit_objects_all is None:
        normed_by = 'SIR'
    else:
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

    figname_pdf = f"Figures/1D_scan/1D_scan_{parameter}"
    for key, val in kwargs.items():
        figname_pdf += f"_{key}_{val}"
    figname_pdf += f'_normed_by_{normed_by}.pdf'

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100) # bbox_inches='tight', pad_inches=0.3
    plt.close('all')
