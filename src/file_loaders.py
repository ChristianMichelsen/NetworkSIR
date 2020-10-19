import pandas as pd
from pathlib import Path
import re
import h5py
import os
from tqdm import tqdm
from src.utils import utils


def pandas_load_file(filename):
    # df_raw = pd.read_csv(file)  # .convert_dtypes()

    with h5py.File(filename, "r") as f:
        df_raw = pd.DataFrame(f["df"][()])

    for state in ["E", "I"]:
        df_raw[state] = sum(
            (df_raw[col] for col in df_raw.columns if state in col and len(col) == 2)
        )

    # only keep relevant columns
    df = df_raw[["Time", "E", "I", "R"]].copy()
    df.rename(columns={"Time": "time"}, inplace=True)
    return df


def path(file):
    if isinstance(file, str):
        file = Path(file)
    return file


def file_is_empty(file):
    return path(file).stat().st_size == 0


def get_all_ABM_filenames(base_dir="Data/ABM", filetype="hdf5"):
    "get all ABM result files with filetype {filetype}"
    files = path(base_dir).rglob(f"*.{filetype}")
    # files = sorted(files, )
    return sorted(
        [str(file) for file in files if not file_is_empty(file)],
        key=os.path.getmtime,
    )


def get_all_ABM_folders(filenames):
    folders = list()
    for filename in filenames:
        folder = str(path(filename).parent)
        if not folder in folders:
            folders.append(folder)
    return folders


def filename_to_hash(filename):
    filename = str(filename)
    # split at "_" and "."
    return re.split("_|\.", filename)[-5]


def filename_to_cfg(filename):
    return hash_to_cfg(filename_to_hash(filename))


def folder_to_hash(folder):
    folder = str(folder)
    # split at "_" and "."
    return folder.split("/")[-1]


from tinydb import Query


def folder_to_cfg(folder):
    hash_ = folder_to_hash(folder)
    return hash_to_cfg(hash_)


def hash_to_cfg(hash_, cfgs_dir="./Data/cfgs"):
    db_cfg = utils.get_db_cfg()
    q = Query()
    q_result = db_cfg.search(q.hash == hash_)
    if len(q_result) == 0:
        cfgs = [str(file) for file in Path(cfgs_dir).rglob(f"*{hash_}.yaml")]
        if len(cfgs) == 1:
            cfg = utils.load_yaml(cfgs[0])
            cfg.hash = utils.cfg_to_hash(cfg)
            return cfg
        else:
            return None
    assert len(q_result) == 1
    cfg = utils.DotDict(q_result[0])
    return cfg


def get_cfgs(all_folders):
    hashes = set()
    cfgs = []
    for folder in all_folders:
        cfg = folder_to_cfg(folder)
        if cfg is not None:
            if not cfg.hash in hashes:
                cfgs.append(cfg)
            hashes.add(cfg.hash)
    return cfgs


class ABM_simulations:
    def __init__(self, base_dir="Data/ABM", filetype="hdf5"):
        self.base_dir = Path(base_dir)
        self.filetype = filetype
        self.all_filenames = get_all_ABM_filenames(base_dir, filetype)
        self.all_folders = get_all_ABM_folders(self.all_filenames)
        self.cfgs = get_cfgs(self.all_folders)
        self.d = self._convert_all_files_to_dict(filetype)

    def _convert_all_files_to_dict(self, filetype):
        """
        Dictionary containing all files related to a given hash:
        d[hash] = list of filenames
        """
        d = {}
        for cfg in self.cfgs:
            d[cfg.hash] = utils.hash_to_filenames(cfg.hash, self.base_dir, self.filetype)
        return d

    def iter_all_files(self):
        for filename in self.all_filenames:
            yield filename

    def iter_folders(self):
        for cfg in self.cfgs:
            filenames = self.d[cfg.hash]
            yield cfg, filenames

    def iter_cfgs(self):
        for cfg in self.cfgs:
            yield cfg

    def cfg_to_filenames(self, cfg):
        cfg = utils.DotDict(cfg)
        cfg_list = utils.query_cfg(cfg)
        if not len(cfg_list) == 1:
            raise AssertionError(
                f"cfg did not give unique results in the database",
                "cfg:",
                cfg,
                "cfg_list:",
                cfg_list,
            )
        return self.d[cfg_list[0].hash]

    # def __getitem__(self, key):
    #     if isinstance(key, int):
    #         return self.all_files[key]
    #     # elif isinstance(key, int):
    #     #     return self.all_files[key]

    def __len__(self):
        return len(self.all_filenames)

    def __repr__(self):
        return (
            f"ABM_simulations(base_dir='{self.base_dir}', filetype='{self.filetype}').\n"
            + f"Contains {len(self)} files with "
            + f"{len(self.cfgs)} different simulation parameters."
        )
