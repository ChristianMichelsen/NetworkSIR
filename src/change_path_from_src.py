from pathlib import Path
import os


def change_path_from_src():
    path_was_changed = False
    path = Path("").cwd()
    if path.stem == "src":
        os.chdir(path.parent)
        path_was_changed = True
    return path_was_changed
