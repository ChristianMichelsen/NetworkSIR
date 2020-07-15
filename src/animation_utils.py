import numpy as np
from pathlib import Path

def get_animation_filenames():
    filenames = Path('Data/animation').glob(f'N_tot__*.animation.hdf5')
    return [str(file) for file in sorted(filenames)]
