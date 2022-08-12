[![DOI](https://zenodo.org/badge/258223118.svg)](https://zenodo.org/badge/latestdoi/258223118)

If first time using:

    git clone https://github.com/ChristianMichelsen/NetworkSIR
    cd NetworkSIR
    conda env create -f environment.yaml

(or, if already installed)

    conda env update --file environment.yaml

Then:

    conda activate NetworkSIR

Generate Simulations:

    python generate_simulations.py

Plot the results:

    python generate_plots.py

Make animations

    python generate_animations.py
