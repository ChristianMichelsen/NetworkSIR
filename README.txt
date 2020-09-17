If first time using:

    >> git clone https://github.com/ChristianMichelsen/NetworkSIR
    >> cd NetworkSIR
    >> conda env create -f environment.yml

(or, if already installed)
    >> conda env update --file environment.yml


>> conda activate NetworkSIR



Generate Simulations:
>> python generate_simulations.py

Plot the results:
>> python generate_plots.py

Make animations
>> python generate_animations.py