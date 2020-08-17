>> git clone https://github.com/ChristianMichelsen/NetworkSIR
>> cd NetworkSIR
>> conda env create -f environment_specific.yml

>> conda activate NetworkSIR

(or, if already installed)

>> conda env update --file environment_specific.yml

Generate Simulations:
>> python run_simulation.py

Plot the results:
>> python plot_results.py

Make animations
>> python generate_animations.py