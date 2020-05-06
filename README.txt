>> git clone https://github.com/ChristianMichelsen/NetworkSIR
>> cd NetworkSIR
>> conda env create -f environment.yaml
>> conda activate NetworkSIR

(or, if already installed)

>> conda env update --file environment.yaml

Simulate data:
>> python SimulateDenmark.py

Analyse results
>> python SIR_advanced.py