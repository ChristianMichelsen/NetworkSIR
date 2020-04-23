>> git clone https://github.com/ChristianMichelsen/PyCorona
>> conda env create -f environment.yaml
>> conda activate PyCorona

create login.yaml in login_credentials/ file with the following content: 

Hostname: [insert hostname here]
Username: [insert username here]
Password: [insert password here]

Run DownloadData.py:
>> python DownloadData.py

Run ModelOfCorona_FitToDanishNumbers.py
>> python ModelOfCorona_FitToDanishNumbers.py