import os

import kaggle

if not os.path.exists('./data'):
    os.mkdir('./data')
    os.mkdir('./data/raw')

kaggle.api.authenticate()

kaggle.api.dataset_download_files('rajathmc/cornell-moviedialog-corpus', './data/raw', quiet=False, unzip=True)
print('Dataset downloaded and saved in data folder!')
