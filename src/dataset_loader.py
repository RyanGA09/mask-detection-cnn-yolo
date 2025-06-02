import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(kaggle_json_path, download_path="data"):
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_json_path)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("andrewmvd/face-mask-detection", path=download_path, unzip=True)
    print("Dataset downloaded and extracted to", download_path)