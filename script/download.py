


import os
from titlequill.utils.io_ import load_json
from titlequill.utils.logger import Logger
from titlequill.dataset import OAGKXDownloader

SCRIPT_DIR     = os.path.abspath(os.path.join(__file__, "..", ".."))
LOCAL_SETTINGS = os.path.abspath(os.path.join(SCRIPT_DIR, "local_settings.json"))

DATASET_DIR = load_json(LOCAL_SETTINGS)["dataset_dir"]


if __name__ == "__main__":
    
    logger = Logger()
    
    downloader = OAGKXDownloader(target_dir=DATASET_DIR, logger=logger)
    
    downloader.download()
    