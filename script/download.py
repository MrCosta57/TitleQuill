import os
from script.utils.settings import DATASET_DIR
from titlequill.utils.io_ import load_json
from titlequill.utils.logger import Logger
from titlequill.dataset import OAGKXDownloader

if __name__ == "__main__":
    
    logger = Logger()
    
    downloader = OAGKXDownloader(target_dir=DATASET_DIR, logger=logger)
    
    downloader.download()
    