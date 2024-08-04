from typing import Dict

from titlequill.dataset import OAGKXRawDataset, OAGKXItem
from titlequill.utils.logger import Logger
from script.utils.settings import DATASET_DIR, DATASET_TSV_FILE

def all_keywords_in_abstract(item: OAGKXItem) -> bool: return all([kw in item.abstract for kw in item.keywords])

def main():
    
    logger = Logger()

    dataset = OAGKXRawDataset(dataset_dir=DATASET_DIR, logger=logger)

    dataset.apply_filter(filter=all_keywords_in_abstract)

    dataset.save_tsv(file_path=DATASET_TSV_FILE)

if __name__ == "__main__": main()
