import logging
from logging import Logger
import os


def make_dir(dir_path: str, logger: Logger = logging.getLogger(__name__)) -> None:
    """
    Creates a directory if it does not exist.

    :param dir_path: Path to the directory.
    """
    
    if not os.path.exists(dir_path):
    
        logger.info(f"Creating directory {dir_path}")
        os.makedirs(name=dir_path, exist_ok=True)