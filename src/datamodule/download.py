"""
This module contains the Downloader class to download a dataset from a URL, save it to a target directory and preprocess it.
"""

import glob
import os
from urllib.parse import urlparse
import zipfile
import logging
from logging import Logger
from tqdm import tqdm
import requests
import argparse


class Downloader:
    """
    Generic class to download a dataset from a URL, save it to a target directory and preprocess it.
    """

    def __init__(
        self,
        url: str,
        target_dir: str,
        unzip: bool = True,
        logger: Logger = logging.getLogger(__name__),
    ):
        """
        Initializes the Downloader object.

        :param url: URL to download the dataset from.
        :param target_dir: Directory to save the dataset.
        :param unzip: Whether to unzip the downloaded file.
        :param logger: Logger object to log messages.
        """

        self.URL = url
        self._target_dir = target_dir
        self._unzip = unzip
        self._logger = logger

        # Create the target directory if it does not exist
        if not os.path.exists(target_dir):
            self._logger.info(f"Creating directory {target_dir}")
            os.makedirs(name=target_dir, exist_ok=True)

    def __str__(self) -> str:
        """ Returns a string representation of the Downloader object. """

        return f"Downloader[target_dir={self._target_dir}]"

    def __repr__(self) -> str:
        """ Returns a string representation of the Downloader object"""
        return str(self)

    def download(self):
        """
        Downloads the dataset from the URL.
        """

        # Parse the URL to get the filename
        parsed_url = urlparse(self.URL)
        filename = os.path.basename(parsed_url.path)
        zip_file = os.path.join(self._target_dir, filename)

        # Stream the download and show progress bar
        response = requests.get(self.URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kilobyte

        # Downloading
        with open(zip_file, "wb") as f:
            self._logger.info(f"Downloading from {self.URL}")
            # Use tqdm to show download progress
            with tqdm(total=total_size, unit="iB", unit_scale=True) as t:

                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        t.update(len(chunk))
        self._logger.info(f"Downloaded {filename} to {self._target_dir}")

        if self._unzip:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Assuming `zip_ref` is your zipfile.ZipFile object
                # and `self._target_dir` is your target directory

                # Exclude the directory itself from the count
                total_files = len(zip_ref.namelist()) - 1
                extracted_files = 0

                self._logger.info(f"Unzipping {total_files} files...")
                with tqdm(total=total_files, unit="file") as pbar:
                    for zip_info in zip_ref.infolist():
                        if zip_info.is_dir():
                            continue
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, self._target_dir)
                        extracted_files += 1
                        pbar.update(1)
                        pbar.set_postfix(extracted=f"{extracted_files}/{total_files}")

            self._logger.info(f"Unzipped {filename} in {self._target_dir}")
            # Remove the zip file
            os.remove(zip_file)
            self._logger.info(f"Removed {filename}")


class OAGKXDownloader(Downloader):
    """
    Class to download the OAGKX dataset and postprocess the files to have the correct extension and a single numeric indexing.
    """

    def __init__(
        self,
        target_dir: str,
        unzip: bool = True,
        logger: Logger = logging.getLogger(__name__),
    ):
        """ 
        Initializes the OAGKXDownloader object.
        
        :param url: URL to download the dataset from.
        :param target_dir: Directory to save the dataset.
        :param unzip: Whether to unzip the downloaded file.
        :param logger: Logger object to log messages.

        """

        url = "https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3062/oagkx.zip?sequence=1&isAllowed=y"

        super().__init__(url, target_dir, unzip, logger)

    def __str__(self) -> str:

        return f"OAGKX{super().__str__()}"

    def postprocess(self, old_ext: str = "txt", new_ext: str = "jsonl"):

        self._logger.info(msg="Postprocessing files...")
        self._logger.info(msg=f" > Old extension: {old_ext}")
        self._logger.info(msg=f" > New extension: {new_ext}")
        self._logger.info(msg=f" > Search directory: {self._target_dir}")

        # Get all files with the old extension in the target directory
        files = glob.glob(os.path.join(self._target_dir, f"*.{old_ext}"))
        files.sort()
        self._logger.info(msg=f"Found {len(files)} new files")

        # Get all files with the new extension in the target directory
        pre_files = glob.glob(os.path.join(self._target_dir, f"*.{new_ext}"))
        self._logger.info(msg=f"Found {len(pre_files)} old files")

        # Create files with the new extension and format name
        for i, old_filename in enumerate(files, start=len(pre_files)):
            # Generate a new name and extension
            new_base_name = f"part_{i}.{new_ext}"
            # Construct the full new file path
            new_filename = os.path.join(self._target_dir, new_base_name)
            # Rename the file
            os.rename(old_filename, new_filename)
            self._logger.info(msg=f"Renamed: {old_filename} -> {new_filename}")

        self._logger.info(msg="Done!")


def main(args: argparse.Namespace):

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download the dataset
    downloader = OAGKXDownloader(target_dir=args.data_dir)
    downloader.download()

    # Preprocess the files
    downloader.postprocess(old_ext=args.old_ext_postproc, new_ext=args.new_ext_postproc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/OAGKX",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3062/oagkx.zip?sequence=1&isAllowed=y",
        help="URL to download the dataset",
    )
    parser.add_argument(
        "--old_ext_postproc",
        type=str,
        default="txt",
        help="Old extension of the files to postprocess",
    )
    parser.add_argument(
        "--new_ext_postproc",
        type=str,
        default="jsonl",
        help="New extension of the files to postprocess",
    )
    args = parser.parse_args()

    main(args)
