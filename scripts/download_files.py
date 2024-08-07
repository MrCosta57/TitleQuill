import os
from urllib.parse import urlparse
import zipfile
import logging
from omegaconf import DictConfig
from tqdm import tqdm
import requests
import hydra


class Downloader:
    """Downloads the OAGKX dataset from the LINDAT repository"""

    def __init__(
        self,
        target_dir: str,
        URL: str = "https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3062/oagkx.zip?sequence=1&isAllowed=y",
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the downloader

        :param target_dir: Directory to save the dataset
        :type target_dir: str
        :param logger: Logger to log i/o operations, defaults to SilentLogger()
        :type logger: Logger, optional
        :raises ValueError: If the target directory is invalid
        """

        # Check if the target directory exists
        if not os.path.exists(target_dir):
            raise ValueError(f"Invalid path: {target_dir}")

        self.URL = URL
        self._target_dir: str = target_dir
        self._logger: logging.Logger = (
            logging.getLogger(__name__) if logger is None else logger
        )

    # --- MAGIC METHODS ---

    def __str__(self) -> str:
        return f"Downloader[target_dir={self._target_dir}]"

    def __repr__(self) -> str:
        return str(self)

    # --- DOWNLOAD METHODS ---

    def download(self):
        """Downloads the dataset from the URL"""

        # Parse the URL to get the filename
        parsed_url = urlparse(self.URL)
        filename = os.path.basename(parsed_url.path)
        file_path = os.path.join(self._target_dir, filename)

        # Stream the download and show progress bar
        response = requests.get(self.URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kilobyte

        # Downloading
        with open(file_path, "wb") as f:

            self._logger.info(f"Downloading from {self.URL}")

            # Use tqdm to show download progress
            with tqdm(total=total_size, unit="iB", unit_scale=True) as t:

                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        t.update(len(chunk))

        self._logger.info(f"Downloaded {filename} to {self._target_dir}")

        # Unzip the file
        with zipfile.ZipFile(file_path, "r") as zip_ref:

            # Assuming `zip_ref` is your zipfile.ZipFile object
            # and `self._target_dir` is your target directory
            total_files = len(zip_ref.namelist())
            extracted_files = 0

            self._logger.info(f"Unzipping {total_files} files...")

            with tqdm(total=total_files, unit="file") as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, self._target_dir)
                    extracted_files += 1
                    pbar.update(1)
                    pbar.set_postfix(extracted=f"{extracted_files}/{total_files}")

        self._logger.info(f"Unzipped {filename} in {self._target_dir}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="scripts")
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Download the dataset
    downloader = Downloader(target_dir=cfg.data.data_dir)
    downloader.download()


if __name__ == "__main__":
    main()
