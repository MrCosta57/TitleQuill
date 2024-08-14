import glob
import os
from urllib.parse import urlparse
import zipfile
import logging
from logging import Logger
from omegaconf import DictConfig
from tqdm import tqdm
import requests
import hydra


class Downloader:
    """
    Generic class to download a dataset from a URL, save it to a target directory and preprocess it.
    """

    def __init__(
        self,
        target_dir: str,
        URL: str,
        logger: Logger = logging.getLogger(__name__),
    ):
        """
        Initialize the downloader

        :param target_dir: Directory to save the dataset
        :type target_dir: str
        :param URL: URL to download the dataset from
        :type URL: str
        :param logger: Logger to log i/o operations, defaults to SilentLogger()
        :type logger: Logger, optional
        :raises ValueError: If the target directory is invalid
        """
        
        self.URL: str = URL
        self._target_dir: str = target_dir
        self._logger: Logger = logger

        # Create the target directory if it does not exist
        if not os.path.exists(target_dir):
            self._logger.info(f"Creating directory {target_dir}")
            os.makedirs(name=target_dir, exist_ok=True)

    # --- MAGIC METHODS ---

    def __str__(self) -> str:
        return f"Downloader[target_dir={self._target_dir}]"

    def __repr__(self) -> str:
        return str(self)

    # --- DOWNLOAD METHODS ---
    
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
        
        # Unzip the file
        with zipfile.ZipFile(zip_file, "r") as zip_ref:

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
        
        # Remove the zip file
        os.remove(zip_file)
        self._logger.info(f"Removed {filename}")
    
    def preprocess(self):
        """
        Preprocess the downloaded files
        By default, no processing is done.
        """
        
        pass

class OAGKXDownloader(Downloader):
    """
    Class to download the OAGKX dataset from the LINDAT repository
        and preprocess the files to have the correct extension and a single numeric indexing.
    """
    
    def __init__(
        self, 
        target_dir: str,
        logger: Logger = logging.getLogger(__name__),        
    ):
    
        URL = "https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3062/oagkx.zip?sequence=1&isAllowed=y"
    
        super().__init__(target_dir=target_dir, URL=URL, logger=logger)
    
    def __str__(self) -> str:
        
        return f'OAGKX{super().__str__()}'
    
    def preprocess(self):
        
        # TODO: Should we pass these argument from main as we did in `preprocess_files.py`?
        old_ext = 'txt'
        new_ext = 'jsonl'
        
        self._logger.info(msg="Preprocessing files...")
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

@hydra.main(version_base="1.3", config_path="../configs", config_name="scripts")
def main(cfg: DictConfig):
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Download the dataset
    downloader = OAGKXDownloader(target_dir=cfg.data.data_dir)
    downloader.download()
    
    # Preprocess the files
    downloader.preprocess()

if __name__ == "__main__":
    main()
