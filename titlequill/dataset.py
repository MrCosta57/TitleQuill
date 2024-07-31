
from dataclasses import dataclass
import json
import os
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse
import zipfile

import requests
from tqdm import tqdm

from transformers     import PreTrainedTokenizer
from torch.utils.data import Dataset

from titlequill.utils.io_ import get_file_lines_count, read_file_lines
from titlequill.utils.logger import Logger, SilentLogger
import re


class OAGKXDownloader:
    ''' Downloads the OAGKX dataset from the LINDAT repository '''
    
    URL = "https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3062/oagkx.zip?sequence=1&isAllowed=y"
    
    def __init__(
        self, 
        target_dir : str, 
        logger     : Logger = SilentLogger()
    ):
        '''
        Initialize the downloader

        :param target_dir: Directory to save the dataset
        :type target_dir: str
        :param logger: Logger to log i/o operations, defaults to SilentLogger()
        :type logger: Logger, optional
        :raises ValueError: If the target directory is invalid
        '''
        
        # Check if the target directory exists
        if not os.path.exists(target_dir): raise ValueError(f"Invalid path: {target_dir}")
        
        self._target_dir = target_dir
        self._logger = logger

    
    def __str__ (self) -> str: return f"OAGKXDownloader[target_dir={self._target_dir}]"
    def __repr__(self) -> str: return str(self)
        
    def download(self):
        ''' Downloads the dataset from the URL '''
        
        # Parse the URL to get the filename
        parsed_url = urlparse(self.URL)
        filename   = os.path.basename(parsed_url.path)
        file_path  = os.path.join(self._target_dir, filename)
        
        # Stream the download and show progress bar
        response   = requests.get(self.URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte

        with open(file_path, 'wb') as f:
            
            self._logger.info(mess=f"Downloading from {self.URL}")

            
            # Use tqdm to show download progress
            with tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        t.update(len(chunk))
        
        self._logger.info(mess=f"Downloaded {filename} to {self._target_dir}")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                
            # Assuming `zip_ref` is your zipfile.ZipFile object and `self._target_dir` is your target directory
            total_files = len(zip_ref.namelist())
            extracted_files = 0
            
            self._logger.info(mess=f"Unzipping {total_files} files...")

            with tqdm(total=total_files, unit='file') as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, self._target_dir)
                    extracted_files += 1
                    pbar.update(1)
                    pbar.set_postfix(extracted=f"{extracted_files}/{total_files}")

        
        self._logger.info(mess=f"Unzipped {filename} in {self._target_dir}")
        


@dataclass
class OAGKXItem:
    ''' Dataclass to represent an item in the OAGKX dataset - i.e. a line in the dataset file '''
    
    title    : str
    ''' Title of the paper '''
    
    abstract : str      
    ''' Abstract of the paper '''
    
    keywords : List[str]
    ''' Keywords associated with the paper '''
    
    def __str__ (self) -> str: return  f'Title: {self.title}\n\nAbstract: {self.abstract}\n\nKeywords: {self.keywords}'
    def __repr__(self) -> str: return str(self)
    
    @property
    def item(self) -> Dict:
        ''' Returns the item as a dictionary for batching '''
        
        return {
            'title'    : self.title,
            'abstract' : self.abstract,
            'keywords' : ", ".join(self.keywords) # NOTE: This is made to avoid different length representation in the batch
        }
    
    @classmethod
    def from_line(
        cls, 
        line      : str
    ) -> 'OAGKXItem':
        ''' Parses a line from the dataset file and returns an OAGKXItem object '''
        
        # Define the delimiter for keywords
        KEYWORDS_DELIMITER = ' , '
        
        # Parse the line  
        line_dict = json.loads(line)
        
        # Extract title and abstract
        title    = line_dict['title']
        abstract = line_dict['abstract']
        
        # Extract keywords
        keywords = [keyword.strip() for keyword in line_dict['keywords'].split(KEYWORDS_DELIMITER)]
        
        return OAGKXItem(
            title    = title,
            abstract = abstract,
            keywords = keywords,
        )


class OAGKXDataset(Dataset):
    '''
    Class to load the OAGKX dataset from the disk and provide an iterable interface to access the items.
    Since the dataset is splitted in chunks, the loader loads the chunks on demand with caching,
    as it holds the last used chunk in memory for faster access.
    '''
    
    PATTERN  = r'part_(\d+)_(\d+)\.txt'
    ''' Pattern to match the file names '''
    
    def __init__(self, data_dir: str, logger: Logger = SilentLogger()):
        '''
        Initialize the loader

        :param data_dir: Directory containing the dataset files
        :type data_dir: str
        :param tokenizer: Tokenizer to tokenize the text data
        :type tokenizer: PreTrainedTokenizer
        '''
        
        self._data_dir : str    = data_dir
        self._logger   : Logger = logger
        
        # We represent each file indexes by part and subpart indexes - e.g. part_1_2.txt has indexes (1, 2) 
        # The mapping maps the key to the file absolute path and the interval of indexes in the file to build a global indexing across files
        self._file_map: Dict[Tuple[int, int], Tuple[str, Tuple[int, int]]] = self._create_file_map()
        
        # We load the first chunk to start with
        first_key = next(iter(self._file_map.keys()))
        self._loaded_idx = None # this is made to trigger the first load as the current index will of course not match the loaded index
        self._load_chunk(key=first_key)
        
    def __str__(self)  -> str: return f'OAGKXLoader[path: {self._data_dir}; files: {len(self._file_map)}; items: {len(self)}]'
    def __repr__(self) -> str: return str(self)
    
    def __len__ (self) -> int: 
        
        # Use the left index of the last chunk to get the total number of items
        _, (_, tot_items) = list(self._file_map.values())[-1] 
        return tot_items
    
    def __iter__(self) -> Iterable[OAGKXItem]: 
        
        for idx in range(len(self)): yield self.get_item(idx)
    
    # NOTE: We create two methods to access the item
    # - get_item(index) that return the `OAGKXItem` object 
    # - __getitem__(index), that is the magic method to use the DataLoader as uses the dictionary representation to be loaded in batches
    
    def get_item(self, idx: int) -> OAGKXItem:
        ''' 
        Load the item corresponding to the global index 
        
        :param idx: Global index of the item
        :type idx: int
        :return: Item at the index
        :rtype: OAGKXItem
        '''
        
        # Find the chunk key corresponding to the index and load the chunk
        key = self._find_chunk_key(idx)
        
        # Load the chunk
        self._load_chunk(key)
        
        # Get the left relative index in the chunk
        loaded_from, _ = self._loaded_interval
        line = self._loaded_lines[idx - loaded_from]
        
        return OAGKXItem.from_line(line)

    def __getitem__(self, idx: int) -> Dict[str, str | List[str]] : return self.get_item(idx=idx).item
    
    def _create_file_map(self) -> Dict[Tuple[int, int], Tuple[str, Tuple[int, int]]]:
        '''
        Create a mapping from the file indexes to the file path and the interval of indexes in the file

        :return: Mapping from the file indexes to the file path and the interval of indexes in the file
        :rtype: Dict[Tuple[int, int], Tuple[str, Tuple[int, int]]]
        '''
        
        def text_to_key(text: str) -> Tuple[int, int]:
            ''' Extracts the indexes from the file name '''
        
            match = re.search(self.PATTERN, text)
            
            if match: 
                N1 = int(match.group(1))
                N2 = int(match.group(2))
                return N1, N2
            
            else: 
                raise ValueError(f"Invalid text format: {text}, expected {self.TEMPLATE} ")
        
        # Maps part_N1_N2.txt as (N1, N2) -> (file_path, (from, to))
        file_map: Dict[Tuple[int, int], Tuple[str, Tuple[int, int]]] = {}
        
        prev_idx = 0
        
        # Sort the files by the indexes
        file_list = sorted(os.listdir(self._data_dir), key=text_to_key)
        
        # Iterate over the files and build the mapping
        self._logger.info(mess=f"Building file map...")
        
        progress_bar = tqdm(file_list, desc="Processing files")

        for file_name in progress_bar:
            
            if file_name.endswith('.txt') and re.match(self.PATTERN, file_name):
                
                progress_bar.set_description(f"Processing {file_name}")
            
                N1, N2 = text_to_key(file_name)
                
                file_path  = os.path.join(self._data_dir, file_name)
                file_lines = get_file_lines_count(file_path=file_path)
                
                # NOTE: The left index is inclusive, the right index is exclusive to avoid overlaps - i.e. from_ <= x < to
                idx_from = prev_idx
                idx_to   = prev_idx + file_lines
                
                # Create a new entry in the mapping with the file path and the interval of indexes
                file_map[(N1, N2)] = (file_path, (idx_from, idx_to))
                
                prev_idx = idx_to
                
        return file_map
    
    def _load_chunk(self, key: Tuple[int, int]):
        ''' 
        Loads the chunk corresponding to the key.
        If already cached, does nothing.
        
        :param key: Key of the chunk to load
        :type key: Tuple[int, int]
        '''
        
        # If the chunk is already loaded, do nothing
        if key == self._loaded_idx: return
        
        # Otherwise, load the chunk
        file_path, interval = self._file_map[key]
        
        self._loaded_idx      = key
        self._loaded_interval = interval
        self._loaded_lines    = read_file_lines(file_path)
    
    def _find_chunk_key(self, idx: int) -> Tuple[int, int]:
        ''' 
        Find the chunk key corresponding to the global index 
        The operation is constant in the case the index lies in the cached chunk,
        otherwise it is linear in the number of files.
        '''
        
        def check_interval(idx: int, interval: Tuple[int, int]) -> bool:
            ''' Checks if the index is in the interval '''
            from_, to = interval
            return from_ <= idx < to
        
        # If the index is in the loaded chunk, return the loaded key
        if check_interval(idx, self._loaded_interval): return self._loaded_idx
        
        # Otherwise, iterate over the files to find the key
        for key, (_, interval) in self._file_map.items():
            if check_interval(idx, interval): return key
        
        raise ValueError(f"Index {idx} out of bounds")

