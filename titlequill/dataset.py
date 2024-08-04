from __future__ import annotations

from abc import ABC
import re
import csv
import os
import json
import zipfile
import requests
from dataclasses  import dataclass
from typing       import Any, Callable, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets         import load_dataset

from titlequill.utils.io_    import get_file_lines_count, read_file_lines
from titlequill.utils.logger import Logger, SilentLogger
from titlequill.utils.misc   import list_of_dict_to_dict_of_list

# Type Alias
IdxMapping = Dict[int, int]
FileKey    = Tuple[int, int]
Interval   = Tuple[int, int]

class OAGKXDownloader:
    ''' Downloads the OAGKX dataset from the LINDAT repository '''
    
    URL = "https://lindat.cz/repository/xmlui/bitstream/handle/11234/1-3062/oagkx.zip?sequence=1&isAllowed=y"
    
    def __init__(
        self, 
        target_dir  : str,
        logger      : Logger = SilentLogger()
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
        
        self._target_dir : str    = target_dir
        self._logger     : Logger = logger

    # --- MAGIC METHODS ---
    
    def __str__ (self) -> str: return f"OAGKXDownloader[target_dir={self._target_dir}]"
    def __repr__(self) -> str: return str(self)
    
    # --- DOWNLOAD METHODS ---
        
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
        
        # Downloading
        with open(file_path, 'wb') as f:
            
            self._logger.info(mess=f"Downloading from {self.URL}")

            
            # Use tqdm to show download progress
            with tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        t.update(len(chunk))
        
        self._logger.info(mess=f"Downloaded {filename} to {self._target_dir}")

        # Unzip the file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                
            # Assuming `zip_ref` is your zipfile.ZipFile object
            # and `self._target_dir` is your target directory
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
    ''' Dataclass to represent an item in the OAGKX dataset - i.e. a line in the raw dataset file '''
    
    title    : str
    ''' Title of the paper '''
    
    abstract : str      
    ''' Abstract of the paper '''
    
    keywords : List[str]
    ''' Keywords associated with the paper '''
    
    _KEYWORDS_DELIMITER = ", "
    
    KEYS = ['title', 'abstract', 'keywords']
    
    @classmethod
    def from_json(
        cls, 
        json_item : Dict[str, Any]
    ) -> 'OAGKXItem':
        ''' Parses a line from the dataset file and returns an OAGKXItem object '''
        
        if not all([key in json_item for key in cls.KEYS]): 
            raise ValueError(f"Invalid JSON item, expected keys: {cls.KEYS}")
        
        # Extract title and abstract
        title    = json_item['title']
        abstract = json_item['abstract']
        
        # Extract keywords
        keywords_line = json_item['keywords']
        keywords = [keyword.strip() for keyword in keywords_line.split(OAGKXItem._KEYWORDS_DELIMITER)]
        
        return OAGKXItem(
            title    = title,
            abstract = abstract,
            keywords = keywords,
        )
    
    # --- MAGIC METHODS ---
    
    def __str__ (self) -> str: return  f'Title: {self.title}\n\nAbstract: {self.abstract}\n\nKeywords: {self.keywords}'
    def __repr__(self) -> str: return str(self)
    
    @property
    def item(self) -> Dict:
        ''' Returns the item as a dictionary for batching '''
        
        # NOTE: Keywords string transformation is made to avoid different length representation in the batch
        # TODO: Is this the correct choice ???
        return {
            'title'    : self.title,
            'abstract' : self.abstract,
            'keywords' : self._KEYWORDS_DELIMITER.join(self.keywords) 
        }

class _OAGKXDataset(Dataset, ABC):
    '''
    Abstract class to define the interface for the OAGKX dataset loaders.
    
    It provides a common interface to access the items in the dataset, that is 
        - the return function type of the `__getitem__` method to be an OAGKXItem object.
        - an interface to provide a DataLoader object for batching that handles the collate function.
    '''
    
    # --- ABSTRACT METHODS ---
    
    def __getitem__(self, idx: int) -> OAGKXItem:
        
        raise NotImplementedError("Cannot instantiate the abstract class `_OACGXDataset`")
    
    # --- DATALOADER FACTORY ---
    
    def get_dataloader(
        self, 
        batch_size : int                         = 32, 
        shuffle    : bool                        = False, 
        collate_fn : Callable[[OAGKXItem], Dict] = None
    ) -> DataLoader:
        '''
        Create and return a DataLoader object for the dataset.
        
        NOTE: The `__getitem__` object returns the custom dataclass OAGKXItem.
            The `collate_fn` is used to a data structure suitable for batching.

        :param batch_size: The number of samples per batch to load, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Whether to shuffle the dataset between epochs, defaults to False
        :type shuffle: bool, optional
        :param collate_fn: A function that takes a list of dataset items and returns a batch, defaults to None
        :type collate_fn: Callable[[OAGKXItem], Dict], optional
        :return: A DataLoader object that can be used for iterating over the dataset
        :rtype: DataLoader
        '''
        
        # Use as default the collate function defined in the class
        collate_fn = collate_fn or self.default_collate_fn
        
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn
        )

    @staticmethod
    def default_collate_fn(batch: List[OAGKXItem]) -> Dict:
        ''' Extract the items in a dictionary-like format suitable for batching '''
        
        return list_of_dict_to_dict_of_list([el.item for el in batch])


class OAGKXRawDataset(_OAGKXDataset):
    '''
    Class to load the RawOAGKX dataset from the disk and provide an iterable interface to access the items.
    
    Since the dataset is splitted in chunks, the loader uses lazy-loading implementing caching,
    as it holds the last used chunk in memory for a faster access.
    
    It also provides a filtering mechanism to filter the items in the dataset.
    '''
    
    PATTERN  = r'part_(\d+)_(\d+)\.txt'
    ''' Pattern to match the file names '''
    
    def __init__(self, dataset_dir: str, logger: Logger = SilentLogger()):
        '''
        Initialize the loader

        :param dataset_dir: Directory containing the dataset files
        :type dataset_dir: str
        :param logger: Logger to log i/o operations.
        :type Logger: Logger
        '''
        
        # Store input parameters
        self._data_dir : str    = dataset_dir
        self._logger   : Logger = logger
        
        # We index each file with a couple of integers the file name is composed by - e.g. part_1_2.txt has key (1, 2) 
        # The mapping maps the key to the file absolute path and the interval of indexes in the file to build a global indexing across files
        # Example mapping {(1, 2) : ('/path/to/part_1_2.txt', (2000, 3000))}, indicates that the file contains indexes from 2000 to 3000
        self._file_map: Dict[FileKey, Tuple[str, Interval]] = self._create_file_map()
        
        # We load the first chunk to start with
        # NOTE: We set `self._loaded_idx` to None to trigger the first load
        first_key = next(iter(self._file_map.keys()))
        self._loaded_idx = None
        self._load_chunk(key=first_key)
        
        # The initial dictionary has no filtering applied
        self.reset_filtering()
    
    def _create_file_map(self) -> Dict[FileKey, Tuple[str, Interval]]:
        '''
        Helper function for the initialization of the loader.
        It creates a mapping from the file indexes to the file path and the interval of indexes in the file

        :return: Mapping from the file indexes to the file path and the interval of indexes in the file
        :rtype: Dict[FileKey, Tuple[str, Interval]]
        '''
        
        def get_file_key(file_name: str) -> FileKey:
            ''' Extracts the indexes from the file name '''
        
            match = re.search(self.PATTERN, file_name)
            
            if match: 
                N1 = int(match.group(1))
                N2 = int(match.group(2))
                return N1, N2
            
            else: 
                raise ValueError(f"Invalid text format: {file_name}, expected {self.TEMPLATE} ")
        
        # Maps part_N1_N2.txt as (N1, N2) -> (file_path, (from, to))
        file_map: Dict[FileKey, Tuple[str, Interval]] = {}
        
        prev_idx = 0
        
        # Sort the files by the indexes
        file_list = sorted(os.listdir(self._data_dir), key=get_file_key)
        
        # Iterate over the files and build the mapping
        self._logger.info(mess=f"Building file map...")
        
        progress_bar = tqdm(file_list, desc="Processing files")

        # Iterate over the files and build the mapping
        for file_name in progress_bar:
            
            # Check if the file name matches the pattern
            if file_name.endswith('.txt') and re.match(self.PATTERN, file_name):
                
                progress_bar.set_description(f"Processing {file_name}")
            
                N1, N2 = get_file_key(file_name)
                
                file_path  = os.path.join(self._data_dir, file_name)
                file_lines = get_file_lines_count(file_path=file_path)
                
                # NOTE: The left index is inclusive, the right index is exclusive to avoid overlaps - i.e. from_ <= x < to
                idx_from = prev_idx
                idx_to   = prev_idx + file_lines
                
                # Create a new entry in the mapping with the file path and the interval of indexes
                file_map[(N1, N2)] = (file_path, (idx_from, idx_to))
                
                prev_idx = idx_to
                
        return file_map
    
    # --- PROPERTIES ---
    
    @property
    def has_filter(self) -> bool: return self._has_filter
    ''' Returns True if a filter is applied '''
    
    @property
    def tot_items(self) -> int:
        ''' Returns the total number of items in the dataset '''
        
        _, (_, tot_items) = list(self._file_map.values())[-1] 
        return tot_items    
    
    # --- MAGIC METHODS ---
        
    def __str__(self)  -> str:
        ''' String representation of the loader, including the number of files and items, and the filtering status '''
        
        return  (
            f'OAGKXLoader[path: {self._data_dir}; ' +\
            f'files: {len(self._file_map)};  '+\
            (f'filtered items: {len(self)}; ' if self.has_filter else '') +\
            f'total items: {self.tot_items}]'
        )
    
    def __repr__(self) -> str: return str(self)
    
    def __len__ (self) -> int: return len(self._idx_mapping) if self.has_filter else self.tot_items
    ''' The dataset length is the total number of items if no filtering is applied, otherwise is the number of filtered items '''
    
    def __iter__(self) -> Iterable[OAGKXItem]:
        ''' Iterate over the items in the dataset '''
        
        for idx in range(len(self)): yield self[idx]
    
    def __getitem__(self, idx: int) -> OAGKXItem: 
        ''' 
        Load the item corresponding to the global index 
        
        :param idx: Global index of the item
        :type idx: int
        :return: Item at the index
        :rtype: OAGKXItem
        '''
        
        # Use the mapping if a filter is applied
        if self.has_filter:
            try:             idx = self._idx_mapping[idx]
            except KeyError: raise ValueError(f"Index {idx} out of bounds")
        
        # Find the chunk key corresponding to the index and load the chunk
        key = self._find_chunk_key(idx)
        
        # Load the chunk
        self._load_chunk(key)
        
        # Get the left relative index in the chunk
        loaded_from, _ = self._loaded_interval
        json_line: str = self._loaded_lines[idx - loaded_from]
        json_item: Dict[str, Any] = json.loads(json_line)
        
        return OAGKXItem.from_json(json_item=json_item)
    
    # --- CHUNK LOADING ---
    
    def _load_chunk(self, key: FileKey):
        ''' 
        Loads the chunk corresponding to the key.
        If already cached, does nothing.
        
        :param key: Key of the chunk to load
        :type key: FileKey
        '''
        
        # If the chunk is already loaded, do nothing
        if key == self._loaded_idx: return
        
        # Otherwise, load the chunk
        file_path, interval = self._file_map[key]
        
        self._loaded_idx      = key
        self._loaded_interval = interval
        self._loaded_lines    = read_file_lines(file_path)
    
    def _find_chunk_key(self, idx: int) -> FileKey:
        ''' 
        Find the chunk key corresponding to the global index 
        The operation is constant in the case the index lies in the cached chunk,
        otherwise it is linear in the number of files.
        '''
        
        def check_interval(idx: int, interval: Interval) -> bool:
            ''' Checks if the index is in the interval '''
            from_, to = interval
            return from_ <= idx < to
        
        # If the index is in the loaded chunk, return the loaded key
        if check_interval(idx, self._loaded_interval): return self._loaded_idx
        
        # Otherwise, iterate over the files to find the key
        for key, (_, interval) in self._file_map.items():
            if check_interval(idx, interval): return key
        
        raise ValueError(f"Index {idx} out of bounds")
    
    # --- FILTERING ---
    
    def reset_filtering(self):
        ''' Reset the filtering '''
        
        self._idx_mapping : IdxMapping = {}; 
        self._has_filter  : bool       = False
    
    def apply_filter(self, filter: Callable[[OAGKXItem], bool]):
        ''' 
        Apply a filter to the dataset.
        
        The filtering is achieved by a remapping function that maps the global index to the filtered index.
        
        :param filter: Filter function that takes an OAGKXItem and returns if the item should be included or not.
        :type filter: Callable[[OAGKXItem], bool]
        '''
        
        self.reset_filtering()
        
        self._logger.info(mess=f"Applying filter...")
        
        curr_idx    : int = 0
        new_mapping : IdxMapping = {}

        for idx in tqdm(range(len(self))):
            
            if filter(self[idx]):
                
                new_mapping[curr_idx] = idx
                curr_idx += 1
        
        if len(new_mapping) == 0:
            self._logger.warn(mess="Filter applied. No items left")
        
        self._logger.info(mess=f"Filter applied. {len(self)} items left")
        self._has_filter  = True
        self._idx_mapping = new_mapping
    
    # --- SAVING ---

    def save_tsv(self, file_path: str) -> OAGKXTSVDataset:
        ''' 
        Save the dataset as a TSV dataset
        
        :param file_path: Path to the TSV file
        :type file_path: str
        :return: OAGKXTSVDataset object
        :rtype: OAGKXTSVDataset
        '''
        
        HEADER = OAGKXItem.KEYS
        
        FILE_ARGS = {
            'newline': '',
            'encoding': 'utf-8'
        }
        
        WRITER_ARGS = {
            'delimiter': '\t',
            'quotechar': '"',
            'quoting': csv.QUOTE_ALL
        }
        
        with open(file_path, 'w', **FILE_ARGS) as tsv_file:
            
            # Create TSV file
            writer = csv.writer(tsv_file, **WRITER_ARGS)
            writer.writerow(HEADER)
            
            self._logger.info(mess=f"Dumping dataset to {file_path}")
            
            # Write each item as a new line
            for el in tqdm(self): 
                values = list(el.item.values())
                writer.writerow(v for v in values)
        
        return OAGKXTSVDataset(file_path=file_path)


class OAGKXTSVDataset(_OAGKXDataset):
    ''' 
    Class to deal with OAGKXTS dataset transposed TSV format, suitable for lazy loading.
    It implements STUBS to HuggingFace Dataset API (see https://huggingface.co/docs/datasets/index)
    '''
    
    ENCODING  = 'utf-8'
    DELIMITER = '\t'
    
    def __init__(self, file_path: str):
        '''
        Initialize the dataset from the TSV file
        
        :param file_path: Path to the TSV file containing the OACKXTSV dataset
        :type file_path: str
        '''
        
        self._dataset: Dataset = load_dataset(
            'csv', 
            data_files = file_path, 
            delimiter  = self.DELIMITER, 
            encoding   = self.ENCODING
        )['train']
        
    # --- MAGIC METHODS ---
    
    def __len__    (self)      -> int       : return len(self._dataset)
    def __getitem__(self, idx) -> OAGKXItem : return OAGKXItem.from_json(json_item=self._dataset[idx])
    