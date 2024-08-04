import json
import os
import platform
import subprocess
from typing import Dict, List
    
from titlequill.utils.logger import Logger, SilentLogger

def make_dir(path: str, logger: Logger = SilentLogger()) -> str:
    
    logger.info(f'Creating directory: {path}')
    os.makedirs(path, exist_ok=True)
    
    return path


def read_file_lines(file_path: str) -> List[str]:
    '''
    Read lines from a file and return them as a list of strings.

    :param file_path: Path to the file to read
    :type file_path: str
    :return: List of lines from the file
    :rtype: List[str]
    '''
    
    lines = []
    
    with open(file_path, 'r') as file:
        for line in file: lines.append(line.strip())
            
    return lines

def load_json(file_path: str) -> Dict:
    '''
    Load JSON data from a file and return it as a dictionary
    
    :param file_path: Path to the JSON file
    :type file_path: str
    :return: JSON data as a dictionary
    :rtype: Dict
    '''
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    return data

def get_file_lines_count(file_path: str) -> int:
    '''
    Get the number of lines in a file
    
    NOTE: The specific implementation depends on the operating system
        as we are able to specify different os-specific commands to speed-up row count.
    
    :param file_path: Path to the file
    :type file_path: str
    :return: Number of lines in the file
    :rtype: int
    '''
    
    # Get the current operating system
    current_os = platform.system()

    # Match case for different operating systems
    match current_os:
        
        case 'Linux':
            
            result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
            output = result.stdout.strip()
            count = int(output.split()[0])
            
        case 'Windows' | 'Darwin':
            
            with open(file_path, 'r') as file:
                count = sum(1 for _ in file)
        
        case _:
            
            raise NotImplementedError(f'Unsupported OS: {current_os}')
    
    return count