import json
import subprocess
from typing import List


def read_file_lines(file_path: str) -> List[str]:
        
    lines = []
    
    with open(file_path, 'r') as file:
        for line in file: lines.append(line.strip())
            
    return lines

def load_json(file_path: str) -> dict:
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    return data

def get_file_lines_count(file_path: str) -> int:
    
    # Uncomment in Windows
    # with open(file_path, 'r') as file:
    #     count = sum(1 for _ in file)
    
    # Uncomment in Linux
    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
    output = result.stdout.strip()
    count = int(output.split()[0])
    
    return count