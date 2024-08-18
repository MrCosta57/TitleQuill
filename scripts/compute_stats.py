import os
import sys
sys.path.append(os.path.join(__file__, '..', '..'))

import pickle
from os import path
from typing import Counter, Dict, Any

from src.datamodule.stats import OAGKXItemStats, plot_stats
from src.datamodule.dataset import load_oagkx_dataset

MODEL_NAME = 'google/flan-t5-small'
OUT_DIR    = r'figures'
DATA_DIR   = r'data\OAGKX'

def mapping_function(el: Dict[str, str]) -> Dict[str, Any]:
    
    item = OAGKXItemStats.from_json(el)
    
    stats = {
        'abstract_length'     : item.abstract_word_count,
        'title_length'        : item.title_word_count,
        'keywords_count'      : len(item.keywords),
        # 'keywords_in_abstract': len(item.keywords_in_abstract),
    }
    
    return stats

if __name__ == "__main__":
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load dataset
    dataset = load_oagkx_dataset(
        data_dir = DATA_DIR,
        split_size = (1., 0., 0.),
        #just_one_file=True
    )['train']
    
    #dataset = dataset.select(range(100))
    
    # Compute stats
    stats_dataset = dataset.map(
        mapping_function, 
        remove_columns=dataset.column_names
    )
    
    # Extract stats count per column
    stats_count = {
        k : dict(sorted(Counter(stats_dataset[k]).items()))
        for k in stats_dataset.column_names
    }
    
    # Save stats for future use
    with open(path.join(OUT_DIR, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats_count, f)
    
    # Plot stats
    plot_stats(
        stats=stats_count,
        out_dir=OUT_DIR
    )
    