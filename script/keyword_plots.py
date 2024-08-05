
from collections import defaultdict
from typing import Dict, TypeVar
from os import path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm              import tqdm

from titlequill.dataset      import OAGKXRawDataset, OAGKXItem, OAGKXDataset
from titlequill.utils.io_ import make_dir
from titlequill.utils.logger import Logger, SilentLogger
from script.utils.settings   import DATASET_DIR, OUT_DIR

T = TypeVar('T')

def dict_to_comulative(dict_: Dict[T, int]) -> Dict[T, int]:
    ''' Transpose a count dictionary into a cumulative-count dictionary '''
    
    sorted_keys     = sorted(dict_.keys())
    cumulative_sum  = 0
    cumulative_dict = {k: 0 for k in sorted_keys}
    
    # Iterate over the sorted keys and calculate the cumulative sum
    # NOTE : We do it in reverse order to have the cumulative sum at the end 
    # TODO : Is this the best view to do it?
    for key in sorted_keys[::-1]:
        cumulative_sum       += dict_[key]
        cumulative_dict[key]  = cumulative_sum
    
    return cumulative_dict

# --- PLOTTING FUNCTION ---

# TODO - Can we maybe adopt other plot types? (e.g. bar plot / histogram)
def plot_document_counts(
    data      : Dict, 
    title     : str    = "", 
    xlabel    : str    = "",
    ylabel    : str    = "", 
    save_path : str    = "plot.png",
    logger    : Logger = SilentLogger()
):
    
    # ----------- HYERPARAMETERS ------------
    
    def format(x, pos): return f'{int(x):,}'

    FIGSIZE    = (12, 7)
    STYLE      = 'ggplot'
    AXES_FONT  = 14
    TITLE_FONT = 16
    TICK_SIZE  = 12
    #YPADDING   = 10 ** 6
    
    PLOT_ARGS = {
        'marker'    : 'o',
        'linestyle' : '-',
        'color'     : 'b'
    }
    
    GRID_ARGS = {
        'which'     :'both', 
        'linestyle' : '--', 
        'linewidth' : 0.5
    }
    
    # ----------------------------------------
    
    # Set the style
    plt.style.use(STYLE)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot the data
    x = list(data.keys  ())
    y = list(data.values())
    
    ax.plot(x, y, **PLOT_ARGS)

    # Customize labels and title
    ax.set_xlabel(xlabel, fontsize=AXES_FONT)
    ax.set_ylabel(ylabel, fontsize=AXES_FONT)
    ax.set_title (title,  fontsize=TITLE_FONT)

    # Set y-axis limits to the original scale of the data
    #ax.set_ylim(min(y) - YPADDING, max(y) + YPADDING)
    ax.yaxis.set_major_formatter(FuncFormatter(format))
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)

    # Add grid lines
    ax.grid(True, **GRID_ARGS)

    # Save the plot if a save_path is provided
    logger.info(f'Saving plot to {save_path}')
    fig.savefig(save_path, bbox_inches='tight')
    

class OAGKXStats:
    
    def __init__(self, logger: Logger = SilentLogger()):
        '''
        Initialize the statistics 
            - Keywords count
            - Keywords in abstract count
            - Keywords in abstract percentage
        '''
        
        self._logger : Logger = logger  
        
        self._keywords_count           : Dict[int,   int]   = defaultdict(int)
        self._keywords_in_abstract     : Dict[int,   int]   = defaultdict(int)
        self._keywords_in_abstract_prc : Dict[float, int]   = defaultdict(int)
    
    # --- RUNNING FUNCTIONS ---
    
    def compute(self, dataset: OAGKXDataset):
        ''' 
        Perform the computation of the statistics breaking the process into 3 steps 
            1. Start the computation
            2. Update the statistics with each item
            3. Postprocess the statistics
        '''
        
        self._start()
        
        self._logger.info(f'Processing {len(dataset)} elements...')
        
        for item in tqdm(dataset): self._update(item=item)
        
        self._postprocess()
        
    def _start(self):
        ''' Clear the statistics '''
        
        self._keywords_count           .clear()
        self._keywords_in_abstract     .clear()
        self._keywords_in_abstract_prc .clear()
    
    def _update(self, item: OAGKXItem):
        ''' Update the statistics with a new item '''
        
        keywords_n      = len(item.keywords)
        in_abstract     = item.keywords_in_abstract_count
        in_abstract_prc = in_abstract / keywords_n
        
        self._keywords_count          [keywords_n]      += 1
        self._keywords_in_abstract    [in_abstract]     += 1
        self._keywords_in_abstract_prc[in_abstract_prc] += 1
    
    def _postprocess(self):
        
        # Sort by key
        self._keywords_count           = {k: self._keywords_count          [k] for k in sorted(self._keywords_count)}
        self._keywords_in_abstract     = {k: self._keywords_in_abstract    [k] for k in sorted(self._keywords_in_abstract)}
        self._keywords_in_abstract_prc = {k: self._keywords_in_abstract_prc[k] for k in sorted(self._keywords_in_abstract_prc)}
    
    def plot(self, plot_dir : str = '.'):
        
        Y_LABEL = "Papers count"
        EXT     = 'png'
        
        make_dir(path=plot_dir, logger=self._logger)
        
        for cumulative in [False, True]:
        
            for data, title, xlabel, file_name in [
                [self._keywords_count,           'Keywords count',                  "Keyword count",                  'keyword_count'          ],
                [self._keywords_in_abstract,     'Keywords count in abstract',      "Keyword in abstract count",      'keyword_in_abstract'    ],
                [self._keywords_in_abstract_prc, 'Keywords percentage in abstract', "Keyword in abstract percentage", 'keyword_in_abstract_prc'],
            ]:
                
                if cumulative:
                    
                    data = dict_to_comulative(dict_=data)
                    title = f'Cumulative {title.lower()}'
                    file_name = f'{file_name}_cumulative'
                
                save_path = path.join(plot_dir, f'{file_name}.{EXT}')
        
                plot_document_counts(
                    data      = data, 
                    title     = title, 
                    xlabel    = xlabel,
                    ylabel    = Y_LABEL, 
                    save_path = save_path,
                    logger    = self._logger
                )
                
def main():
    
    PLOT_DIR = path.join(OUT_DIR, "keywords_plots")
    
    logger  = Logger()
    dataset = OAGKXRawDataset(DATASET_DIR, logger=logger)
    stats   = OAGKXStats(logger=logger)
    
    stats.compute(dataset=dataset)
    stats.plot(plot_dir=PLOT_DIR)

if __name__ == '__main__': main()
