from __future__ import annotations

import sys
sys.path.append('.')

import re
from os import path
from dataclasses import dataclass
from typing import Counter, Dict, Set, Tuple

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from nltk.corpus import stopwords
from transformers import PreTrainedTokenizer


@dataclass
class OAGKXItemStats:
    ''' Dataclass to represent an item in the OAGKX dataset - i.e. a line in the dataset file '''
    
    title    : str
    ''' Title of the paper '''
    
    abstract : str      
    ''' Abstract of the paper '''
    
    keywords : Set[str]
    ''' Keywords associated with the paper '''
    
    _KEYWORDS_DELIMITER  = ' , '
    _SENTENCE_DELIMITERS = r'[.!?]'
    _STOPWORDS           = set(stopwords.words('english'))
    
    def __str__ (self) -> str: return  f'Title: {self.title}\n\nAbstract: {self.abstract}\n\nKeywords: {self.keywords}'
    def __repr__(self) -> str: return str(self)
    
    @classmethod
    def from_json(
        cls, 
        json_item : Dict[str, str]
    ) -> 'OAGKXItemStats':
        ''' Parses a line from the dataset file and returns an OAGKXItem object '''
        
        # Extract title and abstract
        title        = json_item['title']
        abstract     = json_item['abstract']
        keywords_str = json_item['keywords']
        
        # Extract keywords
        keywords = set([keyword.strip() for keyword in keywords_str.split(OAGKXItemStats._KEYWORDS_DELIMITER)])
        
        return OAGKXItemStats(
            title    = title,
            abstract = abstract,
            keywords = keywords,
        )

    @property
    def keywords_in_abstract(self) -> Set[str]:
        ''' Returns the set of keywords that appear in the abstract '''
        return set([kw for kw in self.keywords if kw in self.abstract])
    
    @property
    def keywords_in_abstract_prc(self) -> float:
        ''' Returns the percentage of keywords that appear in the abstract '''
        return len(self.keywords_in_abstract) / len(self.keywords)
    
    @property
    def abstract_first_sentence(self) -> str:
        ''' Returns the first sentence of the abstract '''
        
        return re.split(OAGKXItemStats._SENTENCE_DELIMITERS, self.abstract)[0]
    
    @property
    def sentence_with_more_keywords(self) -> Tuple[str, int]:
        
        # Find the sentence with the most keywords
        sentence = max(
            re.split(OAGKXItemStats._SENTENCE_DELIMITERS, self.abstract),
            key = lambda sentence: len([kw for kw in self.keywords if kw in sentence])
        )
        
        return sentence, len([kw for kw in self.keywords if kw in sentence])
    
    @property
    def title_word_count(self) -> int:
        ''' Returns the number of words in the title '''
        return len(re.findall(r'\w+', self.title))
    
    def get_most_frequent_words(self, min_freq : int = 3)-> Dict[str, int]:
        ''' Returns the k most frequent words in the abstract '''
        
        # Extract words from the abstract
        words = re.findall(r'\w+', self.abstract)
        
        filtered_words = [word for word in words if word.lower() not in self._STOPWORDS]

        # Count the frequency of each word
        word_freq = Counter(filtered_words)
        filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}

        # Return the k most frequent words
        return filtered_word_freq
    
    def get_abstract_tokens(self, tokenizer: PreTrainedTokenizer):
        
        return len(tokenizer(self.abstract)['input_ids'])

def line_plot(
    data      : Dict, 
    title     : str    = "", 
    xlabel    : str    = "",
    ylabel    : str    = "", 
    save_path : str    = "plot.png"
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
    
    # Sort data by key
    data = dict(sorted(data.items()))

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
    print(f'Saving plot to {save_path}')
    fig.savefig(save_path, bbox_inches='tight')


def plot_stats(
    stats: Dict[str, Dict[int, str]],
    out_dir: str
):
    
    def dict_to_cumulative(dict_: Dict[str, int]) -> Dict[str, int]:
        ''' Transpose a count dictionary into a cumulative-count dictionary '''
        
        sorted_keys     = sorted(dict_.keys())
        cumulative_sum  = 0
        cumulative_dict = {k: 0 for k in sorted_keys}
        
        # Iterate over the sorted keys and calculate the cumulative sum
        for key in sorted_keys:
            cumulative_sum       += dict_[key]
            cumulative_dict[key]  = cumulative_sum
        
        return cumulative_dict
    
    for cumulative in [False, True]:
    
        for name, data in stats.items():
            
            if cumulative:
                data = dict_to_cumulative(data)
                name += '_cumulative'
            
            line_plot(
                data      = data,
                title     = f'{name}',
                xlabel    = 'Values',
                ylabel    = 'Count',
                save_path = path.join(out_dir, f'{name}.png')
            )

