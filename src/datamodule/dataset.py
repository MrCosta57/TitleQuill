"""

This module contains the implementation of the OAGKX dataset and related functions.

The OAGKX dataset represents a collection of research papers, where each paper is represented by its title, abstract, and keywords. The dataset provides various methods to extract information from the papers, such as finding keywords in the abstract, counting the number of words in the title and abstract, and more.

The module also includes functions for loading and preprocessing the dataset, as well as applying tokenization for sequence-to-sequence models.

Classes:
- OAGKXItem: Represents an item in the OAGKX dataset, with properties and methods to extract information from the item.

Functions:
- filter_no_keywords: Filters the dataset based on the presence of keywords.
- filter_on_stats: Filters the dataset based on various statistics of the items.
- load_oagkx_dataset: Loads the OAGKX dataset from JSON files, with optional filtering.
- apply_tokenization: Performs tokenization on inputs and labels using a tokenizer.
- custom_collate_seq2seq: Custom collate function for sequence-to-sequence models.

"""

import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Counter, List, Set, Tuple, Dict

from nltk.corpus import stopwords
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizer,
)
from src.utils.general_utils import split_keywords_by_comma


@dataclass
class OAGKXItem:
    """
    Dataclass to represent an item in the OAGKX dataset - i.e. a line in the dataset file
    """

    title: str
    """ Title of the paper """

    abstract: str
    """ Abstract of the paper """

    keywords: Set[str]
    """ Keywords associated with the paper """

    _KEYWORDS_DELIMITER = " , "
    """ Delimiter used to separate keywords in the dataset file """

    _SENTENCE_DELIMITERS = r"[.!?]"
    """ Delimiters used to split the abstract into sentences """

    def __str__(self) -> str:
        """ Returns a string representation of the OAGKXItem object """

        return f"Title: {self.title}\n\nAbstract: {self.abstract}\n\nKeywords: {self.keywords}"

    def __repr__(self) -> str:
        """ Returns a string representation of the OAGKXItem object """

        return str(self)

    @classmethod
    def from_data(cls, title: str, abstract: str, keywords_str: str) -> "OAGKXItem":
        """ Parses a line from the dataset file and returns an OAGKXItem object """

        # Extract keywords
        keywords = set(split_keywords_by_comma(keywords_str))

        return OAGKXItem(
            title=title,
            abstract=abstract,
            keywords=keywords,
        )

    @classmethod
    def from_json(cls, json_item: Dict[str, str]) -> "OAGKXItem":
        """Parses a line from the dataset file and returns an OAGKXItem object"""

        # Extract title and abstract
        title = json_item["title"]
        abstract = json_item["abstract"]
        keywords_str = json_item["keywords"]

        return OAGKXItem.from_data(
            title=title, abstract=abstract, keywords_str=keywords_str
        )

    @property
    def keywords_str(self) -> str:
        """Returns the keywords as a string"""

        return OAGKXItem._KEYWORDS_DELIMITER.join(self.keywords)

    @property
    def keywords_in_abstract(self) -> Set[str]:
        """Returns the set of keywords that appear in the abstract"""

        return set([kw for kw in self.keywords if kw in self.abstract])

    @property
    def keywords_in_abstract_prc(self) -> float:
        """Returns the percentage of keywords that appear in the abstract"""

        return len(self.keywords_in_abstract) / len(self.keywords)

    @property
    def abstract_first_sentence(self) -> str:
        """Returns the first sentence of the abstract"""

        return re.split(OAGKXItem._SENTENCE_DELIMITERS, self.abstract)[0]

    @property
    def sentence_with_more_keywords(self) -> Tuple[str, int]:
        """ 
        Returns the sentence with the most keywords and the number of keywords in that sentence 
        
        :return: Tuple containing the sentence with the most keywords and the number of keywords in that sentence
        """

        # Find the sentence with the most keywords
        sentence = max(
            re.split(OAGKXItem._SENTENCE_DELIMITERS, self.abstract),
            key=lambda sentence: len([kw for kw in self.keywords if kw in sentence]),
        )

        # Return the sentence and the number of keywords in that sentence
        return sentence, len([kw for kw in self.keywords if kw in sentence])

    @property
    def title_word_count(self) -> int:
        """Returns the number of words in the title"""

        return len(re.findall(r"\w+", self.title))

    @property
    def abstract_word_count(self) -> int:
        """Returns the number of words in the title"""

        return len(re.findall(r"\w+", self.abstract))

    def get_most_frequent_words(self, min_freq: int = 3) -> Dict[str, int]:
        """Returns most frequent words (stop words excluded) in the abstract with frequency >= min_freq"""

        # Get the set of stop words
        stop_words = set(stopwords.words("english"))

        # Extract words from the abstract
        words = re.findall(r"\w+", self.abstract)

        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Count the frequency of each word
        word_freq = Counter(filtered_words)
        filtered_word_freq = {
            word: freq for word, freq in word_freq.items() if freq >= min_freq
        }

        # Return the k most frequent words
        return filtered_word_freq

    def get_abstract_tokens(self, tokenizer: PreTrainedTokenizer) -> int:
        """ Returns the tokens of the abstract using the provided tokenizer """

        return len(tokenizer(self.abstract)["input_ids"])  # type: ignore

# --- FILTER FUNCTION ---


def filter_no_keywords(batch: Dict[str, List[str]]) -> List[bool]:
    """ Filter function to remove samples with no keywords """

    return [bool(re.match(r"\w+", elem)) for elem in batch["keywords"]]


def filter_on_stats(batch: Dict[str, List[str]]) -> List[bool]:
    """ Filter function to remove samples based on statistics """

    def filter_fn_aux(sample_triple):

        # Extract title, abstract, and keywords
        title, abstract, keywords = sample_triple
        item = OAGKXItem.from_data(title, abstract, keywords)
        
        # Extract stats
        abstract_words = item.abstract_word_count
        title_length   = item.title_word_count
        keywords_count = len(item.keywords)
        
        return (
            250 <= abstract_words <= 540 and\
             10 <= title_length   <= 25  and\
              3 <= keywords_count <= 5
        )

    # Apply filter function to each sample
    return [
        filter_fn_aux(sample_triple)
        for sample_triple in zip(batch["title"], batch["abstract"], batch["keywords"])
    ]

# --- LOADING DATASET ---

def load_oagkx_dataset(
    data_dir: str,
    split_size: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    first_n_files: int = -1,
    filter_fn: Callable[[Dict[str, List[str]]], List[bool]] | None = None,
    verbose: bool = True,
) -> DatasetDict:
    """
    Load OAGKX dataset from jsonl files with filtering
    
    :param data_dir: Path to the directory containing the dataset files
    :param split_size: Tuple containing the train, validation, and test split sizes
    :param first_n_files: Number of files to load from the dataset directory
    :param filter_fn: Function to filter the dataset
    :param verbose: Flag to print progress information
    """

    def train_test_split(
        dataset, test_size: float
    ) -> Tuple[DatasetDict, DatasetDict | None]:
        """
        Split dataset into train and test sets handling the case of no test set
        
        :param dataset: Dataset to split
        :param test_size: Size of the test set
        """

        # Split the dataset into train and test sets
        if test_size > 0:
            dataset_split = dataset.train_test_split(test_size=test_size)
            return dataset_split["train"], dataset_split["test"]
        
        # In the case of no test set, return None
        else:
            return dataset, None

    train_size, val_size, test_size = split_size

    # Sanity checks
    assert train_size > 0, f"Train size must be greater than 0. Got {train_size}."
    assert all(
        [0 <= x <= 1 for x in split_size]
    ), f"Split sizes must be in [0, 1]. Got {split_size}."
    assert sum(split_size) - 1 < 1e-6, f"Split sizes must sum to 1. Got {split_size}."

    print_fn = print if verbose else lambda x: None

    # Select filtered directory and files
    filtered_dir = os.path.join(
        data_dir,
        f"filtered_{'first_'+str(first_n_files) if first_n_files!=-1 else 'full'}",
    )

    # In the case of a filtered dataset, load it
    if os.path.exists(filtered_dir):
        print_fn("Loading filtered dataset ...")
        dataset = load_from_disk(filtered_dir)
        print_fn(f"Dataset loaded with {len(dataset)} samples.")

    # Otherwise, load the dataset and apply the filter function
    else:

        print_fn(f"Loading dataset from {data_dir} ...")

        data_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        data_files.sort()

        if first_n_files != -1:
            print_fn(f"Using just {first_n_files} files")
            data_files = data_files[:first_n_files]
        dataset = load_dataset(
            "json", data_files=data_files, split="train", streaming=False
        )

        print_fn(f"Dataset loaded with {len(dataset)} samples.")  # type: ignore

        # Apply filter function
        if filter_fn is not None:

            print_fn("Applying filter function ...")
            dataset = dataset.filter(filter_fn, batched=True)

            # Save filtered dataset to disk
            dataset.save_to_disk(filtered_dir)  # type: ignore
            print_fn(f"Dataset saved to {filtered_dir} with {len(dataset)} samples.")  # type: ignore

    # Apply split
    train_val, test = train_test_split(dataset=dataset, test_size=test_size)
    train, val = train_test_split(
        dataset=train_val, test_size=val_size / (1 - test_size)
    )

    # Wrap the split datasets in a DatasetDict
    # NOTE: Empty splits are excluded
    dataset_dict = DatasetDict(
        {
            split_name: ds
            for split_name, ds in [
                ("train", train),
                ("validation", val),
                ("test", test),
            ]
            if ds is not None
        }
    )

    print_fn("Final dataset splits:")
    print_fn({split_name: len(ds) for split_name, ds in dataset_dict.items()})

    return dataset_dict


def apply_tokenization(
    input_str: List[str],
    label_str: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    tokenizer_input_args: Dict[str, Any] = {},
    tokenizer_label_args: Dict[str, Any] = {},
):
    """
    Perform tokenization on inputs and labels.

    :param input_str: List of input strings
    :param label_str: List of label strings
    :param tokenizer: Tokenizer to use
    :param max_length: Maximum length of the input sequence
    :param tokenizer_input_args: Additional arguments for tokenizing the input
    :param tokenizer_label_args: Additional arguments for tokenizing the label
    """

    # Tokenize inputs and labels
    model_inputs = tokenizer(
        input_str,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        **tokenizer_input_args,
    )

    # Tokenize labels
    label_encodings = tokenizer(
        label_str,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        **tokenizer_label_args,
    )

    # Replace padding tokens with -100
    labels = label_encodings["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100  # type: ignore

    # Add labels to model inputs
    model_inputs["labels"] = labels
    return model_inputs

# --- COLLATE FUNCTIONS ---


def custom_collate_seq2seq(
    batch,
    tokenizer,
    max_length: int,
    input_format: str = "Generate title and keywords: {e}",
    output_format: str = "Title: {t}<sep>Keywords: {k}",
    shuffle: bool = False,
):
    """
    Custom collate function for a combined task

    :param batch: List of dataset items
    :param tokenizer: Tokenizer to use
    :param max_length: Maximum length of the input sequence
    :param input_format: Format string for the input
    :param output_format: Format string for the output
    :param shuffle: Flag to shuffle keywords, default is False
    """

    def shuffle_keywords(keywords: str) -> str:
        """Shuffle keywords in a string."""

        SEP = " , "
        keywords_list = split_keywords_by_comma(keywords)
        random.shuffle(keywords_list)
        return SEP.join(keywords_list)

    shuffle_fn = shuffle_keywords if shuffle else lambda x: x

    # Create input and ground truth strings
    inp_gt = [
        (
            input_format.format(e=item["abstract"]),
            output_format.format(t=item["title"], k=shuffle_fn(item["keywords"])),
        )
        for item in batch
    ]

    # Return 2 tuples (like lists)
    inp, gt = zip(*inp_gt)

    # Apply tokenization
    new_row = apply_tokenization(
        inp,
        gt,
        tokenizer,
        max_length,
    )

    return new_row


def custom_collate_seq2seq_2task(
    batch,
    tokenizer,
    max_length: int,
    input_format1: str = "Generate title: {e}",
    input_format2: str = "Generate keywords: {e}",
    output_format1: str = "{t}",
    output_format2: str = "{k}",
):
    """
    Collate function for a combined task with a couple of inputs and outputs
    """

    # Create input and ground truth strings
    x_y_z_w = [
        (
            input_format1.format(e=item["abstract"]),
            output_format1.format(t=item["title"]),
            input_format2.format(e=item["abstract"]),
            output_format2.format(k=item["keywords"]),
        )
        for item in batch
    ]

    # Extract and combine inputs and ground truths for both tasks
    x, y, z, w = zip(*x_y_z_w)
    inp = x + z
    gt = y + w

    # Apply tokenization
    new_row = apply_tokenization(
        inp,
        gt,
        tokenizer,
        max_length,
    )

    return new_row
