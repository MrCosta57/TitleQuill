from dataclasses import dataclass
import glob
import os
import random
import re
from nltk.corpus import stopwords
from typing import Any, Callable, Counter, List, Set, Tuple
from typing import Dict, List
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)
from utils.general_utils import split_keywords_by_comma


@dataclass
class OAGKXItem:
    """Dataclass to represent an item in the OAGKX dataset - i.e. a line in the dataset file"""

    title: str
    """ Title of the paper """

    abstract: str
    """ Abstract of the paper """

    keywords: Set[str]
    """ Keywords associated with the paper """

    _KEYWORDS_DELIMITER = " , "
    _SENTENCE_DELIMITERS = r"[.!?]"

    def __str__(self) -> str:
        return f"Title: {self.title}\n\nAbstract: {self.abstract}\n\nKeywords: {self.keywords}"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_data(cls, title: str, abstract: str, keywords_str: str) -> "OAGKXItem":
        """Parses a line from the dataset file and returns an OAGKXItem object"""

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
        # Find the sentence with the most keywords
        sentence = max(
            re.split(OAGKXItem._SENTENCE_DELIMITERS, self.abstract),
            key=lambda sentence: len([kw for kw in self.keywords if kw in sentence]),
        )
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

    def get_abstract_tokens(self, tokenizer: PreTrainedTokenizer):
        return len(tokenizer(self.abstract)["input_ids"])  # type: ignore


def filter_no_keywords(batch: Dict[str, List[str]]) -> List[bool]:
    return [bool(re.match(r"\w+", elem)) for elem in batch["keywords"]]


def filter_on_stats(batch: Dict[str, List[str]]) -> List[bool]:

    def filter_fn_aux(sample_triple):
        title, abstract, keywords = sample_triple
        item = OAGKXItem.from_data(title, abstract, keywords)
        abstract_words = item.abstract_word_count
        title_length = item.title_word_count
        keywords_count = len(item.keywords)
        # print(abstract_tokens)
        return (
            250 <= abstract_words <= 540
            and 10 <= title_length <= 20
            and 4 <= keywords_count <= 5
        )

    return [
        filter_fn_aux(sample_triple)
        for sample_triple in zip(batch["title"], batch["abstract"], batch["keywords"])
    ]


def load_oagkx_dataset(
    data_dir: str,
    split_size: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    just_one_file: bool = False,
    filter_fn: Callable[[Dict[str, List[str]]], List[bool]] | None = None,
    verbose: bool = True,
) -> DatasetDict:
    """Load OAGKX dataset from jsonl files with filtering"""

    def train_test_split(
        dataset, test_size: float
    ) -> Tuple[DatasetDict, DatasetDict | None]:
        """Split dataset into train and test sets handling the case of no test set"""

        if test_size > 0:
            dataset_split = dataset.train_test_split(test_size=test_size)
            return dataset_split["train"], dataset_split["test"]
        else:
            return dataset, None

    train_size, val_size, test_size = split_size

    assert train_size > 0, f"Train size must be greater than 0. Got {train_size}."
    assert all(
        [0 <= x <= 1 for x in split_size]
    ), f"Split sizes must be in [0, 1]. Got {split_size}."
    assert sum(split_size) - 1 < 1e-6, f"Split sizes must sum to 1. Got {split_size}."

    print_fn = print if verbose else lambda x: None

    if os.path.exists(os.path.join(data_dir, "filtered")):
        print_fn("Loading filtered dataset ...")
        dataset = load_from_disk(os.path.join(data_dir, "filtered"))
        print_fn(f"Dataset loaded with {len(dataset)} samples.")
    else:
        print_fn(f"Loading dataset from {data_dir} ...")
        files = "part_0.jsonl" if just_one_file else "*.jsonl"
        data_files = glob.glob(os.path.join(data_dir, files))
        dataset = load_dataset(
            "json", data_files=data_files, split="train", streaming=False
        )
        print_fn(f"Dataset loaded with {len(dataset)} samples.")  # type: ignore
        # Apply filter function
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn, batched=True)
            dataset.save_to_disk(os.path.join(data_dir, "filtered"))

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
    input_str: str,
    label_str: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    tokenizer_input_args: Dict[str, Any] = {},
    tokenizer_label_args: Dict[str, Any] = {},
):
    """Perform tokenization on inputs and labels."""
    # Tokenize inputs and labels
    model_inputs = tokenizer(
        input_str,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        **tokenizer_input_args,
    )
    label_encodings = tokenizer(
        label_str,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        **tokenizer_label_args,
    )

    # Add labels to model inputs
    model_inputs["labels"] = label_encodings["input_ids"]
    return model_inputs


def custom_collate_seq2seq(
    batch,
    tokenizer,
    model,
    max_length: int,
    input_format: str = "Generate title and keywords: {e}",
    output_format: str = "Title: {t}<sep>Keywords: {k}",
    shuffle: bool = False,
):
    # batch is a list of dataset items
    def shuffle_keywords(keywords: str) -> str:
        SEP = " , "
        """Shuffle keywords in a string."""
        keywords_list = split_keywords_by_comma(keywords)
        random.shuffle(keywords_list)
        return SEP.join(keywords_list)

    shuffle_fn = shuffle_keywords if shuffle else lambda x: x
    new_row = [
        apply_tokenization(
            input_format.format(e=item["abstract"]),
            output_format.format(t=item["title"], k=shuffle_fn(item["keywords"])),
            tokenizer,
            max_length,
        )
        for item in batch
    ]
    default_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return default_collator(new_row)


def custom_collate_seq2seq_2task(
    batch,
    tokenizer,
    model,
    max_length: int,
    input_format1: str = "Generate title: {e}",
    input_format2: str = "Generate keywords: {e}",
    output_format1: str = "Title: {t}",
    output_format2: str = "Keywords: {k}",
):

    # batch is a list of dataset items
    new_row = [
        apply_tokenization(
            input_format1.format(e=item["abstract"]),
            output_format1.format(t=item["title"]),
            tokenizer,
            max_length,
        )
        for item in batch
    ] + [
        apply_tokenization(
            input_format2.format(e=item["abstract"]),
            output_format2.format(k=item["keywords"]),
            tokenizer,
            max_length,
        )
        for item in batch
    ]
    default_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return default_collator(new_row)
