import glob
from os import path
import torch
import re
from typing import Any, Callable, List, Tuple
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, DataCollatorForSeq2Seq


def filter_no_keywords(batch: Dict[str, List[str]]) -> List[bool]:
    return [bool(re.match(r"\w+", elem)) for elem in batch["keywords"]]


def load_oagkx_dataset(
    data_dir: str,
    split_size: Tuple[int, int, int] = (0.7, 0.15, 0.15),
    just_one_file: bool = False,
    filter_fn: Callable[[Dict[str, List[str]]], List[bool]] | None = None,
    verbose: bool = True
) -> DatasetDict:
    """Load OAGKX dataset from jsonl files with filtering"""
    
    def train_test_split(dataset, test_size: float) -> Tuple[DatasetDict, DatasetDict | None]:
        """ Split dataset into train and test sets handling the case of no test set """
        
        if test_size > 0:
            dataset_split = dataset.train_test_split(test_size=test_size)
            return dataset_split["train"], dataset_split["test"]
        else:
            return dataset, None
    
    train_size, val_size, test_size = split_size
    
    assert train_size > 0,                         f"Train size must be greater than 0. Got {train_size}."
    assert all([0 <= x <= 1 for x in split_size]), f"Split sizes must be in [0, 1]. Got {split_size}."
    assert sum(split_size) - 1 < 1e-6,             f"Split sizes must sum to 1. Got {split_size}."
    
    
    print_fn = print if verbose else lambda x: None

    print_fn(f"Loading dataset from {data_dir} ...")
    # Load dataset
    files = 'part_0.jsonl' if just_one_file else '*.jsonl'
    data_files = glob.glob(path.join(data_dir, files))
    dataset = load_dataset(
        "json", data_files=data_files, split="train", streaming=False
    )
    print_fn(f"Dataset loaded with {len(dataset)} samples.")

    # Apply filter function
    if filter_fn:
        dataset = dataset.filter(filter_fn, batched=True)
    
    # Apply split
    train_val, test = train_test_split(dataset=dataset,   test_size=test_size)
    train,     val  = train_test_split(dataset=train_val, test_size=val_size / (1 - test_size))

    # Wrap the split datasets in a DatasetDict
    # NOTE: We exclude empty splits
    dataset_dict = DatasetDict(
        {
            split_name : ds
            for split_name, ds in [
                ("train",      train),
                ("validation", val),
                ("test",       test),
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
    output_format: str = "Title: {t}.\nKeywords: {k}",
):
    # batch is a list of dataset items
    new_row = [
        apply_tokenization(
            input_format.format(e=item["abstract"]),
            output_format.format(t=item["title"], k=item["keywords"]),
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
