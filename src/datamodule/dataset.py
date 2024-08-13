import glob
from os import path
import torch
import re
from typing import Any, Callable, List
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, DataCollatorForSeq2Seq


def filter_no_keywords(batch: Dict[str, List[str]]) -> List[bool]:
    return [bool(re.match(r"\w+", elem)) for elem in batch["keywords"]]


def load_oagkx_dataset(
    data_dir: str,
    split: str = "train",
    test_size: float = 0.2,
    filter_fn: Callable[[Dict[str, List[str]]], List[bool]] | None = None,
) -> DatasetDict:
    """Load OAG-KX dataset from jsonl files with filtering and streaming support."""

    # Load dataset
    # data_files = glob.glob(path.join(data_dir, "*.jsonl"))
    data_files = glob.glob(path.join(data_dir, "part_0.jsonl"))
    dataset = load_dataset("json", data_files=data_files, split=split, streaming=False)
    # Apply filter function
    if filter_fn:
        dataset = dataset.filter(filter_fn, batched=True)

    dataset_split = dataset.train_test_split(test_size=test_size)
    # Wrap the split datasets in a DatasetDict
    dataset_dict = DatasetDict(
        {"train": dataset_split["train"], "test": dataset_split["test"]}
    )

    return dataset_dict


def apply_tokenization(
    input_str: str,
    label_str: str,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_input_args: Dict[str, Any] = {},
    tokenizer_label_args: Dict[str, Any] = {},
):
    """Perform tokenization on inputs and labels."""

    # Check if inputs and labels have the same length
    # assert len(inputs) == len(labels), "Inputs and labels must have the same length"

    # Tokenize inputs and labels
    model_inputs = tokenizer(input_str, **tokenizer_input_args)
    label_encodings = tokenizer(label_str, **tokenizer_label_args)

    # Add labels to model inputs
    model_inputs["labels"] = label_encodings["input_ids"]
    return model_inputs


def custom_collate_seq2seq(
    batch,
    tokenizer,
    model,
    input_format: str = "Abstract: {e}",
    output_format: str = "Title: {t}\nKeywords: {k}",
):
    # batch is a list of dataset items
    new_row = [
        apply_tokenization(
            input_format.format(e=item["abstract"]),
            output_format.format(t=item["title"], k=item["keywords"]),
            tokenizer,
        )
        for item in batch
    ]
    default_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, return_tensors="pt"
    )
    return default_collator(new_row)


def custom_collate_seq2seq_2task(
    batch,
    tokenizer,
    model,
    input_format1: str = "Title task. Abstract: {e}",
    input_format2: str = "Keywords task. Abstract: {e}",
    output_format1: str = "Title: {t}",
    output_format2: str = "Keywords: {k}",
):
    # batch is a list of dataset items
    new_row = [
        (
            apply_tokenization(
                input_format1.format(e=item["abstract"]),
                output_format1.format(t=item["title"]),
                tokenizer,
            )
            if i % 2 == 0
            else apply_tokenization(
                input_format2.format(e=item["abstract"]),
                output_format2.format(k=item["keywords"]),
                tokenizer,
            )
        )
        for i in range(2)
        for item in batch
    ]
    default_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, return_tensors="pt"
    )
    return default_collator(new_row)
