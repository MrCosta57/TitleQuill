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
    val_size: float = 0.1,
    test_size: float = 0.2,
    just_one_file: bool = False,
    filter_fn: Callable[[Dict[str, List[str]]], List[bool]] | None = None,
) -> DatasetDict:
    """Load OAGKX dataset from jsonl files with filtering"""

    print(f"Loading dataset from {data_dir} ...")
    # Load dataset
    data_files = (
        glob.glob(path.join(data_dir, "part_0.jsonl"))
        if just_one_file
        else glob.glob(path.join(data_dir, "*.jsonl"))
    )
    dataset = load_dataset(
        "json", data_files=data_files, split="train", streaming=False
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Apply filter function
    if filter_fn:
        dataset = dataset.filter(filter_fn, batched=True)

    train_test_ds = dataset.train_test_split(test_size=test_size)
    train_val_ds = train_test_ds["train"].train_test_split(test_size=val_size)

    # Wrap the split datasets in a DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_val_ds["train"],
            "validation": train_val_ds["test"],
            "test": train_test_ds["test"],
        }
    )
    print("Final dataset splits:")
    print(dataset_dict)

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
