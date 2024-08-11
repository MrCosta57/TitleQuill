import glob
import re
from typing import Callable, List
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Union
from datasets import load_dataset


def load_oagkx_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = 512,
    max_target_length: int = 128,
    truncation: bool = True,
    split: str = "train",
    filter_fun: Callable[[Dict[str, List[str]]], List[bool]] = lambda batch: [
        bool(re.match(r"\w+", elem)) for elem in batch["keywords"]
    ],
    input_format: str = "Abstract: {e}",
    output_format: str = "Title: {t}\nKeywords: {k}",
):
    data_files = glob.glob(f"{data_dir}/*.jsonl")
    dataset = load_dataset("json", data_files=data_files, split=split, streaming=True)
    if filter_fun:
        dataset = dataset.filter(filter_fun)

    if input_format != "" and output_format != "":

        def process_data(examples):
            inputs = input_format.format(e=examples["abstract"])
            targets = output_format.format(t=examples["title"], k=examples["keywords"])
            model_inputs = tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=truncation,
            )
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=truncation
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = dataset.map(
            process_data,
        )
        dataset = dataset.remove_columns(["abstract", "title", "keywords"])

    return dataset
