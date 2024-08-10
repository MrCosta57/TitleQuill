import glob
from typing import List
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, List, Union
from datasets import load_dataset


def load_oagkx_dataset(data_dir: str, split: str = "train", streaming: bool = False):
    data_files = glob.glob(f"{data_dir}/*.jsonl")
    dataset = load_dataset(
        "json", data_files=data_files, split=split, streaming=streaming
    )
    return dataset


class DataCollatorOAGKX:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str] = True,
        max_input_length: int = 512,
        max_target_length: int = 128,
        truncation: bool = True,
        input_format: str = "Abstract: {e}",
        output_format: str = "Title: {t}\nKeywords: {k}",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_format = input_format
        self.output_format = output_format

    def __call__(self, batch: List[Dict[str, str]]):

        inputs = [self.input_format.format(e=example["abstract"]) for example in batch]
        targets = [
            self.output_format.format(t=example["title"], k=example["keywords"])
            for example in batch
        ]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            targets,
            max_length=self.max_target_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
