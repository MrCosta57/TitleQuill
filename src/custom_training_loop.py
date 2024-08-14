import glob
from os import path
from typing import Any, Callable, Dict, List, Tuple
from functools import partial

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorForSeq2Seq,
)
from src.datamodule.dataset import apply_tokenization, load_oagkx_dataset, filter_no_keywords

# --- TYPE ALIASES ---

Args = Dict[str, Any]  # Example: {'batch_size': 32, 'lr': 0.001, ...}
''' Keyword arguments for a generic function call. '''

DatasetItem = Dict[str, str] # Example: {'abstract': 'in this paper...', 'title': 'let's cure the cancer', ...}
''' A single item in the dataset with keys {`abstract`, `title`, `keywords`} mapped to strings. '''

Batch = Dict[str, Tensor]  # Example: {'input_ids': tensor([[1, 2, 3, ...]]), 'attention_mask': tensor([[1, 1, 1, ...]]), ...}
''' A batch of data with keys mapped to PyTorch tensors. '''

ModelOut = Dict[str, Any]  # Example: {'logits': tensor([[1, 2, 3, ...]]), ...}
''' The output of the model call in the forward step. '''


def collate_function_with_preprocessing(
    items: List[DatasetItem],
    input_labels_fn: Callable[[List[DatasetItem]], Tuple[List[str], List[str]]],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_input_args: Dict[str, Any] = {},
    tokenizer_target_args: Dict[str, Any] = {},
    data_collator: DataCollatorForSeq2Seq | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate function to perform custom preprocessing on the batch before passing it to the standard Seq2Seq collate function.
    
    It leverage on a callback returning the inputs and labels from the dataset items and applies tokenization to them.
    """

    # Get inputs and labels from items
    inputs, labels = input_labels_fn(items)

    # Tokenize inputs and labels
    batch = apply_tokenization(
        input_str=inputs,
        label_str=labels,
        tokenizer=tokenizer,
        tokenizer_input_args=tokenizer_input_args,
        tokenizer_label_args=tokenizer_target_args,
    )
    
    # TODO : which version is more efficient ? 
    # Postprocess the batch to a list of dictionaries
    # batch_dict_list = [{k: v[i] for k, v in batch.items()} for i in range(len(batch["input_ids"]))]

    # Initialize DataCollatorForSeq2Seq only if not provided
    if data_collator is None:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # Apply the data collator
    return data_collator(batch_dict_list)


def get_collate_function_with_preprocessing(
    input_labels_fn: Callable[[List[DatasetItem]], Tuple[List[str], List[str]]],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_input_args: Dict[str, Any] = {},
    tokenizer_target_args: Dict[str, Any] = {},
) -> Callable[[List[DatasetItem]], Batch]:
    """
    Utility function to fix the input arguments of the collate function.
    """

    return partial(
        collate_function_with_preprocessing,
        input_labels_fn=input_labels_fn,
        tokenizer=tokenizer,
        tokenizer_input_args=tokenizer_input_args,
        tokenizer_target_args=tokenizer_target_args,
    )


class CustomTrainingLoop:
    """
    Custom training loop for PyTorch models.

    How to build (x, y) couple through collate function
    How to compute loss
    How to log metrics
    """

    def __init__(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        optimizer: Optimizer,
        collate_fn: Callable[[List[DatasetItem]], Batch],
        loss_fn: Callable[[Batch, ModelOut], Tensor],
        log_fn: Callable[[PreTrainedModel, Batch, ModelOut], None] = lambda x: None,
        epochs: int = 100,
        dataloader_args: Dict[str, Any] = {},
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):

        self._model = model
        self._dataset = dataset
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._collate_fn = collate_fn
        self._log_fn = log_fn
        self._epochs = epochs
        self._dataloader_args = dataloader_args
        self._device = device

    def train(self):

        dataloader = DataLoader(
            self._dataset, collate_fn=self._collate_fn, **self._dataloader_args
        )

        self._model.to(self._device)
        self._model.train()

        for epoch in range(self._epochs):

            print(f"Epoch {epoch + 1}/{self._epochs}")

            for i, batch in enumerate(dataloader):

                print("Iteration", i + 1)

                outputs = self._model(**batch)

                loss = self._loss_fn(batch=batch, outputs=outputs)

                print(f"Loss: {loss.item()}")

                # Backward pass
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._log_fn(self._model, batch, outputs)