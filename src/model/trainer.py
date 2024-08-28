from datetime import datetime
import os
from typing import Callable, Dict, List, Literal, Optional, Tuple
import loguru
import torch
import wandb
from functools import partial
from transformers.optimization import Adafactor
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import DatasetDict
from utils.general_utils import postprocess_validation_text, split_keywords_by_comma
from utils.evaluator import Evaluator


class Trainer:

    def __init__(
        self,
        model: PreTrainedModel,
        dataset_dict: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        shuffle: bool,
        max_length: int,
        max_new_tokens: int,
        max_epochs: int,
        double_task: bool,
        collate_fn: Callable,
        loss_fn: Callable,
        evaluator: Evaluator,
        log_wandb: bool = True,
        lr: float = 5e-4,
        sep_special_tokens: str = "<sep>",
        train_batch_size: int = 64,
        val_batch_size: int = 128,
        log_interval: int = 100,
    ):

        assert all(
            [
                split_name in dataset_dict.keys()
                for split_name in ["train", "validation", "test"]
            ]
        )

        # Model and Device
        self.device = torch.device(device)
        self.model = model.to(self.device)  # type: ignore
        self.log_wandb = log_wandb

        # Tokenization
        self.tokenizer = tokenizer
        if sep_special_tokens != "":
            self.tokenizer.add_tokens(sep_special_tokens, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        # Dataset and Dataloader
        self.dataset_dict = dataset_dict
        self.train_batch_size = (
            train_batch_size // 2 if double_task else train_batch_size
        )
        self.val_batch_size = val_batch_size // 2 if double_task else val_batch_size
        self.shuffle = shuffle

        # Training loop
        self.max_epochs = max_epochs
        self.log_interval = log_interval

        # Training callbacks
        self.double_task = double_task
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn

        # Optimizer
        self.optimizer = Adafactor(self.model.parameters(), lr=lr, relative_step=False)
        self.evaluator = evaluator

    def train(self):
        # Logging function
        # self.log_fn = lambda x: None
        self.log_fn = print
        """ loguru.logger.add(
            f"logs/training-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
            level="INFO",
        )
        self.log_fn = loguru.logger.info """

        # Initialize metrics
        train_dataloader = DataLoader(
            self.dataset_dict["train"],  # type: ignore - interface compatibility
            batch_size=self.train_batch_size,
            collate_fn=partial(
                self.collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                model=self.model,
            ),
            shuffle=self.shuffle,
        )

        # Set model to training mode
        self.model.train()
        train_global_step = 0
        for epoch in range(self.max_epochs):

            self.log_fn(
                f"Epoch {epoch + 1}/{self.max_epochs} ({round((epoch + 1) / self.max_epochs * 100, 2)}%)"
            )
            if self.log_wandb:
                self.train_pred_table = wandb.Table(
                    columns=["prediction", "ground_truth"]
                )
            loss_batches = []
            for batch_id, batch in enumerate(train_dataloader):

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Forward pass
                outputs = self.model(**batch)
                loss = self.loss_fn(batch, outputs)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Add batch loss
                loss_batches.append(loss.item())
                # Log
                if batch_id % self.log_interval == 0:
                    if self.log_wandb:
                        wandb.log({"train/loss": loss.item()}, step=train_global_step)
                    self.log_fn(
                        f" > Training batch {batch_id+1}/{len(train_dataloader)} ({round(batch_id+1 / len(train_dataloader) * 100, 2)}%) - Loss: {loss.item()}"
                    )
                    self._print_eval_train(batch, outputs, epoch, batch_id)

                train_global_step += 1
                # if batch_id == 100: break

            if self.log_wandb:
                wandb.log(
                    {"train/epoch_loss": sum(loss_batches) / len(loss_batches)},
                    step=epoch,
                )
                wandb.log(
                    {"train/pred_target_table": self.train_pred_table}, step=epoch
                )

            # Perform validation
            self.validation(epoch=epoch)

        self.log_fn("Training completed")

    def validation(self, epoch: int):
        val_dataloader = DataLoader(
            self.dataset_dict["validation"],  # type: ignore - interface compatibility
            batch_size=self.val_batch_size,
            collate_fn=partial(
                self.collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                model=self.model,
            ),
            shuffle=False,
        )
        self._common_eval(val_dataloader, "val", epoch=epoch)

    def test(self):
        test_dataloader = DataLoader(
            self.dataset_dict["test"],  # type: ignore - interface compatibility
            batch_size=self.val_batch_size,
            collate_fn=partial(
                self.collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                model=self.model,
            ),
            shuffle=False,
        )
        self._common_eval(test_dataloader, "test")

    @torch.no_grad()
    def _common_eval(
        self,
        dataloader: DataLoader,
        eval_type: Literal["val", "test"],
        epoch: Optional[int] = None,
    ):

        assert eval_type in ["val", "test"]
        assert (
            epoch is not None
            and eval_type == "val"
            or epoch is None
            and eval_type == "test"
        )

        self.model.eval()
        if eval_type == "val":
            self.log_fn("Starting Validation")
        else:
            self.log_fn("Starting Testing")
        self.log_fn(f" - Batch Size:  {self.val_batch_size}")
        self.log_fn(f" - Num Batches: {len(dataloader)}")
        self.log_fn(f" - Device:      {self.device}")

        if self.log_wandb:
            self.eval_pred_table = wandb.Table(
                columns=["GT_Title", "Pred_title", "GT_Keywords", "Pred_keywords"]
            )
        for i, batch in enumerate(dataloader):
            # Put batch on device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Generate predicted tokens
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.max_new_tokens,
            )
            labels = batch["labels"]
            # Replace -100 in the labels as we can't decode them
            labels[labels < 0] = self.tokenizer.pad_token_id
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decoded_labels = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decoded_preds, decoded_labels = postprocess_validation_text(
                decoded_preds, decoded_labels
            )

            if self.double_task:
                half_batch_len = len(batch["labels"]) // 2
                pred_title = decoded_preds[0:half_batch_len]
                pred_keywords = decoded_preds[half_batch_len:]
                pred_keywords = [
                    set(split_keywords_by_comma(kw)) for kw in pred_keywords
                ]
                target_title = decoded_labels[0:half_batch_len]
                target_keywords = decoded_labels[half_batch_len:]
                target_keywords = [
                    set(split_keywords_by_comma(kw)) for kw in target_keywords
                ]
            else:
                pred_split = self.evaluator.split_title_keywords(decoded_preds)
                target_split = self.evaluator.split_title_keywords(decoded_labels)
                pred_title, pred_keywords = zip(*pred_split)
                target_title, target_keywords = zip(*target_split)

            bin_keywords_list = self.evaluator.binary_labels_keywords(
                target_keywords=target_keywords, pred_keywords=pred_keywords
            )
            ref_binary, pred_binary = zip(*bin_keywords_list)
            self.evaluator.add_batch_title(predicted=pred_title, target=target_title)
            self.evaluator.add_batch_keywords(predicted=pred_binary, target=ref_binary)

            if i % self.log_interval == 0:
                if self.log_wandb:
                    self.eval_pred_table.add_data(
                        target_title[0],
                        pred_title[0],
                        target_keywords[0],
                        pred_keywords[0],
                    )

        # Compute metrics
        if eval_type == "val":
            self.log_fn(" > Validation Scores: ")
        else:
            self.log_fn(" > Test Scores: ")

        results_title = self.evaluator.compute_title()
        results_keywords = self.evaluator.compute_keywords()
        if self.log_wandb:
            if eval_type == "val":
                wandb.log({"val/eval_table": self.eval_pred_table})
                wandb.log({"val/title_metrics": results_title}, step=epoch)
                wandb.log({"val/keywords_metrics": results_keywords}, step=epoch)
            else:
                wandb.log({"test/eval_table": self.eval_pred_table})
                wandb.log({"test/title_metrics": results_title})
                wandb.log({"test/keywords_metrics": results_keywords})

        self.log_fn(" > Title Metrics: ")
        for metric_name, metric_value in results_title.items():
            self.log_fn(f"   > {metric_name}: {metric_value}")
        self.log_fn(" > Keywords Metrics: ")
        for metric_name, metric_value in results_keywords.items():
            self.log_fn(f"   > {metric_name}: {metric_value}")

        self.log_fn("")
        self.model.train()

    @torch.no_grad()
    def _print_eval_train(self, batch, outputs, epoch, batch_id):
        self.model.eval()
        if "logits" in outputs:
            # For seq2seq models like T5, the logits usually correspond to decoder outputs
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_text = self.tokenizer.batch_decode(
                predicted_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            # Decode the ground truth labels
            if "labels" in batch:
                ground_truth_ids = batch["labels"]

                # Replace -100 in the labels as we can't decode them
                ground_truth_ids[ground_truth_ids < 0] = self.tokenizer.pad_token_id
                ground_truth_text = self.tokenizer.batch_decode(
                    ground_truth_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                )

                # Print the first example and the one in the middle
                # NOTE: This is mostly for the two-task training
                log_idx = [0, (len(predicted_text) + 1) // 2]

                for idx in log_idx:
                    if self.log_wandb:
                        self.train_pred_table.add_data(
                            predicted_text[idx], ground_truth_text[idx]
                        )

                    self.log_fn(f" > Example {idx} in the batch")
                    self.log_fn(f"   > Prediction:   {predicted_text[idx]}")
                    self.log_fn(f"   > Ground Truth: {ground_truth_text[idx]}")
                    self.log_fn("")
            else:
                self.log_fn("No labels found in batch. Check the batch structure.")
        else:
            self.log_fn(
                "No logits found in model output. Check the model output structure."
            )
        self.model.train()

    def save(self, save_directory: str):
        """
        Save the model and tokenizer to the specified directory.
        Args:
            save_directory (str): The directory where the model and tokenizer will be saved.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Save model
        model_save_path = os.path.join(save_directory, "model")
        self.model.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")
        # Save tokenizer
        tokenizer_save_path = os.path.join(save_directory, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Tokenizer saved to {tokenizer_save_path}")

    def load(self, load_directory: str):
        """
        Load the model and tokenizer from the specified directory.

        Args:
            load_directory (str): The directory where the model and tokenizer are stored.
        """
        model_load_path = os.path.join(load_directory, "model")
        tokenizer_load_path = os.path.join(load_directory, "tokenizer")
        if not os.path.exists(model_load_path) or not os.path.exists(
            tokenizer_load_path
        ):
            raise ValueError(
                f"Model or tokenizer path does not exist at {load_directory}"
            )
