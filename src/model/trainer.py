from datetime import datetime
from typing import Callable, Dict, List, Literal, Tuple
import loguru
import torch
import evaluate
from functools import partial
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import DatasetDict
from utils.general_utils import postprocess_validation_text


class Trainer:

    def __init__(
        self,
        model: PreTrainedModel,
        dataset_dict: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        device: Literal['cpu', 'cuda'],
        train_batch_size: int,
        val_batch_size: int,
        shuffle: bool,
        max_length: int,
        max_new_tokens: int,
        max_epochs: int,
        collate_fn: Callable,
        loss_fn: Callable,
        lr: float = 1e-5,
        log_interval: int = 100,
        metrics = ['rouge', 'bleu', 'meteor'] 
    ):
        
        assert all([split_name in dataset_dict.keys() for split_name in ["train", "validation", "test"]])

        # Model and Device
        self.device = torch.device(device)
        self.model = model.to(self.device)  # type: ignore

        # Tokenization
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        # Dataset and Dataloader
        self.dataset_dict = dataset_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle

        # Training loop
        self.max_epochs = max_epochs
        self.log_interval = log_interval

        # Training callbacks
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Metrics
        self.metrics = {
            metric_name : evaluate.load(metric_name) for metric_name in metrics
        }

    def train(self) -> Tuple[PreTrainedModel, Dict[str, Dict[str, List[float]]]]:

        # Logging function
        # self.print_fn = print
        
        loguru.logger.add(f"logs/training-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log", level="INFO")     
        self.print_fn = loguru.logger.info

        # Initialize history of losses and metrics
        self._history = {
            'train': {'loss': []}, 
            'val': {metric_name: [] for metric_name in self.metrics}
        }

        # Set model to training mode
        self.model.train()

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

        self.print_fn(f"Starting Training")
        self.print_fn(f' - Epochs:            {self.max_epochs}')
        self.print_fn(f' - Train Batch Size:  {self.train_batch_size}')
        self.print_fn(f' - Num Train Batches: {len(train_dataloader)}')
        self.print_fn(f' - Device:            {self.device}')
        self.print_fn(f'')

        for epoch in range(self.max_epochs):

            self.print_fn(f"Epoch {epoch + 1}/{self.max_epochs} ({round((epoch + 1) / self.max_epochs * 100, 2)}%)")

            loss_batches = []

            for i, batch in enumerate(train_dataloader):

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
                if i % self.log_interval == 0:
                    self.print_fn(f' > Training batch {i+1}/{len(train_dataloader)} ({round(i+1 / len(train_dataloader) * 100, 2)}%) - Loss: {loss.item()}')
                    self.print_eval(batch, outputs)
            
            # Add epoch loss as average of batch losses
            self._history['train']['loss'].append(sum(loss_batches) / len(loss_batches))

            # Perform validation
            self.validation()

        self.print_fn("Training completed")

        self.model.eval()

        return self.model, self._history

    @torch.no_grad()
    def validation(self):

        # TODO - Dobbiamo farlo se c'è torch.no_grad()? E se si dobbiamo rimettere self.model.train() alla fine?
        self.model.eval()

        val_dataloader = DataLoader(
            self.dataset_dict["validation"], # type: ignore - interface compatibility
            batch_size=self.val_batch_size,
            collate_fn=partial(
                self.collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                model=self.model,
            ),
            shuffle=False,
        )

        self.print_fn(" Starting validation...")
        for i, batch in enumerate(val_dataloader):

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

            # Add batch to metrics
            for metric in self.metrics.values():
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        self.print_fn(" > Validation Scores: ")
        for metric_name, metric in self.metrics.items():

            result = metric.compute()
            assert result is not None, f"Error computing metric: {metric_name}"

            match metric_name:
                case "rouge" : result_log = result['rougeL']
                case "bleu"  : result_log = result['bleu']
                case "meteor": result_log = result['meteor']
                case _       : raise ValueError(f"Invalid metric name: {metric_name}")

            self.print_fn(f"   > {metric_name.upper()}: {result_log}")
            self._history['val'][metric_name].append(result)
        
        self.print_fn("")

        self.model.train()

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pass

    @torch.no_grad()
    def print_eval(self, batch, outputs):

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

                    self.print_fn(f" > Example {idx} in the batch")
                    self.print_fn(f"   > Prediction:   {predicted_text[idx]}")
                    self.print_fn(f"   > Ground Truth: {ground_truth_text[idx]}")
                    self.print_fn("")
                
            else:
                self.print_fn("No labels found in batch. Check the batch structure.")
        else:
            self.print_fn("No logits found in model output. Check the model output structure.")

        self.model.train()
