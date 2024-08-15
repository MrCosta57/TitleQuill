from typing import Callable
import torch
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        max_epochs: int,
        loss_fn: Callable,
        lr: float = 1e-5,
    ):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.max_epochs = max_epochs
        self.loss_fn = loss_fn
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
    ):
        self.model.train()

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch + 1}/{self.max_epochs}")

            for i, batch in enumerate(train_dataloader):
                print("Iteration", i + 1)
                print([f"{k}: {v.shape}" for k, v in batch.items()])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Forward pass
                outputs = self.model(**batch)
                loss = self.loss_fn(batch, outputs)
                print(f"Loss: {loss.item()}")
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # self.print_eval(batch, outputs)

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        self.model.eval()
        pass

    @torch.no_grad()
    def print_eval(self, batch, outputs):
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
                for i, zipped in enumerate(zip(predicted_text, ground_truth_text)):
                    pred, truth = zipped
                    print(f"Example batch {i + 1}")
                    print(f"Prediction:\n{pred}")
                    print(f"Ground Truth:\n{truth}")
                    print("-" * 50)
            else:
                print("No labels found in batch. Check the batch structure.")
        else:
            print("No logits found in model output. Check the model output structure.")
