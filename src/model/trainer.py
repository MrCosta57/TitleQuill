from typing import Callable
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
        device: str,
        train_batch_size: int,
        train_shuffle: bool,
        val_batch_size: int,
        max_length: int,
        max_new_tokens: int,
        max_epochs: int,
        collate_fn: Callable,
        loss_fn: Callable,
        lr: float = 1e-5,
        log_interval: int = 100,
    ):
        assert "train" in dataset_dict.keys()
        assert "validation" in dataset_dict.keys()
        assert "test" in dataset_dict.keys()

        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.dataset_dict = dataset_dict
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.val_batch_size = val_batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.max_epochs = max_epochs
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.log_interval = log_interval
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.rouge_score = evaluate.load("rouge")
        self.bleu_score = evaluate.load("bleu")
        self.meteor_score = evaluate.load("meteor")

    def train(self):
        self.model.train()
        train_dataloader = DataLoader(
            self.dataset_dict["train"],
            batch_size=self.train_batch_size,
            collate_fn=partial(
                self.collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                model=self.model,
            ),
            shuffle=self.train_shuffle,
        )

        print("Starting training...")
        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch + 1}/{self.max_epochs}")

            for i, batch in enumerate(train_dataloader):
                # print([f"{k}: {v.shape}" for k, v in batch.items()])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Forward pass
                outputs = self.model(**batch)
                loss = self.loss_fn(batch, outputs)

                if i % self.log_interval == 0:
                    print(f"===Train=== Batch {i} Loss: {loss.item()}")
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #if i % self.log_interval == 0:
                #    self.print_eval(batch, outputs)
                if i == 100:
                    break

                self.validation()
        print("Training completed")

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        val_dataloader = DataLoader(
            self.dataset_dict["validation"],
            batch_size=self.val_batch_size,
            collate_fn=partial(
                self.collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                model=self.model,
            ),
            shuffle=False,
        )

        for i, batch in enumerate(val_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.max_new_tokens,
            )
            labels = batch["labels"]

            """ generated_tokens = pad_tensor(
                generated_tokens, pad_token_id=self.tokenizer.pad_token_id
            )
            # If we did not pad to max length, we need to pad the labels too
            labels = pad_tensor(labels, self.tokenizer.pad_token_id)
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy() """

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

            self.rouge_score.add_batch(
                predictions=decoded_preds, references=decoded_labels
            )
            self.bleu_score.add_batch(
                predictions=decoded_preds, references=decoded_labels
            )
            self.meteor_score.add_batch(
                predictions=decoded_preds, references=decoded_labels
            )

        # Compute metrics
        result_rouge = self.rouge_score.compute()
        result_rouge = {
            key: round(value.mid.fmeasure * 100, 4)
            for key, value in result_rouge.items()
        }
        result_bleu = self.bleu_score.compute()
        result_meteor = self.meteor_score.compute()

        print("===Validation=== Scores")
        print(">> ROUGE")
        print(result_rouge)
        print(">> BLEU")
        print(result_bleu)
        print(">> METEOR")
        print(result_meteor)

    @torch.no_grad()
    def test(self):
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
                    print(f"First example batch")
                    print(f"Prediction:\n{pred}")
                    print(f"Ground Truth:\n{truth}")
                    break
            else:
                print("No labels found in batch. Check the batch structure.")
        else:
            print("No logits found in model output. Check the model output structure.")
