from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from utils.general_utils import seed_everything
from datamodule.dataset import (
    load_oagkx_dataset,
    filter_no_keywords,
    custom_collate_seq2seq,
    custom_collate_seq2seq_2task,
)
from torch.optim.adamw import AdamW


def training_loop(model, tokenizer, dataset, num_epochs, device):

    # data_collator = partial(custom_collate_seq2seq, tokenizer=tokenizer, model=model)
    data_collator = partial(
        custom_collate_seq2seq_2task, tokenizer=tokenizer, model=model
    )
    # Define the DataLoader
    train_dataloader = DataLoader(
        dataset, collate_fn=data_collator, shuffle=False, batch_size=2
    )

    for batch in train_dataloader:
        print([f"{k}: {v.shape}" for k, v in batch.items()])
        break
    optimizer = AdamW(model.parameters(), lr=1e-5)

    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(train_dataloader):
            print("Iteration", i + 1)
            # print([f"{k}: {v.shape}" for k, v in batch.items()])
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass

            outputs = model(**batch)
            logits = outputs.logits # 2, N, 32128

            labels = batch['labels'] # 2, N
            batched_logits = logits.view(torch.prod(torch.tensor(logits.shape[:2])), -1)
            batched_labels = labels.view(-1)
            loss = torch.nn.CrossEntropyLoss()(batched_logits,batched_labels)
            print(f"Loss: {loss.item()}")
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_eval(outputs, batch, tokenizer)


@torch.no_grad()
def print_eval(outputs, batch, tokenizer):
    if "logits" in outputs:
        # For seq2seq models like T5, the logits usually correspond to decoder outputs
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.batch_decode(
            predicted_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )
        # Decode the ground truth labels
        if "labels" in batch:
            ground_truth_ids = batch["labels"]
            # Replace -100 in the labels as we can't decode them
            ground_truth_ids[ground_truth_ids < 0] = tokenizer.pad_token_id
            ground_truth_text = tokenizer.batch_decode(
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1
    model_type = "google/flan-t5-small"
    data_dir = "data/OAGKX/oagkx"
    seed_everything(123)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=data_dir, split="train", filter_fn=filter_no_keywords
    )

    training_loop(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_dict["train"],
        num_epochs=num_epochs,
        device=device,
    )


if __name__ == "__main__":
    main()
