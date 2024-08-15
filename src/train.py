from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from utils.general_utils import seed_everything
from utils.loss import twotasks_ce_loss_fn, hf_loss_fn
from datamodule.dataset import (
    load_oagkx_dataset,
    filter_no_keywords,
    custom_collate_seq2seq,
    custom_collate_seq2seq_2task,
)
from model.trainer import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1
    model_type = "google/flan-t5-small"
    data_dir = "data/OAGKX"
    seed_everything(123)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=data_dir, just_one_file=True, filter_fn=filter_no_keywords
    )

    train_dataloader = DataLoader(
        dataset_dict["train"],
        batch_size=2,
        collate_fn=partial(
            custom_collate_seq2seq_2task, tokenizer=tokenizer, model=model
        ),
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_epochs=num_epochs,
        loss_fn=twotasks_ce_loss_fn,
        lr=1e-5,
    )
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()
