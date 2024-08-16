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
from dotenv import load_dotenv
from model.trainer import Trainer

load_dotenv()


def main():
    """if cfg.use_wandb:
    wandb.require("core")
    wandb.login(key=os.getenv("WANDB_API_KEY"))"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1
    model_type = "google/flan-t5-small"
    max_length = 512
    max_new_tokens = 150
    data_dir = "data/OAGKX"
    seed_everything(123)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=data_dir, just_one_file=True, filter_fn=filter_no_keywords
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_epochs=num_epochs,
        dataset_dict=dataset_dict,
        train_batch_size=8,
        train_shuffle=True,
        val_batch_size=8,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        collate_fn=custom_collate_seq2seq,
        loss_fn=hf_loss_fn,
        log_interval=1,
        lr=1e-5,
    )
    trainer.train()


if __name__ == "__main__":
    main()
