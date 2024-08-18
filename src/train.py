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
    filter_on_stats,
    custom_collate_seq2seq,
    custom_collate_seq2seq_2task,
)
from dotenv import load_dotenv
from model.trainer import Trainer

load_dotenv()

TRAINING_STRATEGIES = {
    'combined_tasks': (
        custom_collate_seq2seq, 
        hf_loss_fn
    ),
    'divided_tasks': (
        partial(custom_collate_seq2seq_2task, shuffle=False),
        twotasks_ce_loss_fn
    ),
    'divided_tasks_shuffle': (
        partial(custom_collate_seq2seq_2task, shuffle=True),
        twotasks_ce_loss_fn
    )
}

def main():
    
    """if cfg.use_wandb:
    wandb.require("core")
    wandb.login(key=os.getenv("WANDB_API_KEY"))"""
    
    STRATEGY = 'combined_tasks'
    assert STRATEGY in TRAINING_STRATEGIES, f"Invalid training strategy: {STRATEGY}. Choose one of {list(TRAINING_STRATEGIES.keys())}"
    
    collate_fn, loss_fn = TRAINING_STRATEGIES[STRATEGY]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 100
    model_type = "google/flan-t5-small"
    max_length = 512
    max_new_tokens = 150
    data_dir = "data/OAGKX"
    seed_everything(123)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=data_dir, 
        just_one_file=True, 
        filter_fn=filter_on_stats
    )

    print("Dataset loaded")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_epochs=num_epochs,
        dataset_dict=dataset_dict,
        train_batch_size=64,
        train_shuffle=True,
        val_batch_size=64,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        collate_fn=collate_fn,
        loss_fn=loss_fn,
        log_interval=10,
        lr=1e-5,
    )

    trainer.train()


if __name__ == "__main__":
    main()
