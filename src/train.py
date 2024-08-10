import torch
import hydra
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from utils.general_utils import seed_everything
from transformers import TrainerCallback, TrainerState, TrainerControl
from datamodule.dataset import load_oagkx_dataset, DataCollatorOAGKX


def main():
    model_type = "google/flan-t5-small"
    data_dir = "data/OAGKX"
    seed_everything(123)
    dataset = load_oagkx_dataset(
        data_dir=data_dir, split="train", streaming=False
    ).select(range(10))
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
    data_collator = DataCollatorOAGKX(tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir="out",
        learning_rate=2e-5,
        per_device_train_batch_size=5,
        weight_decay=0.01,
        num_train_epochs=1,
        logging_steps=1,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
