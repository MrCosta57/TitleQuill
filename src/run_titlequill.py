from functools import partial
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
)
from utils.general_utils import seed_everything, setup_nltk
from utils.loss import hf_loss_fn, twotasks_ce_eisl_loss_fn
from utils.evaluator import Evaluator
from datamodule.dataset import (
    load_oagkx_dataset,
    filter_on_stats,
    custom_collate_seq2seq,
    custom_collate_seq2seq_2task,
)
from dotenv import load_dotenv
from model.trainer import Trainer
from omegaconf import OmegaConf, DictConfig
import hydra, wandb

load_dotenv()

# Load config
# (collate_fn, loss_fn, double_task_flag)
TRAINING_STRATEGIES = {
    
    "combined_tasks": (
        custom_collate_seq2seq,
        hf_loss_fn,
        False
    ),
    "combined_tasks_shuffle": (
        partial(custom_collate_seq2seq, shuffle=True),
        hf_loss_fn,
        False,
    ),
    "divided_tasks": (
        custom_collate_seq2seq_2task,
        hf_loss_fn,
        True,
    ),
    "divided_tasks_ce_eisl": (
        custom_collate_seq2seq_2task,
        twotasks_ce_eisl_loss_fn,
        True,
    ),
}


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):
    setup_nltk()
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") is None:
        conf = OmegaConf.load("configs/model/titlequill.yaml")
        cfg["model"] = conf  # type: ignore
    cfg = DictConfig(cfg)
    log_wandb: bool = cfg.get("logger") is not None
    if log_wandb:
        wandb.require("core")
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=cfg.logger.project,
            tags=[cfg.model.model_name],
            dir=cfg.logger.log_dir,
            config={
                k: v
                for k, v in cfg["model"].items()
                if k != "model_name" and k != "model_type"
            },
        )
    assert cfg.model.strategy in TRAINING_STRATEGIES
    assert len(cfg.data.split_size) == 3

    collate_fn, loss_fn, double_task = TRAINING_STRATEGIES[cfg.model.strategy]

    # loss_fn = (
    #     partial(loss_fn, lambda_=cfg.model.lambda_)
    #     if cfg.model.strategy == "divided_tasks_ce_eisl"
    #     else loss_fn
    # )

    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_type)
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        just_one_file=cfg.data.just_one_file,
        filter_fn=filter_on_stats,
    )
    evaluator = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_epochs=cfg.model.max_epochs,
        dataset_dict=dataset_dict,
        train_batch_size=cfg.model.train_batch_size,
        val_batch_size=cfg.model.val_batch_size,
        double_task=double_task,
        log_wandb=True if cfg.get("logger") is not None else False,
        shuffle=True,
        max_length=cfg.model.max_length,
        max_new_tokens=cfg.model.max_new_tokens,
        lr=cfg.model.lr,
        evaluator=evaluator,
        collate_fn=collate_fn,
        loss_fn=loss_fn,
        log_interval=cfg.log_interval,
    )

    trainer.train()
    trainer.save(cfg.output_dir)
    trainer.test()


if __name__ == "__main__":
    main()
