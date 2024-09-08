""" Run the TextRank model on the OAGKX dataset. """

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from datamodule.dataset import OAGKXItem, filter_on_stats, load_oagkx_dataset
from model.text_rank import get_title_and_keywords
from utils.evaluator import Evaluator
from utils.general_utils import postprocess_validation_text, seed_everything, setup_nltk


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):
    setup_nltk()
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/textrank.yaml")
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

    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        first_n_files=cfg.data.first_n_files,
        filter_fn=filter_on_stats,
    )
    dataset = dataset_dict["test"]
    print_fn = print

    evaluator = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )

    print_fn("Starting Testing")
    print_fn(f" - Num Batches: {len(dataset)}")
    eval_table = None
    if log_wandb:
        for metric_name in evaluator.get_metric_names:
            wandb.define_metric(
                f"test/{metric_name}",
                step_metric=f"test_{metric_name}_step",
            )
        eval_table = wandb.Table(
            columns=[
                "GT_Title",
                "Pred_Title",
                "GT_Keywords",
                "Pred_Keywords",
            ]
        )

    for i, data in enumerate(dataset):

        item = OAGKXItem.from_json(data)  # type: ignore
        abstract = item.abstract
        title = item.title
        keywords = item.keywords

        pred_title, pred_keywords = get_title_and_keywords(abstract)
        pred_title, title = postprocess_validation_text([pred_title], [title])

        evaluator.add_batch_title(predicted=pred_title, target=title)

        bin_keywords_list = Evaluator.binary_labels_keywords(
            [keywords], [pred_keywords]
        )
        pred_binary, ref_binary = zip(*bin_keywords_list)

        evaluator.add_batch_keywords(predicted=pred_binary, target=ref_binary)

        if i % cfg.log_interval == 0:
            print_fn(f"Batch {i+1}/{len(dataset)}")
            print_fn(f"True title:\n{title[0]}")
            print_fn(f"Predicted title:\n{pred_title[0]}")
            print_fn(f"True keywords:\n{keywords}")
            print_fn(f"Predicted keywords:\n{pred_keywords}")

            if log_wandb and eval_table is not None:
                eval_table.add_data(
                    title[0],
                    pred_title[0],
                    " , ".join(keywords),
                    " , ".join(pred_keywords),
                )

    result_log_title = evaluator.compute_title()
    result_log_keywords = evaluator.compute_keywords()

    print_fn("\nTitle metrics:")

    for metric_name, result in result_log_title.items():
        print(f">> {metric_name.upper()}: {result}")

    print_fn("\nKeywords metrics:")

    for metric_name, result in result_log_title.items():
        print(f">> {metric_name.upper()}: {result}")

    if log_wandb:
        logs = result_log_title | result_log_keywords
        wandb.log({"test/test_pred_table": eval_table})
        for metric_name in evaluator.get_metric_names:
            wandb.log(
                {
                    f"test/{metric_name}": logs[metric_name],
                    f"test_{metric_name}_step": 0,
                }
            )


if __name__ == "__main__":
    main()
