import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from datamodule.dataset import OAGKXItem, filter_on_stats, load_oagkx_dataset
from model.text_rank import get_title_and_keywords
from utils.evaluator import Evaluator
from utils.general_utils import seed_everything, setup_nltk


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
        just_one_file=cfg.data.just_one_file,
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
        eval_table = wandb.Table(
            columns=[
                "GT_Title",
                "Predicted Title",
                "GT_Keywords",
                "Predicted Keywords",
            ]
        )
    for i, data in enumerate(dataset):

        item = OAGKXItem.from_json(data)  # type: ignore
        abstract = item.abstract
        title = item.title
        keywords = item.keywords

        pred_title, pred_keywords = get_title_and_keywords(abstract)

        evaluator.add_batch_title(predicted=[pred_title], target=[title])

        bin_keywords_list = Evaluator.binary_labels_keywords(
            [keywords], [pred_keywords]
        )
        pred_binary, ref_binary = zip(*bin_keywords_list)

        evaluator.add_batch_keywords(predicted=pred_binary, target=ref_binary)

        if i % cfg.log_interval == 0:
            print_fn(f"Batch {i+1}")
            print_fn(f"Predicted title:\n{pred_title}")
            print_fn(f"TRUE title:\n{title}")
            print_fn(f"Predicted keywords:\n{pred_keywords}")
            print_fn(f"TRUE keywords:\n{keywords}")

            if log_wandb and eval_table is not None:
                eval_table.add_data(
                    title,
                    pred_title,
                    " , ".join(keywords),
                    " , ".join(pred_keywords),
                )

        break

    result_log_title = evaluator.compute_title()
    print_fn("Title metrics:")
    print_fn(result_log_title)

    result_log_keywords = evaluator.compute_keywords()
    print_fn("Keywords metrics:")
    print_fn(result_log_keywords)

    if log_wandb:
        wandb.log({"test/eval_table": eval_table})
        wandb.log({"test/title": result_log_title})
        wandb.log({"test/keywords": result_log_keywords})


if __name__ == "__main__":
    main()
