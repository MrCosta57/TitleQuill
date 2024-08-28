import os

from omegaconf import DictConfig, OmegaConf
from utils.general_utils import seed_everything, setup_nltk
from utils.evaluator import Evaluator
from datamodule.dataset import OAGKXItem, filter_on_stats
from datamodule.dataset import load_oagkx_dataset
import hydra
import wandb


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):
    setup_nltk()
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/baseline.yaml")
        cfg["model"] = conf  # type: ignore
    cfg = DictConfig(cfg)
    log_wandb: bool = cfg.get("logger") != None
    if log_wandb:
        wandb.require("core")
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=cfg.logger.project,
            tags=[cfg.model.model_name],
            dir=cfg.logger.log_dir,
        )

    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        just_one_file=cfg.data.just_one_file,
        filter_fn=filter_on_stats,
    )
    dataset = dataset_dict["test"]
    eval_first_baseline = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )
    eval_second_baseline = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )
    eval_keywords_baseline = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )
    test_wandb_table = None
    if log_wandb:
        test_wandb_table = wandb.Table(
            columns=[
                "GT_Title",
                "Sentence with more keywords",
                "First sentence of abstract",
                "GT_Keywords",
                "Most frequent words",
            ]
        )
    for i, data in enumerate(dataset):
        item = OAGKXItem.from_json(data)  # type: ignore

        true_title = item.title
        true_keywords = item.keywords

        first_baseline_title, _ = item.sentence_with_more_keywords
        second_baseline_title = item.abstract_first_sentence
        baseline_keywords = set(item.get_most_frequent_words().keys())

        # Convert lists to binary format
        pred_binary, ref_binary = Evaluator.binary_labels_keywords(
            [true_keywords], [baseline_keywords]
        )
        eval_first_baseline.add_batch_title([first_baseline_title], [true_title])
        eval_second_baseline.add_batch_title([second_baseline_title], [true_title])
        eval_keywords_baseline.add_batch_keywords(pred_binary, ref_binary)

        if i % cfg.log_interval == 0:
            print(f"Batch {i}/{len(dataset)}")
            print(f"True Title: {true_title}")
            print(f"Sentence with more keywords: {first_baseline_title}")
            print(f"First sentence of abstract: {second_baseline_title}")
            print(f"True Keywords: {true_keywords}")
            print(f"Most frequent words: {baseline_keywords}")

            if log_wandb:
                assert test_wandb_table is not None
                test_wandb_table.add_data(
                    true_title,
                    first_baseline_title,
                    second_baseline_title,
                    true_keywords,
                    baseline_keywords,
                )

    log_title_first = eval_first_baseline.compute_title()
    log_title_second = eval_second_baseline.compute_title()
    log_keywords = eval_keywords_baseline.compute_keywords()

    for metric_name, result in log_title_first.items():
        print(f"Title - First Baseline   > {metric_name.upper()}: {result}")

    for metric_name, result in log_title_second.items():
        print(f"Title - Second Baseline   > {metric_name.upper()}: {result}")

    for metric_name, result in log_keywords.items():
        print(f"Keywords   > {metric_name.upper()}: {result}")

    if log_wandb:
        wandb.log({"test/eval_table": test_wandb_table})
        wandb.log("test/sent_more_keywords", log_title_first)
        wandb.log("test/abstract_first_sentence", log_title_second)
        wandb.log("test/most_freq_words", log_keywords)


if __name__ == "__main__":
    main()
