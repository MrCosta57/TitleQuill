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
    cfg['model'] = OmegaConf.load("configs/model/baseline.yaml")

    assert cfg is not None
    print(cfg)
    assert cfg['model']['title_strategy'] in ['first_sentence', 'with_more_kw']
    
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/baseline.yaml")
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

    evaluator = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )

    if log_wandb:

        # Add metrics
        for metric_name in evaluator.get_metric_names:
            wandb.define_metric(
                f"test/{metric_name}",
                step_metric=f"test_{metric_name}_step",
            )

        test_wandb_table = wandb.Table(
            columns=[
                "GT_Title",
                "Pred_title",
                "GT_Keywords",
                "Pred_keywords",
            ]
        )

    else: 
        test_wandb_table = None

    
    for i, data in enumerate(dataset):

        item = OAGKXItem.from_json(data)  # type: ignore

        true_title = item.title
        true_keywords = item.keywords

        if cfg['model']['title_strategy'] == 'first_sentence':
            baseline_title, _ = item.sentence_with_more_keywords
        elif cfg['model']['title_strategy'] == 'with_more_kw':
            baseline_title = item.abstract_first_sentence
        
        baseline_keywords = set(item.get_most_frequent_words().keys())

        # Convert lists to binary format
        bin_keywords_list = Evaluator.binary_labels_keywords(
            [true_keywords], [baseline_keywords]
        )
        pred_binary, ref_binary = zip(*bin_keywords_list)

        evaluator.add_batch_title([baseline_title], [true_title])
        evaluator.add_batch_keywords(pred_binary, ref_binary)

        if i % cfg.log_interval == 0:

            print(f"Batch {i+1}/{len(dataset)}")
            print(f"True Title: {true_title}")
            print(f"Predicted title: {baseline_title}")
            print(f"True Keywords: {true_keywords}")
            print(f"Most frequent words: {baseline_keywords}")

            if log_wandb and test_wandb_table is not None:
                test_wandb_table.add_data(
                    true_title,
                    baseline_title,
                    " , ".join(true_keywords),
                    " , ".join(baseline_keywords),
                )

            # if i > 0: break
            

    log_title    = evaluator.compute_title()
    log_keywords = evaluator.compute_keywords()

    print("\nTitle Metrics")

    for metric_name, result in log_title.items():
        print(f">> {metric_name.upper()}: {result}")

    print("\nKeywords Metrics:")

    for metric_name, result in log_keywords.items():
        print(f">> {metric_name.upper()}: {result}")

    if log_wandb:

        logs = log_title | log_keywords

        wandb.log({"test/eval_table": test_wandb_table})

        for metric_name in evaluator.get_metric_names:
            wandb.log(
                {
                    f"test/{metric_name}": logs[metric_name],
                    f"test_{metric_name}_step": 0,
                }
            )

if __name__ == "__main__":
    main()
