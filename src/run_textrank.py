import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from datamodule.dataset import OAGKXItemStats, filter_on_stats, load_oagkx_dataset
from model.text_rank import get_title_and_keywords, setup_environment
from utils.evaluator import Evaluator
from utils.general_utils import seed_everything


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/textrank.yaml")
        cfg["model"] = conf  # type: ignore
    cfg = DictConfig(cfg)
    
    if cfg.get("logger") != None:

        wandb.require("core")
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=cfg.logger.project,
            tags=cfg.model.model_name,
            dir=cfg.logger.log_dir,
        )
        pass

    setup_environment()
    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        just_one_file=cfg.data.just_one_file,
        # filter_fn=filter_on_stats,
    )
    dataset = dataset_dict["test"]
    print_fn = print

    evaluator = Evaluator(cfg.eval_metrics)
    
    print_fn("Starting Testing")
    print_fn(f" - Num Batches: {len(dataset)}")
    for i, data in enumerate(tqdm(dataset)):

        el = OAGKXItemStats.from_json(data)

        pred_title, pred_keywords = get_title_and_keywords(el.abstract)

        if i % cfg.log_interval == 0:
            print_fn(f"Batch {i+1}")
            print_fn(f"Predicted title:\n{pred_title}")
            print_fn(f"TRUE title:\n{el.title}")
            print_fn(f"Predicted keywords:\n{pred_keywords}")
            print_fn(f"TRUE keywords:\n{el.keywords_str}")

        evaluator.add_batch_title(predicted=[pred_title], target=[el.title])

        labels = sorted(list(el.keywords.union(pred_keywords)))
        pred_binary = [1 if label in pred_keywords else 0 for label in labels]
        ref_binary = [1 if label in el.keywords else 0 for label in labels]

        evaluator.add_batch_keywords(predicted=pred_binary, target=ref_binary)

    result_log_title = evaluator.compute_title()
    print_fn(result_log_title)

    result_log_keywords = evaluator.compute_keywords()
    print_fn(result_log_keywords)


if __name__ == "__main__":
    main()
