import os
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hydra
import wandb
from datamodule.dataset import filter_on_stats, load_oagkx_dataset, OAGKXItem
from utils.evaluator import Evaluator
from utils.general_utils import postprocess_validation_text, seed_everything, setup_nltk


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):
    setup_nltk()
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/qwen2.yaml")
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

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_type)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        first_n_files=cfg.data.first_n_files,
        filter_fn=filter_on_stats,
    )
    dataset = dataset_dict["test"]
    # Encode input prompt
    template_prompt = "Generate the title and the keywords from the below abstract. Do not add any other information and separate the keywords by the comma character.\
        Output must be in the format:\nTitle: [title]\nKeywords: [keywords]\n\
        The abstract is:"
    print_fn = print

    evaluator = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )
    print_fn("Starting Testing")
    print_fn(f" - Num Batches: {len(dataset)}")
    print_fn(f" - Device:      {device}")

    eval_table = None

    if log_wandb:
        # Add metrics
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

    model = model.to(device)
    model.eval()
    for i, data in enumerate(dataset):
        item = OAGKXItem.from_json(data)  # type: ignore
        abstract = item.abstract
        title = "Title: " + item.title
        keywords = item.keywords

        prompt = template_prompt + abstract
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)  # type: ignore
        generated_ids = model.generate(
            model_inputs.input_ids, max_new_tokens=cfg.model.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        pred_split = Evaluator.split_title_keywords([response])
        pred_title, pred_keywords = zip(*pred_split)

        pred_title, title = postprocess_validation_text(pred_title, [title])

        bin_keywords_list = Evaluator.binary_labels_keywords(
            target_keywords=[keywords], pred_keywords=pred_keywords
        )
        pred_binary, ref_binary = zip(*bin_keywords_list)

        evaluator.add_batch_title(predicted=pred_title, target=title)
        evaluator.add_batch_keywords(predicted=pred_binary, target=ref_binary)

        if i % cfg.log_interval == 0:

            print_fn(f"Batch {i+1}/{len(dataset)}")
            print_fn(f"True title:\n{title[0]}")
            print_fn(f"True keywords:\n{keywords}")
            print_fn(f"Prediction:\n{response}")

            if log_wandb and eval_table is not None:
                eval_table.add_data(
                    title[0],
                    pred_title[0],
                    " , ".join(keywords),
                    " , ".join(pred_keywords[0]),
                )

        break

    result_title = evaluator.compute_title()
    result_keywords = evaluator.compute_keywords()

    print_fn("\nTitle metrics:")

    for metric_name, result in result_title.items():
        print(f">> {metric_name.upper()}: {result}")

    print_fn("\nKeywords metrics:")

    for metric_name, result in result_keywords.items():
        print(f">> {metric_name.upper()}: {result}")

    if log_wandb:
        logs = result_title | result_keywords
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
