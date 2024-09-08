""" Run the Qwen2 model on the OAGKX dataset. """

from functools import partial
import os
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hydra
import wandb
from datamodule.dataset import filter_on_stats, load_oagkx_dataset
from utils.evaluator import Evaluator
from torch.utils.data import DataLoader
from utils.general_utils import (
    postprocess_validation_text,
    seed_everything,
    setup_nltk,
    split_keywords_by_comma,
)

template_prompt = (
    "Generate the title and the keywords from the provided abstract."
    "Do not add any other information. Keywords should be comma-separated, NOT LISTED."
    "Output MUST be as in this format example:\n"
    "Title: <title>. Keywords: <keyword_1>, <keyword_2>, <keyword_3>, ...]\n"
    "The abstract is: {abstract}"
)


def custom_qwen2_collate_fn(batch, tokenizer, max_length):
    """ 
    Custom collate function for Qwen2 model. 
    
    :param batch: List of dictionaries with keys: "abstract", "title", "keywords"
    :param tokenizer: Tokenizer object
    :param max_length: Maximum length of the input sequence
    """

    prompts = [template_prompt.format(abstract=item["abstract"]) for item in batch]

    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


    result = tokenizer(texts, padding=True, max_length=max_length, truncation=True, return_tensors="pt")  # type: ignore
    result["titles"] = [item["title"] for item in batch]
    result["keywords"] = [item["keywords"] for item in batch]
    return result


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):

    # Initialize
    setup_nltk()
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/qwen2.yaml")
        cfg["model"] = conf  # type: ignore
    cfg = DictConfig(cfg)
    log_wandb: bool = cfg.get("logger") is not None
    device = torch.device(cfg.device)

    # Initialize wandb
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

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_type)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_type,
        padding_side="left",
    )

    # Load dataset test instances
    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        first_n_files=cfg.data.first_n_files,
        filter_fn=filter_on_stats,
    )
    dataset = dataset_dict["test"]
    print_fn = print

    # Load evaluator
    evaluator = Evaluator(
        metrics_title=cfg.metrics_title, metrics_keywords=cfg.metrics_keywords
    )

    # Wanb table and metrics
    if log_wandb:
        
        eval_table = None

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
                "Pred_text",
            ]
        )

    dataloader = DataLoader(
        dataset,  # type: ignore
        batch_size=cfg.model.batch_size,
        shuffle=False,
        collate_fn=partial(
            custom_qwen2_collate_fn,
            tokenizer=tokenizer,
            max_length=cfg.model.max_length,
        ),
    )
    model = model.to(device)
    model.eval()

    print_fn("Starting Testing")
    print_fn(f" - Num Batches: {len(dataloader)}")
    print_fn(f" - Device:      {device}")
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            titles = batch["titles"]
            keywords = [set(split_keywords_by_comma(kw)) for kw in batch["keywords"]]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids_batch = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.model.max_new_tokens,
            )
            generated_ids_batch = generated_ids_batch[:, input_ids.shape[1] :]
            responses = tokenizer.batch_decode(
                generated_ids_batch,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            pred_split = Evaluator.split_title_keywords(responses)
            pred_title, pred_keywords = zip(*pred_split)

            pred_title, titles = postprocess_validation_text(pred_title, titles)

            bin_keywords_list = Evaluator.binary_labels_keywords(
                target_keywords=keywords, pred_keywords=pred_keywords
            )
            pred_binary, ref_binary = zip(*bin_keywords_list)

            evaluator.add_batch_title(predicted=pred_title, target=titles)
            evaluator.add_batch_keywords(predicted=pred_binary, target=ref_binary)

            if batch_id % cfg.log_interval == 0:
                print_fn(f"Batch {batch_id+1}/{len(dataloader)}")
                print_fn(f"True title:\n{titles[0]}")
                print_fn(f"True keywords:\n{keywords[0]}")
                print_fn(f"Prediction:\n{responses[0]}")

                if log_wandb and eval_table is not None:
                    eval_table.add_data(
                        titles[0],
                        pred_title[0],
                        " , ".join(keywords[0]),
                        " , ".join(pred_keywords[0]),
                        responses[0],
                    )

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
