import glob
import os
import re
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from datasets import load_dataset
import hydra
import wandb
from datamodule.dataset import filter_on_stats, load_oagkx_dataset
from utils.evaluator import Evaluator
from utils.general_utils import seed_everything


@hydra.main(version_base="1.3", config_path="../configs", config_name="run")
def main(cfg):
    seed_everything(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg is not None
    if cfg.get("model") == None:
        conf = OmegaConf.load("configs/model/qwen2.yaml")
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

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_type)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_type)
    dataset_dict = load_oagkx_dataset(
        data_dir=cfg.data.data_dir,
        split_size=tuple(cfg.data.split_size),
        just_one_file=cfg.data.just_one_file,
        filter_fn=filter_on_stats,
    )
    dataset = dataset_dict["test"]
    # Encode input prompt
    template_prompt = "Generate the title and the keywords from the below abstract. Do not add any other information.\
        Output must be in the format:\nTitle: [title]\nKeywords: [keywords]\n\
        The abstract is:"
    print_fn = print

    evaluator = Evaluator(cfg.eval_metrics)
    print_fn("Starting Testing")
    print_fn(f" - Num Batches: {len(dataset)}")
    print_fn(f" - Device:      {device}")
    for i, data in enumerate(dataset):
        abstract = data["abstract"]  # type: ignore
        title = data["title"]  # type: ignore
        keywords = data["keywords"]  # type: ignore

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
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        match = re.match(r"(.*)Keywords:\s*(.*)", response)
        if match:
            pred_title = match.group(1).strip()
            pred_keywords = match.group(2).strip()
        else:
            # If "Keywords:" is not found, everything is before
            pred_title = response.strip()
            pred_keywords = ""

        if i % cfg.log_interval == 0:
            print_fn(f"Batch {i+1}")
            print_fn(f"Prediction:\n{response}")
            print_fn(f"TRUE title:\n{title}")
            print_fn(f"TRUE keywords:\n{keywords}")

        evaluator.add_batch(predicted=[pred_title], target=[title])

    result_log = evaluator.compute()
    print_fn(result_log)


if __name__ == "__main__":
    main()
