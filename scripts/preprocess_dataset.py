import os
import re, glob
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="scripts")
def main(cfg: DictConfig):
    # Features: {"title", "keywords", "abstract"}

    data_dir = cfg.data.data_dir
    file_parts = glob.glob(f"{data_dir}/*.jsonl")
    file_parts.sort()
    print(f"Total files found: {len(file_parts)}")

    dataset = load_dataset("json", data_files=file_parts)
    print("Dataset info:")
    print(dataset)

    # Filter out rows with empty keywords
    filtered_dataset = dataset.filter(
        lambda batch: [bool(re.match(r"\w+", elem)) for elem in batch["keywords"]],
        batched=True,
    )

    print("Dataset after filtering:")
    print(filtered_dataset)

    filtered_dataset.save_to_disk(f"{data_dir}/filtered_dataset")

    """
    def add_prefix(example):
        example["sentence1"] = 'My sentence: ' + example["sentence1"]
        return example
    updated_dataset = small_dataset.map(add_prefix)
    """
    # dataset.train_test_split(test_size=0.2)


if __name__ == "__main__":
    main()
