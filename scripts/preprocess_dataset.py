import os
import re, glob
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


""" def format_dataset(examples):
    return {
        "input": [f"Abstract: {e}" for e in examples["abstract"]],
        "gt": [
            f"Title: {t}\nKeywords: {k}"
            for t, k in zip(examples["title"], examples["keywords"])
        ],
    } """


@hydra.main(version_base="1.3", config_path="../configs", config_name="scripts")
def main(cfg: DictConfig):
    # Features: {"title", "keywords", "abstract"}

    print("Preprocessing started...")
    data_dir = cfg.data.data_dir
    file_parts = glob.glob(f"{data_dir}/*.jsonl")
    file_parts.sort()
    print(f"Total files found: {len(file_parts)}")

    dataset = load_dataset("json", data_files=file_parts)
    print("Dataset info:")
    print(dataset)
    prev_len = len(dataset["train"])

    # Filter out rows with empty keywords
    print("Filter out rows with empty keywords...")
    filtered_dataset = dataset.filter(
        lambda batch: [bool(re.match(r"\w+", elem)) for elem in batch["keywords"]],
        batched=True,
    )
    """ print("Making dataset in right format...")
    filtered_dataset = filtered_dataset.map(
        format_dataset,
        batched=True,
        remove_columns=filtered_dataset["train"].column_names,
    )
    print("Dataset after preprocessing:")
    print(filtered_dataset)
    print("First row:")
    print(filtered_dataset["train"][0]) """
    curr_len = len(filtered_dataset["train"])

    print(f"Removed {prev_len-curr_len} rows ({(prev_len-curr_len)/prev_len*100:.3f}%)")
    print("Preprocessing completed! Saving to disk...")
    filtered_dataset.save_to_disk(f"{data_dir}/filtered_dataset")
    print("Done!")

    # dataset.train_test_split(test_size=0.2)


if __name__ == "__main__":
    main()
