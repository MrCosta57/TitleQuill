from typing import Dict
from utils.evaluator import Evaluator
from datamodule.dataset import OAGKXItemStats
from tqdm import tqdm
from datamodule.dataset import load_oagkx_dataset

eval_first_baseline = Evaluator()
eval_second_baseline = Evaluator()
eval_keywords_baseline = Evaluator()

data_dir = "data/OAGKX"

def main():

    dataset = load_oagkx_dataset(
        data_dir,
        just_one_file=True,
        verbose=True
    )
    
    for entry in tqdm(dataset['test'], desc="Evaluating Baseline"):

        item = OAGKXItemStats.from_json(entry)

        true_title = item.title
        true_keywords = item.keywords

        first_baseline_title, _ = item.sentence_with_more_keywords
        second_baseline_title = item.abstract_first_sentence
        baseline_keywords = set(item.get_most_frequent_words().keys())


        # Convert lists to binary format
        labels = sorted(list(baseline_keywords.union(true_keywords)))
        pred_binary = [1 if label in baseline_keywords else 0 for label in labels]
        ref_binary = [1 if label in true_keywords else 0 for label in labels]

        eval_first_baseline.add_batch_title([first_baseline_title], [true_title])
        eval_second_baseline.add_batch_title([second_baseline_title], [true_title])
        eval_keywords_baseline.add_batch_keywords(pred_binary, ref_binary)

    log_title_first = eval_first_baseline.compute_title()
    log_title_second = eval_second_baseline.compute_title()
    log_keywords = eval_keywords_baseline.compute_keywords()

    for metric_name, result in log_title_first.items():
        print(f"Title - First Baseline   > {metric_name.upper()}: {result}")

    for metric_name, result in log_title_second.items():
        print(f"Title - Second Baseline   > {metric_name.upper()}: {result}")

    for metric_name, result in log_keywords.items():
        print(f"Keywords   > {metric_name.upper()}: {result}")
    

if __name__ == "__main__":
    main()