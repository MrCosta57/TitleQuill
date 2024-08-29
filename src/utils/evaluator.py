from typing import List, Set
import evaluate
from datamodule.dataset import OAGKXItem
import re
from utils.general_utils import split_keywords_by_comma


class Evaluator:
    def __init__(
        self,
        metrics_title=["rouge", "bleu", "meteor"],
        metrics_keywords=["f1", "precision", "recall"],
    ):  # jaccard missing

        self.metrics_title = {
            metric_name: evaluate.load(metric_name) for metric_name in metrics_title
        }

        self.metrics_keywords = {
            metric_name: evaluate.load(metric_name) for metric_name in metrics_keywords
        }

    @property
    def get_metrics_title(self):
        return self.metrics_title

    @property
    def get_metrics_keywords(self):
        return self.metrics_keywords

    def add_batch_title(self, predicted, target):
        for metric in self.metrics_title.values():
            metric.add_batch(predictions=predicted, references=target)

    def add_batch_keywords(self, predicted, target):
        for metric in self.metrics_keywords.values():
            metric.add_batch(predictions=predicted, references=target)

    @staticmethod
    def binary_labels_keywords(
        target_keywords: List[Set[str]], pred_keywords: List[Set[str]]
    ):
        labels_list = [
            list(target.union(pred))
            for target, pred in zip(target_keywords, pred_keywords)
        ]
        result = [
            (
                1 if label in pred_keywords[i] else 0,
                1 if label in target_keywords[i] else 0,
            )
            for i in range(len(labels_list))
            for label in labels_list[i]
        ]
        return result

    @staticmethod
    def split_title_keywords(responses: List[str]):
        pattern = r"\s*\.?\s*\n?\s*Keywords:\s*"
        parts = [re.split(pattern, r, maxsplit=1) for r in responses]
        result = [
            (
                (p[0].strip(), set(split_keywords_by_comma(p[1])))
                if len(p) == 2
                else (r.strip(), set())
            )
            for p, r in zip(parts, responses)
        ]
        return result

    def compute_title(self):
        result_log = {}
        for metric_name, metric in self.metrics_title.items():
            result = metric.compute()
            assert result is not None, f"Error computing metric: {metric_name}"
            match metric_name:
                case "rouge":
                    result_log["rouge1"] = result["rouge1"]
                    result_log["rouge2"] = result["rouge2"]
                    result_log["rougeL"] = result["rougeL"]
                case "bleu":
                    result_log[metric_name] = result["bleu"]
                case "meteor":
                    result_log[metric_name] = result["meteor"]
                case _:
                    raise ValueError(f"Invalid metric name: {metric_name}")
        return result_log

    def compute_keywords(self):
        result_log = {}
        for metric_name, metric in self.metrics_keywords.items():
            match metric_name:
                case "f1":
                    result = metric.compute()
                    assert result is not None, f"Error computing metric: {metric_name}"
                    result_log[metric_name] = result["f1"]
                case "precision":
                    result = metric.compute(zero_division=0.0)
                    assert result is not None, f"Error computing metric: {metric_name}"
                    result_log[metric_name] = result["precision"]
                case "recall":
                    result = metric.compute(zero_division=0.0)
                    assert result is not None, f"Error computing metric: {metric_name}"
                    result_log[metric_name] = result["recall"]
                # case "jaccard":
                #     result_log[metric_name] = result["jaccard"]
                case _:
                    raise ValueError(f"Invalid metric name: {metric_name}")
        return result_log
