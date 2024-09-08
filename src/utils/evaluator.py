from typing import Dict, List, Set, Tuple
import evaluate
from evaluate import EvaluationModule
import re
from utils.general_utils import split_keywords_by_comma


class Evaluator:
    """
    Evaluator class to compute metrics for title and keywords.
    """

    def __init__(
        self,
        metrics_title=["rouge", "bleu", "meteor"],
        metrics_keywords=["f1", "precision", "recall"],
    ):  
        """
        Initialize the evaluator with the metrics to compute.

        :param metrics_title: List of metrics to compute for the title.
        :param metrics_keywords: List of metrics to compute for the keywords.
        """
        
        self.metrics_title = {
            metric_name: evaluate.load(metric_name) for metric_name in metrics_title
        }

        self.metrics_keywords = {
            metric_name: evaluate.load(metric_name) for metric_name in metrics_keywords
        }

    @property
    def get_metrics_title(self) -> Dict[str, EvaluationModule]:
        """Return the metrics for the title."""

        return self.metrics_title

    @property
    def get_metrics_keywords(self) -> Dict[str, EvaluationModule]:
        """Return the metrics for the keywords."""

        return self.metrics_keywords

    @property
    def get_metric_names(self) -> List[str]:
        """ Return the names of the metrics to compute. """

        names = []

        for name in self.get_metrics_title:

            if name == "rouge":
                names.extend(["rouge1", "rouge2", "rougeL"])
            else:
                names.append(name)

        for name in self.get_metrics_keywords:
            names.append(name)

        return names

    def add_batch_title(self, predicted: str | List[str], target: str | List[str]):
        """ 
        Add a batch of predictions and targets for the title. 
        
        :param predicted: Predicted titles.
        :param target: Target titles.
        """
        
        for metric in self.metrics_title.values():
            metric.add_batch(predictions=predicted, references=target)

    def add_batch_keywords(self, predicted: Set[int] | List[Set[int]], target: Set[int] | List[Set[int]]):
        """
        Add a batch of predictions and targets for the keywords.

        :param predicted: Predicted keywords.
        :param target: Target keywords.
        """

        for metric in self.metrics_keywords.values():
            metric.add_batch(predictions=predicted, references=target)

    @staticmethod
    def binary_labels_keywords(
        target_keywords: List[Set[str]], pred_keywords: List[Set[str]]
    ) -> List[Tuple[int, int]]:
        """
        Convert the target and predicted keywords to binary labels.

        :param target_keywords: The target keywords.
        :param pred_keywords: The predicted keywords.
        """

        # Get the union of the target and predicted keywords
        labels_list = [
            list(target.union(pred))
            for target, pred in zip(target_keywords, pred_keywords)
        ]

        # Create the binary labels
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
        """ 
        Extract the title and keywords from the responses using a pattern
        """

        PATTERN = r"(?:Title:\s*(.*?))?\s*(?:Keywords:\s*(.*))?$"

        def extract_title_keywords(text: str) -> Tuple[str, Set[str]]:
            """
            Extract the title and keywords from the text.

            :param text: The text to extract the title and keywords from.
            """

            # Match the text with the pattern
            match = re.match(PATTERN, text)

            if match:
                title = match.group(1).strip() if match.group(1) else ""
                keywords = match.group(2).strip() if match.group(2) else ""
                result = (title, keywords)
            else:
                result = (text.strip(), "")

            t, k = result
            k = set(split_keywords_by_comma(k))

            return t, k

        return [extract_title_keywords(response) for response in responses]

    def compute_title(self) -> Dict[str, float]:
        """
        Compute the metrics for the title.

        :return: The computed metrics.
        """

        result_log = {}

        for metric_name, metric in self.metrics_title.items():

            match metric_name:

                case "rouge":

                    result = metric.compute()
                    assert result is not None, "Error computing metric: rouge"

                    result_log["rouge1"] = result["rouge1"]
                    result_log["rouge2"] = result["rouge2"]
                    result_log["rougeL"] = result["rougeL"]

                case "bleu":
                    try:
                        result = metric.compute()
                    except:
                        result = {"bleu": 0.0}
                    assert result is not None, "Error computing metric: bleu"
                    result_log[metric_name] = result["bleu"]

                case "meteor":
                    result = metric.compute()
                    assert result is not None, "Error computing metric: meteor"
                    result_log[metric_name] = result["meteor"]

                case _:
                    raise ValueError(f"Invalid metric name: {metric_name}")
                
        return result_log

    def compute_keywords(self) -> Dict[str, float]:
        """
        Compute the metrics for the keywords.

        :return: The computed metrics.
        """

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
                
                case _:
                    raise ValueError(f"Invalid metric name: {metric_name}")
        
        return result_log

    def compute(self) -> Dict[str, float]:
        """
        Compute the metrics for the title and keywords.

        :return: The computed metrics.
        """

        return self.compute_title() | self.compute_keywords()
