import evaluate


class Evaluator:
    def __init__(self, 
                metrics_title=["rouge", "bleu", "meteor"], 
                metrics_keywords=["f1", "precision", "recall"]): # jaccard missing

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


    def compute_title(self):
        result_log = {}
        for metric_name, metric in self.metrics_title.items():
            result = metric.compute()
            assert result is not None, f"Error computing metric: {metric_name}"
            match metric_name:
                case "rouge":
                    result_log[metric_name] = result["rougeL"]
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
            result = metric.compute()
            assert result is not None, f"Error computing metric: {metric_name}"
            match metric_name:
                case "f1":
                    result_log[metric_name] = result["f1"]
                case "precision":
                    result_log[metric_name] = result["precision"]
                case "recall":
                    result_log[metric_name] = result["recall"]
                # case "jaccard":
                #     result_log[metric_name] = result["jaccard"]
                case _:
                    raise ValueError(f"Invalid metric name: {metric_name}")
        return result_log