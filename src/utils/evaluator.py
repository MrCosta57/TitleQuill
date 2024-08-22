import evaluate


class Evaluator:
    def __init__(self, metrics=["rouge", "bleu", "meteor"]):
        self.metrics = {
            metric_name: evaluate.load(metric_name) for metric_name in metrics
        }

    @property
    def get_metrics(self):
        return self.metrics

    def add_batch(self, predicted, target):
        for metric in self.metrics.values():
            metric.add_batch(predictions=predicted, references=target)

    def compute(self):
        result_log = {}
        for metric_name, metric in self.metrics.items():
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
