import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricNormalizer:
    """
    Normalize metrics to [0, 1] range using dataset-specific min/max bounds.
    Assumes 1 is best, 0 is worst.
    """

    def __init__(self, bounds):
        """
        Args:
            bounds (dict): {dataset: {metric: {"min": ..., "max": ...}}}, as in
                config/metrics.yaml.
        """
        if not bounds:
            raise ValueError("No metric bounds given (is config/metrics.yaml empty?)")
        self.bounds = bounds

    def normalize(self, metric_name, value, dataset):
        """
        Normalize a metric value to the [0, 1] range using dataset-specific bounds.

        Args:
            metric_name (str): The name of the metric (e.g., 'accuracy', 'latency').
            value (float): The raw value of the metric to normalize.
            dataset (str): The name of the dataset the metric value belongs to.

        Returns:
            float: The normalized value clamped between 0.0 and 1.0.
                   Returns 0.5 if bounds are missing or invalid, or if min == max.
        """
        if value is None or np.isnan(value):
            logger.warning(
                f"Cannot normalize None or NaN value for metric '{metric_name}' on dataset '{dataset}'. Returning 0.5."
            )
            return 0.5
        dataset_bounds = self.bounds.get(dataset, self.bounds.get("default"))

        if not dataset_bounds:
            logger.warning(
                f"No bounds found for dataset '{dataset}' (and no default). Cannot normalize '{metric_name}'. Returning 0.5."
            )
            return 0.5

        metric_bounds = dataset_bounds.get(metric_name)

        if (
            not metric_bounds
            or "min" not in metric_bounds
            or "max" not in metric_bounds
        ):
            logger.warning(
                f"Incomplete/missing bounds for metric '{metric_name}' in dataset '{dataset}' config. Cannot normalize. Returning 0.5."
            )
            return 0.5

        min_val = metric_bounds["min"]
        max_val = metric_bounds["max"]
        if max_val == min_val:

            logger.debug(
                f"Min and Max bounds are identical ({min_val}) for metric '{metric_name}' on dataset '{dataset}'. Returning 0.5."
            )
            return 0.5
        try:
            value_f = float(value)
            min_val_f = float(min_val)
            max_val_f = float(max_val)

            normalized = (value_f - min_val_f) / (max_val_f - min_val_f)
        except ZeroDivisionError:
            logger.warning(
                f"Division by zero during normalization for metric '{metric_name}' on dataset '{dataset}'. Min={min_val}, Max={max_val}. Returning 0.5."
            )
            return 0.5
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Type error during normalization calculation for metric '{metric_name}' on dataset '{dataset}'. Value={value}, Min={min_val}, Max={max_val}. Error: {e}. Returning 0.5."
            )
            return 0.5
        clamped_value = max(0.0, min(1.0, normalized))
        return clamped_value

    def update_bounds(self, metric_name, value, task_type=None):
        """Update bounds if value extends the range. Nothing in this repository
        calls this."""
        key = f"{metric_name}_{task_type}" if task_type else metric_name

        if key not in self.bounds:
            self.bounds[key] = {"min": value, "max": value}
            return
        if value < self.bounds[key]["min"]:
            self.bounds[key]["min"] = value
        elif value > self.bounds[key]["max"]:
            self.bounds[key]["max"] = value
