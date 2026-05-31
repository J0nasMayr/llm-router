from src.metrics.accuracy import QualityMetrics
from src.metrics.energy import EnergyMetrics
from src.metrics.extractors import get_extractor
from src.metrics.normalizer import MetricNormalizer
from src.metrics.text_utils import TextNormalizer


class EvaluationService:
    """Service for evaluating model responses and calculating rewards."""

    def __init__(self, metrics_config=None):
        self.metrics_config = metrics_config or {}
        self.quality_metrics = QualityMetrics()
        self.text_normalizer = TextNormalizer()
        self.metric_normalizer = MetricNormalizer(metrics_config)
        self.energy_metrics = EnergyMetrics()

    def evaluate(
        self,
        response,
        reference,
        task_type="default",
        extraction_method="raw",
        evaluation_metric="exact_match",
    ):
        """Evaluate response against reference for a specific task type."""

        extractor = get_extractor(extraction_method)
        extracted_response = extractor.extract(response)
        skip_normalization = extraction_method == "numeric" or (
            extraction_method.startswith("regex_") and "(?!.*\\d)" in extraction_method
        )

        if skip_normalization:
            normalized_response = extracted_response
            normalized_reference = reference
        else:
            normalized_response = self.text_normalizer.normalize(extracted_response)
            normalized_reference = self.text_normalizer.normalize(reference)
        score = self.quality_metrics.calculate(
            normalized_response, normalized_reference, metric_type=evaluation_metric
        )

        return score

    def normalize_metric(self, metric_name, value, task_type=None):
        """Normalize a metric value to [0,1] range."""
        normalized = self.metric_normalizer.normalize(metric_name, value, task_type)
        self.metric_normalizer.update_bounds(metric_name, value, task_type)
        return normalized

    def calculate_energy_metrics(self, energy_consumption, input_tokens, output_tokens):
        """Calculate energy-related metrics."""
        energy_per_token = self.energy_metrics.calculate_per_token(
            energy_consumption, input_tokens, output_tokens
        )
        normalized_energy = self.normalize_metric("energy_per_token", energy_per_token)

        return {
            "energy_consumption": energy_consumption,
            "energy_per_token": energy_per_token,
            "normalized_energy": normalized_energy,
        }

    def evaluate_and_calculate_reward(
        self,
        response,
        reference,
        task_type,
        energy_consumption,
        input_tokens,
        output_tokens,
        evaluation_metric="exact_match",
        extraction_method="raw",
        alpha=0.5,
        beta=0.5,
    ):
        """Evaluate response and calculate reward r = alpha*Acc - beta*E.

        Both accuracy and energy are normalized to [0,1]. We keep the
        energy-efficiency convention (1 - normalized_energy) so larger reward is
        better, which matches `r = alpha*Acc - beta*E` up to an additive constant
        — the bandit only cares about relative differences between arms.
        """

        accuracy = self.evaluate(
            response, reference, task_type, extraction_method, evaluation_metric
        )
        energy_metrics = self.calculate_energy_metrics(
            energy_consumption, input_tokens, output_tokens
        )
        normalized_accuracy = self.normalize_metric(
            evaluation_metric, accuracy, task_type
        )
        reward = alpha * normalized_accuracy + beta * (
            1 - energy_metrics["normalized_energy"]
        )

        return {
            "accuracy": accuracy,
            "normalized_accuracy": normalized_accuracy,
            "energy_consumption": energy_consumption,
            "energy_per_token": energy_metrics["energy_per_token"],
            "normalized_energy": energy_metrics["normalized_energy"],
            "reward": reward,
        }
