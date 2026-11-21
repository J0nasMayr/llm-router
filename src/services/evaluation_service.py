from src.metrics.accuracy import QualityMetrics
from src.metrics.energy import EnergyMetrics
from src.metrics.extractors import get_extractor
from src.metrics.normalizer import MetricNormalizer
from src.metrics.text_utils import TextNormalizer


class EvaluationService:
    """Service for evaluating model responses and calculating rewards."""

    def __init__(self, metric_bounds, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.quality_metrics = QualityMetrics()
        self.energy_metrics = EnergyMetrics()
        self.metric_normalizer = MetricNormalizer(metric_bounds)
        self.text_normalizer = TextNormalizer()

    def _evaluate(
        self,
        response,
        reference,
        extraction_method,
        evaluation_metric,
    ):
        """Score the response against the reference."""

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

    def _calculate_energy_metrics(
        self, energy_consumption, input_tokens, output_tokens, task_type=None
    ):
        """Calculate energy-related metrics."""
        energy_per_token = self.energy_metrics.calculate_per_token(
            energy_consumption, input_tokens, output_tokens
        )
        # Bounds in metrics.yaml are per dataset; without task_type the
        # normalizer finds none and returns a constant 0.5.
        normalized_energy = self.metric_normalizer.normalize(
            "energy_per_token", energy_per_token, task_type
        )

        return {
            "energy_consumption": energy_consumption,
            "energy_per_token": energy_per_token,
            "normalized_energy": normalized_energy,
        }

    def reward(self, normalized_accuracy, normalized_energy):
        """r = alpha*Acc + beta*(1 - E); see evaluate_and_calculate_reward."""
        return self.alpha * normalized_accuracy + self.beta * (1 - normalized_energy)

    def evaluate_and_calculate_reward(
        self,
        response,
        reference,
        task_type,
        energy_consumption,
        input_tokens,
        output_tokens,
        evaluation_metric,
        extraction_method,
    ):
        """Evaluate response and calculate reward r = alpha*Acc - beta*E.

        Both accuracy and energy are normalized to [0,1]. We keep the
        energy-efficiency convention (1 - normalized_energy) so larger reward is
        better, which matches `r = alpha*Acc - beta*E` up to an additive constant
        — the bandit only cares about relative differences between arms.
        """

        accuracy = self._evaluate(
            response, reference, extraction_method, evaluation_metric
        )
        energy_metrics = self._calculate_energy_metrics(
            energy_consumption, input_tokens, output_tokens, task_type
        )
        # metrics.yaml keys the quality bounds as "accuracy" for every dataset,
        # whichever evaluation_metric (exact_match, rouge, ...) produced it.
        normalized_accuracy = self.metric_normalizer.normalize(
            "accuracy", accuracy, task_type
        )
        reward = self.reward(normalized_accuracy, energy_metrics["normalized_energy"])

        return {
            "accuracy": accuracy,
            "normalized_accuracy": normalized_accuracy,
            "energy_consumption": energy_consumption,
            "energy_per_token": energy_metrics["energy_per_token"],
            "normalized_energy": energy_metrics["normalized_energy"],
            "reward": reward,
        }
