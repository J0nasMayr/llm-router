class QualityMetrics:
    """Calculate various quality metrics for model responses."""

    def calculate(self, prediction, reference, metric_type="exact_match"):
        """Calculate quality metric between prediction and reference."""
        metric_functions = {
            "exact_match": self._exact_match,
            "f1_score": self._f1_score,
            "rouge": self._rouge_score,
        }

        metric_fn = metric_functions.get(metric_type, self._exact_match)
        return metric_fn(prediction, reference)

    def _exact_match(self, prediction, reference):
        """Calculate exact match score (1.0 if exact match, 0.0 otherwise)."""
        return 1.0 if prediction.strip() == reference.strip() else 0.0

    def _f1_score(self, prediction, reference):
        """Calculate F1 score for question answering."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        common = len(pred_tokens.intersection(ref_tokens))
        if common == 0:
            return 0.0

        precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = common / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _rouge_score(self, prediction, reference):
        """Calculate ROUGE score for summarization (simplified)."""

        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        common = len(pred_tokens.intersection(ref_tokens))
        return common / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
