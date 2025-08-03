# src/feature_extractors/complexity.py
import textstat

from src.feature_extractor.base import BaseFeatureExtractor


class ComplexityExtractor(BaseFeatureExtractor):
    """Extract text complexity score and bin using textstat package"""

    def __init__(self, config=None):
        super().__init__(config)
        print(f"ComplexityExtractor: {self.config.get('bins', None)}")
        self.complexity_bins = self.config.get("bins", [3])

    def extract(self, query_text, metadata=None):
        """Extract complexity score and bin"""

        score = textstat.flesch_reading_ease(query_text)
        inverted_score = max(0, min(100, 100 - score))
        bin_idx = self._bin_complexity(inverted_score)

        return {"score": inverted_score, "bin": bin_idx}

    def _bin_complexity(self, score):
        """Bin complexity score into discrete categories"""
        for i, threshold in enumerate(self.complexity_bins):
            if score < threshold:
                return i
        return len(self.complexity_bins)
