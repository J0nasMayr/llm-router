import logging

from src.feature_extractor.complexity import ComplexityExtractor
from src.feature_extractor.semantic_cluster import SemanticClusterExtractor
from src.feature_extractor.task_type import TaskTypeExtractor

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for extracting features from queries using specialized extractors"""

    def __init__(self, config):
        self.task_extractor = TaskTypeExtractor(config["task_classification"])
        self.semantic_extractor = SemanticClusterExtractor(config["semantic_clustering"])
        self.complexity_extractor = ComplexityExtractor(config["complexity"])
        self.task_types = self.task_extractor.task_types
        self.num_clusters = self.semantic_extractor.k
        self.num_complexity_bins = len(self.complexity_extractor.complexity_bins)
        print(
            f"Task types: {self.task_types}; "
            f"Semantic clusters: {self.num_clusters}; "
            f"Complexity bins: {self.complexity_extractor.complexity_bins}"
        )

    @property
    def context_dimension(self):
        """Length of the context vector built by `_get_context_vector`:
        three one-hot blocks in a row (task type, cluster, complexity bin)."""
        return len(self.task_types) + self.num_clusters + self.num_complexity_bins

    def extract_features(self, query_text, query_metadata=None):
        """Extract features from query text and metadata"""
        features = {}
        query_metadata = query_metadata or {}
        dataset_name = query_metadata.get("dataset", None)

        if dataset_name and dataset_name in self.task_types:
            features["task_type"] = dataset_name
            logger.debug(f"Using dataset '{dataset_name}' directly as task_type.")
        else:
            logger.debug(
                f"Dataset '{dataset_name}' is not one of the task types "
                f"{self.task_types}; classifying the query from its text."
            )
            features["task_type"] = self.task_extractor.extract(
                query_text, query_metadata
            )
        features["semantic_cluster"] = self.semantic_extractor.extract(
            query_text, query_metadata
        )
        complexity_result = self.complexity_extractor.extract(
            query_text, query_metadata
        )
        features["complexity_score"] = complexity_result["score"]
        features["complexity_bin"] = complexity_result["bin"]
        features["context_vector"] = self._get_context_vector(features)

        return features

    def _get_context_vector(self, features):
        """Three one-hot blocks in a row (task type, cluster, complexity bin),
        of length context_dimension. A task type outside task_types leaves its
        block all zero."""
        task = features["task_type"]
        task_index = self.task_types.index(task) if task in self.task_types else None
        return (
            _one_hot(task_index, len(self.task_types))
            + _one_hot(features["semantic_cluster"], self.num_clusters)
            + _one_hot(features["complexity_bin"], self.num_complexity_bins)
        )


def _one_hot(index, size):
    """[0, …, 1 at index, …, 0] of length size; all zero if index is None."""
    return [1.0 if i == index else 0.0 for i in range(size)]
