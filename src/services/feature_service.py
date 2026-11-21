import logging

from src.feature_extractor.complexity import ComplexityExtractor
from src.feature_extractor.semantic_cluster import SemanticClusterExtractor

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for extracting features from queries using specialized extractors"""

    def __init__(self, config):
        self.task_types = list(config["task_classification"]["task_types"])
        self.num_task_types = len(self.task_types)
        self.semantic_extractor = SemanticClusterExtractor(config["semantic_clustering"])
        self.num_clusters = self.semantic_extractor.k
        self.complexity_extractor = ComplexityExtractor(config["complexity"])
        self.num_complexity_bins = len(self.complexity_extractor.complexity_bins)
        logger.info(
            f"Task types: {self.task_types}; "
            f"Semantic clusters: {self.num_clusters}; "
            f"Complexity bins: {self.complexity_extractor.complexity_bins}"
        )

    @property
    def context_dimension(self):
        """Length of the context vector built by `_get_context_vector`:
        three one-hot blocks in a row (task type, cluster, complexity bin)."""
        return self.num_task_types + self.num_clusters + self.num_complexity_bins

    def extract_features(self, query_text, query_metadata=None):
        """Extract features from query text and metadata. The task type is the
        query's dataset label if it is one of the task types, else None: no
        classifier is used, so an unlabelled query gets an all-zero task-type
        block (as GreenServ's experiment code, which always sends the label)."""
        features = {}
        query_metadata = query_metadata or {}
        dataset_name = query_metadata.get("dataset")
        features["task_type"] = dataset_name if dataset_name in self.task_types else None
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
        of length context_dimension. A task type of None leaves its block all
        zero."""
        task = features["task_type"]
        task_index = None if task is None else self.task_types.index(task)
        return (
            _one_hot(task_index, self.num_task_types)
            + _one_hot(features["semantic_cluster"], self.num_clusters)
            + _one_hot(features["complexity_bin"], self.num_complexity_bins)
        )


def _one_hot(index, size):
    """[0, …, 1 at index, …, 0] of length size; all zero if index is None."""
    return [1.0 if i == index else 0.0 for i in range(size)]
