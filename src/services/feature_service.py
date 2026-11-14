# src/services/feature_service.py
import logging

from src.feature_extractor.complexity import ComplexityExtractor
from src.feature_extractor.semantic_cluster import SemanticClusterExtractor
from src.feature_extractor.task_type import TaskTypeExtractor

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for extracting features from queries using specialized extractors"""

    def __init__(self, config):
        feature_config = config.get("feature_extraction", {})
        self.task_extractor = TaskTypeExtractor(
            feature_config.get("task_classification", {})
        )
        self.semantic_extractor = SemanticClusterExtractor(
            feature_config.get("semantic_clustering", {})
        )
        self.complexity_extractor = ComplexityExtractor(
            feature_config.get("complexity", {})
        )
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
            logger.warning(
                f"Dataset '{dataset_name}' not in configured task_types {self.task_types} or not provided. Setting task_type to None."
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
        """Convert features dictionary to a context vector for MAB algorithms"""
        context_vector = []
        if "task_type" in features:
            task_type = features["task_type"]
            for task in self.task_types:
                context_vector.append(1.0 if task == task_type else 0.0)
        if "semantic_cluster" in features:
            cluster_id = features["semantic_cluster"]
            for i in range(self.num_clusters):
                context_vector.append(1.0 if i == cluster_id else 0.0)
        if "complexity_bin" in features:
            bin_id = features["complexity_bin"]
            for i in range(self.num_complexity_bins):
                context_vector.append(1.0 if i == bin_id else 0.0)

        return context_vector
