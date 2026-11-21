# src/feature_extractors/semantic_cluster.py
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extractor.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class SemanticClusterExtractor(BaseFeatureExtractor):
    """
    Extracts semantic cluster from query using online K-means with a fixed number of clusters (k).
    Initializes centroids using the first k unique query embeddings encountered.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.k = self.config.get("num_clusters", 2)
        if self.k <= 0:
            logger.warning(
                f"num_clusters must be > 0, defaulting to 2. Got: {self.k}"
            )
            self.k = 2

        self.embedding_model_name = self.config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dim = None
        self.embedding_model = None
        self.model_loaded = False
        self.centroids = None
        self.counts = None
        self._initial_embeddings = []
        self.initialized = False

        self._load_model()

    def _load_model(self):
        """Loads the sentence embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.centroids = np.zeros((self.k, self.embedding_dim))
            self.counts = np.zeros(self.k)
            self.model_loaded = True
            logger.info(
                f"Embedding model loaded successfully. Dimension: {self.embedding_dim}, k={self.k}"
            )
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{self.embedding_model_name}': {e}",
                exc_info=True,
            )
            self.model_loaded = False

    def extract(self, query_text, metadata=None):
        """Extract semantic cluster ID (0 to k-1) using online k-means."""

        try:
            embedding = self.embedding_model.encode(query_text, show_progress_bar=False)
            if not self.initialized:
                is_unique = True
                for existing_emb in self._initial_embeddings:
                    if np.allclose(embedding, existing_emb, atol=1e-6):
                        is_unique = False
                        break

                if is_unique:
                    self._initial_embeddings.append(embedding)
                    logger.debug("Collected unique initial embedding")

                if len(self._initial_embeddings) == self.k:
                    logger.info(
                        f"Initializing {self.k} centroids with the first {self.k} unique embeddings."
                    )
                    self.centroids = np.array(self._initial_embeddings)
                    self.counts = np.ones(self.k)
                    self.initialized = True
                    self._initial_embeddings = []
                    logger.info(
                        f"*** Centroids Initialized with {self.k} points. Initial counts: {self.counts}"
                    )
                    similarities = cosine_similarity([embedding], self.centroids)[0]
                    nearest_cluster_id = np.argmax(similarities)
                    return nearest_cluster_id
                else:

                    logger.debug(
                        "Waiting for more unique embeddings to initialize centroids. Assigning temp cluster 0."
                    )
                    return 0

            else:
                similarities = cosine_similarity([embedding], self.centroids)[0]
                nearest_cluster_id = np.argmax(similarities)
                self.counts[nearest_cluster_id] += 1
                learning_rate = 1.0 / self.counts[nearest_cluster_id]
                self.centroids[nearest_cluster_id] += learning_rate * (
                    embedding - self.centroids[nearest_cluster_id]
                )
                total_count = np.sum(self.counts)
                if total_count % 100 == 0:
                    centroid_norms = np.linalg.norm(self.centroids, axis=1)
                    logger.info(
                        f"Online K-Means Update #{int(total_count)}: Cluster counts: {self.counts}, Centroid Norms: {centroid_norms}"
                    )

                if total_count <= self.k + 10:
                    logger.info("Assigning to cluster")

                return nearest_cluster_id

        except Exception as e:
            logger.error(
                f"Error during semantic clustering extraction for query: '{query_text[:50]}...': {e}",
                exc_info=True,
            )
