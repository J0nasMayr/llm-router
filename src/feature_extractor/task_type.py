# src/feature_extractor/task_type.py
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.feature_extractor.base import BaseFeatureExtractor


class TaskTypeExtractor(BaseFeatureExtractor):
    """
    Extracts task type by classifying the instruction part of queries
    using a Logistic Regression model on top of BERT embeddings.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.task_types = self.config.get(
            "task_types", ["mmlu", "gsm8k", "hellaswag", "winogrande", "cnn_dailymail"]
        )
        self.embedding_model_name = self.config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dim = self.config.get("embedding_dim", 384)
        self.max_instruction_length = self.config.get("instruction_max_length", 200)

        try:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name, device="cuda"
            )
            if (
                self.embedding_model.get_sentence_embedding_dimension()
                != self.embedding_dim
            ):
                print(
                    f"Warning: Expected embedding dimension {self.embedding_dim} but model outputs "
                    f"{self.embedding_model.get_sentence_embedding_dimension()}"
                )
                self.embedding_dim = (
                    self.embedding_model.get_sentence_embedding_dimension()
                )

            self.model_loaded = True
            print(f"Loaded embedding model: {self.embedding_model_name} on GPU")
        except Exception as e:
            print(f"Error loading embedding model on GPU: {e}")
            self.model_loaded = False
        self.classifier_path = self.config.get(
            "classifier_path", "models/proper_task_classifier.pkl"
        )
        self.classifier = self._load_classifier()

    def _load_classifier(self):
        """Load trained classifier from disk or create a new one"""
        if os.path.exists(self.classifier_path):
            print(f"Loading classifier from {self.classifier_path}")
            try:
                with open(self.classifier_path, "rb") as f:
                    classifier = pickle.load(f)
                print(f"Loaded task classifier from {self.classifier_path}")
                return classifier
            except Exception as e:
                print(f"Error loading classifier: {e}")

        print("Creating new task classifier (needs training)")
        return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")

    def _extract_instruction(self, text):
        """Extract the instruction part from the query text (first few lines)"""

        lines = text.split("\n")
        instruction = "\n".join(lines[: min(3, len(lines))])
        if len(instruction) > self.max_instruction_length:
            instruction = instruction[: self.max_instruction_length]

        return instruction

    def extract(self, query_text, metadata=None):
        """Extract task type from query text or metadata"""

        if metadata and "task_type" in metadata:
            task_type = metadata["task_type"]
            if task_type in self.task_types:
                return task_type
        if not self.model_loaded or not hasattr(self, "classifier"):
            return self._keyword_based_classification(query_text)
        instruction = self._extract_instruction(query_text)
        try:
            embedding = self.embedding_model.encode(
                instruction, show_progress_bar=False
            )
            embedding = embedding.reshape(1, -1)
            predicted_label_idx = self.classifier.predict(embedding)[0]

            if isinstance(predicted_label_idx, (int, np.integer)):
                if 0 <= predicted_label_idx < len(self.task_types):
                    return self.task_types[predicted_label_idx]
                else:
                    return self._keyword_based_classification(query_text)
            else:
                return predicted_label_idx

        except Exception as e:
            print(f"Error during task classification: {e}")
            return self._keyword_based_classification(query_text)

    def _keyword_based_classification(self, query_text):
        """Fallback using keyword-based classification"""
        query_lower = query_text.lower()

        if any(x in query_lower for x in ["summarize", "summary", "summarization"]):
            return "summarization"
        elif any(x in query_lower for x in ["solve", "calculate", "math", "equation"]):
            return "reasoning"
        elif any(
            x in query_lower
            for x in ["if", "imagine", "would", "scenario", "hypothetical"]
        ):
            return "common_sense"
        else:
            return "question_answering"

    def train(self, texts, labels, save=True):
        """
        Train the classifier on instruction texts and their labels

        Parameters:
        - texts: list of query texts
        - labels: corresponding task type labels
        - save: whether to save the trained model
        """
        if not self.model_loaded:
            print("Embedding model not loaded, can't train classifier")
            return False
        instructions = [self._extract_instruction(text) for text in texts]
        print(f"Generating embeddings for {len(instructions)} examples...")
        embeddings = self.embedding_model.encode(instructions, show_progress_bar=False)
        if isinstance(labels[0], str):
            label_indices = []
            for label in labels:
                if label in self.task_types:
                    label_indices.append(self.task_types.index(label))
                else:
                    label_indices.append(self.task_types.index("question_answering"))
            labels = label_indices
        print("Training logistic regression classifier...")
        self.classifier = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced"
        )
        self.classifier.fit(embeddings, labels)
        if save:
            try:
                os.makedirs(os.path.dirname(self.classifier_path), exist_ok=True)
                with open(self.classifier_path, "wb") as f:
                    pickle.dump(self.classifier, f)
                print(f"Saved task classifier to {self.classifier_path}")
            except Exception as e:
                print(f"Error saving classifier: {e}")

        return True
