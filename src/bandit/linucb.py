# src/bandit/linucb.py
import logging

import numpy as np

from src.bandit.base import BaseBandit

logger = logging.getLogger(__name__)


class LinUCB(BaseBandit):
    """Linear Upper Confidence Bound algorithm for contextual bandits."""

    requires_context = True

    def __init__(self, model_ids, context_dimension, alpha, regularization, seed=None):
        """Initialize LinUCB.

        Args:
            model_ids (list): List of model identifiers (arms).
            context_dimension (int): Dimensionality of context vectors.
            alpha (float): Exploration parameter.
            regularization (float): Regularization parameter (lambda).
            seed (int, optional): Random seed.
        """
        super().__init__(model_ids, context_dimension)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError("LinUCB alpha must be a non-negative number.")
        if not isinstance(regularization, (int, float)) or regularization < 0:

            raise ValueError("LinUCB regularization must be a non-negative number.")
        if context_dimension is None or context_dimension <= 0:
            raise ValueError("LinUCB requires a positive context_dimension.")

        self.alpha = alpha
        self.regularization = regularization

        if seed is not None:
            np.random.seed(seed)

        self.A = {}
        self.b = {}
        self.theta = {}

        for model_id in model_ids:
            self._initialize_model(model_id)

        self.decisions = []
        self.contexts = []
        self.rewards = {}
        self.ucb_scores = {}

    def select_model(self, context=None, available_models=None):
        if context is None:
            context = np.zeros(self.context_dimension)

        context = np.asarray(context).reshape(-1)

        allowed = set(available_models) if available_models else None
        best_model = None
        highest_ucb = float("-inf")
        all_ucb_scores = {}

        for model_id in self.model_ids:
            ucb = self._compute_ucb(model_id, context)
            if allowed is not None and model_id not in allowed:
                ucb = float("-inf")
            all_ucb_scores[model_id] = ucb

            if ucb > highest_ucb:
                highest_ucb = ucb
                best_model = model_id

        if best_model is None and allowed:
            best_model = next(iter(allowed))

        self.decisions.append(
            {
                "model": best_model,
                "context": context.tolist(),
                "ucb_scores": all_ucb_scores.copy(),
            }
        )
        self.contexts.append(context.tolist())
        self.ucb_scores = all_ucb_scores

        return best_model

    def update(self, model_id, reward, context=None):
        if context is None:
            context = np.zeros(self.context_dimension)

        context = np.asarray(context).reshape(-1)

        self.A[model_id] += np.outer(context, context)
        self.b[model_id] += reward * context
        self.theta[model_id] = None

        if model_id not in self.rewards:
            self.rewards[model_id] = []
        self.rewards[model_id].append(reward)

    def _compute_ucb(self, model_id, context):

        if self.theta[model_id] is None:
            A_inv = np.linalg.inv(self.A[model_id])
            self.theta[model_id] = A_inv.dot(self.b[model_id])

        mean_estimate = np.dot(self.theta[model_id], context)
        A_inv = np.linalg.inv(self.A[model_id])
        variance = np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
        ucb = mean_estimate + self.alpha * variance

        return ucb

    def reset(self):
        for model_id in self.model_ids:
            self.A[model_id] = np.identity(self.context_dimension) * self.regularization
            self.b[model_id] = np.zeros(self.context_dimension)
            self.theta[model_id] = None

        self.decisions = []
        self.contexts = []
        self.rewards = {}
        self.ucb_scores = {}

    def get_state(self):
        state = super().get_state()

        A_serializable = {
            model_id: self.A[model_id].tolist() for model_id in self.model_ids
        }
        b_serializable = {
            model_id: self.b[model_id].tolist() for model_id in self.model_ids
        }

        state.update(
            {
                "alpha": self.alpha,
                "regularization": self.regularization,
                "A": A_serializable,
                "b": b_serializable,
            }
        )

        return state

    def set_state(self, state):
        super().set_state(state)

        self.alpha = state.get("alpha", self.alpha)
        self.regularization = state.get("regularization", self.regularization)

        if "A" in state:
            for model_id in self.model_ids:
                if model_id in state["A"]:
                    self.A[model_id] = np.array(state["A"][model_id])

        if "b" in state:
            for model_id in self.model_ids:
                if model_id in state["b"]:
                    self.b[model_id] = np.array(state["b"][model_id])

        for model_id in self.model_ids:
            self.theta[model_id] = None

    def _initialize_model(self, model_id):
        """Initializes the parameters for a single model ID."""
        if model_id not in self.A:
            logger.debug(f"Initializing parameters for model: {model_id}")

            self.A[model_id] = np.identity(self.context_dimension) * self.regularization
            self.b[model_id] = np.zeros(self.context_dimension)
            self.theta[model_id] = None
        else:
            logger.warning(
                f"Attempted to re-initialize model {model_id}, which already exists."
            )

    def add_model(self, model_id):
        """Adds a new model arm to the LinUCB bandit."""
        if model_id in self.model_ids:
            logger.warning(
                f"Model '{model_id}' already exists in LinUCB. No action taken."
            )
            return False

        logger.info(f"Adding new model '{model_id}' to LinUCB.")
        self.model_ids.append(model_id)
        self.n_models = len(self.model_ids)
        self._initialize_model(model_id)

        self.rewards.setdefault(model_id, [])
        logger.info(
            f"Model pool size is now {self.n_models}. Current models: {self.model_ids}"
        )
        return True

    def remove_model(self, model_id):
        """Removes a model arm from the LinUCB bandit."""
        if model_id not in self.model_ids:
            logger.warning(f"Model '{model_id}' not found in LinUCB. Cannot remove.")
            return False

        if self.n_models <= 1:
            logger.error(
                f"Cannot remove model '{model_id}'. At least one model must remain."
            )
            return False

        logger.info(f"Removing model '{model_id}' from LinUCB.")
        try:
            self.model_ids.remove(model_id)
            self.n_models = len(self.model_ids)

            if model_id in self.A:
                del self.A[model_id]
            if model_id in self.b:
                del self.b[model_id]
            if model_id in self.theta:
                del self.theta[model_id]
            if model_id in self.rewards:
                del self.rewards[model_id]

            logger.info(
                f"Successfully removed '{model_id}'. Model pool size is now {self.n_models}. Remaining models: {self.model_ids}"
            )
            return True
        except Exception as e:
            logger.error(f"Error removing model '{model_id}': {e}", exc_info=True)

            return False

    def get_theta(self):
        """Calculates and returns the learned linear parameters (theta) for all *current* arms."""
        calculated_theta = {}
        for model_id in self.model_ids:
            try:

                if self.theta[model_id] is None:

                    A_inv = np.linalg.inv(self.A[model_id])
                    self.theta[model_id] = A_inv @ self.b[model_id]
                calculated_theta[model_id] = self.theta[model_id]
            except np.linalg.LinAlgError:
                logger.error(
                    f"Matrix inversion failed for arm {model_id} in get_theta. Returning zeros."
                )

                self.theta[model_id] = np.zeros(self.context_dimension)
                calculated_theta[model_id] = self.theta[model_id]
            except Exception as e:
                logger.error(
                    f"Error calculating theta for arm {model_id}: {e}", exc_info=True
                )

                self.theta[model_id] = np.zeros(self.context_dimension)
                calculated_theta[model_id] = self.theta[model_id]
        return calculated_theta

    def get_metrics(self):
        metrics = {
            "alpha": self.alpha,
            "decisions_count": len(self.decisions),
            "model_selection_counts": self._get_model_selection_counts(),
            "model_average_rewards": self._get_model_average_rewards(),
        }

        feature_influence = self._get_features_influence()
        metrics["feature_influence"] = feature_influence

        if hasattr(self, "contexts") and len(self.contexts) > 0:
            latest_context = self.contexts[-1]

            if isinstance(latest_context, list):
                latest_context = np.array(latest_context)

            confidence_bounds = {}
            for model_id in self.model_ids:
                if self.theta[model_id] is not None:
                    try:
                        mean_estimate = np.dot(self.theta[model_id], latest_context)
                        A_inv = np.linalg.inv(self.A[model_id])

                        x = latest_context.reshape(-1, 1)
                        uncertainty = float(np.sqrt(np.dot(x.T, np.dot(A_inv, x))))
                        confidence_bounds[model_id] = {
                            "mean": float(mean_estimate),
                            "uncertainty": uncertainty,
                            "ucb": float(mean_estimate + self.alpha * uncertainty),
                        }
                    except Exception as e:

                        confidence_bounds[model_id] = {"mean": "error", "error": str(e)}
            metrics["latest_confidence_bounds"] = confidence_bounds

        if hasattr(self, "ucb_scores"):
            metrics["latest_ucb_scores"] = self.ucb_scores

        return metrics

    def _get_model_selection_counts(self):
        counts = {model_id: 0 for model_id in self.model_ids}
        for decision in self.decisions:
            counts[decision["model"]] += 1
        return counts

    def _get_model_average_rewards(self):
        avg_rewards = {}
        for model_id in self.model_ids:
            if model_id in self.rewards and len(self.rewards[model_id]) > 0:
                avg_rewards[model_id] = np.mean(self.rewards[model_id])
            else:
                avg_rewards[model_id] = None
        return avg_rewards

    def _get_features_influence(self):
        feature_influence = {}
        for model_id in self.model_ids:
            if (
                self.theta[model_id] is None
                and model_id in self.A
                and np.sum(self.A[model_id])
                > self.regularization * self.context_dimension
            ):

                if model_id in self.b:
                    A_inv = np.linalg.inv(self.A[model_id])
                    self.theta[model_id] = A_inv.dot(self.b[model_id])

            if self.theta[model_id] is not None:
                feature_influence[model_id] = self.theta[model_id].tolist()
            else:
                feature_influence[model_id] = [0] * self.context_dimension

        return feature_influence
