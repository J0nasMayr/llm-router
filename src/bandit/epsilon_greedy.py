# src/bandit/epsilon_greedy.py
import ast
import logging
import random
from collections import defaultdict

import numpy as np

from src.bandit.base import BaseBandit

logger = logging.getLogger(__name__)


class EpsilonGreedy(BaseBandit):
    def __init__(
        self,
        model_ids,
        initial_epsilon=0.1,
        decay_factor=0.995,
        min_epsilon=0.01,
        seed=None,
        context_dimension=None,
    ):
        super().__init__(model_ids, context_dimension)
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon
        self.reward_sums = {model: 0.0 for model in model_ids}
        self.counts = {model: 0 for model in model_ids}
        logger.info(f"Instantiate {self.__class__.__name__}")
        if seed is not None:
            random.seed(seed)

        self.decisions = []

    def select_model(self, context=None, available_models=None):
        candidates = self._candidates(available_models)
        if random.random() < self.epsilon:
            selected_model = random.choice(candidates)
            self.decisions.append({"type": "exploration", "model": selected_model})
            return selected_model
        selected_model = self._get_best_model(candidates)
        self.decisions.append({"type": "exploitation", "model": selected_model})
        return selected_model

    def update(self, model_id, reward, context=None):
        self.counts[model_id] += 1
        self.reward_sums[model_id] += reward
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)

    def _candidates(self, available_models):
        if not available_models:
            return list(self.model_ids)
        allowed = set(available_models)
        filtered = [m for m in self.model_ids if m in allowed]
        return filtered if filtered else list(self.model_ids)

    def _get_best_model(self, candidates=None):
        candidates = candidates or list(self.model_ids)
        avg_rewards = {}
        untried_models = []
        for model in candidates:
            if self.counts[model] > 0:
                avg_rewards[model] = self.reward_sums[model] / self.counts[model]
            else:
                untried_models.append(model)

        if untried_models:
            return random.choice(untried_models)
        if avg_rewards:
            max_reward = max(avg_rewards.values())
            best_models = [
                model for model, reward in avg_rewards.items() if reward == max_reward
            ]
            return random.choice(best_models)
        else:

            logger.warning("_get_best_model fallback triggered. Choosing random model.")
            return random.choice(candidates)

    def reset(self):
        self.reward_sums = {model: 0.0 for model in self.model_ids}
        self.counts = {model: 0 for model in self.model_ids}
        self.epsilon = self.initial_epsilon
        self.decisions = []

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "epsilon": self.epsilon,
                "initial_epsilon": self.initial_epsilon,
                "decay_factor": self.decay_factor,
                "min_epsilon": self.min_epsilon,
                "reward_sums": self.reward_sums,
                "counts": self.counts,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.initial_epsilon = state.get("initial_epsilon", self.initial_epsilon)
        self.decay_factor = state.get("decay_factor", self.decay_factor)
        self.min_epsilon = state.get("min_epsilon", self.min_epsilon)
        self.reward_sums = state.get("reward_sums", self.reward_sums)
        self.counts = state.get("counts", self.counts)

    def get_metrics(self):
        metrics = {
            "current_epsilon": self.epsilon,
            "exploration_count": sum(
                1 for d in self.decisions if d["type"] == "exploration"
            ),
            "exploitation_count": sum(
                1 for d in self.decisions if d["type"] == "exploitation"
            ),
            "model_selection_counts": {m: self.counts[m] for m in self.model_ids},
        }
        avg_rewards = {}
        for model in self.model_ids:
            if self.counts[model] > 0:
                avg_rewards[model] = self.reward_sums[model] / self.counts[model]
            else:
                avg_rewards[model] = None

        metrics["average_rewards"] = avg_rewards
        return metrics

    def add_model(self, model_id):
        """Adds a new model arm to the EpsilonGreedy bandit."""
        if model_id not in self.model_ids:
            self.model_ids.append(model_id)
            self.n_models = len(self.model_ids)
            self.reward_sums[model_id] = 0.0
            self.counts[model_id] = 0
            logger.info(
                f"Added model {model_id} to EpsilonGreedy. Total models: {self.n_models}"
            )
        else:
            logger.info(f"Model {model_id} already exists in EpsilonGreedy.")


class ContextualEpsilonGreedy(BaseBandit):
    def __init__(
        self,
        model_ids,
        context_dimension=3,
        initial_epsilon=0.1,
        decay_factor=0.995,
        min_epsilon=0.01,
        seed=None,
    ):
        super().__init__(model_ids, context_dimension)
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.context_reward_sums = defaultdict(
            lambda: {model: 0.0 for model in model_ids}
        )
        self.context_counts = defaultdict(lambda: {model: 0 for model in model_ids})
        self.choices_per_context = defaultdict(list)

        self.global_reward_sums = {model: 0.0 for model in model_ids}
        self.global_counts = {model: 0 for model in model_ids}

        self.decisions = []

    def _context_key(self, context):
        if context is None:
            return None

        def _make_hashable(item):
            if isinstance(item, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in item.items()))
            elif isinstance(item, list):
                return tuple(_make_hashable(i) for i in item)
            elif isinstance(item, tuple):
                return tuple(_make_hashable(i) for i in item)
            else:
                return item

        if isinstance(context, np.ndarray):
            return tuple(context.tolist())
        elif isinstance(context, dict):
            return _make_hashable(context)
        elif isinstance(context, (list, tuple)):
            return _make_hashable(context)

        return context

    def select_model(self, context=None, available_models=None):
        context_key = self._context_key(context)
        candidates = self._candidates(available_models)
        if random.random() < self.epsilon:
            selected_model = random.choice(candidates)
            self.decisions.append(
                {"type": "exploration", "model": selected_model, "context": context_key}
            )
            return selected_model
        selected_model = self._get_best_model_for_context(context_key, candidates)
        self.decisions.append(
            {"type": "exploitation", "model": selected_model, "context": context_key}
        )
        self.choices_per_context[context_key].append(selected_model)
        return selected_model

    def _candidates(self, available_models):
        if not available_models:
            return list(self.model_ids)
        allowed = set(available_models)
        filtered = [m for m in self.model_ids if m in allowed]
        return filtered if filtered else list(self.model_ids)

    def update(self, model_id, reward, context=None):
        context_key = self._context_key(context)
        if context_key is not None:
            self.context_counts[context_key][model_id] += 1
            self.context_reward_sums[context_key][model_id] += reward
        self.global_counts[model_id] += 1
        self.global_reward_sums[model_id] += reward
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)

    def _get_best_model_for_context(self, context_key, candidates=None):
        candidates = candidates or list(self.model_ids)
        if context_key is not None:
            for model_id in candidates:
                if self.context_counts[context_key][model_id] == 0:
                    return model_id
            avg_rewards = {}
            for model_id in candidates:
                avg_rewards[model_id] = self._get_context_average_reward(
                    context_key, model_id
                )
            return max(avg_rewards, key=avg_rewards.get)
        return self._get_best_global_model(candidates)

    def _get_context_average_reward(self, context_key, model_id):
        if context_key is None:
            return self._get_global_average_reward(model_id)

        count = self.context_counts[context_key][model_id]
        if count > 0:
            return self.context_reward_sums[context_key][model_id] / count
        else:
            return self._get_global_average_reward(model_id)

    def _get_global_average_reward(self, model_id):
        if self.global_counts[model_id] > 0:
            return self.global_reward_sums[model_id] / self.global_counts[model_id]
        else:
            return float("inf")

    def _get_best_global_model(self, candidates=None):
        candidates = candidates or list(self.model_ids)
        for model_id in candidates:
            if self.global_counts[model_id] == 0:
                return model_id
        avg_rewards = {
            model: self._get_global_average_reward(model) for model in candidates
        }
        return max(avg_rewards, key=avg_rewards.get)

    def reset(self):
        self.context_reward_sums = defaultdict(
            lambda: {model: 0.0 for model in self.model_ids}
        )
        self.context_counts = defaultdict(
            lambda: {model: 0 for model in self.model_ids}
        )
        self.global_reward_sums = {model: 0.0 for model in self.model_ids}
        self.global_counts = {model: 0 for model in self.model_ids}
        self.epsilon = self.initial_epsilon
        self.decisions = []
        self.choices_per_context.clear()

    def get_state(self):
        state = super().get_state()
        context_reward_sums = {str(k): v for k, v in self.context_reward_sums.items()}
        context_counts = {str(k): v for k, v in self.context_counts.items()}

        state.update(
            {
                "initial_epsilon": self.initial_epsilon,
                "epsilon": self.epsilon,
                "decay_factor": self.decay_factor,
                "min_epsilon": self.min_epsilon,
                "context_reward_sums": context_reward_sums,
                "context_counts": context_counts,
                "global_reward_sums": self.global_reward_sums,
                "global_counts": self.global_counts,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_epsilon = state.get("initial_epsilon", self.initial_epsilon)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.decay_factor = state.get("decay_factor", self.decay_factor)
        self.min_epsilon = state.get("min_epsilon", self.min_epsilon)
        if "context_reward_sums" in state:
            for k_str, v in state["context_reward_sums"].items():
                try:
                    k = ast.literal_eval(k_str)
                    self.context_reward_sums[k] = v
                except (ValueError, SyntaxError):
                    self.context_reward_sums[k_str] = v

        if "context_counts" in state:
            for k_str, v in state["context_counts"].items():
                try:
                    k = ast.literal_eval(k_str)
                    self.context_counts[k] = v
                except (ValueError, SyntaxError):
                    self.context_counts[k_str] = v

        self.global_reward_sums = state.get(
            "global_reward_sums", self.global_reward_sums
        )
        self.global_counts = state.get("global_counts", self.global_counts)

    def get_metrics(self):
        metrics = {
            "current_epsilon": self.epsilon,
            "exploration_count": sum(
                1 for d in self.decisions if d["type"] == "exploration"
            ),
            "exploitation_count": sum(
                1 for d in self.decisions if d["type"] == "exploitation"
            ),
            "model_selection_counts": {
                m: self.global_counts[m] for m in self.model_ids
            },
            "context_count": len(self.context_counts),
        }
        global_avg_rewards = {}
        for model in self.model_ids:
            if self.global_counts[model] > 0:
                global_avg_rewards[model] = (
                    self.global_reward_sums[model] / self.global_counts[model]
                )
            else:
                global_avg_rewards[model] = None

        metrics["global_average_rewards"] = global_avg_rewards

        return metrics

    def add_model(self, model_id):
        """Adds a new model arm to the ContextualEpsilonGreedy bandit."""
        if model_id not in self.model_ids:
            self.model_ids.append(model_id)
            self.n_models = len(self.model_ids)
            self.global_reward_sums[model_id] = 0.0
            self.global_counts[model_id] = 0

            self.context_reward_sums.default_factory = lambda: {
                model: 0.0 for model in self.model_ids
            }
            self.context_counts.default_factory = lambda: {
                model: 0 for model in self.model_ids
            }

            logger.info(
                f"Added model {model_id} to ContextualEpsilonGreedy. Total models: {self.n_models}"
            )
        else:
            logger.info(f"Model {model_id} already exists in ContextualEpsilonGreedy.")

    def get_context_decisions(self):
        """Returns the dictionary mapping context keys to lists of chosen models."""

        return dict(self.choices_per_context)

    def get_context_average_rewards(self):
        """Calculates and returns the average reward per model for each learned context."""
        context_avg_rewards = defaultdict(dict)
        for context_key, counts_dict in self.context_counts.items():
            for model_id, count in counts_dict.items():
                if count > 0:
                    total_reward = self.context_reward_sums.get(context_key, {}).get(
                        model_id, 0.0
                    )
                    context_avg_rewards[context_key][model_id] = total_reward / count
                else:
                    context_avg_rewards[context_key][model_id] = None
        return dict(context_avg_rewards)


class LinearEpsilonGreedy(BaseBandit):
    """
    Epsilon-Greedy bandit that uses a linear model per arm to predict rewards
    based on context, allowing generalization. Uses Ridge Regression for updates.
    """

    def __init__(
        self,
        model_ids,
        context_dimension,
        initial_epsilon,
        decay_factor,
        min_epsilon,
        lambda_,
        seed=None,
    ):
        """
        Args:
            model_ids (list): List of model identifiers (arms).
            context_dimension (int): Dimensionality of the context vectors.
            initial_epsilon (float): Starting exploration rate.
            decay_factor (float): Multiplicative factor for epsilon decay.
            min_epsilon (float): Floor for epsilon value.
            lambda_ (float): Regularization parameter for Ridge Regression.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(model_ids, context_dimension)
        if context_dimension is None or context_dimension <= 0:
            raise ValueError(
                "LinearEpsilonGreedy requires a positive context_dimension."
            )
        if not 0 <= initial_epsilon <= 1:
            raise ValueError("initial_epsilon must be between 0 and 1")
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be between 0 (exclusive) and 1")
        if not 0 <= min_epsilon <= 1:
            raise ValueError("min_epsilon must be between 0 and 1")
        if initial_epsilon < min_epsilon:
            raise ValueError("initial_epsilon cannot be less than min_epsilon")
        if not isinstance(lambda_, (int, float)) or lambda_ < 0:
            raise ValueError("lambda_ (regularization) must be non-negative")

        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon
        self.lambda_ = lambda_

        self.A = {
            mid: self.lambda_ * np.identity(context_dimension) for mid in model_ids
        }
        self.b = {mid: np.zeros(context_dimension) for mid in model_ids}
        self.theta = {mid: np.zeros(context_dimension) for mid in model_ids}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        logger.info(
            f"LinearEpsilonGreedy initialized with d={context_dimension}, lambda={lambda_}"
        )

    def select_model(self, context, available_models=None):
        """Selects a model using epsilon-greedy strategy based on linear reward prediction."""
        allowed = set(available_models) if available_models else None
        candidates = (
            [m for m in self.model_ids if m in allowed] if allowed else list(self.model_ids)
        )
        if not candidates:
            candidates = list(self.model_ids)

        if context is None:

            logger.warning(
                "LinearEpsilonGreedy received None context, selecting randomly."
            )
            return random.choice(candidates)
        context = np.asarray(context).reshape(-1)
        if context.shape[0] != self.context_dimension:
            logger.error(
                f"Context dimension mismatch. Expected {self.context_dimension}, got {context.shape}. Selecting randomly."
            )
            return random.choice(candidates)
        if random.random() < self.epsilon:
            return random.choice(candidates)
        else:
            predictions = {}
            for mid in candidates:
                predictions[mid] = context @ self.theta[mid]
            max_pred = -np.inf
            best_models = []
            for mid, pred in predictions.items():
                if pred > max_pred:
                    max_pred = pred
                    best_models = [mid]
                elif pred == max_pred:
                    best_models.append(mid)
            return random.choice(best_models)

    def update(self, model_id, reward, context):
        """Updates the linear model parameters for the chosen arm."""
        if context is None:
            logger.warning(
                f"Cannot update LinearEpsilonGreedy for arm {model_id} with None context."
            )
            return

        context = np.asarray(context).reshape(-1)
        if context.shape[0] != self.context_dimension:
            logger.error(
                f"Cannot update: Context dimension mismatch. Expected {self.context_dimension}, got {context.shape}."
            )
            return

        if model_id not in self.model_ids:
            logger.error(f"Attempted to update non-existent model_id '{model_id}'")
            return

        context_col = context.reshape(-1, 1)
        self.A[model_id] += context_col @ context_col.T
        self.b[model_id] += reward * context

        try:
            A_inv = np.linalg.pinv(self.A[model_id])
            self.theta[model_id] = A_inv @ self.b[model_id]
        except np.linalg.LinAlgError:
            logger.error(
                f"Matrix inversion failed for arm {model_id}. Theta not updated."
            )

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)

    def reset(self):
        """Resets the bandit state to initial conditions."""
        self.epsilon = self.initial_epsilon
        self.A = {
            mid: self.lambda_ * np.identity(self.context_dimension)
            for mid in self.model_ids
        }
        self.b = {mid: np.zeros(self.context_dimension) for mid in self.model_ids}
        self.theta = {mid: np.zeros(self.context_dimension) for mid in self.model_ids}
        logger.info("LinearEpsilonGreedy reset.")

    def get_state(self):
        """Returns the current state of the bandit for persistence."""
        state = super().get_state()
        state.update(
            {
                "initial_epsilon": self.initial_epsilon,
                "epsilon": self.epsilon,
                "decay_factor": self.decay_factor,
                "min_epsilon": self.min_epsilon,
                "lambda_": self.lambda_,
                "A": {mid: A.tolist() for mid, A in self.A.items()},
                "b": {mid: b.tolist() for mid, b in self.b.items()},
            }
        )
        return state

    def set_state(self, state):
        """Restores the bandit state from a saved state."""
        super().set_state(state)
        self.initial_epsilon = state.get("initial_epsilon", self.initial_epsilon)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.decay_factor = state.get("decay_factor", self.decay_factor)
        self.min_epsilon = state.get("min_epsilon", self.min_epsilon)
        self.lambda_ = state.get("lambda_", self.lambda_)
        self.A = {
            mid: np.array(A_list)
            for mid, A_list in state.get("A", {}).items()
            if mid in self.model_ids
        }
        self.b = {
            mid: np.array(b_list)
            for mid, b_list in state.get("b", {}).items()
            if mid in self.model_ids
        }
        self.theta = {}
        for mid in self.model_ids:
            if mid not in self.A:
                self.A[mid] = self.lambda_ * np.identity(self.context_dimension)
                self.b[mid] = np.zeros(self.context_dimension)
                self.theta[mid] = np.zeros(self.context_dimension)
            else:
                try:
                    A_inv = np.linalg.pinv(self.A[mid])
                    self.theta[mid] = A_inv @ self.b[mid]
                except np.linalg.LinAlgError:
                    logger.error(
                        f"Matrix inversion failed for arm {mid} during set_state. Setting theta to zeros."
                    )
                    self.theta[mid] = np.zeros(self.context_dimension)

    def add_model(self, model_id):
        """Adds a new model arm."""
        if model_id not in self.model_ids:
            super().add_model(model_id)

            self.A[model_id] = self.lambda_ * np.identity(self.context_dimension)
            self.b[model_id] = np.zeros(self.context_dimension)
            self.theta[model_id] = np.zeros(self.context_dimension)
            logger.info(
                f"Initialized state for new arm {model_id} in LinearEpsilonGreedy."
            )

    def remove_model(self, model_id):
        """Removes a model arm."""
        if model_id in self.model_ids:
            super().remove_model(model_id)

            del self.A[model_id]
            del self.b[model_id]
            del self.theta[model_id]
            logger.info(f"Removed state for arm {model_id} from LinearEpsilonGreedy.")

    def get_theta(self):
        """Returns the learned linear parameter vectors for all arms."""
        return self.theta
