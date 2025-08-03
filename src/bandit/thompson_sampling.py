# src/bandit/thompson.py
import logging

import numpy as np

from src.bandit.base import BaseBandit

logger = logging.getLogger(__name__)


class ThompsonSampling(BaseBandit):
    """Contextual Thompson Sampling for multi-armed bandits."""

    requires_context = True

    def __init__(self, model_ids, context_dimension, sigma, prior_variance, seed=None):
        """Initialize Thompson Sampling.

        Args:
            model_ids (list): List of model identifiers (arms).
            context_dimension (int): Dimensionality of context vectors.
            sigma (float): Standard deviation of the reward noise.
            prior_variance (float): Initial variance of the prior over theta.
            seed (int, optional): Random seed.
        """
        super().__init__(model_ids, context_dimension)
        if context_dimension is None or context_dimension <= 0:
            raise ValueError("ThompsonSampling requires a positive context_dimension.")
        if sigma <= 0:
            raise ValueError("Sigma (reward noise std dev) must be positive.")
        if prior_variance <= 0:
            raise ValueError("Prior variance must be positive.")

        self.sigma = sigma
        self.prior_variance = prior_variance
        self.lambda_reg = (self.sigma**2) / self.prior_variance
        logger.info(
            f"Thompson Sampling initialized with sigma={sigma}, prior_variance={prior_variance}, lambda_reg={self.lambda_reg}"
        )

        self.B = {}
        self.f = {}
        self.mu = {}
        self.cov = {}

        for model_id in model_ids:
            self._initialize_model(model_id)

        if seed is not None:
            np.random.seed(seed)

    def _initialize_model(self, model_id):
        """Initializes the parameters for a single model ID."""
        if model_id not in self.B:
            logger.debug(f"Initializing parameters for model: {model_id}")

            prior_precision = 1.0 / self.prior_variance

            self.B[model_id] = np.identity(self.context_dimension) * prior_precision
            self.f[model_id] = np.zeros(self.context_dimension)
            self.mu[model_id] = None
            self.cov[model_id] = None

            if not hasattr(self, "rewards"):
                self.rewards = {}
            self.rewards.setdefault(model_id, [])
        else:
            logger.warning(
                f"Attempted to re-initialize model {model_id}, which already exists."
            )

    def select_model(self, context=None):
        if context is None:
            context = np.zeros(self.context_dimension)
        context = np.asarray(context).reshape(-1)
        if context.shape[0] != self.context_dimension:
            logger.error(
                f"Context dim mismatch! Expected {self.context_dimension}, got {context.shape[0]}. Using zeros."
            )
            context = np.zeros(self.context_dimension)

        best_model, highest_reward = None, float("-inf")
        sampled_weights_all = {}
        for model_id in self.model_ids:
            if self.mu[model_id] is None or self.cov[model_id] is None:
                self._update_posterior(model_id)
            try:
                if self.mu[model_id] is None or self.cov[model_id] is None:
                    logger.error(
                        f"Posterior mean or cov is None for {model_id} even after update attempt. Using fallback."
                    )
                    weights = np.zeros(self.context_dimension)
                else:
                    weights = np.random.multivariate_normal(
                        self.mu[model_id], self.cov[model_id]
                    )
            except Exception as e:
                logger.warning(f"Sampling failed for {model_id}: {e}. Using mean.")
                weights = (
                    self.mu[model_id]
                    if self.mu[model_id] is not None
                    else np.zeros(self.context_dimension)
                )

            expected_reward = np.dot(weights, context)
            sampled_weights_all[model_id] = weights.tolist()
            if expected_reward > highest_reward:
                highest_reward, best_model = expected_reward, model_id

        if best_model is None:
            best_model = np.random.choice(self.model_ids)
        self.decisions.append({"model": best_model, "context": context.tolist()})
        return best_model

    def update(self, model_id, reward, context=None):
        if context is None:
            context = np.zeros(self.context_dimension)
        context = np.asarray(context).reshape(-1)
        if context.shape[0] != self.context_dimension:
            logger.error("Update dim mismatch! Skipping.")
            return
        try:
            outer_product = np.outer(context, context)

            sigma_sq = self.sigma**2
            if sigma_sq < 1e-9:
                sigma_sq = 1e-9
            self.B[model_id] += outer_product / sigma_sq
            self.f[model_id] += context * reward / sigma_sq
            self.mu[model_id] = None
            self.cov[model_id] = None
            self.rewards[model_id].append(reward)
        except Exception as e:
            logger.error(f"Error during update for {model_id}: {e}")

    def _update_posterior(self, model_id):
        try:

            B_inv = np.linalg.inv(self.B[model_id])

            sigma_sq = self.sigma**2
            if sigma_sq < 1e-12:
                sigma_sq = 1e-12

            if not hasattr(self, "cov"):
                self.cov = {}
            self.cov[model_id] = sigma_sq * B_inv

            self.mu[model_id] = B_inv.dot(self.f[model_id])
        except np.linalg.LinAlgError:
            logger.warning(
                f"LinAlgError during posterior update for {model_id}. Using regularized fallback."
            )
            try:
                epsilon_identity = 1e-6 * np.identity(self.context_dimension)
                B_inv_reg = np.linalg.inv(self.B[model_id] + epsilon_identity)
                sigma_sq = self.sigma**2
                if sigma_sq < 1e-12:
                    sigma_sq = 1e-12
                if not hasattr(self, "cov"):
                    self.cov = {}
                self.cov[model_id] = sigma_sq * B_inv_reg

                self.mu[model_id] = B_inv_reg.dot(self.f[model_id])
            except Exception as e_fallback:
                logger.error(
                    f"Regularized fallback posterior update failed for {model_id}: {e_fallback}"
                )
                self.mu[model_id] = np.zeros(self.context_dimension)
                if not hasattr(self, "cov"):
                    self.cov = {}
                self.cov[model_id] = np.identity(self.context_dimension)

        except Exception as e_gen:
            logger.error(
                f"General error during posterior update for {model_id}: {e_gen}",
                exc_info=True,
            )
            self.mu[model_id] = np.zeros(self.context_dimension)
            if not hasattr(self, "cov"):
                self.cov = {}
            self.cov[model_id] = np.identity(self.context_dimension)

    def reset(self):
        if self.prior_variance <= 0:

            logger.error(
                f"Invalid prior_variance ({self.prior_variance}) during reset. Using fallback."
            )
            prior_precision = 1e-6
        else:
            prior_precision = 1.0 / self.prior_variance

        for model_id in self.model_ids:
            self.B[model_id] = np.identity(self.context_dimension) * prior_precision
            self.f[model_id] = np.zeros(self.context_dimension)
            self.mu[model_id] = None

            if not hasattr(self, "cov"):
                self.cov = {}
            self.cov[model_id] = None

        self.decisions = []
        if not hasattr(self, "rewards"):
            self.rewards = {}
        self.rewards = {mid: [] for mid in self.model_ids}
        logger.info("ThompsonSampling reset complete.")

    def get_state(self):
        state = super().get_state()
        B_serializable = {
            model_id: self.B[model_id].tolist() for model_id in self.model_ids
        }
        f_serializable = {
            model_id: self.f[model_id].tolist() for model_id in self.model_ids
        }

        state.update(
            {
                "sigma": self.sigma,
                "prior_variance": self.prior_variance,
                "B": B_serializable,
                "f": f_serializable,
            }
        )

        return state

    def set_state(self, state):
        super().set_state(state)

        self.sigma = state.get("sigma", self.sigma)
        self.prior_variance = state.get("prior_variance", self.prior_variance)

        if "B" in state:
            for model_id in self.model_ids:
                if model_id in state["B"]:
                    self.B[model_id] = np.array(state["B"][model_id])

        if "f" in state:
            for model_id in self.model_ids:
                if model_id in state["f"]:
                    self.f[model_id] = np.array(state["f"][model_id])

        if not hasattr(self, "cov"):
            self.cov = {}
        for model_id in self.model_ids:
            self.mu[model_id] = None
            self.cov[model_id] = None

    def get_metrics(self):
        metrics = {
            "sigma": self.sigma,
            "decisions_count": len(self.decisions),
            "model_selection_counts": self._get_model_selection_counts(),
            "model_average_rewards": self._get_model_average_rewards(),
        }
        posterior_distribution = {}
        for model_id in self.model_ids:
            if (
                hasattr(self, "cov")
                and model_id in self.cov
                and self.mu.get(model_id) is not None
                and self.cov.get(model_id) is not None
            ):
                posterior_distribution[model_id] = {
                    "mean": self.mu[model_id].tolist(),
                    "uncertainty": float(np.trace(self.cov[model_id])),
                    "condition_number": float(np.linalg.cond(self.cov[model_id])),
                }
        metrics["posterior_distribution"] = posterior_distribution
        if hasattr(self, "sampled_weights"):
            metrics["latest_sampled_weights"] = self.sampled_weights
        feature_importance = {}
        for model_id in self.model_ids:
            if self.mu[model_id] is not None:
                importance = np.abs(self.mu[model_id])
                feature_importance[model_id] = importance.tolist()
        metrics["feature_importance"] = feature_importance

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

    def _get_uncertainty_evolution(self):
        uncertainty = {}
        for model_id in self.model_ids:
            if (
                hasattr(self, "cov")
                and model_id in self.cov
                and (self.mu.get(model_id) is None or self.cov.get(model_id) is None)
            ):
                try:
                    self._update_posterior(model_id)
                except Exception as e:
                    logger.error(f"Failed updating posterior: {e}")
                    continue
            if (
                hasattr(self, "cov")
                and model_id in self.cov
                and self.cov.get(model_id) is not None
            ):
                uncertainty[model_id] = float(np.trace(self.cov[model_id]))
            else:
                uncertainty[model_id] = None
        return uncertainty

    def add_model(self, model_id):
        """Adds a new model arm to the Thompson Sampling bandit."""
        if model_id in self.model_ids:
            logger.warning(
                f"Model '{model_id}' already exists in ThompsonSampling. No action taken."
            )
            return False

        logger.info(f"Adding new model '{model_id}' to ThompsonSampling.")
        self.model_ids.append(model_id)
        self.n_models = len(self.model_ids)
        if not hasattr(self, "cov"):
            self.cov = {}
        self._initialize_model(model_id)
        logger.info(
            f"Model pool size is now {self.n_models}. Current models: {self.model_ids}"
        )
        return True

    def remove_model(self, model_id):
        """Removes a model arm from the Thompson Sampling bandit."""
        if model_id not in self.model_ids:
            logger.warning(
                f"Model '{model_id}' not found in ThompsonSampling. Cannot remove."
            )
            return False

        if self.n_models <= 1:
            logger.error(
                f"Cannot remove model '{model_id}'. At least one model must remain."
            )
            return False

        logger.info(f"Removing model '{model_id}' from ThompsonSampling.")
        try:
            self.model_ids.remove(model_id)
            self.n_models = len(self.model_ids)
            if model_id in self.B:
                del self.B[model_id]
            if model_id in self.f:
                del self.f[model_id]
            if model_id in self.mu:
                del self.mu[model_id]
            if model_id in self.cov:
                del self.cov[model_id]
            if model_id in self.rewards:
                del self.rewards[model_id]
            if hasattr(self, "cov") and model_id in self.cov:
                del self.cov[model_id]

            logger.info(
                f"Successfully removed '{model_id}'. Model pool size is now {self.n_models}. Remaining models: {self.model_ids}"
            )
            return True
        except Exception as e:
            logger.error(f"Error removing model '{model_id}': {e}", exc_info=True)
            return False  # Indicate removal failed
