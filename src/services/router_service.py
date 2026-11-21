from src.bandit.epsilon_greedy import ContextualEpsilonGreedy, EpsilonGreedy
from src.bandit.linucb import LinUCB
from src.bandit.thompson_sampling import ThompsonSampling


class RouterService:
    """Service for selecting models using MAB algorithms"""

    ALGORITHM_TO_CLASS_MAP = {
        "epsilon_greedy": EpsilonGreedy,
        "contextual_epsilon_greedy": ContextualEpsilonGreedy,
        "linucb": LinUCB,
        "thompson_sampling": ThompsonSampling,
    }

    def __init__(self, algorithm_name, algorithm_params, model_ids, context_dimension):
        self.bandit = self.ALGORITHM_TO_CLASS_MAP[algorithm_name](
            model_ids, context_dimension=context_dimension, **algorithm_params
        )

    def select_model(self, features, available_models=None):
        """Select a model for the features, among `available_models` only
        (all models if None)."""
        return self.bandit.select_model(
            features["context_vector"], available_models=available_models
        )

    def preferred_model(self):
        """The model the bandit would have chosen in its last selection if every
        model had been available, or None if the bandit does not report it."""
        return getattr(self.bandit, "last_preferred_model", None)

    def update(self, features, model_id, reward):
        """Update the bandit algorithm with the observed reward.

        Extracts the numeric context vector from the features dict, mirroring
        `select_model` — the bandit must see the same context in both calls.
        """
        self.bandit.update(model_id, reward, context=features["context_vector"])
