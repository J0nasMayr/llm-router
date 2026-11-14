# src/services/router_service.py
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

    def __init__(self, algorithm_name, model_ids, algorithm_params, context_dimension):
        self.bandit = self.ALGORITHM_TO_CLASS_MAP[algorithm_name](
            model_ids, context_dimension=context_dimension, **algorithm_params
        )

    def select_model(self, features, available_models=None):
        """Select model based on features, optionally constrained to a subset of arms.

        available_models: iterable of model_ids that are physically viable right now
            (e.g. surviving the orchestrator's resource-state filter). Bandits that
            support masking will set scores of excluded arms to -inf; if a bandit
            ignores the kwarg, behavior reduces to the unmasked select.
        """
        context = features.get("context_vector") if features else None
        try:
            return self.bandit.select_model(context, available_models=available_models)
        except TypeError:
            return self.bandit.select_model(context)

    def preferred_model(self):
        """The model the bandit would have chosen in its last selection if every
        model had been available, or None if the bandit does not report it."""
        return getattr(self.bandit, "last_preferred_model", None)

    def update(self, features, model_id, reward):
        """Update the bandit algorithm with the observed reward.

        Extracts the numeric context vector from the features dict, mirroring
        `select_model` — the bandit must see the same context in both calls.
        """
        context = features.get("context_vector") if features else None
        self.bandit.update(model_id, reward, context=context)
