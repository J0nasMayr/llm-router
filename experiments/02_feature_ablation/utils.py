# --- Imports ---
import logging

import numpy as np

from src.bandit.epsilon_greedy import (
    ContextualEpsilonGreedy,
    EpsilonGreedy,
    LinearEpsilonGreedy,
)
from src.bandit.linucb import LinUCB
from src.bandit.thompson_sampling import ThompsonSampling

logger = logging.getLogger(__name__)


def create_bandit_instance(
    algo_name: str,
    algo_params: dict,
    model_ids: list,
    context_dimension: int,
    reg_lambda: float,
    seed: int,
):
    """
    Factory function to create a bandit instance based on configuration.

    Args:
        algo_name (str): Name of the algorithm (e.g., 'linear_epsilon_greedy').
        algo_params (dict): Dictionary of parameters specific to the algorithm.
        model_ids (list): List of model IDs (arms).
        context_dimension (int): The context dimension required for this instance.
        reg_lambda (float): Regularization lambda (used by linear models).
        seed (int): Random seed for the run.

    Returns:
        An initialized bandit instance (subclass of BaseBandit) or None if error.
    """

    algo_name_lower = algo_name.lower()
    logger.debug(
        f"Attempting to create bandit: {algo_name_lower} with params: {algo_params}, dim: {context_dimension}, lambda: {reg_lambda}, seed: {seed}"
    )

    try:
        if algo_name_lower == "linear_epsilon_greedy":
            return LinearEpsilonGreedy(
                model_ids=model_ids,
                context_dimension=context_dimension,
                initial_epsilon=algo_params["initial_epsilon"],
                decay_factor=algo_params["decay_factor"],
                min_epsilon=algo_params["min_epsilon"],
                lambda_=reg_lambda,
                seed=seed,
            )
        elif algo_name_lower == "contextual_epsilon_greedy":

            return ContextualEpsilonGreedy(
                model_ids=model_ids,
                context_dimension=context_dimension,
                initial_epsilon=algo_params["initial_epsilon"],
                decay_factor=algo_params["decay_factor"],
                min_epsilon=algo_params["min_epsilon"],
                seed=seed,
            )
        elif algo_name_lower == "epsilon_greedy":
            return EpsilonGreedy(
                model_ids=model_ids,
                initial_epsilon=algo_params["initial_epsilon"],
                decay_factor=algo_params["decay_factor"],
                min_epsilon=algo_params["min_epsilon"],
                seed=seed,
            )
        elif algo_name_lower == "linucb":
            return LinUCB(
                model_ids=model_ids,
                context_dimension=context_dimension,
                alpha=algo_params["alpha"],
                regularization=algo_params["regularization"],
                seed=seed,
            )
        elif algo_name_lower == "thompson_sampling":
            return ThompsonSampling(
                model_ids=model_ids,
                context_dimension=context_dimension,
                sigma=algo_params["sigma"],
                prior_variance=algo_params["prior_variance"],
                seed=seed,
            )
        else:
            logger.error(f"Unsupported algorithm name for A3 experiment: {algo_name}")
            return None
    except KeyError as e:
        logger.error(
            f"Missing required parameter '{e}' for algorithm '{algo_name}' in config."
        )
        return None
    except Exception as e:
        logger.error(f"Error creating bandit instance {algo_name}: {e}", exc_info=True)
        return None


def get_bandit_parameters(algo_instance):
    """
    Extracts the learned linear parameters (theta estimates) from various bandit types.

    Args:
        algo_instance: An instance of a bandit algorithm (subclass of BaseBandit).

    Returns:
        dict: A dictionary mapping model_id to its learned parameter vector (numpy array),
              or None if parameters are not applicable or calculable.
    """

    algo_name = algo_instance.__class__.__name__
    logger.debug(f"Attempting to get parameters for bandit type: {algo_name}")
    params = None
    try:
        if isinstance(algo_instance, LinearEpsilonGreedy):
            params = algo_instance.get_theta()
            logger.debug("Extracted theta from LinearEpsilonGreedy.")

        elif isinstance(algo_instance, LinUCB):
            if hasattr(algo_instance, "get_theta") and callable(
                getattr(algo_instance, "get_theta")
            ):
                params = algo_instance.get_theta()
                logger.debug("Extracted theta from LinUCB using get_theta().")
            else:
                logger.warning(
                    "LinUCB instance does not have the expected get_theta() method. Cannot extract parameters."
                )
                params = None

        elif isinstance(algo_instance, ThompsonSampling):
            params = {}
            for mid in algo_instance.model_ids:
                if algo_instance.mu[mid] is None:
                    algo_instance._update_posterior(mid)

                if algo_instance.mu[mid] is not None:
                    params[mid] = algo_instance.mu[mid]
                else:
                    logger.warning(
                        f"Could not calculate posterior mean for {mid} in ThompsonSampling."
                    )
                    params[mid] = np.zeros(algo_instance.context_dimension)
            logger.debug("Extracted posterior means (mu) from ThompsonSampling.")

        else:
            logger.info(
                f"Parameter extraction not implemented for algorithm type: {algo_name}"
            )
    except AttributeError as e:
        logger.error(f"AttributeError getting parameters for {algo_name}: {e}")
        params = None
    except Exception as e:
        logger.error(
            f"Unexpected error getting parameters for {algo_name}: {e}", exc_info=True
        )
        params = None
    if not isinstance(params, dict) and params is not None:
        logger.warning(
            f"Parameter extraction for {algo_name} returned unexpected type {type(params)}. Returning None."
        )
        return None

    return params


def calculate_performance_gaps(learned_params, unique_contexts, model_ids):
    """
    Calculates the performance gap (best model prediction - average prediction)
    for each unique context vector based on learned parameters.

    Args:
        learned_params (dict): Dict mapping model_id to learned parameter vector.
        unique_contexts (iterable): An iterable (set, list) of unique context vector tuples.
        model_ids (list): List of all model IDs.

    Returns:
        list: A list of dictionaries, each containing 'context', 'best_model',
              'best_pred', 'avg_pred', and 'gap' for a unique context.
              Returns an empty list if learned_params is invalid.
    """

    if not learned_params or not isinstance(learned_params, dict):
        logger.warning(
            "Invalid or empty learned_params provided to calculate_performance_gaps."
        )
        return []
    gap_results = []
    logger.debug(
        f"Calculating performance gaps for {len(unique_contexts)} unique contexts..."
    )

    for context_tuple in unique_contexts:
        context_vector = np.array(context_tuple)
        predictions = {}
        valid_models = 0
        for model_id in model_ids:
            if model_id in learned_params:
                theta_vector = learned_params[model_id]
                if (
                    theta_vector is not None
                    and theta_vector.shape[0] == context_vector.shape[0]
                ):
                    pred = context_vector @ theta_vector
                    predictions[model_id] = pred
                    valid_models += 1
                else:
                    logger.warning(
                        f"Shape mismatch or None params for {model_id}. Context: {context_vector.shape}, Theta: {theta_vector.shape if theta_vector is not None else 'None'}"
                    )
                    predictions[model_id] = -np.inf
            else:
                logger.warning(
                    f"Parameters missing for model {model_id} in learned_params dict."
                )
                predictions[model_id] = -np.inf
        if valid_models == 0:
            logger.warning(
                f"No valid model predictions for context: {context_tuple}. Skipping gap calculation."
            )
            continue
        best_model = max(predictions, key=predictions.get)
        best_model_pred = predictions[best_model]
        valid_preds = [p for p in predictions.values() if p > -np.inf]
        avg_pred = np.mean(valid_preds) if valid_preds else 0.0

        gap = best_model_pred - avg_pred

        gap_results.append(
            {
                "context": context_tuple,
                "best_model": best_model,
                "best_pred": best_model_pred,
                "avg_pred": avg_pred,
                "gap": gap,
            }
        )

    return gap_results
