# experiments/04_hyperparameter_tuning/utils.py
import itertools
import logging

logger = logging.getLogger(__name__)


def generate_hyperparameter_configs(base_params: dict):
    """
    Generates specific hyperparameter configurations from a base dictionary
    containing lists of values for parameters to be tuned.

    Args:
        base_params (dict): The algorithm's parameter dictionary from the config,
                          where keys to be tuned have lists as values.
                          Example: {'alpha': [0.1, 1.0], 'regularization': [0.1]}

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents
                    a unique combination of hyperparameter values.
                    Example output for above:
                    [
                        {'alpha': 0.1, 'regularization': 0.1},
                        {'alpha': 1.0, 'regularization': 0.1}
                    ]
    """

    if not base_params:
        return [{}]
    params_to_tune = {}
    fixed_params = {}
    for key, value in base_params.items():
        if isinstance(value, list):
            params_to_tune[key] = value
        else:
            fixed_params[key] = value

    if not params_to_tune:
        return [fixed_params]
    param_names = list(params_to_tune.keys())
    value_combinations = list(itertools.product(*params_to_tune.values()))

    configs = []
    for combo in value_combinations:
        specific_config = dict(zip(param_names, combo))
        specific_config.update(fixed_params)
        configs.append(specific_config)

    logger.debug(
        f"Generated {len(configs)} hyperparameter configurations from base: {base_params}"
    )
    return configs
