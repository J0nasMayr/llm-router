import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

CONFIG_FILES = {
    "models": "models.yaml",
    "datasets": "datasets.yaml",
    "mab": "mab.yaml",
    "metrics": "metrics.yaml",
    "baselines": "baselines.yaml",
    "feature_extraction": "feature_extraction.yaml",
    "experiments": "experiments.yaml",
}


def load_config(*config_keys, config_dir=None):
    """
    Load specified configuration files.

    Args:
        *config_keys (str): Keys identifying which configs to load (e.g., 'models', 'datasets').
                            If empty, attempts to load all known configs.
        config_dir (Path or str, optional): Directory containing config files.
                                             Defaults to project's config/ directory.

    Returns:
        dict: A dictionary where keys are config names (e.g., 'models')
              and values are the loaded config dictionaries. Returns {} on error.
    """
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    else:
        config_dir = Path(config_dir)

    if not config_dir.is_dir():
        logger.error(f"Configuration directory not found: {config_dir}")
        return {}

    configs_to_load = config_keys if config_keys else CONFIG_FILES.keys()
    loaded_configs = {}

    for key in configs_to_load:
        logger.debug(f"Loading config '{key}' from {config_dir}")
        if key not in CONFIG_FILES:
            logger.warning(f"Unknown config key '{key}'. Skipping.")
            continue

        file_path = config_dir / CONFIG_FILES[key]
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    loaded_configs[key] = yaml.safe_load(f)
                    logger.info(f"Loaded config '{key}' from {file_path}")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {file_path}: {e}")
                loaded_configs[key] = None
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                loaded_configs[key] = None
        else:
            logger.warning(f"Config file not found for key '{key}': {file_path}")
            loaded_configs[key] = None

    return loaded_configs


def setup_experiment_dirs(experiment_name: str, base_dir: str = "experiments"):
    """
    Creates directories for experiment results and plots if they don't exist.

    Args:
        experiment_name (str): The name of the experiment (e.g., 'a1_static_sanity_check').
        base_dir (str): The base directory containing the experiment subdirectories.

    Returns:
        dict: A dictionary containing Path objects for 'base', 'results', and 'plots' directories.
              Returns None if the base experiment directory doesn't exist.
    """
    base_path = Path(base_dir) / experiment_name
    if not base_path.is_dir():
        logger.error(f"Base experiment directory not found: {base_path}")

        return None

    results_path = base_path / "results"
    plots_path = base_path / "plots"

    results_path.mkdir(exist_ok=True)
    plots_path.mkdir(exist_ok=True)

    return {"base": base_path, "results": results_path, "plots": plots_path}
