import logging
import os

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(name):
    """Load config/<name>.yaml; an empty file loads as {}."""
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    logger.debug(f"Loading config from {path}")
    with open(path) as f:
        # An empty file (or one with only comments) loads as None.
        return yaml.safe_load(f) or {}
