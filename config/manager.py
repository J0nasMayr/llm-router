# src/config/manager.py
import yaml
import os

def load_config(name):
    """Load config file, return empty dict if it fails or the file is empty"""
    path = os.path.join('config', f'{name}.yaml')
    print(f"Loading config from {path}")
    try:
        with open(path, 'r') as f:
            # An empty file (or one with only comments) loads as None.
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config {path}: {e}")
        return {}
