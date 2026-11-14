# data/load.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import load_config
from datasets import load_dataset

def load_datasets(dataset_names=None):
    """Load datasets from HuggingFace based on config"""
    config = load_config("datasets")
    print(config)
    
    # Use all datasets if none specified
    if dataset_names is None:
        dataset_names = list(config.keys())
    
    loaded_datasets = {}
    for name in dataset_names:
        if name not in config:
            print(f"Dataset {name} not found in config")
            continue
            
        dataset_config = config[name]
        load_params = dataset_config.get("load_params", {})
        split = dataset_config.get("split", "")
        try:
            dataset = load_dataset(**load_params)
            dataset = dataset[split] if split else dataset
            loaded_datasets[name] = dataset
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    return loaded_datasets


if __name__ == "__main__":
    load_datasets(["hellaswag"])