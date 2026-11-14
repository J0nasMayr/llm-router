import pandas as pd
from config.manager import load_config
from datasets import Dataset


dataset_config = load_config("datasets")


def restructure_mmlu(entry, config):
    """Convert mmlu format to standard format"""
    input1_col = config.get("input1", "")
    input2_col = config.get("input2", [])
    reference_col = config.get("reference", "")

    return {
        'input1': entry[input1_col],
        'input2': entry[input2_col],
        'reference': entry[reference_col]
    }


def restructure_hellaswag(entry, config):
    """Convert hellaswag format to standard format"""
    input1_col = config.get("input1", "")
    input2_col = config.get("input2", [])
    reference_col = config.get("reference", "")

    return {
        'input1': entry[input1_col],
        'input2': entry[input2_col],
        'reference': entry[reference_col]
    }


def restructure_winogrande(entry, config):
    """Convert winogrande format to standard format"""
    input1_col = config.get("input1", "")
    input2 = [entry[col] for col in config.get("input2", [])]
    reference_col = config.get("reference", "")
    reference = str(int(entry[reference_col]) - 1)  # Convert to 0-based index

    return {
        'input1': entry[input1_col],
        'input2': input2,
        'reference': reference
    }


def restructure_gsm8k(entry, config):
    """Convert gsm8k format to standard format"""
    input1_col = config.get("input1", "")
    input1 = entry[input1_col] + "\n\nAnswer:\n"
    reference_col = config.get("reference", "") 
    reference = entry[reference_col].split("####")[1].strip()

    return {
        'input1': input1,
        'reference': reference
    }


def restructure_cnn_dailymail(entry, config):
    """Convert cnn_dailymail format to standard format"""
    input1_col = config.get("input1", "")
    input1 = entry[input1_col] + "\n\nSummary:\n"
    reference_col = config.get("reference", "")

    return {
        'input1': input1,
        'reference': entry[reference_col]
    }


def preprocess_dataset(dataset_name, dataset):
    """Apply dataset-specific preprocessing"""
    print("Preprocessing dataset: ", dataset_name)
    print("First entry:", dataset[0])
    
    # Create a new list to hold processed entries
    processed_entries = []
    config = dataset_config[dataset_name]["columns"]

    if dataset_name == "hellaswag":
        for entry in dataset:
            processed_entries.append(restructure_hellaswag(entry, config))
    elif dataset_name == "winogrande":
        for entry in dataset:
            processed_entries.append(restructure_winogrande(entry, config))
    elif dataset_name == "gsm8k":
        for entry in dataset:
            processed_entries.append(restructure_gsm8k(entry, config))
    elif dataset_name == "mmlu":
        for entry in dataset:
            processed_entries.append(restructure_mmlu(entry, config))
    elif dataset_name == "cnn_dailymail":
        for entry in dataset:
            processed_entries.append(restructure_cnn_dailymail(entry, config))
    else:
        raise ValueError(f"No preprocessing function for {dataset_name} found")

    # Convert processed entries to a new dataset
    new_dataset = Dataset.from_list(processed_entries)
    
    print("First entry after preprocessing:", new_dataset[0])
    # Default: return as-is
    print("Processed Dataset: ", new_dataset)
    return new_dataset
