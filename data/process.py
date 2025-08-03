import re
import os
import sys
import pandas as pd
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.manager import load_config
from data.load import load_datasets
from data.helper import preprocess_dataset

class DataProcessor:
    """Process datasets according to configuration"""
    
    def __init__(self, dataset_name):
        self.name = dataset_name
        config = load_config("datasets")
        self.config = config["datasets"].get(dataset_name, {})
        self.columns = self.config.get("columns", {})
        self.preprocessing = self.config.get("preprocessing", {})
        
    def process_example(self, example):
        """Process a single example to standardized format"""
        processed = {
            "dataset": self.name,
        }
        
        input_text = example["input1"]
        if "input1_prefix" in self.preprocessing:
            input_text = self.preprocessing["input1_prefix"] + input_text
            
        if "input2" in self.columns:
            input2 = example["input2"]
            
            if "input2_prefix" in self.preprocessing:
                input2_text = self.preprocessing["input2_prefix"]
                
                if isinstance(input2, list):
                    for i, choice in enumerate(input2):
                        option = chr(65 + i)  
                        input2_text += f"{option}. {choice}\n"
                else:
                    input2_text += str(input2)
                    
                input_text += "\n" + input2_text

        if "output_prefix" in self.preprocessing:
            input_text += self.preprocessing["output_prefix"]
        
        processed["processed_input"] = input_text
        reference = example["reference"]
            
        if self.preprocessing.get("reference_mapping") == "index_to_letter":
            try:
                if isinstance(reference, int) or (isinstance(reference, str) and reference.isdigit()):
                    idx = int(reference)
                    reference = chr(65 + idx)  
            except (ValueError, TypeError):
                pass
                
        processed["reference"] = reference
        
        generation = self.config.get("generation", {})
        
        processed["max_new_tokens"] = generation.get("max_new_tokens", 100)
        processed["stop_sequence"] = str(generation.get("stop_sequence", ["\n"]))
        
        extraction = self.config.get("extraction", {})
        extraction_method = extraction.get("method", "raw")

        if extraction_method == "regex":
            extraction_method = f"{extraction_method}_{extraction.get('pattern', '')}"
            
        processed["extraction_method"] = extraction_method
        
        processed["evaluation_metric"] = self.config.get("evaluation_metric", "exact_match")
        
        return processed
        
    def process_dataset(self, dataset, n_samples=None, seed=None):
        """Process dataset and convert to DataFrame"""        
        if seed is None:
            general_config = load_config("general")
            seed = general_config.get("random_seed", 40)
        
        if n_samples is not None and n_samples < len(dataset):
            random.seed(seed)
            indices = random.sample(range(len(dataset)), n_samples)
            dataset = dataset.select(indices)
        
        # Process all examples
        processed_data = []
        for i, example in enumerate(dataset):
            processed = self.process_example(example)
            processed["id"] = f"{self.name}_{i+1}"
            processed_data.append(processed)
        
        # Convert to DataFrame
        return pd.DataFrame(processed_data)
    
    def save_to_csv(self, df, output_dir=None):
        """Save processed dataset to CSV"""
        if output_dir is None:
            general_config = load_config("general")
            output_dir = general_config.get("preprocessed_dir", "preprocessed")
            
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} samples to {output_path}")
        return output_path
        
    # def extract_answer(self, prediction):
    #     """Extract answer from model prediction using configured method"""
    #     extraction = self.config.get("extraction", {})
    #     method = extraction.get("method", "raw")
        
    #     if method == "multiple_choice":
    #         pattern = extraction.get("pattern", "([A-D])")
    #         match = re.search(pattern, prediction)
    #         return match.group(1) if match else None
            
    #     elif method == "regex":
    #         pattern = extraction.get("pattern", "(.+)")
    #         match = re.search(pattern, prediction)
    #         return match.group(1) if match else None
        
    #     elif method == "raw":
    #         return prediction.strip()
            
    #     return prediction.strip()


def main():
    """Download, process and save datasets as CSV"""
    parser = argparse.ArgumentParser(description="Process datasets into CSV files")
    parser.add_argument("--datasets", nargs="*", help="Specific datasets to process")
    parser.add_argument("--samples", type=int, help="Number of samples per dataset")
    
    args = parser.parse_args()
    
    # Get config
    general_config = load_config("general")
    samples_per_dataset = args.samples or general_config.get("samples_per_dataset", 500)
    output_dir = general_config.get("preprocessed_dir", "data/preprocessed")
    
    # Load datasets from HuggingFace
    datasets = load_datasets(args.datasets)
    
    all_data = []
    # Process each dataset
    for name, dataset in datasets.items():
        print(f"Processing {name}...")
        
        # Apply dataset-specific preprocessing
        preprocessed_dataset = preprocess_dataset(name, dataset)
        
        processor = DataProcessor(name)
        
        # Process dataset
        df = processor.process_dataset(
            preprocessed_dataset, 
            n_samples=samples_per_dataset
        )
        print("df:", df.head())
        # Save individual dataset
        processor.save_to_csv(df, output_dir)
        all_data.append(df)
    
    # Create combined CSV with all samples
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(output_dir, "combined.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined dataset with {len(combined_df)} samples to {output_path}")

if __name__ == "__main__":
    main()