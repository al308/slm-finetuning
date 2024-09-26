# src/data_prep.py
from datasets import load_dataset

def load_and_split_data(dataset_name="HuggingFaceH4/ultrachat_200k", split_size=0.2):
    """
    Loads the dataset from Hugging Face and splits it into train and test sets.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face hub.
        split_size (float): The proportion of the dataset to include in the test split.
    
    Returns:
        train_dataset, test_dataset: Training and testing datasets.
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split='train_sft[:2%]')
    
    # Split the dataset
    print(f"Splitting the dataset with test size {split_size}...")
    dataset = dataset.train_test_split(test_size=split_size)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    return train_dataset, test_dataset

def save_datasets_to_json(train_dataset, test_dataset, train_file="data/train.jsonl", test_file="data/eval.jsonl"):
    """
    Saves the train and test datasets to jsonl format.
    
    Args:
        train_dataset: The training dataset.
        test_dataset: The testing dataset.
        train_file (str): Path to save the training dataset.
        test_file (str): Path to save the test dataset.
    """
    print(f"Saving datasets to {train_file} and {test_file}...")
    train_dataset.to_json(train_file)
    test_dataset.to_json(test_file)
