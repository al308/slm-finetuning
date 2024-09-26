# train.py
from src.data_prep import load_and_split_data, save_datasets_to_json
from src.model_setup import load_model_and_tokenizer, apply_lora_config
from src.training import get_training_args, train_model
from src.config import *

def main():
    # Step 1: Load and preprocess data
    train_dataset, test_dataset = load_and_split_data(split_size=TEST_SPLIT_SIZE)
    save_datasets_to_json(train_dataset, test_dataset, TRAIN_FILE, EVAL_FILE)
    
    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=MODEL_CHECKPOINT)
    
    # Step 3: Set up LoRA configuration
    peft_config = apply_lora_config(r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    
    # Step 4: Get training arguments
    training_args = get_training_args(output_dir=OUTPUT_DIR, num_epochs=1)
    
    # Step 5: Start training the model
    train_model(model, tokenizer, train_dataset, test_dataset, training_args, peft_config)

if __name__ == "__main__":
    main()
