# src/config.py

# Model Checkpoint (Phi-3)
MODEL_CHECKPOINT = "microsoft/Phi-3-mini-4k-instruct"

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Dataset split ratio
TEST_SPLIT_SIZE = 0.2

# Output directories
TRAIN_FILE = "data/train.jsonl"
EVAL_FILE = "data/eval.jsonl"
OUTPUT_DIR = "./checkpoint_dir"
