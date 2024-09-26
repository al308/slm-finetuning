# src/training.py
from transformers import TrainingArguments
from trl import SFTTrainer
import logging

def get_training_args(output_dir="./checkpoint_dir", logging_steps=20, num_epochs=1):
    """
    Sets up training arguments for Hugging Face Trainer.

    Args:
        output_dir (str): The directory to store model checkpoints.
        logging_steps (int): Number of steps before logging metrics.
        num_epochs (int): Number of epochs for fine-tuning.

    Returns:
        TrainingArguments: The configuration for the Trainer.
    """
    print("Setting up training arguments...")
    
    training_args = TrainingArguments(
        bf16=True,
        do_eval=False,
        learning_rate=5e-06,
        log_level="info",
        logging_steps=logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        max_steps=-1,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        remove_unused_columns=True,
        save_steps=100,
        save_total_limit=1,
        seed=0,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        warmup_ratio=0.2,
    )
    print("Training arguments set.")
    return training_args

def train_model(model, tokenizer, train_dataset, test_dataset, training_args, peft_config):
    """
    Trains the model using SFTTrainer with the provided training arguments and PEFT (LoRA) configuration.

    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        train_dataset: The dataset to train the model on.
        test_dataset: The dataset to evaluate the model on.
        training_args: TrainingArguments object for the Trainer.
        peft_config: The LoRA configuration for fine-tuning.
    """
    print("Starting training...")

    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    # Start training
    train_result = trainer.train()
    metrics = train_result.metrics
    print(f"Training metrics: {metrics}")

    # Save the model and metrics
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Training finished.")
