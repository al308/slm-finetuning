# src/model_setup.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

def load_model_and_tokenizer(checkpoint_path="microsoft/Phi-3-mini-4k-instruct", use_flash_attention=True):
    """
    Loads the pretrained Phi-3 model and tokenizer, with optional flash attention.

    Args:
        checkpoint_path (str): The model checkpoint path on Hugging Face.
        use_flash_attention (bool): If True, use Flash Attention (requires GPU support).

    Returns:
        model, tokenizer: The model and tokenizer objects.
    """
    print("Loading model and tokenizer...")
    
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        torch_dtype="bfloat16",
        device_map="auto"
    )
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Set tokenizer settings
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    
    print("Model and tokenizer loaded.")
    return model, tokenizer

def apply_lora_config(r=16, alpha=32, dropout=0.05):
    """
    Sets up LoRA (Low-Rank Adapters) configuration for fine-tuning.

    Args:
        r (int): LoRA rank (default 16).
        alpha (int): LoRA alpha (default 32).
        dropout (float): LoRA dropout rate (default 0.05).

    Returns:
        LoraConfig: The LoRA configuration object.
    """
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        modules_to_save=None,
    )
    print("LoRA configuration set.")
    return lora_config
