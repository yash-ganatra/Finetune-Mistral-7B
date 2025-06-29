from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# === Config ===
MODEL_PATH = "./Mistral-7B-Instruct-v0.2"  
DATA_PATH = "final_finetune_prompts.jsonl"       
OUTPUT_DIR = "./mistral7b-finetuned"

print("=" * 60)
print("MISTRAL 7B FINE-TUNING SCRIPT")
print("=" * 60)

# === Step 1: Load and Prepare Dataset ===
print("Step 1: Preparing dataset...")

# === Step 2: Load Tokenizer and Model ===
print("Step 2: Loading Model....")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

data = load_dataset("json", data_files=DATA_PATH)["train"]

def format_and_tokenize(example):
    # Format the text
    formatted_text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    
    # Tokenize and return only the tokenized fields
    tokenized = tokenizer(
        formatted_text, 
        padding=False, 
        truncation=True, 
        max_length=512,  # Add max_length to prevent extremely long sequences
        return_tensors=None  # Keep as lists, not tensors
    )
    
    # Set labels to be the same as input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Apply formatting and tokenization, removing original columns
data = data.map(
    format_and_tokenize,
    remove_columns=data.column_names,  # Remove original columns to avoid conflicts
    batched=False
)

print(f"Dataset size: {len(data)} examples")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

# === Step 3: Apply QLoRA Configuration ===
print("Step 3: Applying QLoRA Config")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# === Step 4: Set Training Arguments ===
print('Step 4: Setting training Arguments')
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_32bit",
    remove_unused_columns=False,
    dataloader_drop_last=True,  # Drop incomplete batches
    report_to=None,  # Disable wandb/tensorboard logging
)

# === Step 5: Fine-Tune ===
print("Step 5: Starting Fine-tuning...")
print("This may take a while depending on your dataset size and hardware...")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8  # Optional: pad to multiple of 8 for better performance
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=data_collator
)

trainer.train()

# === Step 6: Save Fine-Tuned Model ===
print("Step 6: Saving Fine-Tuned Model....")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print(f"✅ Model saved to: {OUTPUT_DIR}")
print("=" * 60)
print("Next steps:")
print("1. Restart your Python environment to clear GPU memory")
print("2. Run 'python inference_mistral.py' to test your fine-tuned model")
print("=" * 60)