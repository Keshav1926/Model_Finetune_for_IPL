#!/usr/bin/env python3
# /workspace/train_lora.py
import os
import subprocess
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import get_last_checkpoint

# ---------------- Config (tweak via env if you like) ----------------
MODEL_ID = os.getenv("MODEL_ID", "microsoft/phi-3-mini-4k-instruct")
TOKENIZED_DS_DIR = os.getenv("TOKENIZED_DS_DIR", "/workspace/tokenized_dataset")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACC = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
LR = float(os.getenv("LEARNING_RATE", "2e-4"))
DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "8"))

# ---------------- helpers ----------------
def get_gpu_free_mb(gpu=0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        )
        free = int(out.decode().strip().split("\n")[gpu].strip())
        return free
    except Exception:
        return 0

# ---------------- load tokenized dataset ----------------
print("Loading tokenized dataset from:", TOKENIZED_DS_DIR)
train_dataset = load_from_disk(TOKENIZED_DS_DIR)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print("Rows:", len(train_dataset))

# ---------------- tokenizer & collator ----------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------- BitsAndBytes (QLoRA) config ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ---------------- determine memory map ----------------
free_mb = get_gpu_free_mb(0)
target = max(8000, max(0, free_mb - 1500))
max_memory = {0: f"{int(target)}MB", "cpu": "60000MB"}
print("Loading model with max_memory:", max_memory)

# ---------------- load model ----------------
print("Loading model (QLoRA 4-bit)... this may take a minute")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# ---------------- Compatibility tweaks (non-invasive) ----------------
# Disable attention cache to avoid calling DynamicCache.get_usable_length
# and force 'eager' attention implementation (avoids flash-attn/window-size checks).
print("Applying compatibility flags: disabling use_cache, setting attn_implementation='eager'")
try:
    # these settings are read by model forward and the model code
    model.config.use_cache = False
except Exception as e:
    print("Warning: could not set model.config.use_cache:", e)
try:
    model.config.attn_implementation = "eager"
except Exception:
    # if attribute doesn't exist, ignore
    pass

# Prepare model for k-bit training (PEFT helper)
model = prepare_model_for_kbit_training(model)

# If GPU has ample free memory, disable gradient checkpointing (speed)
if free_mb > 9000:
    try:
        model.gradient_checkpointing_disable()
        print("Disabled gradient checkpointing for speed (enough GPU memory).")
    except Exception:
        pass

# ---------------- LoRA (r=4, minimal modules) ----------------
print("Applying LoRA adapters (r=4) on q/k/v/o ...")
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------- TrainingArguments & Trainer ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    save_strategy="steps",
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# ---------------- Resume logic ----------------
print("Checking for existing checkpoints in:", OUTPUT_DIR)
last_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt:
        print("Found checkpoint:", last_ckpt)
    else:
        print("No checkpoint found; starting fresh.")
else:
    print("Output dir not present; starting fresh.")

# ---------------- Train (resume if checkpoint found) ----------------
print("Starting training (trainer will resume if checkpoint is present)...")
trainer.train(resume_from_checkpoint=last_ckpt)

# ---------------- Save final adapter ----------------
final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
os.makedirs(final_dir, exist_ok=True)
print("Saving final adapter to:", final_dir)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print("Done. Adapter saved at", final_dir)

