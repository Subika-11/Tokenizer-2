import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# -------------------------------
# Step 1: Load tokenizer
# -------------------------------
tokenizer = LlamaTokenizer.from_pretrained("llama_tokenizer")
tamil_tokens = [line.strip() for line in open("agathyam_tokens.txt", "r", encoding="utf-8")]
tokenizer.add_tokens(tamil_tokens)

# -------------------------------
# Step 2: Load pretrained LLaMA
# -------------------------------
model = LlamaForCausalLM.from_pretrained("llama_pretrained")

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# -------------------------------
# Step 3: Replace embeddings with augmented Tamil embeddings
# -------------------------------
augmented_embeddings = torch.load("llama_augmented_embeddings.pt")
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    model.get_input_embeddings().weight.copy_(augmented_embeddings)
print("✅ Embeddings replaced")

# -------------------------------
# Step 4: Load and tokenize dataset
# -------------------------------
dataset = load_dataset("text", data_files={"train": "tamil_corpus.txt"})
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)  # shorter seqs save RAM
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# -------------------------------
# Step 5: Data collator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# Step 6: Training arguments (CPU-optimized)
# -------------------------------
training_args = TrainingArguments(
    output_dir="./tamil_llama_cpu",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,       # tiny batch for CPU
    gradient_accumulation_steps=16,      # simulate larger batch
    learning_rate=5e-5,
    save_total_limit=2,
    logging_steps=50,
    save_steps=200,
    fp16=False,                           # CPU cannot do FP16
    dataloader_num_workers=2,             # reduce CPU overhead
    report_to="none",
)

# -------------------------------
# Step 7: Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# -------------------------------
# Step 8: Train
# -------------------------------
trainer.train()
trainer.save_model("./tamil_llama_cpu")
print("✅ CPU pretraining complete!")
