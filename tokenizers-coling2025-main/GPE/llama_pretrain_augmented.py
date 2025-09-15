import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# -------------------------------
# Step 1: Load tokenizer
# -------------------------------
tokenizer = LlamaTokenizer.from_pretrained("llama_tokenizer")
tokenizer.add_tokens([line.strip() for line in open("agathyam_tokens.txt", "r", encoding="utf-8")])

# -------------------------------
# Step 2: Load pretrained LLaMA
# -------------------------------
model = LlamaForCausalLM.from_pretrained("llama_pretrained")
print("Original vocab size:", model.get_input_embeddings().num_embeddings)

# -------------------------------
# Step 3: Replace embeddings
# -------------------------------
augmented_embeddings = torch.load("llama_augmented_embeddings.pt")
model.resize_token_embeddings(len(tokenizer))  # ensure vocab size matches
with torch.no_grad():
    model.get_input_embeddings().weight.copy_(augmented_embeddings)

print("✅ Embeddings replaced with augmented Tamil embeddings")

# -------------------------------
# Step 4: Load training corpus
# -------------------------------
# Example: text file or JSONL
dataset = load_dataset("text", data_files={"train": "tamil_corpus.txt"})
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
)

# -------------------------------
# Step 5: Data collator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# Step 6: Training setup
# -------------------------------
training_args = TrainingArguments(
    output_dir="./tamil_llama_augmented",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,   # adjust for your GPU/CPU memory
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    save_total_limit=2,
    fp16=False,                      # set True if using GPU with FP16
    logging_steps=100,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# -------------------------------
# Step 7: Start pretraining
# -------------------------------
trainer.train()
trainer.save_model("./tamil_llama_augmented")

print("✅ Pretraining complete!")
