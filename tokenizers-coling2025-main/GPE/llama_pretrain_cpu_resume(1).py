import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

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
model.gradient_checkpointing_enable()  # memory efficient
augmented_embeddings = torch.load("llama_augmented_embeddings.pt")
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    model.get_input_embeddings().weight.copy_(augmented_embeddings)
print("✅ Embeddings replaced")

# -------------------------------
# Step 3: Load dataset
# -------------------------------
dataset = load_dataset("text", data_files={"train": "tamil_corpus.txt"})
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# Step 4: Training arguments (CPU)
# -------------------------------
output_dir = "./tamil_llama_cpu"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,          # important to keep checkpoints
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    save_total_limit=5,
    logging_steps=50,
    save_steps=200,
    fp16=False,
    dataloader_num_workers=2,
    report_to="none",
)

# -------------------------------
# Step 5: Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# -------------------------------
# Step 6: Resume checkpoint if exists
# -------------------------------
last_checkpoint = None
if os.path.isdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=os.path.getctime)[-1]  # latest checkpoint
        print(f"Resuming from checkpoint: {last_checkpoint}")

# -------------------------------
# Step 7: Start training
# -------------------------------
trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model(output_dir)
print("✅ CPU pretraining complete (with checkpoint support)!")
