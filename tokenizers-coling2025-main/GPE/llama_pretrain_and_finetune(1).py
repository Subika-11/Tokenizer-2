import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
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
# Step 3: Load pretraining dataset
# -------------------------------
dataset = load_dataset("text", data_files={"train": "tamil_corpus.txt"})
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# Step 4: Pretraining arguments (CPU)
# -------------------------------
output_dir = "./tamil_llama_cpu"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# -------------------------------
# Step 5: Resume checkpoint if exists
# -------------------------------
last_checkpoint = None
if os.path.isdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=os.path.getctime)[-1]
        print(f"Resuming from checkpoint: {last_checkpoint}")

# -------------------------------
# Step 6: Start pretraining
# -------------------------------
trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model(output_dir)
print("✅ CPU pretraining complete!")

# -------------------------------
# Step 7: Downstream fine-tuning (example: classification)
# -------------------------------
# Assume you have a dataset with 'text' and 'label' columns
task_dataset = load_dataset("csv", data_files={"train": "task_train.csv", "test": "task_test.csv"})

def tokenize_task(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
tokenized_task = task_dataset.map(tokenize_task, batched=True)

# Create classification model using the pretrained weights as backbone
num_labels = len(set(task_dataset["train"]["label"]))
classification_model = AutoModelForSequenceClassification.from_pretrained(
    output_dir,
    num_labels=num_labels
)

task_training_args = TrainingArguments(
    output_dir="./tamil_llama_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    save_total_limit=2,
    logging_steps=50,
    fp16=False,
    dataloader_num_workers=2,
    report_to="none",
)

task_trainer = Trainer(
    model=classification_model,
    args=task_training_args,
    train_dataset=tokenized_task["train"],
    eval_dataset=tokenized_task["test"],
    tokenizer=tokenizer,
)

task_trainer.train()
task_trainer.save_model("./tamil_llama_finetuned")
print("✅ Fine-tuning complete! Model ready for downstream tasks.")
