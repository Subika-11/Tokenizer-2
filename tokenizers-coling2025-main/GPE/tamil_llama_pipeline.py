# tamil_llama_pipeline.py
"""
Full pipeline: Pretraining + Fine-tuning for LLaMA-Tamil

Requirements:
- Python >= 3.10
- pip install torch transformers datasets sentencepiece python-Levenshtein
- (Optional for CPU speedups) pip install accelerate

Data files:
- tamil_corpus.txt: pretraining corpus
- agathyam_tokens.txt: mined Tamil tokens
- task_train.csv & task_test.csv: downstream task (with text and label columns)
"""

import torch
from transformers import (
    GemmaForCausalLM,
    GemmaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
import os

# -------------------------------
# Step 1: Load tokenizer and add Tamil tokens
# -------------------------------
tokenizer = GemmaTokenizer.from_pretrained("gemma-3-270m")
tamil_tokens = [line.strip() for line in open("agathyam_tokens.txt", "r", encoding="utf-8")]
tokenizer.add_tokens(tamil_tokens)

# -------------------------------
# Step 2: Load pretrained LLaMA and replace embeddings
# -------------------------------
model = GemmaForCausalLM.from_pretrained("gemma-3-270m")
model.gradient_checkpointing_enable()
augmented_embeddings = torch.load("llama_augmented_embeddings.pt")
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    model.get_input_embeddings().weight.copy_(augmented_embeddings)
print("✅ Embeddings replaced")

# -------------------------------
# Step 3: Pretraining on Tamil corpus
# -------------------------------
dataset = load_dataset("text", data_files={"train": "tamil_corpus.txt"})
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
tokenized_dataset = dataset.map(tokenize_fn, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

last_checkpoint = None
if os.path.isdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=os.path.getctime)[-1]
        print(f"Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model(output_dir)
print("✅ CPU pretraining complete!")

# -------------------------------
# Step 4: Downstream fine-tuning (classification)
# -------------------------------
task_dataset = load_dataset("csv", data_files={"train": "task_train.csv", "test": "task_test.csv"})
def tokenize_task(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
tokenized_task = task_dataset.map(tokenize_task, batched=True)

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

# -------------------------------
# Step 5: Evaluation example
# -------------------------------
if __name__ == "__main__":
    from transformers import pipeline, AutoModelForSequenceClassification
    tokenizer = LlamaTokenizer.from_pretrained("./tamil_llama_finetuned")
    model = AutoModelForSequenceClassification.from_pretrained("./tamil_llama_finetuned")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    examples = [
        "தமிழ் மொழி மிக அழகாக உள்ளது.",
        "This is a test sentence with code-switching."
    ]
    results = classifier(examples)
    for sent, res in zip(examples, results):
        print(f"Text: {sent}\nPrediction: {res}\n")
