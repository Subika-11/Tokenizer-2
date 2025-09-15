from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name
model_id = "google/gemma-3-270m"

# Load tokenizer & model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer loaded.")

print("Loading model (this may take a few minutes on first run)...")
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model loaded successfully.")

# Quick test
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)

print("\nGenerated output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
