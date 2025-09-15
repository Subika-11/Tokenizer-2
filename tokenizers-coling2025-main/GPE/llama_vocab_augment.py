import sentencepiece as spm
import json

# -------------------------------
# Step 1: Load existing tokenizer
# -------------------------------
tokenizer_model = "llama_tokenizer.model"
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model)

# -------------------------------
# Step 2: Load Agathyam Tamil tokens
# -------------------------------
# Assume you have a text file with 1 token per line
tamil_tokens_file = "agathyam_tokens.txt"
with open(tamil_tokens_file, "r", encoding="utf-8") as f:
    tamil_tokens = [line.strip() for line in f if line.strip()]

# -------------------------------
# Step 3: Add Tamil tokens to tokenizer
# -------------------------------
# SentencePiece allows adding new tokens via vocab_size increase
# We'll append tokens manually to vocab.json (or a new tokenizer)
# Option 1: Create a separate tokenizer with new tokens
new_tokens = tamil_tokens
current_vocab_size = sp.get_piece_size()
new_vocab_size = current_vocab_size + len(new_tokens)

print(f"Current vocab: {current_vocab_size}, Adding: {len(new_tokens)}, New vocab: {new_vocab_size}")

# You can also write them into a vocab file for SentencePiece training
with open("combined_vocab.txt", "w", encoding="utf-8") as f:
    # Write original vocab
    for i in range(current_vocab_size):
        f.write(sp.id_to_piece(i) + "\n")
    # Write Tamil tokens
    for token in new_tokens:
        f.write(token + "\n")

# -------------------------------
# Step 4: Train new tokenizer (optional)
# -------------------------------
# This step re-trains SentencePiece with combined vocab
# spm.SentencePieceTrainer.Train(
#     input='your_corpus.txt',
#     model_prefix='tamil_llama',
#     vocab_size=new_vocab_size,
#     user_defined_symbols=tamil_tokens
# )

print("✅ Vocab augmentation ready! Now embeddings can be initialized.")
