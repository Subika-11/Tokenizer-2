import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import json

# -------------------------------
# Step 1: Load LLaMA embeddings
# -------------------------------
# Assume embeddings are a PyTorch tensor: (vocab_size, embedding_dim)
llama_embeddings = torch.load("llama_embeddings.pt")  # shape: [V, D]
vocab = json.load(open("vocab.json", "r", encoding="utf-8"))  # id->token mapping

# Convert to list for easier processing
tokens = [vocab[str(i)] for i in range(len(vocab))]

# -------------------------------
# Step 2: Load new Tamil tokens
# -------------------------------
new_tokens = [line.strip() for line in open("agathyam_tokens.txt", "r", encoding="utf-8")]

# -------------------------------
# Step 3: Define similarity function
# -------------------------------
def grapheme_similarity(token1, token2):
    # Convert edit distance to similarity
    dist = levenshtein_distance(token1, token2)
    max_len = max(len(token1), len(token2))
    return 1 - (dist / max_len) if max_len > 0 else 0

# -------------------------------
# Step 4: Initialize embeddings for new tokens
# -------------------------------
new_embeddings = []

N = 5  # number of nearest neighbors

for new_token in new_tokens:
    # Compute grapheme similarity with all existing tokens
    similarities = [grapheme_similarity(new_token, tok) for tok in tokens]
    
    # Get indices of top N neighbors
    top_indices = np.argsort(similarities)[-N:]
    
    # Take mean of their embeddings
    neighbor_embs = llama_embeddings[top_indices, :]
    mean_emb = neighbor_embs.mean(dim=0)
    
    new_embeddings.append(mean_emb)

# Stack into tensor
new_embeddings_tensor = torch.stack(new_embeddings)  # shape: [num_new_tokens, embedding_dim]

# -------------------------------
# Step 5: Combine with existing embeddings
# -------------------------------
updated_embeddings = torch.cat([llama_embeddings, new_embeddings_tensor], dim=0)

# Save for pretraining
torch.save(updated_embeddings, "llama_augmented_embeddings.pt")

print("✅ Embedding initialization complete!")
