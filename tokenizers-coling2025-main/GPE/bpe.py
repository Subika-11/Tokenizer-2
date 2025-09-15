import re
import pickle
from collections import Counter
from itertools import tee

# ---------------- BPE TOKENIZER ----------------
class BPETokenizer:
    def __init__(self, token_to_id, merges):
        """
        token_to_id: dict mapping token -> ID
        merges: dict mapping (token1, token2) -> merged_id
        """
        self.token_to_id = token_to_id
        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        self.merges = merges  # {(tok1, tok2): merged_id}

    def encode(self, text):
        # Step 1: Start with character-level tokens
        tokens = list(text)

        # Step 2: Apply merges iteratively
        changed = True
        while changed:
            changed = False
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in self.merges:
                    merged_id = self.merges[(tokens[i], tokens[i+1])]
                    new_tokens.append(self.id_to_token[merged_id])
                    i += 2
                    changed = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Step 3: Convert tokens to IDs
        ids = [self.token_to_id.get(tok, 0) for tok in tokens]
        return tokens, ids

    def decode(self, ids):
        return ''.join([self.id_to_token.get(i, "<UNK>") for i in ids])


# ---------------- TRAIN BPE ----------------
def train_bpe(corpus, num_merges=200):
    """
    corpus: list of strings
    num_merges: number of merge operations
    """
    # Step 1: Build initial char-level vocab
    vocab = Counter()
    for line in corpus:
        for c in line:
            vocab[c] += 1

    token_to_id = {tok: idx for idx, tok in enumerate(vocab.keys())}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    merges = {}

    # Helper to get consecutive pairs
    def get_pairs(tokens):
        a, b = tee(tokens)
        next(b, None)
        return [(x, y) for x, y in zip(a, b)]

    # Step 2: Iterative merges
    for _ in range(num_merges):
        pairs_count = Counter()
        for line in corpus:
            tokens = list(line)
            for pair in get_pairs(tokens):
                pairs_count[pair] += 1

        if not pairs_count:
            break

        # Most frequent pair
        most_freq = max(pairs_count, key=pairs_count.get)
        merged_token = ''.join(most_freq)
        new_id = len(token_to_id)

        # Add merged token
        token_to_id[merged_token] = new_id
        id_to_token[new_id] = merged_token
        merges[most_freq] = new_id

        # Replace in corpus
        new_corpus = []
        for line in corpus:
            line_tokens = list(line)
            i = 0
            new_line_tokens = []
            while i < len(line_tokens):
                if i < len(line_tokens) - 1 and (line_tokens[i], line_tokens[i+1]) == most_freq:
                    new_line_tokens.append(merged_token)
                    i += 2
                else:
                    new_line_tokens.append(line_tokens[i])
                    i += 1
            new_corpus.append(''.join(new_line_tokens))
        corpus = new_corpus

    return token_to_id, merges


# ---------------- SAVE/LOAD ----------------
def save_bpe(tokenizer, vocab_file="vocab_bpe.pkl", merges_file="merges_bpe.pkl"):
    with open(vocab_file, "wb") as f:
        pickle.dump(tokenizer.token_to_id, f)
    with open(merges_file, "wb") as f:
        pickle.dump(tokenizer.merges, f)


def load_bpe(vocab_file="vocab_bpe.pkl", merges_file="merges_bpe.pkl"):
    with open(vocab_file, "rb") as f:
        token_to_id = pickle.load(f)
    with open(merges_file, "rb") as f:
        merges = pickle.load(f)
    return BPETokenizer(token_to_id, merges)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load corpus from local file (example: first 500 lines)
    corpus_file = r"C:\Users\ROSHINI PRIYA\Downloads\tokenizers-coling2025-main (4)\tokenizers-coling2025-main\GPE\samanantar_eng_90_percent_cleaned1.txt"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()][:500]

    # Train BPE
    token_to_id, merges = train_bpe(corpus, num_merges=200)
    tokenizer = BPETokenizer(token_to_id, merges)

    # Save trained tokenizer
    save_bpe(tokenizer)

    # Test encoding
    sample_text = "தமிழ் ஒரு செழுமையான மொழி."
    tokens, ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)

    print("="*60)
    print("BPE Tokens:", tokens)
    print("BPE Token IDs:", ids)
    print("BPE Decoded:", decoded)
    print("BPE Number of Tokens:", len(tokens))
