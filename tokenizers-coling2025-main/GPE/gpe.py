import pickle
from collections import Counter
import grapheme  # make sure you have installed it via pip

class GPETokenizer:
    def __init__(self, vocab, merges=None):
        self.vocab = vocab
        self.merges = merges if merges else {}
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab.keys())}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def encode(self, text):
        tokens = list(grapheme.graphemes(text))  # <--- correct grapheme splitting
        ids = [self.token_to_id.get(tok, 0) for tok in tokens]
        return tokens, ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return ''.join(tokens)


# ---------------- TRAINER ----------------
def train_gpe(corpus):
    """
    corpus: list of strings
    Returns vocab and merges (empty for now)
    """
    # Use grapheme-aware tokenization
    vocab_counter = Counter()
    for line in corpus:
        tokens = list(grapheme.graphemes(line))
        for t in tokens:
            vocab_counter[t] += 1

    vocab = {tok: idx for idx, tok in enumerate(vocab_counter.keys())}
    merges = {}  # can implement merges later if desired
    return vocab, merges


# ---------------- SAVE/LOAD ----------------
def save_gpe(tokenizer, vocab_file="vocab_gpe.pkl", merges_file="merges_gpe.pkl"):
    with open(vocab_file, "wb") as f:
        pickle.dump(tokenizer.token_to_id, f)
    with open(merges_file, "wb") as f:
        pickle.dump(tokenizer.merges, f)


def load_gpe(vocab_file="vocab_gpe.pkl", merges_file="merges_gpe.pkl"):
    with open(vocab_file, "rb") as f:
        token_to_id = pickle.load(f)
    with open(merges_file, "rb") as f:
        merges = pickle.load(f)
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return GPETokenizer(id_to_token, merges)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load corpus from a local file
    with open(r"C:\Users\HP\Documents\vs code\tokenizers-coling2025-main\GPE\samanantar_eng_90_percent_cleaned1.txt",
              "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()][:500]  # limit to 500 lines for testing

    vocab, merges = train_gpe(corpus)
    tokenizer = GPETokenizer(vocab, merges)

    save_gpe(tokenizer)

    # Test
    sample_text = "தமிழ் ஒரு செழுமையான மொழி."
    tokens, ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)
    print("GPE Tokens:", tokens)
    print("GPE Token IDs:", ids)
    print("GPE Decoded:", decoded)
    print("Number of Tokens:", len(tokens))
