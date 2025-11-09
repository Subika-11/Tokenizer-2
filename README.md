# Tamil Tokenizer Repository

This repository provides a comprehensive suite of tools and scripts for tokenizing Tamil text, handling sandhi rules, out-of-vocabulary (OOV) detection, and evaluating tokenization models. It is designed for research and practical applications in Tamil NLP, including experiments, model training, and evaluation.

## Directory Structure

- *core/*: Main tokenization logic and utilities
  - [core/bpe.py](core/bpe.py): Byte Pair Encoding tokenizer implementation
  - [core/compare_tokenizers.py](core/compare_tokenizers.py): Compare different tokenizers
  - [core/GPE_sandhi.py](core/GPE_sandhi.py), [core/gpe.py](core/gpe.py), [core/sandhi.py](core/sandhi.py): Sandhi rule handling
  - [core/oov.py](core/oov.py): Out-of-vocabulary detection
  - [core/trim_tokens.py](core/trim_tokens.py): Token trimming utilities
- *data/*: Sample datasets and checkpoints
  - [data/sample_tamil.txt](data/sample_tamil.txt): Example Tamil text file
  - [data/agathyam_tokens.txt](data/agathyam_tokens.txt), [data/agathyam_tokens_trimmed.txt](data/agathyam_tokens_trimmed.txt): Token lists
  - [data/checkpoint.pkl](data/checkpoint.pkl): Model checkpoint
  - [data/vocab_bpe.pkl](data/vocab_bpe.pkl): BPE vocabulary
  - [data/merges_bpe.pkl](data/merges_bpe.pkl): BPE merge rules
  - [data/flores/](data/flores/): Additional multilingual datasets
- *experiments/*: Scripts for training, evaluation, and embedding initialization
  - [experiments/eval_perplexity.py](experiments/eval_perplexity.py): Perplexity evaluation
  - [experiments/initialize_embeddings.py](experiments/initialize_embeddings.py): Embedding initialization
  - [experiments/lightweight_pretrain_fixed.py](experiments/lightweight_pretrain_fixed.py): Lightweight pretraining
- *tamiltokenizer/*: Additional tokenizer modules
- *tests/*: Unit tests
- *jupyter/, **Lib/, **models/, **results/, **Scripts/, **share/*: Supporting scripts, notebooks, and outputs

## Installation

1. *Clone the repository:*
   sh
   git clone https://github.com/yourusername/tamiltokenizer.git
   cd tamiltokenizer
   

2. *Install dependencies:*
   sh
   pip install -r requirements.txt
   

## Usage

### Tokenize Tamil Text

->To tokenize a Tamil text file using BPE:
python core/bpe.py 

->To tokenize a Tamil text file using GPE:
python core/gpe.py 

->To tokenize a Tamil text file using Sandhi-GPE:
python core/sandhi.py 
python core/GPE-Sandhi.py 


### Compare Tokenizers

->To compare different tokenizers:
python core/compare_tokenizers.py 

### OOV Detection

To detect out-of-vocabulary words:

python core/oov.py

### Trim Tokens

To trim tokens in a file:
python core/trim_tokens.py 

### Evaluate Perplexity

To evaluate model perplexity:
python experiments/eval_perplexity.py 


### Initialize Embeddings

To initialize embeddings:
sh
python experiments/initialize_embeddings.py 


## Testing

Run all unit tests with:
pytest tests/


## Data

Sample data files are provided in the data/ directory. You can use your own datasets by placing them in this folder.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

See [LICENSE](LICENSE) for details.

## Contact

For questions or contributions, please open an issue or contact the repository maintainer.
