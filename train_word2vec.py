"""
Train a Word2Vec model on the JSTOR Humanities full-text corpus (~1M articles, ~25GB JSONL).

Designed to run on an M1 Mac with 16GB RAM by streaming the corpus line-by-line.
Estimated training time: 3-8 hours. Peak RAM: ~4-6 GB.

Usage:
    pip install gensim
    python train_word2vec.py

The script:
1. Detects multi-word phrases ("machine learning" -> "machine_learning")
2. Trains a Word2Vec skip-gram model (300 dims, window=10)
3. Saves the model and phrase detectors for later use

After training, explore with:
    from gensim.models import Word2Vec
    model = Word2Vec.load('jstor_humanities_w2v.model')
    model.wv.most_similar('machine_learning', topn=20)
    model.wv.similarity('algorithm', 'method')
"""

import json
import re
import os
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

# --- Configuration ---

JSONL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "2026-actual-jstor-full-texts",
    "e119d16a-1426-497d-a42a-a239d354c137.jsonl",
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data_output", "word2vec")

# Word2Vec hyperparameters
VECTOR_SIZE = 300
WINDOW = 10        # broader window captures thematic associations
MIN_COUNT = 20     # ignore very rare words
WORKERS = 8        # use all M1 cores
EPOCHS = 5
SG = 1             # skip-gram (better for rare words)

# Phrase detection parameters
PHRASE_MIN_COUNT = 30
PHRASE_THRESHOLD = 10
TRIGRAM_MIN_COUNT = 20
TRIGRAM_THRESHOLD = 10

# Reference removal: strip everything after these markers if they appear
# in the last 30% of the text
REFERENCE_MARKERS = ["references", "bibliography", "works cited", "literature cited"]
REFERENCE_CUTOFF = 0.7  # marker must be past this fraction of the text


# --- Corpus streaming classes ---

class JSTORCorpus:
    """Streams tokenized documents from the JSONL without loading all into RAM."""

    def __init__(self, path, strip_references=True):
        self.path = path
        self.strip_references = strip_references

    def __iter__(self):
        with open(self.path, "r") as f:
            for line in f:
                item = json.loads(line)
                # full_text is a list of page strings
                text = " ".join(item.get("full_text", []))

                # Optionally strip the references/bibliography section
                if self.strip_references:
                    text = _remove_references(text)

                # Basic tokenization: lowercase, keep only alphabetic tokens (2+ chars)
                tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
                yield tokens


class PhrasedCorpus:
    """Wraps a corpus with bigram and trigram phrase detection."""

    def __init__(self, corpus, bigram_model, trigram_model):
        self.corpus = corpus
        self.bigram = bigram_model
        self.trigram = trigram_model

    def __iter__(self):
        for tokens in self.corpus:
            yield self.trigram[self.bigram[tokens]]


# --- Helpers ---

def _remove_references(text):
    """Heuristic: strip everything after a reference-section marker
    that appears in the last 30% of the text."""
    text_lower = text.lower()
    for marker in REFERENCE_MARKERS:
        idx = text_lower.rfind(marker)
        if idx > len(text) * REFERENCE_CUTOFF:
            return text[:idx]
    return text


# --- Main pipeline ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Corpus: {JSONL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Step 1: Detect bigram phrases ("machine learning" -> "machine_learning")
    print("=== Step 1/4: Detecting bigram phrases ===")
    corpus = JSTORCorpus(JSONL_PATH)
    bigram_phrases = Phrases(
        corpus,
        min_count=PHRASE_MIN_COUNT,
        threshold=PHRASE_THRESHOLD,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    bigram = bigram_phrases.freeze()
    bigram_path = os.path.join(OUTPUT_DIR, "bigram_phrases.model")
    bigram.save(bigram_path)
    print(f"Bigram phrase model saved to {bigram_path}")
    print()

    # Step 2: Detect trigram phrases ("natural language processing" -> "natural_language_processing")
    print("=== Step 2/4: Detecting trigram phrases ===")
    bigram_corpus = (bigram[tokens] for tokens in JSTORCorpus(JSONL_PATH))
    trigram_phrases = Phrases(
        bigram_corpus,
        min_count=TRIGRAM_MIN_COUNT,
        threshold=TRIGRAM_THRESHOLD,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    trigram = trigram_phrases.freeze()
    trigram_path = os.path.join(OUTPUT_DIR, "trigram_phrases.model")
    trigram.save(trigram_path)
    print(f"Trigram phrase model saved to {trigram_path}")
    print()

    # Step 3: Train Word2Vec
    print("=== Step 3/4: Training Word2Vec ===")
    print(f"  vector_size={VECTOR_SIZE}, window={WINDOW}, min_count={MIN_COUNT}")
    print(f"  workers={WORKERS}, epochs={EPOCHS}, sg={SG} (skip-gram)")
    phrased = PhrasedCorpus(JSTORCorpus(JSONL_PATH), bigram, trigram)

    model = Word2Vec(
        phrased,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        sg=SG,
    )
    print()

    # Step 4: Save
    print("=== Step 4/4: Saving model ===")
    model_path = os.path.join(OUTPUT_DIR, "jstor_humanities_w2v.model")
    model.save(model_path)
    print(f"Word2Vec model saved to {model_path}")
    print(f"Vocabulary size: {len(model.wv)}")
    print()

    # Quick sanity check
    print("=== Sanity check: nearest neighbors ===")
    for term in ["algorithm", "machine_learning", "digital", "artificial_intelligence"]:
        if term in model.wv:
            neighbors = model.wv.most_similar(term, topn=5)
            print(f"  {term}: {[w for w, _ in neighbors]}")
        else:
            print(f"  {term}: not in vocabulary")


if __name__ == "__main__":
    main()
