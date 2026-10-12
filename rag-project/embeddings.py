"""Turn text into vectors (lists of numbers) with a small model that runs on your computer.

An embedding model reads a piece of text and outputs a vector: 384 numbers that
describe what the text is about. Texts with similar meanings get vectors that
point in similar directions, even when they share no words. "How much PTO do I
get?" lands close to "full PTO accrual after the introductory period".

We use BAAI/bge-small-en-v1.5 through the `fastembed` library. It is small
(~67 MB), runs on a normal CPU, and needs no API key. The first run downloads
it into .model_cache/; after that everything works offline.
"""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CACHE_DIR = Path(__file__).parent / ".model_cache"

_model = None


def _get_model():
    """Load the model once and reuse it (loading takes a second or two)."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    return _model


def _normalize(vectors):
    """Scale every vector to length 1, so a dot product equals cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def embed_passages(texts):
    """Embed document chunks. Returns an array of shape (len(texts), 384)."""
    vectors = np.array(list(_get_model().passage_embed(texts)), dtype=np.float32)
    return _normalize(vectors)


def embed_query(question):
    """Embed a question. Returns an array of shape (384,).

    Questions and passages go through slightly different methods: the model was
    trained to match a short question against the longer passage that answers it.
    """
    vector = np.array(next(iter(_get_model().query_embed(question))), dtype=np.float32)
    return _normalize(vector)
