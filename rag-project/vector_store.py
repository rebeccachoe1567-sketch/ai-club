"""A tiny vector store: a NumPy matrix of vectors plus a list of the chunks they came from.

Real projects often use a vector database (Chroma, pgvector, Qdrant...). They do
the same thing as this file, with extra machinery for millions of vectors. For a
few hundred chunks, one matrix multiplication searches everything in under a
millisecond, so we keep it simple enough to read in one sitting.

On disk the store is two files in index/:
  vectors.npy  - the matrix, one row per chunk
  chunks.json  - the chunk texts and where each came from (row i <-> chunk i)
"""

import json
from pathlib import Path

import numpy as np

INDEX_DIR = Path(__file__).parent / "index"


class VectorStore:
    def __init__(self, vectors, chunks, model_name):
        if len(vectors) != len(chunks):
            raise ValueError("Need exactly one vector per chunk.")
        self.vectors = vectors  # shape (number of chunks, 384), each row length 1
        self.chunks = chunks    # list of dicts: text, document, page, ...
        self.model_name = model_name

    def save(self, folder=INDEX_DIR):
        folder.mkdir(exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        data = {"model": self.model_name, "chunks": self.chunks}
        (folder / "chunks.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, folder=INDEX_DIR):
        if not (folder / "vectors.npy").exists():
            raise SystemExit("No index found. Run `python ingest.py` first.")
        vectors = np.load(folder / "vectors.npy")
        data = json.loads((folder / "chunks.json").read_text())
        return cls(vectors, data["chunks"], data["model"])

    def search(self, query_vector, k=5):
        """Return the k chunks whose vectors are most similar to query_vector.

        Every vector has length 1, so the dot product of two vectors is their
        cosine similarity: 1.0 means "same direction" (very similar meaning),
        around 0 means unrelated. `self.vectors @ query_vector` computes that
        score for every chunk at once.
        """
        scores = self.vectors @ query_vector
        best = np.argsort(scores)[::-1][:k]  # indexes of the k highest scores
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in best]
