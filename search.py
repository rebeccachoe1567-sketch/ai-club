"""Step 2 of RAG: find the chunks most relevant to a question.

This uses BM25, a classic keyword-ranking formula (it's what many search
engines used before AI). It needs no API key and no extra libraries.

Try it on its own:  python search.py "when is the invoice due?"
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

INDEX_PATH = Path(__file__).parent / "index.json"

# Common words that don't help tell chunks apart.
STOPWORDS = set("""
a an and are as at be by did do does for from has have how i in is it its
of on or our that the their them there these this to was we were what when
where which who why will with would you your can about any
""".split())


def tokenize(text):
    """Lowercase words and numbers, e.g. "Q3 revenue: $4,820,000" -> ["q3", "revenue", "4820000"]."""
    text = text.lower().replace(",", "")
    return [w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*[a-z0-9]|[a-z0-9]", text) if w not in STOPWORDS]


class Searcher:
    def __init__(self, index_path=INDEX_PATH):
        if not index_path.exists():
            raise SystemExit("index.json not found. Run `python ingest.py` first.")
        self.chunks = json.loads(index_path.read_text())
        # Search the document title too, so "invoice" matches the invoice's chunks.
        self.docs = [tokenize(c["document"] + " " + c["text"]) for c in self.chunks]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs)
        # How many chunks contain each word (rare words count for more).
        self.doc_freq = Counter(word for d in self.docs for word in set(d))

    def search(self, question, k=5):
        """Return the top `k` chunks, each with a relevance `score`."""
        query = tokenize(question)
        n = len(self.docs)
        k1, b = 1.5, 0.75  # standard BM25 settings
        scored = []
        for chunk, words in zip(self.chunks, self.docs):
            counts = Counter(words)
            score = 0.0
            for term in query:
                if term not in counts:
                    continue
                idf = math.log(1 + (n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                tf = counts[term]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / self.avg_len))
            if score > 0:
                scored.append({**chunk, "score": round(score, 2)})
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "When is the inventory system cutover?"
    print(f"Question: {question}\n")
    for i, hit in enumerate(Searcher().search(question), start=1):
        print(f"[{i}] score={hit['score']}  {hit['document']} (page {hit['page']})")
        print("    " + hit["text"][:300].replace("\n", "\n    ") + "\n")
