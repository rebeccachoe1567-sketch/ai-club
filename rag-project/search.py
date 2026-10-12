"""Step 2 of RAG: find the chunks most likely to answer a question.

Run:  python search.py "How many attendance points lead to termination?"

This step doesn't use Claude or an API key. Use it to check what the retriever
finds: if the right chunk isn't in this list, Claude can't answer correctly,
however good its prompt is.
"""

import sys

from embeddings import MODEL_NAME, embed_query
from vector_store import VectorStore

TOP_K = 5  # how many chunks to retrieve per question


class Retriever:
    def __init__(self):
        self.store = VectorStore.load()
        if self.store.model_name != MODEL_NAME:
            # Vectors from two different models can't be compared.
            raise SystemExit("The index was built with a different model. Run `python ingest.py` again.")

    def search(self, question, k=TOP_K):
        return self.store.search(embed_query(question), k=k)


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python search.py "your question"')
    question = " ".join(sys.argv[1:])

    for rank, chunk in enumerate(Retriever().search(question), start=1):
        print(f"[{rank}] score {chunk['score']}  {chunk['document']}, page {chunk['page']}")
        print("    " + chunk["text"].replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
