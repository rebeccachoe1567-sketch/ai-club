"""Step 1 of RAG: turn the PDFs in docs/ into a searchable vector store.

Run:  python ingest.py

Run it again whenever you add, remove or change a file in docs/.

What it does:
  1. LOAD   - read the text of every page of every PDF
  2. CLEAN  - drop page headers, footers and other repeated noise
  3. CHUNK  - cut each page into pieces of a few paragraphs
  4. EMBED  - turn each chunk into a vector with the local embedding model
  5. SAVE   - write vectors + chunk texts to index/
"""

import re
from pathlib import Path

from pypdf import PdfReader

from embeddings import MODEL_NAME, embed_passages
from vector_store import VectorStore

DOCS_DIR = Path(__file__).parent / "docs"

# Roughly how many characters go in one chunk. Smaller chunks match a question
# more precisely; bigger ones give Claude more surrounding context. ~800
# characters (about 150 words) suits short policy documents like these.
CHUNK_SIZE = 800

# "Page 2 of 5" footers, and the "Oct 11, 2026 · @Rebecca" byline under each title.
NOISE_LINE = re.compile(r"Page \d+ of \d+|[A-Z][a-z]{2} \d{1,2}, \d{4}\s+·.*@\w+")


# 1. LOAD ---------------------------------------------------------------------

def read_pdf(path):
    """Return (title, list of page texts) for one PDF."""
    reader = PdfReader(path)
    # Use the title stored inside the PDF, or build one from the file name.
    title = (reader.metadata and reader.metadata.title) or path.stem.replace("_", " ")
    pages = []
    for page in reader.pages:
        # "layout" mode keeps each table row on one line instead of one cell per line.
        text = page.extract_text(extraction_mode="layout")
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if NOISE_LINE.fullmatch(line):
                continue
            # Runs of 3+ spaces are gaps between table columns: show them as " | "
            # so "Barista I    $18.50 + tips    $19.75" stays readable.
            lines.append(re.sub(r" {3,}", " | ", line))
        pages.append("\n".join(lines))
    return title, pages


# 2. CLEAN --------------------------------------------------------------------

def remove_repeated_lines(pages):
    """Drop lines that appear on every page of a document (the running header).

    A header like "Summit Ridge Coffee Co. — Employee Handbook" on every page
    adds nothing to any one chunk, and would make every chunk look alike.
    """
    if len(pages) < 2:
        return pages
    on_every_page = set.intersection(*(set(p.splitlines()) for p in pages)) - {""}
    return ["\n".join(l for l in p.splitlines() if l not in on_every_page) for p in pages]


# 3. CHUNK --------------------------------------------------------------------

def split_into_chunks(text, size=CHUNK_SIZE):
    """Group paragraphs (separated by blank lines) into chunks of about `size` characters.

    Each new chunk starts with the last paragraph of the previous one. That
    overlap means a fact sitting on a chunk boundary still appears together
    with the sentence before it.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for para in paragraphs:
        if current and len("\n\n".join(current + [para])) > size:
            chunks.append("\n\n".join(current))
            current = [current[-1]]  # overlap: carry the last paragraph forward
        current.append(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_chunks():
    """Steps 1-3 for every PDF in docs/. Returns a list of chunk dicts."""
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}")

    chunks = []
    for path in pdf_paths:
        title, pages = read_pdf(path)
        pages = remove_repeated_lines(pages)
        for page_number, page_text in enumerate(pages, start=1):
            for text in split_into_chunks(page_text):
                chunks.append({
                    "id": len(chunks),
                    "document": title,
                    "file": path.name,
                    "page": page_number,
                    "text": text,
                })
    return chunks


# 4. EMBED and 5. SAVE --------------------------------------------------------

def main():
    chunks = load_chunks()
    print(f"Read {len({c['file'] for c in chunks})} PDFs into {len(chunks)} chunks.")

    print(f"Embedding with {MODEL_NAME} (the first run downloads the model)...")
    # Put the document title on top of each chunk before embedding it. A chunk
    # that is just a table row ("Mugs, tumblers | 30 days | Unused") is much
    # easier to match once the vector also "knows" it is from the refund policy.
    vectors = embed_passages([f"{c['document']}\n\n{c['text']}" for c in chunks])

    VectorStore(vectors, chunks, MODEL_NAME).save()
    print(f"Saved {vectors.shape[0]} vectors of {vectors.shape[1]} numbers each to index/")
    for name in sorted({c["file"] for c in chunks}):
        print(f"  {name}: {sum(c['file'] == name for c in chunks)} chunks")


if __name__ == "__main__":
    main()
