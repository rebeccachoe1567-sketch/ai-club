"""Step 1 of RAG: turn the PDFs into small, searchable text chunks.

Run:  python ingest.py

What it does:
  1. Reads every PDF in the documents/ folder.
  2. Removes boilerplate lines that repeat across documents (letterhead, footers).
  3. Splits each page into chunks of a few paragraphs.
  4. Saves all chunks to index.json, which search.py and ask.py read.
"""

import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

DOCS_DIR = Path(__file__).parent / "documents"
INDEX_PATH = Path(__file__).parent / "index.json"

# Roughly how many characters go in one chunk. Small chunks make search more
# precise; big chunks give Claude more surrounding context. ~1000 is a good start.
CHUNK_SIZE = 1000


def read_pdf(path):
    """Return a list of page texts. Layout mode keeps table rows on one line."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout")
        lines = []
        for line in text.splitlines():
            line = line.replace("\x7f", "•")  # the PDFs' bullet glyph extracts as \x7f
            # Wide gaps between words are table columns: show them as " | ".
            line = re.sub(r"\s{3,}", " | ", line.strip())
            lines.append(line)
        pages.append("\n".join(lines))
    return pages


def find_boilerplate(all_pages):
    """Lines that show up in 3+ different documents are letterhead/footer noise."""
    seen_in = Counter()
    for pages in all_pages.values():
        lines = {line for page in pages for line in page.splitlines() if line}
        seen_in.update(lines)
    return {line for line, count in seen_in.items() if count >= 3}


def split_into_chunks(text, size=CHUNK_SIZE):
    """Group paragraphs (separated by blank lines) into chunks of about `size` chars.

    Each chunk repeats the last paragraph of the previous chunk ("overlap") so a
    fact that sits on a chunk boundary is still findable with its context.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for para in paragraphs:
        if current and len("\n\n".join(current + [para])) > size:
            chunks.append("\n\n".join(current))
            current = [current[-1]]  # overlap
        current.append(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def main():
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}")

    all_pages = {path: read_pdf(path) for path in pdf_paths}
    boilerplate = find_boilerplate(all_pages)

    chunks = []
    for path, pages in all_pages.items():
        # "03_Master_Services_Agreement_MSA-2026-007.pdf" -> "Master Services Agreement MSA-2026-007"
        title = re.sub(r"^\d+_", "", path.stem).replace("_", " ")
        for page_number, page_text in enumerate(pages, start=1):
            kept = [l for l in page_text.splitlines() if l not in boilerplate]
            kept = [l for l in kept if not re.fullmatch(r"\|?\s*Page \d+", l)]
            for text in split_into_chunks("\n".join(kept)):
                chunks.append({
                    "id": len(chunks),
                    "document": title,
                    "file": path.name,
                    "page": page_number,
                    "text": text,
                })

    INDEX_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"Indexed {len(pdf_paths)} documents into {len(chunks)} chunks -> {INDEX_PATH.name}")
    for path in pdf_paths:
        n = sum(1 for c in chunks if c["file"] == path.name)
        print(f"  {path.name}: {n} chunks")


if __name__ == "__main__":
    main()
