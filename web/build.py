"""Embed index.json into the web page: python ingest.py && python web/build.py"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
chunks = json.loads((root / "index.json").read_text())
data = json.dumps(chunks, ensure_ascii=False).replace("</", "<\\/")
html = (root / "web" / "template.html").read_text().replace("__CHUNKS__", data)
(root / "web" / "document-desk.html").write_text(html)
print(f"Wrote web/document-desk.html with {len(chunks)} chunks")
