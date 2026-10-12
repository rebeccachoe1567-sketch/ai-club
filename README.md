# RAG with Claude

A small, beginner-friendly **RAG** (Retrieval-Augmented Generation) system that
answers questions about the PDFs in `documents/` using Claude.

## What is RAG?

Claude doesn't know what's in your private documents. RAG fixes that in three steps:

```
  documents/*.pdf
        │  1. INGEST  (ingest.py)
        ▼     read the PDFs, cut them into small "chunks", save to index.json
   index.json
        │  2. RETRIEVE  (search.py)
        ▼     for a question, find the few chunks most likely to hold the answer
   top 5 chunks
        │  3. GENERATE  (ask.py)
        ▼     send question + chunks to Claude: "answer using only these sources"
   answer with citations like [1], [2]
```

Why not just send Claude every document? For 5 short PDFs you could. But real
collections have thousands of pages. Retrieval keeps each request small, cheap
and focused, and the citations let you check where each fact came from.

## Setup

1. **Install Python 3.10+**, then the two libraries:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get an API key** at <https://platform.claude.com/> (Settings → API Keys),
   then put it in your environment. Never commit it to git.

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
   ```

## Usage

```bash
python ingest.py                                  # step 1: build index.json (re-run when PDFs change)
python search.py "when is the invoice due?"       # step 2 only: see what gets retrieved (no API key needed)
python ask.py "when is the invoice due?"          # full RAG: retrieve + Claude answers
python ask.py --show-sources "..."                # also list the chunks Claude was given
python ask.py                                     # interactive mode
```

### Questions to try

| Question | Why it's interesting |
|---|---|
| What is the total due on invoice LV-2026-10418, and when is it due? | Reads values from a table |
| How much notice is needed to end the Bluefinch agreement early? | The contract says "terminate", not "end early", so keyword search ranks the right clause only 3rd. This is where semantic search helps |
| Why were overflow transfers paused on October 24–25? | Combines the meeting minutes and the memo |
| What was wholesale revenue in Q3, and how did it change from Q2? | Numbers from a table |
| How was the extra $35,000 for marketing funded? | Detail found in only one document |
| What is the company's dental plan? | Not in the documents: Claude should say so |

## Files

| File | What it does |
|---|---|
| `ingest.py` | Reads PDFs with `pypdf`, strips repeated letterhead/footers, splits pages into ~1000-character chunks |
| `search.py` | Ranks chunks with BM25 keyword search (pure Python, no API key) |
| `ask.py` | Sends the top chunks and your question to Claude and streams back a cited answer |
| `documents/` | The PDFs to search. Add your own and re-run `ingest.py` |

## Ideas for next steps

- **Tune the knobs:** change `CHUNK_SIZE` in `ingest.py` or `TOP_K` in `ask.py`,
  and use `--show-sources` to see how the retrieved chunks change.
- **Semantic search:** keyword search misses synonyms (e.g. "cancel" vs.
  "terminate"). Embeddings (from a provider such as Voyage AI, or a local model
  like `sentence-transformers`) search by meaning. You would replace `Searcher`.
- **Test it:** write 20 questions with known answers and check how many come
  back right after each change.
- **Web or chat UI:** wrap `answer()` in a small Streamlit or Flask app.

> The sample PDFs describe a fictional company. All names and figures are made up.
