# Summit Ridge RAG

A small RAG (Retrieval-Augmented Generation) system that answers questions about
the five Summit Ridge Coffee Co. PDFs in `docs/`, using:

- **local embeddings**: a small model that runs on your computer, no API key needed
- **a simple vector store**: a NumPy matrix saved to disk
- **the Claude API**: writes the final answer, with citations

## How it works

```
 docs/*.pdf
     │
     │  ingest.py   (run once, and again when the documents change)
     │    1. load    read the text of every PDF page
     │    2. clean   drop running headers, page footers, the byline
     │    3. chunk   cut each page into ~800-character pieces
     │    4. embed   turn each chunk into 384 numbers (a vector)
     ▼
 index/vectors.npy + index/chunks.json      ← the vector store
     │
     │  search.py   (every question)
     │    embed the question the same way, then find the 5 chunks
     │    whose vectors point in the most similar direction
     ▼
 top 5 chunks
     │
     │  ask.py      (every question)
     │    send question + chunks to Claude: "answer only from these, cite [n]"
     ▼
 answer with citations like [1], [3]
```

| File | What it does |
|---|---|
| `ingest.py` | Steps 1–4 above, then saves the vector store |
| `embeddings.py` | Loads the embedding model and turns text into vectors |
| `vector_store.py` | Saves, loads and searches the vectors (about 40 lines) |
| `search.py` | Retrieval only: prints the best chunks for a question. No API key needed |
| `ask.py` | Retrieval and then Claude's answer |

## Setup

1. **Install Python 3.13** from https://www.python.org/downloads/ (3.10–3.13 all work).
   Avoid 3.14 for now: the embedding library needs `onnxruntime`, which has no
   Python 3.14 build for Intel Macs or for macOS versions before 14 (Sonoma).

2. **Create a virtual environment and install the libraries:**

   ```bash
   cd rag-project
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Get an Anthropic API key** at https://console.anthropic.com and set it:

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."       # macOS / Linux
   setx ANTHROPIC_API_KEY "sk-ant-..."         # Windows (then open a new terminal)
   ```

   Only `ask.py` needs it. `ingest.py` and `search.py` run fully offline once
   the model is downloaded.

## Usage

```bash
python ingest.py
```

The first run downloads the embedding model (BAAI/bge-small-en-v1.5, about
67 MB) into `.model_cache/`. Later runs reuse it.

```bash
# See what retrieval finds (no Claude involved):
python search.py "How many attendance points lead to termination?"

# Ask Claude:
python ask.py "Can I return a Breville espresso machine after 3 weeks?"
python ask.py --show-sources 'Who approves a $400 refund?'   # single quotes: in double quotes the shell would eat "$4"
python ask.py          # interactive: keep asking until you enter a blank line
```

Questions to try:

- Who manages the Lakewood café, and how did it do in Q3 2025? *(combines two documents)*
- What does a Summit Honey Latte cost in a 16 oz size?
- How quickly is access removed when an employee leaves?
- What is the minimum password length?
- What is the CEO's favorite coffee? *(not in the documents: Claude should say so)*

## Key ideas

**Embeddings capture meaning, not just words.** "What happens if I skip a shift
without telling anyone?" can match the handbook row "No-call, no-show | 4"
(points), although the two share almost no words. Keyword search depends on
shared words, so it would likely miss it.

**Cosine similarity.** Every vector is scaled to length 1, so the dot product
of two vectors equals the cosine of the angle between them: 1.0 means the same
direction (very similar meaning) and near 0 means unrelated. `vector_store.py`
scores every chunk with one line: `self.vectors @ query_vector`.

**Same model for chunks and questions.** Vectors from different models aren't
comparable. The store records which model built it, and `search.py` refuses to
search with a different one.

**Chunks get their document title.** Before embedding, each chunk is prefixed
with its document's title. A bare table row like "Mugs, tumblers, apparel |
30 days" then still "knows" it belongs to the refund policy.

**Grounding and citations.** Claude is told to use only the provided sources,
cite them as [n], and say so when the answer isn't there. The citations let you
check every fact against the page it came from.

## Things to experiment with

- Change `CHUNK_SIZE` in `ingest.py` (try 400 and 1500), re-run ingest, and
  compare the results of `search.py`.
- Change `TOP_K` in `search.py`. More chunks give Claude more context but cost
  more tokens and add noise.
- Remove the title prefix in `ingest.py` and see which searches get worse.
- Add your own PDF to `docs/` and re-run `python ingest.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No index found` | Run `python ingest.py` first |
| `built with a different model` | You changed `MODEL_NAME`: re-run `python ingest.py` |
| `No matching distribution found for onnxruntime` | Your Python is too new for your Mac. Install Python 3.13, delete `.venv`, and create it again with `python3.13 -m venv .venv` |
| Model download fails | The first run needs access to huggingface.co. Check your internet connection or proxy |
| `No API key found` / `Invalid API key` | Set `ANTHROPIC_API_KEY` (see Setup), then open a new terminal on Windows |
| An answer is wrong | Run `search.py` with the same question. If the right chunk isn't listed, the problem is retrieval, not Claude |
