"""Step 3 of RAG: answer a question with Claude, using only the retrieved chunks.

Run:
  python ask.py "What is the total on invoice LV-2026-10418?"
  python ask.py                      # interactive mode: keep asking questions
  python ask.py --show-sources "..." # also print the chunks Claude was given

Needs an Anthropic API key in the ANTHROPIC_API_KEY environment variable.
"""

import argparse

import anthropic

from search import Searcher

MODEL = "claude-opus-5-5"
TOP_K = 5  # how many chunks to give Claude per question

SYSTEM_PROMPT = """You answer questions about a company's internal documents.

Use only the numbered sources provided with each question. After every fact, cite \
the source it came from in square brackets, like [2]. If the sources don't contain \
the answer, say so plainly instead of guessing. When sources need to be combined \
(for example, a date from one document and a decision from another), combine them \
and cite each one. Keep answers short and direct."""


def build_prompt(question, chunks):
    """Put the retrieved chunks and the question into one message for Claude."""
    sources = "\n\n".join(
        f'<source id="{i}" document="{c["document"]}" page="{c["page"]}">\n{c["text"]}\n</source>'
        for i, c in enumerate(chunks, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


def answer(client, searcher, question, show_sources=False):
    chunks = searcher.search(question, k=TOP_K)
    if not chunks:
        print("No matching passages found in the documents.\n")
        return

    if show_sources:
        print("--- Sources given to Claude ---")
        for i, c in enumerate(chunks, start=1):
            print(f"[{i}] {c['document']} (page {c['page']}, score {c['score']})")
        print("-------------------------------\n")

    # Stream the answer so it prints as Claude writes it.
    # `fallbacks="default"`: if Claude declines a request for safety reasons, the
    # API automatically retries it on a recommended fallback model.
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        response = stream.get_final_message()
    print()

    if response.stop_reason == "refusal":
        print("(Claude declined to answer this question.)")
    elif response.stop_reason == "max_tokens":
        print("(Answer was cut off: it hit the max_tokens limit.)")

    # Show which document each [n] refers to.
    print("\nSources:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['document']}, page {c['page']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ask questions about the PDFs in documents/.")
    parser.add_argument("question", nargs="*", help="your question (leave empty for interactive mode)")
    parser.add_argument("--show-sources", action="store_true", help="print retrieved chunks first")
    args = parser.parse_args()

    searcher = Searcher()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    try:
        if args.question:
            answer(client, searcher, " ".join(args.question), args.show_sources)
            return

        print("Ask a question about the documents (blank line or Ctrl+C to quit).\n")
        while True:
            question = input("> ").strip()
            if not question:
                break
            answer(client, searcher, question, args.show_sources)
    except anthropic.AuthenticationError:
        raise SystemExit("Invalid API key. Check ANTHROPIC_API_KEY (see README.md).")
    except TypeError as e:
        # The SDK raises TypeError when it can't find any credentials at all.
        if "authentication" not in str(e):
            raise
        raise SystemExit("No API key found. Set ANTHROPIC_API_KEY (see README.md).")
    except anthropic.RateLimitError:
        raise SystemExit("Rate limited by the API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise SystemExit("Couldn't reach the Anthropic API. Check your internet connection.")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
