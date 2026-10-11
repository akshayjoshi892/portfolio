# Part XII — Document Ingestion

## Before you start

**What you need already:** [Part XI](part11-rag-fundamentals.md) — you know RAG needs clean text before anything else can work. This part is about getting that clean text out of messy real-world sources.

**What this part gives you:** Part XI's project used documents that were already clean plain text. Real documents aren't. This part handles PDFs, HTML pages full of navigation clutter, and the question of what to do when a document you already indexed changes.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Turn a messy folder of real files — PDFs, web pages, spreadsheets — into clean text ready for chunking, and re-index only what actually changed instead of reprocessing everything every time.

---

## Stage 1 — Why this part exists: garbage in, garbage forever

Here's a fact worth internalizing before touching any code: **no retrieval technique, no reranking model, no clever prompt can recover information that was destroyed during ingestion.** If a PDF's table gets flattened into an unreadable run of numbers with no structure, or an HTML page's navigation menu gets indexed as if it were real content, that damage is permanent from that point forward — every later stage inherits it.

This means ingestion quality sets a *ceiling* on your entire RAG system's quality. You can have a great embedding model and a great chunking strategy, and still get bad answers, because the text going in was already broken.

**Check yourself before Stage 2:** Why can't a good embedding model or reranker compensate for badly-parsed source documents?

---

## Stage 2 — One function per file type, dispatched by extension

Different formats need genuinely different parsing logic. The clean way to organize this: a dictionary mapping file extension to the function that knows how to handle it.

```python
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable


def parse_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    # Strip script/style blocks entirely — their content isn't readable text
    text = re.sub(r"<script.*?</script>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL)
    # Strip remaining tags, collapse whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_csv(path: Path) -> str:
    import csv
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # Turn each row into a readable "key: value" line, not a raw comma-separated
    # mess — a chunker later will handle this as prose, not as tabular data,
    # so it needs to already read like a sentence.
    return "\n".join("; ".join(f"{k}: {v}" for k, v in row.items()) for row in rows)


PARSERS: dict[str, Callable[[Path], str]] = {
    ".md": parse_markdown,
    ".html": parse_html,
    ".csv": parse_csv,
}


def parse_file(path: Path) -> str | None:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        print(f"skipping unsupported file type: {path}")
        return None
    return parser(path)
```

The `parse_html` function here is a naive, regex-based stripper — good enough to understand the *idea* (remove non-content markup, keep the text), but real HTML parsing should use a proper library (`trafilatura` or `readability` are common choices) because regex-based HTML parsing breaks on edge cases regex can't reliably handle — nested tags, malformed markup, and content that only appears via JavaScript. You'll swap this out in the project.

**Check yourself before Stage 3:** Why does `parse_csv` turn each row into a "key: value" sentence instead of just keeping the raw CSV text?

---

## Stage 3 — Metadata: remembering where each piece of text came from

A chunk of text alone isn't enough — you'll want to know *where it came from* later, for citations (Part XI), for filtering (Part XV), and for debugging when something goes wrong.

```python
from dataclasses import dataclass, field


@dataclass
class NormalizedDoc:
    doc_id: str
    source_path: str
    content: str
    metadata: dict = field(default_factory=dict)
```

The natural question: how do you generate `doc_id`? Using the filename seems obvious, but it has a real problem you'll hit in Stage 4 — so instead, generate it from the *content*:

```python
import hashlib


def compute_doc_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
```

A **content hash** is a fixed-length fingerprint of the text — the same content always produces the same hash, and any change to the content, even a single character, produces a completely different hash. This one property is doing two jobs at once, which Stage 4 uses directly.

**Check yourself before Stage 4:** If two files at different paths happen to contain byte-for-byte identical text, what would `compute_doc_id` return for each? Why might that be useful rather than a problem?

---

## Stage 4 — Don't index the same thing twice, and don't re-index what hasn't changed

Here's a problem you'll hit the moment you run ingestion more than once: if you naively re-run your pipeline every day over the same folder, you'll re-embed every single document every single day — including the 99% that haven't changed since yesterday. Embedding calls cost money and time. Doing this at real scale, daily, over thousands of documents, is wasteful in a way that adds up fast.

The content hash from Stage 3 fixes this cleanly.

```python
def detect_changes(current_docs: dict[str, str], known_hashes: dict[str, str]) -> tuple[list[str], list[str]]:
    """
    current_docs: {source_path: content} for everything currently on disk
    known_hashes: {source_path: doc_id} from the last successful ingestion run

    Returns (paths_to_reindex, paths_that_were_removed)
    """
    to_reindex = []
    for path, content in current_docs.items():
        new_hash = compute_doc_id(content)
        if known_hashes.get(path) != new_hash:
            to_reindex.append(path)   # new file, or content changed since last run

    removed = [path for path in known_hashes if path not in current_docs]

    return to_reindex, removed
```

Run this before doing any embedding work, and you only pay the embedding cost for files that are genuinely new or genuinely changed — everything else is skipped entirely. `removed` matters too: if a source document was deleted, its chunks need to be deleted from your vector index as well, or you'll keep retrieving and citing a document that no longer exists.

This same content-hash approach also gives you Stage 3's deduplication for free: if the exact same content shows up at two different file paths, `compute_doc_id` produces the same hash for both, and you can detect and skip the duplicate rather than storing (and paying to embed) it twice.

**Check yourself before Stage 5:** Why does comparing content hashes let you skip re-embedding unchanged files, when comparing file modification timestamps alone would not reliably tell you the same thing? (Hint: think about what happens if a file gets re-saved with identical content — does its timestamp change even though nothing meaningful did?)

---

## Stage 5 — Cleaning: the boring step that quietly matters

Real files carry junk that doesn't belong in your index: null bytes from encoding issues, triple-or-more blank lines from PDF extraction artifacts, boilerplate that shows up on every single page of a website (navigation bars, footers, cookie notices).

```python
def clean_text(text: str) -> str:
    text = text.replace("\x00", "")             # stray null bytes from some extractors
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive blank lines
    return text.strip()
```

This looks trivial, and it mostly is — but skipping it has a real, compounding cost. If every page of a scraped website includes the same 200-word navigation menu, and you index 500 pages, that navigation text now appears 500 times across your corpus. During retrieval, a query that happens to match navigation-menu wording (which is common, generic language) can pull in irrelevant boilerplate chunks ahead of the one page that actually contains the answer — purely because the boilerplate is *everywhere*, and everywhere-ness looks like relevance to a naive retrieval system.

**Check yourself before the project:** Why does repeated boilerplate across many documents actively hurt retrieval quality, rather than just being harmlessly ignored?

---

# The Project: A Real Ingestion Pipeline With Incremental Updates

## What you're building

A pipeline that walks a real folder of mixed file types, parses each correctly, cleans the text, deduplicates, and — critically — on a second run, only processes what actually changed.

```bash
$ python ingest.py ./docs
```

```
scanning ./docs...
  found: policy.md, handbook.html, pricing.csv, notes.pdf
  parsed: 4 files -> 4 documents
  new/changed: 4  (first run)
  removed: 0
indexed 4 documents, 23 chunks, in 3.2s
```

```bash
$ python ingest.py ./docs        # run again, nothing changed
```

```
scanning ./docs...
  new/changed: 0
  removed: 0
nothing to do. index unchanged.
```

```bash
$ echo "Updated policy text" >> ./docs/policy.md
$ python ingest.py ./docs
```

```
new/changed: 1  (policy.md)
removed: 0
re-indexed 1 document, 6 chunks, in 0.4s
```

## Why this project

This is what actually runs, on a schedule, in front of any real RAG system — Part XI's project assumed documents were static and pre-cleaned; this is where you deal with reality. You'll reuse this exact incremental-update pattern in [Part XVI](part16-qdrant-deep-dive.md) when the target is a real vector database instead of an in-memory list.

## Before you start

- A folder with at least one file of each type: `.md`, `.html`, `.csv`, and a PDF (`pip install pdfplumber` for the PDF stage)
- Your indexing code from Part XI's project

## Build it in stages

---

### Stage A — Parse every file type in your test folder

**Goal:** Every file in your folder produces clean text, or is clearly skipped with a message explaining why.

**Do:**
1. Implement `PARSERS` and `parse_file` for `.md`, `.html`, `.csv`
2. Add PDF support using `pdfplumber` — extract text page by page, and keep track of which page each chunk of text came from as metadata
3. Walk your test folder, parse everything, print a summary of what was parsed vs. skipped

**You're done when:** Every real file in your folder either produces sensible clean text, or is clearly and deliberately skipped (not silently dropped with no explanation).

---

### Stage B — Real HTML cleaning

**Goal:** Replace the naive regex-based `parse_html` with something that actually handles real, messy HTML correctly.

**Do:**
1. Install `trafilatura` (or a similar content-extraction library)
2. Use it to extract just the main content from an HTML page, discarding navigation, ads, and footers
3. Test it against a real saved web page with visible navigation/boilerplate, and manually confirm the extracted text is just the actual article content

**You're done when:** You can point to a real webpage's HTML, run extraction, and confirm the navigation menu and footer are gone but the actual article text survived intact.

---

### Stage C — Content-hash deduplication and metadata

**Goal:** Every parsed document gets a content-based `doc_id` and useful metadata.

**Do:**
1. Implement `compute_doc_id` and `NormalizedDoc`
2. Build a `NormalizedDoc` for every successfully parsed file, with metadata including at least: file extension, character count, and (for PDFs) page number if you tracked it
3. Add a duplicate file (copy one file's content to a new filename) to your test folder, re-run, and confirm the duplicate is detected via matching content hash

**You're done when:** Duplicate content across different filenames is correctly identified as the same `doc_id`, and you can see this happening in your output.

---

### Stage D — Incremental re-indexing, for real

**Goal:** The exact behavior shown in the example output above — a second run does nothing, and a changed file triggers exactly one re-index.

**Do:**
1. Persist `known_hashes` (source path → doc_id) to a small JSON file after every successful run
2. On startup, load it, and use `detect_changes` to figure out what actually needs processing
3. Only run parsing, cleaning, chunking, and embedding for files in `to_reindex`
4. Handle `removed`: delete their chunks from your index

**You're done when:** You've watched, for real, three consecutive runs: first run processes everything, second run (no changes) processes nothing and says so clearly, third run (one file edited) processes only that one file.

---

### Stage E — Measure the savings

**Goal:** Quantify what incremental updates actually bought you, not just confirm the mechanism works.

**Do:**
1. Time a full re-index of your whole test folder from scratch
2. Change one file, time an incremental update
3. Report the speedup as a ratio

**You're done when:** You have a real number — e.g., "full re-index: 4.1s for 20 documents; incremental update after 1 change: 0.3s" — and can explain why the difference scales even more dramatically as the document count grows.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Same document shows up twice with different `doc_id`s | Content differs by something trivial (trailing whitespace, line-ending differences) before hashing | Run `clean_text` *before* computing the hash, not after, so trivial formatting differences don't produce different hashes |
| Incremental update reprocesses everything every time, even unchanged files | Comparing file modification time instead of content hash, or not persisting `known_hashes` between runs | Confirm the JSON file with known hashes is actually being written and re-loaded correctly |
| PDF text comes out as a jumbled, unreadable mess | Multi-column PDF layout confusing simple text extraction | Check `pdfplumber`'s layout-aware extraction options, or accept that heavily-formatted PDFs need extra handling and flag them for manual review rather than silently indexing garbage |
| HTML extraction includes navigation text despite using `trafilatura` | Some pages don't follow common content-vs-boilerplate patterns the library expects | Spot-check extraction output on a sample of your real pages; don't assume the library is perfect on 100% of inputs |

## Going further (optional)

1. **OCR fallback for scanned PDFs.** Detect a PDF page with no extractable text layer (an image-only scan) and route it through OCR (`pytesseract`) instead of failing silently. Track and store a rough "confidence" flag on OCR'd content, since it's more error-prone than native text extraction.
2. **A `--dry-run` flag** that shows what *would* change without actually re-indexing — useful for reviewing before committing to an expensive re-embedding run on a large corpus.
3. **Scheduled ingestion.** Wire this into a cron job or scheduled task that runs nightly against a real, growing folder, and watch the incremental-update behavior play out over several real days rather than a single test session.

---

## Checkpoint

**Knowledge:**
1. Why does bad ingestion set a hard ceiling on RAG quality that no later stage can fix?
2. Why use a content hash rather than a file modification timestamp to detect changes?
3. Why does repeated boilerplate text across many pages actively hurt retrieval, not just add harmless noise?

**Implementation:** Build `detect_changes` and the incremental-update flow from memory.

**Debugging:** After ingesting a folder of 50 mixed-format documents, retrieval quality on questions about one specific PDF is noticeably worse than for the `.md` files. Walk through how you'd isolate whether the problem is in parsing, cleaning, or something downstream in chunking/embedding.

**Architecture:** You're ingesting 10,000 documents a day from an external source that occasionally re-sends the same document under a different filename. Sketch how the content-hash approach from this part handles that case without any extra code.

---

**Next:** [Part XIII — Chunking](part13-chunking.md), where you'll learn exactly how to split this clean text without breaking the facts inside it.
