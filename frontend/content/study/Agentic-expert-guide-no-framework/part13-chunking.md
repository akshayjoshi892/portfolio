# Part XIII — Chunking

## Before you start

**What you need already:** [Part XI](part11-rag-fundamentals.md) (you used a simple fixed-size chunker there without examining it closely), [Part XII](part12-document-ingestion.md) (clean text to chunk).

**What this part gives you:** A close look at the one step in the whole RAG pipeline that's easiest to get wrong without noticing, because a badly-chunked system doesn't crash — it just quietly gives worse answers, and you may never trace the cause back here unless you know to look.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Explain exactly why chunk boundaries matter, choose between chunking strategies based on evidence rather than habit, and build a chunker that respects document structure instead of cutting blindly by character count.

---

## Stage 1 — A chunk is the smallest thing you can ever retrieve

Here's the idea to hold onto through this whole part: **retrieval never returns part of a chunk, or several chunks merged together. It returns whole chunks, exactly as you stored them.** Whatever boundary you draw at chunking time is a boundary you're stuck with at retrieval time, permanently, for that document.

So if a fact spans a boundary — imagine a chunk that ends with "...the maximum refund amount is" and the next chunk begins with "$500, applicable only to annual plans" — retrieval can return one chunk or the other, but the complete fact only exists split across both. If only the first chunk gets retrieved (because "maximum refund amount" scores well against the query, but the number itself isn't present to help), the model sees an incomplete sentence and either can't answer or, worse, guesses the missing number.

This isn't a hypothetical. It's the single most common reason a RAG system fails on a question whose answer is clearly, visibly present in the source document — the answer was there, but it got cut in half.

**Check yourself before Stage 2:** Why can't retrieval "put back together" a fact that got split across two chunks, even if both chunks are retrieved?

---

## Stage 2 — Fixed-size chunking, and watching it fail on purpose

The chunker from Part XI split every N characters, with no awareness of what's actually in the text. Build it again, deliberately, and then break it on purpose so the problem is visible rather than theoretical.

```python
def fixed_size_chunk(text: str, size: int = 200, overlap: int = 0) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
```

Now feed it something with a clear, checkable fact right around where a chunk boundary is likely to land:

```python
text = (
    "Refund Policy. Customers on annual plans are eligible for a full refund "
    "within 30 days of purchase. After 30 days, the maximum refund amount is "
    "$500, applicable only to annual plans purchased directly through our website."
)

chunks = fixed_size_chunk(text, size=120, overlap=0)
for i, c in enumerate(chunks):
    print(f"chunk {i}: {c!r}")
```

```
chunk 0: 'Refund Policy. Customers on annual plans are eligible for a full refund within 30 days of purchase. After 30 days, the max'
chunk 1: 'imum refund amount is $500, applicable only to annual plans purchased directly through our website.'
```

Look at chunk 0. It ends mid-word: "...the max". The actual number, $500, is entirely in chunk 1. If a query about "maximum refund after 30 days" happens to score chunk 0 more highly (it does contain "maximum refund" and "30 days," after all) and chunk 1 doesn't make the cut, the model gets a chunk that raises the question but never answers it.

**Try it:** Set `overlap=0` and `size=80` on the text above and look for a case where a fact is fully absent from any single chunk. This is worth seeing with your own eyes, on your own text, at least once.

**Check yourself before Stage 3:** Why does `size=120` split "maximum" mid-word here? What determines exactly where a fixed-size chunker cuts, and does it know or care what it's cutting through?

---

## Stage 3 — Overlap: a partial fix, and why it's only partial

The obvious first fix: make chunks share some text at their boundaries, so a fact near a boundary has a better chance of being intact in at least one chunk.

```python
chunks = fixed_size_chunk(text, size=120, overlap=40)
for i, c in enumerate(chunks):
    print(f"chunk {i}: {c!r}")
```

```
chunk 0: 'Refund Policy. Customers on annual plans are eligible for a full refund within 30 days of purchase. After 30 days, the max'
chunk 1: 's. After 30 days, the maximum refund amount is $500, applicable only to annual plans purchased dir'
chunk 2: 'to annual plans purchased directly through our website.'
```

Now chunk 1 contains the full fact — "maximum refund amount is $500" — intact. Overlap bought you this by duplicating the last 40 characters of one chunk at the start of the next, so a boundary that would have split a fact in chunk 0's version now has a second chance in chunk 1.

But notice the cost: you're now storing roughly 133% of your original text (every chunk after the first repeats 40 characters from before it), which means more chunks, more embedding calls, more storage, and more near-duplicate content in your index. Overlap reduces the *probability* of a split fact being unretrievable — it doesn't eliminate it. A large enough fact, or unlucky enough boundary placement, can still get split even with generous overlap.

**Check yourself before Stage 4:** What does overlap actually cost you, in exchange for reducing (not eliminating) the split-fact problem?

---

## Stage 4 — The better fix: cut where the document already has boundaries

Here's the deeper problem with fixed-size chunking: it treats the text as an undifferentiated stream of characters, completely ignoring that the *document itself* already has natural structural boundaries — paragraph breaks, section headings, sentence endings — that were put there specifically to separate distinct ideas.

**Recursive chunking** respects this: try splitting on the largest, most meaningful separator first (like a double newline, signaling a paragraph break). If a resulting piece is still too big, only *then* fall back to a smaller separator (a single newline, then a sentence boundary, then — only as a last resort — a hard character cut).

```python
def recursive_chunk(text: str, size: int = 200,
                     separators: list[str] = None) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= size:
        return [text] if text.strip() else []

    for sep_index, sep in enumerate(separators):
        if sep not in text:
            continue

        pieces = text.split(sep)
        chunks: list[str] = []
        buffer = ""

        for piece in pieces:
            candidate = (buffer + sep + piece) if buffer else piece
            if len(candidate) <= size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(piece) > size:
                    # this single piece is still too big — recurse with
                    # the NEXT, smaller separator
                    chunks.extend(recursive_chunk(piece, size, separators[sep_index + 1:]))
                    buffer = ""
                else:
                    buffer = piece

        if buffer:
            chunks.append(buffer)
        return chunks

    # nothing left to split on and still oversized — last resort, hard cut
    return [text[i:i + size] for i in range(0, len(text), size)]
```

Run it against text with real paragraph structure:

```python
sample = (
    "# Refund Policy\n\n"
    "Customers on annual plans are eligible for a full refund within 30 days of purchase. "
    "After 30 days, the maximum refund amount is $500.\n\n"
    "# Support Hours\n\n"
    "Our support team is available Monday through Friday, 9am to 6pm Eastern Time."
)

for i, c in enumerate(recursive_chunk(sample, size=140)):
    print(f"chunk {i}: {c!r}")
```

```
chunk 0: '# Refund Policy\n\nCustomers on annual plans are eligible for a full refund within 30 days of purchase.'
chunk 1: 'After 30 days, the maximum refund amount is $500.'
chunk 2: '# Support Hours\n\nOur support team is available Monday through Friday, 9am to 6pm Eastern Time.'
```

Notice the refund policy section and the support hours section never get mixed into the same chunk — the `\n\n` paragraph/section boundary is respected first, and only where a section is *still* too big does it fall back to splitting on sentences. Compare this to fixed-size chunking's chunk 0 from Stage 2, which cut mid-word with no regard for what it was cutting through.

**Check yourself before Stage 5:** Why does recursive chunking try the largest separator (`\n\n`) first, rather than starting with the smallest? What would go wrong if it tried them in the opposite order?

---

## Stage 5 — Small chunks for precision, big chunks for context: having both

Here's a genuine tension you'll run into: small chunks make retrieval more *precise* (a tightly focused chunk about one specific fact scores cleanly against a matching query), but they give the model less surrounding context to correctly interpret that fact once retrieved. Large chunks give more context but dilute the embedding — a chunk covering five different subtopics doesn't score cleanly against any single one of them.

**Parent-child chunking** resolves this by not choosing — it keeps two sizes simultaneously, one for searching and a different one for reading.

```python
from dataclasses import dataclass


@dataclass
class DocChunk:
    text: str
    parent_id: str | None = None


def parent_child_chunk(text: str, parent_size: int = 1000, child_size: int = 200) -> tuple[list[DocChunk], list[DocChunk]]:
    parents = [DocChunk(text=t) for t in recursive_chunk(text, parent_size)]

    children = []
    for i, parent in enumerate(parents):
        parent_id = f"parent_{i}"
        for child_text in recursive_chunk(parent.text, child_size):
            children.append(DocChunk(text=child_text, parent_id=parent_id))

    return parents, children
```

The idea: **embed and search only the small children** — precise, focused vectors that score well against specific queries. **But when a child is retrieved, hand the model its larger parent, not the tiny child itself** — so generation gets the surrounding context a small chunk alone would have lost.

```python
parents, children = parent_child_chunk(sample, parent_size=200, child_size=60)
print(f"{len(parents)} parents, {len(children)} children")

# at retrieval time: search `children`, then look up each hit's parent by parent_id
# and hand the PARENT text to the model, not the child that matched
```

This is often called "small-to-big" retrieval, and it's genuinely one of the more effective, low-effort improvements you can make to a RAG system once fixed-size or plain recursive chunking stops being enough.

**Check yourself before Stage 6:** Why search on the small children but generate using the large parent, rather than doing both with the same chunk size?

---

## Stage 6 — Stop guessing: measure which strategy actually wins on your data

Everything above is reasoning about chunking in the abstract. The only way to know which strategy is actually better *for your specific documents and your specific questions* is to measure it — exactly the discipline [Part II](part02-ml-foundations.md)'s search project and [Part V](part05-prompt-engineering.md)'s prompt harness both taught you.

```python
def evaluate_chunking_strategy(text: str, chunk_fn, test_questions: list[tuple[str, str]], embed_fn) -> dict:
    """test_questions: list of (question, exact_phrase_the_answer_should_contain)"""
    chunks = chunk_fn(text)
    chunk_embeddings = [embed_fn(c) for c in chunks]

    hits = 0
    for question, expected_phrase in test_questions:
        q_emb = embed_fn(question)
        scored = sorted(zip(chunks, chunk_embeddings),
                         key=lambda pair: float(q_emb @ pair[1]), reverse=True)
        top_chunk = scored[0][0]
        if expected_phrase.lower() in top_chunk.lower():
            hits += 1

    return {
        "num_chunks": len(chunks),
        "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks),
        "accuracy": hits / len(test_questions),
    }
```

Run this same function against fixed-size, recursive, and parent-child (checking the *child*'s parent, in that case) chunking on the *same* document and the *same* test questions, and you get a real, evidence-based answer to "which strategy is actually better here" — not a guess based on which one sounds more sophisticated.

**Check yourself before the project:** Why is testing chunking strategies against your own specific documents more trustworthy than following a general rule of thumb about which strategy is "best"?

---

# The Project: Chunking Strategies, Measured Against Each Other

## What you're building

An evaluation harness that runs fixed-size, recursive, and parent-child chunking against the same real multi-section document and the same test questions, and reports which one actually wins — with real numbers, on your own data.

```bash
$ python chunking_eval.py --doc handbook.md --questions test_questions.json
```

```
fixed_size (size=200, overlap=0):
  12 chunks, avg 198 chars
  accuracy: 5/10 (50%)

fixed_size (size=200, overlap=50):
  15 chunks, avg 197 chars
  accuracy: 7/10 (70%)

recursive (size=200):
  10 chunks, avg 182 chars
  accuracy: 9/10 (90%)

parent_child (parent=800, child=150):
  6 parents, 21 children
  accuracy: 9/10 (90%)

WINNER on this document: recursive (tied with parent_child), both +40pp over naive fixed-size
```

## Why this project

This is exactly the kind of evidence-based comparison this whole part argued for. You'll reuse this harness's shape directly in [Part XVIII](part18-rag-evaluation.md), where it grows into a full RAG evaluation framework.

## Build it in stages

---

### Stage A — A real multi-section test document, and real test questions

**Goal:** A document with genuine structure — several distinct sections, at least one table or list, facts near section boundaries — and 10 questions with known, checkable answers.

**Do:**
1. Find or write a real document with at least 4 distinct sections (a real policy doc, a product FAQ, documentation you actually have)
2. Write 10 questions, each with a specific expected phrase that should appear in whatever chunk correctly answers it
3. Deliberately include a couple of questions whose answers sit right near a natural section boundary — these are the ones most likely to expose a bad chunking strategy

**You're done when:** You have a document and question set specific enough that a human reading them could confidently say whether a given chunk answers a given question.

---

### Stage B — Implement all three strategies

**Goal:** `fixed_size_chunk`, `recursive_chunk`, and `parent_child_chunk`, all working against your real document.

**Do:**
1. Implement each as shown in this part
2. Run each against your document and print the resulting chunks — read them. Do the recursive chunks look meaningfully more coherent than the fixed-size ones on your actual text?

**You're done when:** You can look at output from all three strategies on the same document and describe, in your own words, how they differ.

---

### Stage C — Score all three against your test questions

**Goal:** Real accuracy numbers, per strategy.

**Do:**
1. Implement `evaluate_chunking_strategy`
2. For parent-child, make sure you're checking the *parent* text (what actually gets handed to the model), not just the child that matched during search
3. Run all three strategies (fixed-size with two different overlap settings counts as two runs) against your 10 questions
4. Report the results in a table like the example output

**You're done when:** You have real percentages for each strategy, from your own document and your own questions — not from this file.

---

### Stage D — Investigate every failure

**Goal:** For every question any strategy got wrong, understand *why* — don't just record the failure and move on.

**Do:**
1. For each miss, print which chunk was actually retrieved as the top match
2. Determine: was the fact actually present in that chunk (a retrieval-ranking problem) or was it split across chunks, absent from the top match entirely (a chunking-boundary problem)?

**You're done when:** You can classify every failure as one or the other, with the actual retrieved chunk as evidence — this is the habit that turns "the RAG system is bad at this" into "the RAG system fails at this specific, fixable thing."

---

### Stage E — Tune based on what you found

**Goal:** Use your Stage D findings to make one deliberate improvement, then re-measure.

**Do:**
1. Pick one change suggested by your failure analysis (larger chunk size? more overlap? smaller parent-child child size?)
2. Re-run the evaluation
3. Confirm — with the same numbers as before — whether the change actually helped, on your own data

**You're done when:** You've made at least one evidence-driven improvement and can quote the before/after accuracy numbers.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| All strategies score identically | Test questions are too easy — their answers exist as a single unambiguous phrase regardless of chunk boundaries | Add harder questions whose answers sit near natural section transitions |
| Recursive chunking doesn't outperform fixed-size at all | Test document lacks real structural boundaries (no paragraph breaks, no headings) to exploit | Use a document with genuine section structure, not one big block of undifferentiated text |
| Parent-child scoring seems to cheat | Accidentally checking the small child's text for the expected phrase instead of its parent's | Make sure evaluation looks up and checks the *parent*, matching what would actually be handed to the model |
| Accuracy is inconsistent across re-runs | Embedding calls aren't deterministic in some setup, or test questions are ambiguous | Confirm you're using the same embedding model/settings consistently; sharpen ambiguous questions |

## Going further (optional)

1. **Semantic chunking.** Embed individual sentences, compute similarity between consecutive sentences, and cut a new chunk wherever similarity drops sharply (a topic shift). Compare its accuracy and its embedding-call cost against recursive chunking on the same test set.
2. **Code-aware chunking.** If you have a codebase handy, try chunking a source file by character count vs. by function/class boundaries (using Python's `ast` module, similar to what you'll do in [Part XLIV](part44-code-agents.md)). Confirm fixed-size chunking regularly splits functions mid-body, while AST-aware chunking never does.
3. **Chunk size sweep.** Run recursive chunking at sizes 100, 200, 400, 800 against your test set. Plot accuracy against chunk size. Is there a clear sweet spot, or does it plateau? This kind of sweep is standard practice before locking in a production chunking config.

---

## Checkpoint

**Knowledge:**
1. Why is a chunk "the smallest thing you can ever retrieve," and what does that imply about where you draw boundaries?
2. What does overlap actually fix, and what does it not fix — why is it a partial solution rather than a complete one?
3. Why does recursive chunking try the largest separator first?
4. In parent-child chunking, what gets searched and what gets handed to the model, and why are they different?

**Implementation:** Build `recursive_chunk` and `parent_child_chunk` from memory.

**Debugging:** A RAG system fails to answer a question whose answer is clearly, visibly present in the source document. Walk through how you'd determine whether the cause is chunking (the fact got split across a boundary) or something else (retrieval ranking, or the embedding model itself).

**Architecture:** You're chunking a corpus that mixes long-form prose documents, structured tables, and source code. Sketch a chunking strategy — it likely isn't one single strategy — for handling all three well within one pipeline.

---

**Next:** [Part XIV — Embeddings](part14-embeddings.md), where you'll go deeper on the piece that determines what "similar" even means — and what happens when you change embedding models on a corpus that's already indexed.
