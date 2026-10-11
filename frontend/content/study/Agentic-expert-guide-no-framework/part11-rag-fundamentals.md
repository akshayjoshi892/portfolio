# Part XI — RAG Fundamentals

## Before you start

**What you need already:** [Part II](part02-ml-foundations.md)'s search project (you built a real, working retrieval system there — this part explains what you were actually building and why it matters). [Part III](part03-llm-fundamentals.md) (why models hallucinate, and why their knowledge is frozen at training time).

**What this part gives you:** The reason RAG exists, stated precisely, and a complete, honest end-to-end pipeline — including the parts that go wrong, which most introductions skip.

**Time:** 3 days.

**When you finish, you'll be able to:** Explain exactly what problem RAG solves and what it doesn't, build a working retrieve-then-generate pipeline with source citations, and recognize the difference between "the system retrieved something" and "the system retrieved the *right* thing."

---

## Stage 1 — The problem, stated precisely

You already know two facts from [Part III](part03-llm-fundamentals.md):

1. A model's knowledge is frozen at the moment training data was collected. It cannot know about your company's internal policies, anything that happened after training, or anything that was simply never in its training data.
2. When the model doesn't actually know something, it doesn't reliably say "I don't know" — it predicts likely-looking tokens, which can produce a fluent, confident, completely wrong answer.

Put these together and you get a genuinely dangerous combination: ask an LLM "what's our company's refund policy?" and if it has no idea, it might not tell you that. It might *invent* a plausible-sounding policy, phrased with exactly the same confidence as a true one.

RAG's fix is almost embarrassingly simple to state: **before answering, find the actual relevant text, and put it directly in front of the model.** Instead of relying on frozen training-time knowledge, ground the answer in real, current, specific information supplied at the moment of the question.

```
Without RAG:  "What's our refund policy?" → model guesses from training data → maybe wrong
With RAG:     "What's our refund policy?" → find the actual policy doc → model reads it → answers from it
```

This reframes what an LLM is doing. It's not "the knowledge base." It's a *reading and reasoning engine* — you hand it the facts, it produces a coherent answer from them. That's a more reliable job for it than "recall a fact from memory," because recall is exactly where hallucination lives.

**Check yourself before Stage 2:** Why is "the model might invent a plausible-sounding wrong policy" specifically more dangerous than "the model says it doesn't know"?

---

## Stage 2 — The two phases, and why they're separate

RAG has two genuinely different phases, running at genuinely different times, and conflating them is a common source of confusion.

**Indexing** — done once, or periodically, whenever your documents change. Take documents, break them into pieces, turn each piece into a vector, store them. This is offline work; nobody's waiting on it in real time.

**Query** — done every single time a user asks something. Turn their question into a vector, find the most similar stored pieces, hand those pieces to the model along with the question.

```
INDEXING (happens ahead of time, offline)
  documents → parse → chunk → embed → store in a vector index

QUERY (happens live, every request)
  question → embed → search the index → get top matches → build a prompt → generate answer
```

You built the query phase's core mechanism already, in [Part II](part02-ml-foundations.md)'s search project — embed a query, find the closest matches by cosine similarity. RAG adds one more step after that: instead of just *showing* the matches to a human, you hand them to an LLM and ask it to *answer using them*.

**Check yourself before Stage 3:** Why does it matter that indexing and query happen at different times? What would go wrong if you tried to re-index your entire document set on every single query?

---

## Stage 3 — Build the whole pipeline, end to end

You've built pieces of this already. Now assemble the complete thing, including the generation step that's new here.

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Chunk:
    doc_id: str
    text: str
    embedding: np.ndarray


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def index_documents(docs: dict[str, str], embed_fn) -> list[Chunk]:
    index = []
    for doc_id, text in docs.items():
        for piece in chunk_text(text):
            index.append(Chunk(doc_id=doc_id, text=piece, embedding=embed_fn(piece)))
    return index


def retrieve(index: list[Chunk], query: str, embed_fn, k: int = 3) -> list[Chunk]:
    q_emb = embed_fn(query)
    scored = [(c, float(q_emb @ c.embedding)) for c in index]   # both normalized -> dot product = cosine
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in scored[:k]]


def build_grounded_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[Source: {c.doc_id}]\n{c.text}" for c in chunks)
    return f"""Answer the question using ONLY the information in the context below.
If the context doesn't contain the answer, say "I don't have information about that."
Cite which source(s) you used, like [Source: filename].

Context:
{context}

Question: {query}
"""
```

Notice the prompt does two specific things that matter, drawing directly on [Part V](part05-prompt-engineering.md): it explicitly tells the model to use *only* the provided context (constraining it away from falling back on frozen training knowledge when the retrieved text doesn't actually cover the question), and it asks for citations, so a human reading the answer can verify where each claim came from rather than taking it on faith.

```python
async def answer_with_rag(index: list[Chunk], query: str, embed_fn, client) -> str:
    retrieved = retrieve(index, query, embed_fn, k=3)
    prompt = build_grounded_prompt(query, retrieved)
    answer, _ = await client.ask(prompt)
    return answer
```

That's a complete, working RAG pipeline. Every framework you'll meet later — LangChain's retrievers, LlamaIndex, a managed RAG service — is doing exactly this underneath, with more configuration surface and more edge-case handling, but the same four steps: chunk, embed, retrieve, generate-with-context.

**Check yourself before Stage 4:** What does telling the model to answer "using ONLY the context" actually prevent? What happens if you omit that instruction?

---

## Stage 4 — Watch it work, and then watch it fail

Run the pipeline on a real, small document set:

```python
docs = {
    "leave_policy.md": "Employees may take 15 days of paid leave per year. "
                        "Leave must be requested at least 2 weeks in advance.",
    "handbook.md": "The company was founded in 2015 and is headquartered "
                    "in Austin, Texas. We have 40 employees.",
}

index = index_documents(docs, real_embed_fn)
answer = await answer_with_rag(index, "How much paid leave do employees get?", real_embed_fn, client)
print(answer)
```

```
Employees get 15 days of paid leave per year, and it must be requested at
least 2 weeks in advance. [Source: leave_policy.md]
```

Good — grounded, correct, cited. Now ask something the documents genuinely don't cover:

```python
answer = await answer_with_rag(index, "What is the CEO's name?", real_embed_fn, client)
print(answer)
```

If your prompt instruction is doing its job, you should get something like "I don't have information about that" — because neither document mentions a CEO. If instead you get a confident, invented name, that's exactly the failure RAG was supposed to prevent, still slipping through. It usually means one of two things: the prompt instruction wasn't strong enough, or — more subtly — retrieval returned *something* (the closest-matching chunks, even if none are actually relevant), and the model treated "I was given some context" as license to answer from it rather than checking whether the context actually addresses the question.

That second failure mode is worth sitting with, because it's the most common real bug in RAG systems, and it's the subject of the next stage.

**Check yourself before Stage 5:** Why might retrieval return chunks even when none of them are actually relevant to the question? What would you expect to happen to the similarity scores in that case, compared to a genuinely relevant match?

---

## Stage 5 — The gap between "retrieved something" and "retrieved the right thing"

Here's the trap: `retrieve()` in Stage 3 *always* returns `k` chunks. It has no concept of "nothing here is actually relevant" — it just returns whatever scored highest among what exists, even if the highest score is mediocre.

```python
retrieved = retrieve(index, "What is the CEO's name?", embed_fn, k=3)
for c in retrieved:
    score = float(embed_fn("What is the CEO's name?") @ c.embedding)
    print(f"{score:.3f}  {c.doc_id}: {c.text[:60]}...")
```

```
0.312  handbook.md: The company was founded in 2015 and is headquar...
0.298  leave_policy.md: Employees may take 15 days of paid leave...
0.201  leave_policy.md: Leave must be requested at least 2 weeks...
```

Those scores are low — nothing here genuinely matches "CEO's name" well. Compare against the earlier, actually-relevant query about leave policy, where you'd expect scores well above 0.7 or 0.8. The *system* has no built-in sense of "these scores are too low to be useful" — that judgment has to be added explicitly.

### The fix: a relevance threshold

```python
def retrieve_with_threshold(index: list[Chunk], query: str, embed_fn,
                             k: int = 3, min_score: float = 0.5) -> list[Chunk]:
    q_emb = embed_fn(query)
    scored = [(c, float(q_emb @ c.embedding)) for c in index]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    above_threshold = [(c, s) for c, s in scored if s >= min_score]
    return [c for c, _ in above_threshold[:k]]


async def answer_with_rag_honest(index: list[Chunk], query: str, embed_fn, client) -> str:
    retrieved = retrieve_with_threshold(index, query, embed_fn, k=3)
    if not retrieved:
        return "I don't have any relevant information to answer that question."
    prompt = build_grounded_prompt(query, retrieved)
    answer, _ = await client.ask(prompt)
    return answer
```

Now the system has an honest way to say "nothing relevant was found" *before* the question ever reaches the model — rather than hoping the model notices the context doesn't help and declines to answer anyway. This is strictly more reliable, because it doesn't depend on the model consistently following an instruction; it's a hard check in your own code.

The right threshold depends on your embedding model and your data — there's no universal number. You find it by testing, which is exactly what [Part XVIII](part18-rag-evaluation.md) builds a proper framework for. For now, the important thing is recognizing that **a similarity score is a *continuous* number, and "relevant enough to trust" is a judgment call you have to make explicitly, not something that falls out automatically from retrieval returning results.**

**Check yourself before Stage 6:** Why does `retrieve()` always return `k` results even when none of them are actually relevant? What does adding a threshold change about the system's behavior?

---

## Stage 6 — Why chunking, embeddings, and the vector store each deserve their own part

You've now built a full, working, honest RAG pipeline. Three of its components are simple in this stage's version — a fixed-size chunker, a cosine-similarity loop, an in-memory list — and each one is a real lever on quality that gets its own dedicated part:

**Chunking** ([Part XIII](part13-chunking.md)) determines what's *possible* to retrieve at all. If a fact gets split across a chunk boundary — say, a table row separated from its header — no amount of clever retrieval afterward can put it back together, because retrieval only ever returns whole chunks as they were stored.

**Embeddings** ([Part XIV](part14-embeddings.md)) determine what "similar" even means. A general-purpose embedding model can badly underperform on jargon-heavy domains — legal contracts, medical terminology, your company's internal acronyms — because it was never trained on enough text like that to learn good distinctions within it.

**The vector store** ([Part XV](part15-vector-databases.md)) determines whether any of this still works once you have millions of chunks instead of a handful. You already measured, in [Part II Stage 5](part02-ml-foundations.md), exactly where a simple in-memory loop like `retrieve()` above stops being fast enough — that's precisely the problem a real vector database solves.

This stage's pipeline is complete and honest, but it's the small-scale, simplified version. Each of the next few parts takes one piece of it and makes it production-real.

**Check yourself before the project:** In your own words, what does each of chunking, embeddings, and the vector store control, and what specifically breaks if each one is done badly?

---

# The Project: A RAG System You Can Trust

## What you're building

Extend the search engine from Part II's project into a complete RAG system — with generation, citations, and an honest "I don't know" path, tested against questions the documents actually can and can't answer.

```bash
$ python rag.py "What's the refund policy for annual plans?"
```

```
retrieved 3 chunks, top score: 0.84
answer: Annual plans can be refunded within 30 days of purchase, prorated
based on unused months. [Source: pricing_policy.md]
```

```bash
$ python rag.py "What's the CEO's favorite color?"
```

```
retrieved 3 chunks, top score: 0.29 (below threshold 0.5)
answer: I don't have any relevant information to answer that question.
```

## Why this project

This becomes the foundation for every later RAG part — chunking strategies get compared against this pipeline's output in Part XIII, embedding models get swapped in and out in Part XIV, and the whole thing gets a real evaluation framework in Part XVIII. Build it carefully once.

## Before you start

- Your working search project from Part II (real embeddings, a small real document corpus)
- Your `LLMClient` from Part IV

## Build it in stages

---

### Stage A — Wire retrieval into generation

**Goal:** The full pipeline from Stage 3, working against your real documents and a real model.

**Do:**
1. Reuse your Part II indexing/retrieval code
2. Add `build_grounded_prompt` and `answer_with_rag`
3. Ask 3 questions your documents genuinely can answer, confirm the answers are correct and grounded

**You're done when:** You get correct, cited answers to real questions, and you can point to exactly which chunk each answer's facts came from.

---

### Stage B — Deliberately ask what it can't answer

**Goal:** See the hallucination-under-RAG failure with your own eyes before fixing it.

**Do:**
1. Ask 3 questions your documents clearly do *not* cover
2. Without a threshold yet, see what the model does — note whether it correctly declines or invents an answer anyway

**You're done when:** You have real evidence, from your own documents and your own model, of whether the "answer only from context" instruction alone was sufficient — or whether you witnessed it fail.

---

### Stage C — Add the relevance threshold

**Goal:** Make "I don't have relevant information" a property enforced by your code, not just hoped for from the model.

**Do:**
1. Implement `retrieve_with_threshold`
2. Print the top similarity score for every query, so you can see the numbers, not just the outcome
3. Re-run Stage B's unanswerable questions and confirm they now trigger the honest decline path *before* the model is even called

**You're done when:** All of Stage B's unanswerable questions are now caught by the threshold, and you've tuned `min_score` by actually looking at real scores for both relevant and irrelevant queries on your data — not guessed.

---

### Stage D — Verify citations are actually correct

**Goal:** Don't just check that a citation exists — check that it's the *right* one.

**Do:**
1. For 5 answered questions, manually verify: does the cited source actually contain the fact stated in the answer?
2. If you find a mismatch (the answer is correct but cites the wrong source, or vice versa), investigate whether it's a retrieval problem (wrong chunk returned) or a generation problem (model cited the wrong one despite having the right chunk in context)

**You're done when:** You've done this check for real, and you understand from firsthand testing — not assumption — how reliable your citations currently are.

---

### Stage E — A held-out test set

**Goal:** Move from "I tried some questions" to a repeatable, honest measurement.

**Do:**
1. Write 10 questions: 5 answerable from your documents, 5 not
2. For each, record whether the system's behavior was correct: an answerable question got a correct, cited answer; an unanswerable one got the honest decline
3. Report your score as a fraction, e.g., "9/10 correct"

**You're done when:** You have a real number, and you know exactly which case (if any) failed and why — this is a small hand-built version of exactly what [Part XVIII](part18-rag-evaluation.md) formalizes.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Model answers unanswerable questions confidently anyway, even with a threshold | Threshold set too low for your embedding model's actual score distribution | Print real scores for known-irrelevant queries and set the threshold above their range |
| Model declines to answer questions it should be able to answer | Threshold set too high, or chunks are too large/small for the fact to be captured well | Lower the threshold slightly; check whether the relevant fact is intact within a single chunk |
| Citations point to the wrong document | Retrieval brought back the right document but with a nearly-tied-score chunk from a different document also present in context | Reduce `k`, or check whether your chunking is splitting single facts oddly |
| Everything scores similarly regardless of query | Embeddings aren't actually differentiating content — possible bug in the embedding call itself | Verify with a sanity check: does searching for a chunk's *exact* text return that chunk with a near-1.0 score? |

## Going further (optional)

1. **Multi-document conflict.** Add two documents that state slightly different things about the same topic (e.g., an old and a new pricing page). Ask a question that would surface both. Does your prompt handle this gracefully, or does the model just pick one silently? This foreshadows real production messiness that clean single-source examples hide.
2. **Streaming the grounded answer.** Combine this with Part IV's streaming client so the answer streams token by token, while citations are still verifiable at the end. Notice you can't apply the threshold check *after* streaming has started — it has to happen before generation begins.
3. **Compare `k` values.** Run your 10-question test set with `k=1`, `k=3`, and `k=6`. Does more context help, hurt, or make no difference? You now have the tools to answer this with evidence instead of intuition.

---

## Checkpoint

**Knowledge:**
1. State precisely what problem RAG solves, referencing what you know from Part III about how models generate text.
2. Why are indexing and query genuinely separate phases, not just two halves of one process?
3. Why does `retrieve()` always return `k` results even when nothing is actually relevant, and what does adding a threshold fix?
4. Name the three components (beyond retrieval and generation themselves) that this part identified as deserving their own deeper treatment, and briefly state what each one controls.

**Implementation:** Build the full pipeline — chunk, embed, retrieve with threshold, generate with citations — from memory.

**Debugging:** A RAG answer confidently cites a real source document, but the fact stated isn't actually in that document. Walk through how you'd determine whether the bug is in retrieval (wrong chunk surfaced) or generation (model misattributed the source despite correct retrieval).

**Architecture:** You're designing a RAG system for a knowledge base that gets updated daily with new documents. Sketch, at a high level, how indexing needs to change from this part's "index everything once at startup" approach to handle ongoing updates — you'll build this properly in [Part XII](part12-document-ingestion.md).

---

**Next:** [Part XII — Document Ingestion](part12-document-ingestion.md), where you'll deal with the real-world mess this part's clean, pre-written `.md` documents skipped over — PDFs, HTML, tables, and documents that change over time.
