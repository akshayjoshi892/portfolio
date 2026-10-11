# Part XVII — Advanced RAG

## Before you start

**What you need already:** [Part XI](part11-rag-fundamentals.md)–[XVI] — a working, filtered, Qdrant-backed RAG pipeline. This part is about the specific ways that pipeline still fails, and the fixes for each.

**What this part gives you:** Your Part XI pipeline does one embed-and-retrieve pass per question. That's often not enough. This part covers the real failure cases — a query that doesn't match the document's phrasing at all, a question needing facts from two different documents, a retrieval that comes back empty or wrong — and a specific technique for each.

**Time:** 3–4 days.

**When you finish, you'll be able to:** Recognize which specific retrieval failure you're looking at, and reach for the right fix instead of guessing.

---

## Stage 1 — A failure your current pipeline can't handle: the wording gap

Try this against your Part XI/XVI pipeline:

```python
# Your document says:
# "Employees may take 15 days of paid leave per year."

# You ask:
question = "How many vacation days do I accrue annually?"
```

Depending on your embedding model, this might work fine — "vacation days" and "paid leave" are close enough in meaning that a good model bridges the gap. But push it further:

```python
question = "What's the PTO allowance?"
```

"PTO" (paid time off) is an abbreviation that may or may not be well-represented in your embedding model's training data, depending on how much workplace-jargon text it saw. If the model's understanding of "PTO" is weak, this query's embedding might land meaningfully far from the document's embedding — not because the topic is actually unrelated, but because the *specific phrasing* the user chose happens to sit awkwardly in the model's vector space.

This is a real, common failure: the answer is in your corpus, but the user's specific wording doesn't embed close enough to the document's specific wording, even though a human reading both would immediately see they're about the same thing.

**Check yourself before Stage 2:** Why might "PTO allowance" embed farther from "paid leave" than you'd expect, even though they mean essentially the same thing to a person?

---

## Stage 2 — Fix: rewrite the query before embedding it

If the user's exact wording is the problem, don't search with their exact wording — use an LLM call to rephrase the question into something more likely to match how the answer is actually written.

```python
async def rewrite_query(question: str, client) -> str:
    prompt = f"""Rewrite this question to use more common, formal phrasing,
as it might appear in a company policy document. Keep the same meaning.
Return only the rewritten question.

Question: {question}
"""
    rewritten, _ = await client.ask(prompt)
    return rewritten.strip()
```

```python
original = "What's the PTO allowance?"
rewritten = await rewrite_query(original, client)
print(rewritten)
# "How many days of paid leave are employees entitled to per year?"
```

Now embed and search with the *rewritten* version instead of the original. This is a small, cheap extra LLM call that can meaningfully improve recall on exactly the kind of wording-mismatch case from Stage 1 — without touching your chunking, embedding model, or index at all.

**Check yourself before Stage 3:** Why does rewriting the query fix a wording-mismatch problem, when the underlying document text and embedding model haven't changed at all?

---

## Stage 3 — A more aggressive version: HyDE

Here's a technique that sounds backwards at first but works surprisingly well for exactly this problem. Instead of rewriting the *question*, have the model **write a hypothetical answer** to it — a plausible-sounding passage that might appear in a real document — and search using *that*, instead of the question itself.

```python
async def hyde_search(question: str, embed_fn, index, client, k=3):
    prompt = f"""Write a short, plausible passage (2-3 sentences) that might
appear in a company policy document, answering this question. It doesn't
need to be factually correct — it just needs to be written in the style
of a real policy document.

Question: {question}
"""
    hypothetical_answer, _ = await client.ask(prompt)

    # search using the HYPOTHETICAL ANSWER's embedding, not the question's
    hyp_embedding = embed_fn(hypothetical_answer)
    return retrieve_by_embedding(index, hyp_embedding, k=k)
```

Why would searching with a made-up answer work better than searching with the actual question? Because a question and its answer are often phrased very differently — "What's the PTO allowance?" is a question; "Employees are entitled to 15 days of paid leave annually" is a statement, in the same grammatical register real documents are written in. **A hypothetical answer, even a wrong one, is often closer in phrasing and structure to the real answer than the original question is** — HyDE stands for Hypothetical Document Embeddings, and this is exactly the intuition behind the name.

This is called **HyDE** (Hypothetical Document Embeddings). It costs an extra LLM call per query, and it's worth testing on your own data rather than assuming it'll help — for some corpora it's a clear win, for others plain query rewriting is enough and HyDE's extra cost isn't justified. Measure, per the evaluation discipline from [Part V](part05-prompt-engineering.md).

**Check yourself before Stage 4:** Why might a hypothetical (even factually wrong) answer embed closer to the real answer than the original question does?

---

## Stage 4 — A failure your pipeline can't handle at all: multi-hop questions

Try this:

```python
question = "Who is the CEO of the company that acquired our main competitor?"
```

Your documents might separately state: "Our main competitor is Beta Corp" (in one document) and "Beta Corp was acquired by Gamma Industries" (in a different document) and "Gamma Industries' CEO is John Smith" (in yet another). No single chunk contains the full answer. A single retrieve-and-generate pass, no matter how well-tuned, cannot answer this — because the answer genuinely requires chaining three separate facts together, and standard retrieval finds chunks similar to the *original* question, which doesn't closely resemble any single one of those three facts on its own.

**Check yourself before Stage 5:** Why can't better chunking or a better embedding model alone fix this kind of question? What's fundamentally different about it compared to the wording-gap problem from Stage 1?

---

## Stage 5 — Fix: decompose, retrieve per sub-question, then combine

The fix mirrors how a person would actually solve this: break it into sub-questions, answer each with its own retrieval pass, then combine.

```python
async def decompose_question(question: str, client) -> list[str]:
    prompt = f"""This question may require multiple pieces of information found
in different places. Break it into a sequence of simpler sub-questions that,
answered in order, would let you answer the original question. If it's already
simple, return just the original question. Return one sub-question per line.

Question: {question}
"""
    response, _ = await client.ask(prompt)
    return [line.strip() for line in response.strip().split("\n") if line.strip()]


async def multi_hop_answer(question: str, index, embed_fn, client) -> str:
    sub_questions = await decompose_question(question, client)
    print(f"decomposed into: {sub_questions}")

    all_evidence = []
    for sq in sub_questions:
        chunks = retrieve(index, sq, embed_fn, k=2)
        all_evidence.extend(chunks)

    context = "\n\n".join(f"[Source: {c.doc_id}]\n{c.text}" for c in all_evidence)
    final_prompt = f"""Using the evidence below, answer the original question.
Cite sources.

Evidence:
{context}

Original question: {question}
"""
    answer, _ = await client.ask(final_prompt)
    return answer
```

Each sub-question gets its *own* retrieval pass — "Who is our main competitor?" retrieves the Beta Corp fact, "Who acquired Beta Corp?" retrieves the Gamma Industries fact, and so on. By the time all the sub-questions have been retrieved for, the combined evidence actually contains the full chain, even though no single retrieval pass on the *original* question ever could have found it.

**Watch out for an unbounded chain:** a genuinely hard question might decompose into sub-questions that themselves need decomposing. Bound this, exactly like every loop in this curriculum — cap the number of hops, and if you're not converging, fail honestly rather than looping.

**Check yourself before Stage 6:** Why does decomposing into sub-questions solve a problem that better chunking couldn't? What specifically would you need to add to make this safe against a runaway decomposition chain?

---

## Stage 6 — A failure your pipeline hides instead of surfacing: bad retrieval that looks fine

Here's the subtlest failure in this whole part. Your Part XI pipeline has a relevance threshold — if nothing scores high enough, it honestly says "I don't have information about that." But what about the case where retrieval returns something *above* the threshold, but it's still not actually the right thing? The system won't complain. It'll confidently generate an answer from mediocre context, and nothing in the pipeline flags that anything went wrong.

### Fix: grade what came back before trusting it

```python
async def grade_retrieval(question: str, chunks: list, client) -> str:
    """Ask the model itself to judge whether the retrieved chunks are
    actually sufficient to answer the question — a distinct check from
    the similarity-score threshold, catching cases where the score was
    high enough to pass but the content still isn't actually useful."""
    context = "\n\n".join(c.text for c in chunks)
    prompt = f"""Given this retrieved context, can the following question be
fully and confidently answered? Respond with exactly one word: "good",
"partial", or "bad".

Context:
{context}

Question: {question}
"""
    grade, _ = await client.ask(prompt)
    return grade.strip().lower()


async def corrective_rag(question: str, index, embed_fn, client, web_search_fn=None) -> str:
    chunks = retrieve(index, question, embed_fn, k=3)
    grade = await grade_retrieval(question, chunks, client)
    print(f"retrieval grade: {grade}")

    if grade == "bad" and web_search_fn is not None:
        print("falling back to web search")
        chunks = await web_search_fn(question)

    return await answer_with_rag(chunks, question, client)
```

This is called **Corrective RAG**: retrieve, grade the retrieval itself (a distinct step from grading the *final answer*), and only fall back to an alternate source (a web search, a broader index, or an honest decline) if the grade comes back poor. It costs one extra LLM call per question, in exchange for catching the specific failure mode where similarity score alone wasn't a reliable enough signal of actual usefulness.

**Check yourself before the project:** Why is grading the *retrieved chunks* a meaningfully different check from the relevance-score threshold you built in Part XI? What kind of failure does each one catch that the other misses?

---

# The Project: A Corrective, Multi-Hop RAG Pipeline

## What you're building

Extend your Qdrant-backed RAG pipeline with query rewriting, multi-hop decomposition, and retrieval grading — tested against questions specifically designed to break the plain Part XI version.

```bash
$ python advanced_rag.py "Who is the CEO of the company that acquired our biggest rival?"
```

```
decomposed into: ['Who is our biggest rival?', 'What company acquired that rival?', "Who is that company's CEO?"]
[hop 1] retrieved: rival = Beta Corp
[hop 2] retrieved: Beta Corp was acquired by Gamma Industries
[hop 3] retrieved: Gamma Industries' CEO is John Smith

answer: John Smith is the CEO of Gamma Industries, which acquired Beta Corp,
your biggest rival. [Sources: competitors.md, acquisitions.md, leadership.md]
```

## Why this project

This is where a RAG system stops being a demo and starts handling the messy, real questions users actually ask — ones that don't map cleanly onto a single retrieval pass. You'll reuse this evaluation-driven approach directly in [Part XVIII](part18-rag-evaluation.md).

## Build it in stages

---

### Stage A — Build a corpus that requires multiple hops

**Goal:** A small set of documents where at least 3 questions genuinely require combining facts from 2+ separate documents — not answerable from any single chunk.

**Do:**
1. Write 4–5 short documents with facts that chain together (like the competitor/acquisition/CEO example)
2. Write 3 multi-hop questions and manually verify the answer requires facts from multiple documents

**You're done when:** You've confirmed, by trying it against your Part XI pipeline, that these questions genuinely fail with plain single-pass retrieval.

---

### Stage B — Query rewriting

**Goal:** `rewrite_query`, measurably improving recall on wording-mismatch questions.

**Do:**
1. Implement it against a real LLM call
2. Write 3 questions using deliberately different vocabulary than your documents (abbreviations, informal phrasing, synonyms)
3. Compare retrieval scores with and without rewriting

**You're done when:** You have real before/after similarity scores showing whether rewriting helped on your specific wording-mismatch test cases.

---

### Stage C — Multi-hop decomposition

**Goal:** Answer your Stage A multi-hop questions correctly.

**Do:**
1. Implement `decompose_question` and `multi_hop_answer`
2. Run all 3 multi-hop questions from Stage A
3. Confirm each sub-question's retrieval actually found the relevant fact, and the final synthesis correctly chains them

**You're done when:** All 3 previously-failing multi-hop questions now get correctly answered, with sources cited for each hop.

---

### Stage D — Retrieval grading and fallback

**Goal:** `corrective_rag`, catching the "retrieval passed threshold but is still bad" case.

**Do:**
1. Implement `grade_retrieval`
2. Construct a deliberate test case: a question whose top retrieved chunk scores above your relevance threshold but doesn't actually answer the question well (e.g., a related-but-not-quite-right document)
3. Confirm the grading step catches this and either falls back or clearly signals low confidence, rather than confidently answering from insufficient context

**You're done when:** You've caused this specific failure deliberately and watched the grading step catch it.

---

### Stage E — Measure the cost of all this sophistication

**Goal:** Know exactly what these techniques cost, not just that they help.

**Do:**
1. For your full test question set, run the plain Part XI pipeline and the advanced pipeline from this part
2. Record total LLM calls and total latency for each
3. Report the accuracy improvement against the added cost

**You're done when:** You can state, with real numbers, something like "advanced RAG improved accuracy from 60% to 90% on multi-hop questions, at 3x the LLM calls per question" — an honest trade-off, not just "it's better."

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Decomposition produces sub-questions that are just rephrasings of the original | The question wasn't actually multi-hop, or the decomposition prompt is too aggressive | Confirm with Stage A that your test questions genuinely require multiple documents before troubleshooting decomposition itself |
| Multi-hop synthesis cites the wrong source for a given fact | Evidence from different hops gets merged without hop-level tracking | Keep each hop's evidence separately labeled in the final synthesis prompt, not just concatenated |
| Corrective RAG's grading step always says "good" regardless of actual quality | Grading prompt too lenient, or grading against the same context that generation will use, without a genuinely independent check | Test the grading prompt in isolation against a deliberately bad context, confirm it correctly says "bad" |

## Going further (optional)

1. **Bound the decomposition depth.** Add a hop limit and test what happens with a question that decomposes into sub-questions that themselves need decomposing — confirm it fails safely rather than looping.
2. **Multi-query retrieval.** Instead of one rewritten query, generate 3 differently-phrased variants and retrieve for each, merging results. Compare recall against single-query rewriting.
3. **Real web search fallback** for Corrective RAG, if you have access to a search API — confirm the fallback actually triggers correctly when your indexed corpus genuinely lacks the answer.

---

## Checkpoint

**Knowledge:**
1. Why does query rewriting help with a wording-mismatch problem that better chunking or embeddings alone can't fix?
2. Why might a hypothetical (possibly wrong) answer embed closer to the real answer than the original question does?
3. Why can't a single retrieval pass answer a genuinely multi-hop question, regardless of how good the retrieval is?
4. What does retrieval grading catch that a similarity-score threshold alone misses?

**Implementation:** Build query decomposition and multi-hop retrieval from memory.

**Debugging:** A RAG system gives a confidently wrong answer to a question that should be answerable from your documents. Walk through how you'd determine whether the cause is a wording mismatch (Stage 1), a multi-hop question your system isn't decomposing (Stage 4), or a retrieval that passed the threshold but was still low-quality (Stage 6).

**Architecture:** Design an "adaptive" retrieval router that classifies incoming questions as simple, needing rewriting, or needing multi-hop decomposition, and routes each to the appropriate pipeline — rather than always running every question through the most expensive path.

---

**Next:** [Part XVIII — RAG Evaluation](part18-rag-evaluation.md), where the ad-hoc "try a few questions and eyeball it" testing you've been doing throughout Parts XI–XVII becomes a real, repeatable measurement framework.
