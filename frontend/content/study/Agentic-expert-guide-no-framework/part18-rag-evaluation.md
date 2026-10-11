# Part XVIII — RAG Evaluation

## Before you start

**What you need already:** [Part XI](part11-rag-fundamentals.md)–[XVII] — a working advanced RAG pipeline. [Part V](part05-prompt-engineering.md)'s test harness, which this part extends specifically for RAG.

**What this part gives you:** Throughout Parts XI–XVII, you've been testing with a handful of hand-picked questions and eyeballing the results. That works for building intuition, but it doesn't scale, doesn't catch regressions, and can't tell you objectively whether a change actually helped. This part turns "I tried some questions and it seemed fine" into a repeatable measurement you can run before every change.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Build a real evaluation suite that separately measures whether retrieval found the right thing and whether generation used it correctly, and set up a check that blocks a change from shipping if it makes things worse.

---

## Stage 1 — Why "I tried a few questions" isn't enough

Here's the problem with informal testing, made concrete. Suppose you change your chunk size from 300 to 500 characters. You try your usual 3 test questions. They all still work. Ship it.

Two things could be true here, and you can't tell which from 3 questions: the change genuinely didn't hurt anything, or the change broke 2 out of your other 50 real users' common question types — ones you just didn't happen to try. Small, informal test sets have no statistical power to catch a regression that affects a minority of cases, and in a real system, "a minority of cases" can still mean thousands of frustrated users.

The fix isn't complicated — it's discipline: build a real test set, run it every time, and compare numbers, not vibes. This mirrors exactly what [Part V](part05-prompt-engineering.md)'s prompt harness did for prompts; this part does the same for the whole RAG pipeline.

**Check yourself before Stage 2:** Why can a change pass your usual 3 test questions and still be a real regression for other users?

---

## Stage 2 — Two separate things can go wrong, and you need to measure them separately

A RAG answer can be wrong for two genuinely different reasons, and conflating them wastes time debugging the wrong layer:

**Retrieval failure:** the system didn't find the right chunk. The generation step never had a chance — you can't generate a correct answer from context that doesn't contain it.

**Generation failure:** the system found the right chunk, but the model still got the answer wrong — misread it, ignored it, or answered from its own frozen training knowledge instead of the provided context.

These need different fixes. A retrieval failure points you back to chunking (Part XIII) or embeddings (Part XIV). A generation failure points you to prompting (Part V) or the "answer using only the context" instruction. Debugging one when the real problem is the other wastes time — you need a way to measure each independently.

**Check yourself before Stage 3:** Give an example scenario for each: one where retrieval is the problem, and one where generation is the problem, with the same wrong final answer in both cases.

---

## Stage 3 — Measuring retrieval alone

The question retrieval evaluation answers: **for this query, did the right chunk actually come back?** Nothing about generation is involved yet.

```python
from dataclasses import dataclass


@dataclass
class RetrievalTestCase:
    query: str
    relevant_doc_ids: set[str]   # which documents SHOULD be in the results


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Of the documents that SHOULD have been found, what fraction actually were,
    within the top k results?"""
    top_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 1.0
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """How close to the TOP was the first relevant result? Rank 1 scores
    1.0, rank 2 scores 0.5, rank 4 scores 0.25, not found scores 0."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

Why two different metrics, not just one? They catch different problems. **Recall@k** tells you "did we find it at all, somewhere in the top k" — useful when you're going to show several results and any of them being present is a win. **MRR** cares about exact rank — useful when only the *first* result really matters, because a relevant document buried at rank 8 is much less useful in practice than one at rank 1, even though both count identically toward recall@10.

```python
def run_retrieval_eval(cases: list[RetrievalTestCase], retrieve_fn, k: int = 5) -> dict:
    recalls, mrrs = [], []
    for case in cases:
        retrieved_ids = [c.doc_id for c in retrieve_fn(case.query)]
        recalls.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrrs.append(mrr(retrieved_ids, case.relevant_doc_ids))

    n = len(cases)
    return {f"recall@{k}": sum(recalls) / n, "mrr": sum(mrrs) / n}
```

This is the metric you should have run, and could have used to *predict* a regression, in Stage 1's chunk-size-change scenario — rather than discovering it later from user complaints.

**Check yourself before Stage 4:** Why might a chunking change hurt recall@5 while leaving MRR roughly unchanged, or vice versa? What would each pattern tell you?

---

## Stage 4 — Measuring generation alone: did it actually use what it was given?

The question generation evaluation answers, given that retrieval succeeded and the right context *was* provided: **did the final answer actually reflect that context, faithfully, without inventing anything?**

```python
FAITHFULNESS_PROMPT = """Given the CONTEXT and the ANSWER below, identify each
distinct factual claim made in the ANSWER. For each claim, determine whether
it is directly supported by the CONTEXT. Return JSON:
{{"claims": [{{"claim": "...", "supported": true/false}}, ...]}}

CONTEXT:
{context}

ANSWER:
{answer}
"""


async def faithfulness_score(context: str, answer: str, client) -> float:
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response, _ = await client.ask(prompt)
    import json
    result = json.loads(response)
    claims = result["claims"]
    if not claims:
        return 1.0
    return sum(c["supported"] for c in claims) / len(claims)
```

This is **LLM-as-judge**: using a model call to evaluate another model's output against a specific, narrow criterion — here, "is every claim in this answer actually backed by the provided context." It's checking something a simple string match couldn't: not whether the answer contains certain words, but whether its *claims* are genuinely grounded, which requires the kind of reading comprehension only a model (or a human) can do.

### The part people skip, and shouldn't: calibrate the judge

A judge model can be wrong, just like any other LLM call — miscalibrated, too lenient, too strict, or inconsistent. Before trusting its scores, check them against your own judgment on a handful of cases.

```python
def calibrate_judge(judge_scores: list[float], human_scores: list[float]) -> float:
    """Simple correlation check — do the judge's scores actually track
    human judgment, or is it just producing plausible-looking numbers?"""
    import numpy as np
    return float(np.corrcoef(judge_scores, human_scores)[0, 1])
```

Score 10 answers yourself, run the same 10 through `faithfulness_score`, and check the correlation. If it's low, the judge prompt needs work — don't trust its numbers on your full test set until it's been checked against real human judgment at least once. This is the exact same discipline as validating any automated test: an automated check that doesn't correlate with reality is worse than no check at all, because it creates false confidence.

**Check yourself before Stage 5:** Why must an LLM judge be calibrated against human judgment before you trust its scores? What would you conclude if the correlation came back near zero?

---

## Stage 5 — Assembling a real evaluation suite

Put retrieval and generation metrics together, running against a real, growing set of test cases.

```python
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class RAGTestCase:
    query: str
    relevant_doc_ids: set[str]
    expected_facts: list[str]   # phrases that should appear if correctly answered


async def run_full_eval(cases: list[RAGTestCase], rag_pipeline, client) -> dict:
    recalls, mrrs, faithfulness_scores, fact_hits = [], [], [], []

    for case in cases:
        retrieved_chunks = retrieve(rag_pipeline.index, case.query, rag_pipeline.embed_fn, k=5)
        retrieved_ids = [c.doc_id for c in retrieved_chunks]

        recalls.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k=5))
        mrrs.append(mrr(retrieved_ids, case.relevant_doc_ids))

        answer = await answer_with_rag(retrieved_chunks, case.query, client)
        context = "\n\n".join(c.text for c in retrieved_chunks)

        faithfulness_scores.append(await faithfulness_score(context, answer, client))
        fact_hits.append(all(fact.lower() in answer.lower() for fact in case.expected_facts))

    n = len(cases)
    return {
        "recall@5": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "faithfulness": sum(faithfulness_scores) / n,
        "fact_accuracy": sum(fact_hits) / n,
    }
```

Four separate numbers, each telling you something different: `recall@5` and `mrr` are purely about retrieval; `faithfulness` is purely about whether generation stayed grounded; `fact_accuracy` is your own hard check on whether the specific facts you expected actually showed up. If a change makes `fact_accuracy` drop but `recall@5` stays the same, you know immediately to look at generation, not retrieval — the separation of concerns from Stage 2, made operational.

**Check yourself before Stage 6:** If `recall@5` drops after a change but `faithfulness` stays high, what does that tell you about where the problem is — and where it isn't?

---

## Stage 6 — The regression gate: don't let a bad change ship

The whole point of measuring is to use the measurement *before* shipping, not just after something breaks in production.

```python
def check_for_regression(new_scores: dict, baseline_scores: dict, tolerance: float = 0.05) -> list[str]:
    regressions = []
    for metric, baseline_value in baseline_scores.items():
        new_value = new_scores.get(metric)
        if new_value is not None and new_value < baseline_value - tolerance:
            regressions.append(
                f"{metric} regressed: {baseline_value:.3f} -> {new_value:.3f} "
                f"(dropped by {baseline_value - new_value:.3f}, tolerance is {tolerance})"
            )
    return regressions


def save_baseline(scores: dict, path: Path = Path("rag_eval_baseline.json")) -> None:
    path.write_text(json.dumps(scores, indent=2))
```

The workflow this enables: save your current scores as a baseline. Make a change — new chunk size, new embedding model, a tweaked prompt. Re-run the eval suite. Compare against the baseline. If anything regressed beyond your tolerance, you know *before* deploying, not after users notice.

This is exactly the same discipline [Part V Stage 7](part05-prompt-engineering.md) introduced for prompts, applied to the whole RAG pipeline — and it directly foreshadows [Part XLVIII](part48-deployment.md), where this kind of check becomes an automated gate that blocks a deployment outright.

**Check yourself before the project:** Why compare against a saved baseline rather than just checking whether the current scores look "good enough" in isolation?

---

# The Project: A RAG Evaluation Suite That Catches Real Regressions

## What you're building

A complete evaluation harness for your Qdrant-backed RAG pipeline, with a calibrated faithfulness judge and a regression gate — then use it to catch a regression you deliberately introduce.

```bash
$ python rag_eval.py --save-baseline
```

```
Running 20 test cases...
recall@5: 0.85   mrr: 0.79   faithfulness: 0.91   fact_accuracy: 0.80
baseline saved.
```

```bash
$ python rag_eval.py --compare
```

```
Running 20 test cases...
recall@5: 0.60   mrr: 0.55   faithfulness: 0.88   fact_accuracy: 0.55

REGRESSIONS DETECTED:
  recall@5 regressed: 0.850 -> 0.600 (dropped by 0.250)
  mrr regressed: 0.790 -> 0.550 (dropped by 0.240)
  fact_accuracy regressed: 0.800 -> 0.550 (dropped by 0.250)

exit code: 1
```

## Why this project

This is the tool you should run before *any* change to your RAG pipeline for the rest of this curriculum — chunking changes, embedding model swaps, prompt tweaks. Build it carefully now and it pays off for everything downstream.

## Before you start

- Your full Qdrant-backed RAG pipeline from Parts XI–XVII
- At least 15–20 real questions you can build proper test cases for

## Build it in stages

---

### Stage A — Build a real test set

**Goal:** 20 test cases with known relevant documents and expected facts — not made up on the spot, but genuinely reflective of questions your system should handle.

**Do:**
1. For each question, identify exactly which document(s) should be retrieved
2. Write specific expected facts/phrases that should appear in a correct answer
3. Include at least 3 multi-hop questions (from Part XVII) and 3 questions with no answer in your corpus at all (testing the honest-decline path)

**You're done when:** You have a JSON file of test cases specific enough that a human could grade any given answer against it unambiguously.

---

### Stage B — Retrieval metrics

**Goal:** `recall_at_k` and `mrr`, running against your real pipeline.

**Do:**
1. Implement both functions
2. Run them against all 20 test cases
3. Look at the individual per-question results, not just the average — which specific questions have low scores, and why?

**You're done when:** You have real recall@5 and MRR numbers, and you've manually inspected at least 2 low-scoring cases to understand what went wrong.

---

### Stage C — Calibrated faithfulness judge

**Goal:** A working `faithfulness_score`, checked against your own judgment before you trust it.

**Do:**
1. Implement the judge prompt and scoring
2. Run it against 10 real answers from your pipeline
3. Score those same 10 yourself, blind (don't look at the judge's scores first)
4. Compute the correlation

**You're done when:** You have a real correlation number, and if it's low, you've revised the judge prompt and re-checked until it's reasonably aligned with your own judgment.

---

### Stage D — Full suite and baseline

**Goal:** All four metrics running together, with a saved baseline.

**Do:**
1. Implement `run_full_eval` combining everything
2. Run it once, save the results as your baseline
3. Confirm the saved JSON file has sensible values for all four metrics

**You're done when:** You have a committed baseline file you could hand to a teammate as "this is what correct looks like."

---

### Stage E — Deliberately break something, and watch the gate catch it

**Goal:** Prove the regression gate actually works, not just trust that it should.

**Do:**
1. Make a deliberately bad change — set your chunk size absurdly small, or lower your relevance threshold to something clearly too permissive
2. Re-run the full eval suite
3. Confirm `check_for_regression` correctly flags the damage, and that the specific metrics that dropped point you toward the actual cause (chunking vs. generation)
4. Revert the change and confirm scores return to baseline

**You're done when:** You've watched a real regression get caught by your own tooling, with the flagged metrics correctly pointing at the layer you broke.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| All scores are suspiciously perfect on first run | Test set too easy, or test cases were written by looking at what the system already returns rather than independently | Write test cases by hand, from the documents, before ever running them against the system |
| Faithfulness judge correlates poorly with your own scoring | Judge prompt too vague about what counts as "supported" | Tighten the prompt with a clear definition and, if needed, a worked example of both a faithful and an unfaithful answer |
| Regression gate never triggers even on an obviously bad change | Tolerance set too loose, or averaging across too few test cases hides the damage | Lower the tolerance; ensure your test set is large enough that a real regression moves the average meaningfully |
| Retrieval scores look fine but overall answer quality is clearly worse | The regression is in generation, not retrieval — check `faithfulness` and `fact_accuracy` specifically, not just recall/MRR | This is expected behavior from Stage 2's separation — use it to correctly localize the problem |

## Going further (optional)

1. **Synthetic test case generation.** Use an LLM to generate candidate (question, expected answer) pairs directly from your document chunks, as a way to bootstrap a larger test set faster than writing every case by hand. Manually review the generated cases before trusting them — auto-generated tests can be subtly too easy.
2. **Per-category breakdown.** Tag your test cases by type (simple factual, multi-hop, unanswerable) and report scores per category, not just an overall average — a regression hiding in one category can be invisible in an aggregate number.
3. **Wire the gate into a script that exits non-zero on regression** — this is literally what you'd hook into a CI pipeline later, in [Part XLVIII](part48-deployment.md), to block a bad deploy automatically rather than relying on someone remembering to run the eval manually.

---

## Checkpoint

**Knowledge:**
1. Why can a change pass a small, informal test set and still be a real regression?
2. Why measure retrieval and generation separately rather than just checking whether the final answer is correct?
3. Why must an LLM judge be calibrated against human scoring before you trust it?
4. Why compare against a saved baseline rather than just checking if current scores look acceptable?

**Implementation:** Build `run_retrieval_eval`, a calibrated faithfulness judge, and `check_for_regression` from memory.

**Debugging:** After a routine dependency update, your evaluation suite shows `recall@5` unchanged but `faithfulness` dropped significantly. What component would you investigate first, and why does this pattern point you there specifically?

**Architecture:** Design a CI pipeline stage that runs this evaluation suite automatically on every pull request that touches the RAG pipeline, and blocks the merge if any metric regresses beyond tolerance. What would you need to make this fast enough to run on every PR without becoming a bottleneck?

---

**Next:** [Part XIX — LangChain](part19-langchain.md), where you'll see how the retrieval, chunking, and generation pieces you've built by hand across Parts XI–XVIII map onto a real, widely-used framework's abstractions.
