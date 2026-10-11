# Part XXXI — Reasoning

## Before you start

**What you need already:** [Part V Stage 4](part05-prompt-engineering.md) (chain-of-thought, and the explicit warning that longer reasoning doesn't automatically mean better results), [Part VIII Stage 6](part08-agent-architectures.md) (reflection, and its limitation: it can only catch what the model already, in some sense, knows is wrong).

**What this part gives you:** The actual fix for reflection's limitation — grounding a claim in something *external and checkable*, not just another round of the model reading its own work.

**Time:** 2 days.

**When you finish, you'll be able to:** Build a reasoning step that verifies a claim against a real, deterministic check instead of trusting the model's self-assessment, and know precisely when that extra step is worth its cost.

---

## Stage 1 — The gap reflection can't close

Recall Part VIII's honest limitation: reflection asks the model to check its own work, but if the model is genuinely wrong about something — not violating a stated instruction, but factually mistaken — asking it to "double-check" gives it no new information. It's the same model, using the same (mistaken) understanding, re-reading the same claim. It often just confirms its own error with added confidence.

Watch this concretely:

```python
async def naive_reflection(claim: str, client) -> str:
    prompt = f"""Double-check this claim for accuracy: "{claim}"
Is it correct? If not, provide the correction."""
    response, _ = await client.ask(prompt)
    return response
```

If the model's original claim was "6 × 7 = 43" because of an arithmetic slip, asking it to "double-check" doesn't hand it a calculator — it hands it the same flawed process that produced the error in the first place. Sometimes it'll catch its own mistake on a second pass (rereading can occasionally surface an error), but there's no structural reason it reliably will, and for genuinely non-obvious errors, it often won't.

**Check yourself before Stage 2:** Why does asking a model to "double-check" its own claim not give it any new information to work with?

---

## Stage 2 — The fix: check against something that isn't the model

The fix is exactly what [Part VI](part06-structured-outputs-tool-calling.md) already gave you the mechanism for: a **tool**. Not the model reasoning about its own claim — a deterministic function that actually computes the real answer, which the model's claim then gets checked against.

```python
def calculator(expr: str) -> float:
    allowed = set("0123456789+-*/(). ")
    if not set(expr) <= allowed:
        raise ValueError("disallowed characters")
    return eval(expr, {"__builtins__": {}}, {})


async def verified_claim(question: str, client) -> str:
    # Step 1: model proposes an answer AND the expression that justifies it
    proposal_prompt = f"""Answer this, and show the exact arithmetic expression
you used. Format: ANSWER: <number> EXPRESSION: <expression>

Question: {question}
"""
    response, _ = await client.ask(proposal_prompt)
    # (parsing omitted for brevity — extract claimed_answer and expression)
    claimed_answer, expression = parse_answer_and_expression(response)

    # Step 2: check the claim against a REAL, deterministic computation —
    # not another round of the model reading its own work
    actual_answer = calculator(expression)

    if abs(actual_answer - claimed_answer) > 1e-6:
        return f"Self-correction: claimed {claimed_answer}, but {expression} actually equals {actual_answer}"
    return f"Verified: {actual_answer}"
```

The critical difference from Stage 1: `calculator(expression)` doesn't care what the model believes. It computes the real result, using ordinary, deterministic arithmetic — the same trustworthy mechanism from [Part VI](part06-structured-outputs-tool-calling.md). If the model's claimed answer doesn't match, you *know* something is wrong, with certainty, not with another guess.

**Check yourself before Stage 3:** Why does checking against `calculator(expression)` give you certainty that reflection alone couldn't?

---

## Stage 3 — Not every claim has a calculator: what verification actually requires

Here's the honest limit of this technique, worth stating clearly before you reach for it everywhere. Tool-assisted verification requires that the claim be checkable by some deterministic or reliably-authoritative process. Arithmetic has a calculator. Code has a test suite (you'll build this properly in [Part XLIV](part44-code-agents.md)). A fact about a document has the document itself (this is exactly what RAG's groundedness check does, from [Part XVIII Stage 4](part18-rag-evaluation.md)).

But "is this the best word choice for this sentence" or "is this argument persuasive" has no calculator. There's no deterministic function that checks subjective quality. For these, you're back to reflection's weaker guarantee — useful for catching *stated, checkable* criteria (length, required sections, format), unreliable for anything requiring judgment the model itself doesn't reliably have.

```python
def choose_verification_strategy(claim_type: str) -> str:
    """A simple, explicit decision — not a technique to apply uniformly."""
    verifiable_types = {"arithmetic", "code_correctness", "factual_from_document"}
    if claim_type in verifiable_types:
        return "tool_assisted_verification"
    return "reflection_only"   # weaker guarantee, use with appropriate skepticism
```

**Check yourself before Stage 4:** Give one example of a claim that tool-assisted verification can reliably check, and one that it fundamentally cannot — and explain what makes the difference.

---

## Stage 4 — Self-consistency: a different kind of check, for a different kind of claim

For claims where there's no single deterministic tool to check against, but the task *does* have a real, single correct answer (a logic puzzle, a multiple-choice-style classification), there's a middle-ground technique: ask multiple times, independently, and see if the answers agree.

```python
import asyncio
from collections import Counter


async def self_consistency(question: str, client, n_samples: int = 5) -> tuple[str, float]:
    # Sampled at a nonzero temperature so the n runs can actually differ —
    # recall from Part III that temperature=0 gives you the same answer every time
    responses = await asyncio.gather(*(
        client.ask(question, temperature=0.7) for _ in range(n_samples)
    ))
    answers = [extract_final_answer(r) for r, _ in responses]

    counts = Counter(answers)
    best_answer, agreement_count = counts.most_common(1)[0]
    confidence = agreement_count / n_samples

    return best_answer, confidence
```

The logic: if the model reliably lands on the same answer across several independent attempts, that answer is more likely to be genuinely correct than a one-off guess — random errors tend to scatter across different wrong answers, while a genuinely well-supported answer tends to converge. If instead you get 5 different answers across 5 attempts, that low agreement is itself useful information: it tells you the question is likely ambiguous, hard, or outside the model's reliable competence, and a human should probably look at it rather than trusting whichever answer happened to come first.

**The honest cost:** this multiplies your API calls by `n_samples`. It's only worth it for genuinely important, verifiable-in-principle decisions — not applied as a blanket default, echoing exactly the same evidence-before-adoption discipline [Part V](part05-prompt-engineering.md) asked of chain-of-thought.

**Check yourself before the project:** Why does low agreement across self-consistency samples carry real information, rather than being pure noise?

---

# The Project: Measure Verification Techniques Against a Real Baseline

## What you're building

An evaluation harness comparing plain answers, reflection, tool-assisted verification, and self-consistency across both verifiable and subjective questions — settling, with your own numbers, when each technique is actually worth its cost.

```bash
$ python reasoning_eval.py
```

```
=== Verifiable questions (arithmetic + logic, n=10) ===
plain:                    60% correct
reflection:               70% correct
tool-assisted:           100% correct   (4.2x cost of plain)
self-consistency (n=5):   90% correct   (5.1x cost of plain)

=== Subjective questions (writing quality, n=10, human-scored) ===
plain:                    avg score 3.4/5
reflection:                avg score 3.6/5
self-consistency:          N/A — no single "correct" answer to converge on
```

## Why this project

This is the direct, evidence-based settling of the question [Part V](part05-prompt-engineering.md) and [Part VIII](part08-agent-architectures.md) both flagged but deferred: which reasoning techniques actually help, on which kinds of tasks, at what cost — measured on real questions, not assumed from general principle.

## Build it in stages

---

### Stage A — A verifiable test set

**Goal:** 10 arithmetic or logic questions with known correct answers.

**Do:**
1. Write 10 questions genuinely requiring multi-step computation (not trivial one-step arithmetic — you want a real chance of the model making a mistake)
2. Record the correct answer for each

**You're done when:** You have a test set where you're confident every answer is objectively checkable.

---

### Stage B — Plain and reflection baselines

**Goal:** Real accuracy numbers for the two weakest techniques, establishing your baseline.

**Do:**
1. Run all 10 questions with a direct "give me the answer" prompt, score accuracy
2. Run all 10 again with the reflection pattern from Stage 1, score accuracy
3. Record both percentages

**You're done when:** You have real numbers for both, from your own test set.

---

### Stage C — Tool-assisted verification

**Goal:** Real accuracy for the calculator-grounded approach from Stage 2.

**Do:**
1. Implement `verified_claim` for your test set (extend the calculator to handle multi-step reasoning if needed)
2. Run all 10 questions, score accuracy
3. Track how many API/tool calls this technique used, total, compared to the plain baseline

**You're done when:** You have a real accuracy number and a real cost multiplier, comparable to Stages A and B.

---

### Stage D — Self-consistency

**Goal:** Real accuracy for majority-vote sampling.

**Do:**
1. Implement `self_consistency` with `n_samples=5`
2. Run all 10 questions, score accuracy
3. For questions where agreement was low (below, say, 60%), manually inspect — was the question genuinely ambiguous, or did the model just get unlucky?

**You're done when:** You have a real accuracy number, a real cost multiplier, and at least one concrete example of low agreement correctly flagging a genuinely hard or ambiguous question.

---

### Stage E — The subjective control group

**Goal:** Confirm, with evidence, that these techniques behave differently on subjective tasks.

**Do:**
1. Write 10 subjective questions (e.g., "write a compelling opening sentence for an article about X")
2. Score plain and reflection outputs yourself, 1–5, blind if possible
3. Note explicitly that self-consistency doesn't meaningfully apply here (there's no single "correct" answer to converge on) — this is itself a finding, not a gap in your project

**You're done when:** You can state, from your own measurements, exactly which techniques helped on which category of question — with real numbers for both categories side by side.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Tool-assisted verification doesn't reach 100% | The model's *expression* is wrong (misunderstood the question), not just its arithmetic — verification catches computation errors, not comprehension errors | This is an honest, correct finding — note it, don't force the number up artificially |
| Self-consistency samples are identical every time | Accidentally using `temperature=0` — samples need to be able to actually differ | Set a nonzero temperature for the sampling calls specifically |
| Reflection scores worse than plain on some questions | Reflection occasionally "corrects" an already-right answer into a wrong one — a real, known failure mode | Note this as a finding rather than a bug; it's consistent with reflection's known unreliability without external verification |

## Going further (optional)

1. **Combine techniques.** Try tool-assisted verification *plus* self-consistency (verify each of the 5 samples, take a majority vote among the verified ones) and see if it beats either alone.
2. **Cost-adjusted comparison.** Instead of just accuracy, compute accuracy-per-dollar for each technique, and identify which one has the best return on cost for your specific test set.
3. **A real code-verification example**, previewing Part XLIV: have the model write a small function and a test for it, verified by actually running the test — the same tool-assisted-verification idea, applied to code instead of arithmetic.

---

## Checkpoint

**Knowledge:**
1. Why doesn't asking a model to "double-check" its own claim give it genuinely new information?
2. What determines whether a claim can be tool-assistedly verified versus requiring plain reflection?
3. Why does low agreement across self-consistency samples carry real signal, not just noise?

**Implementation:** Build `verified_claim` (tool-grounded) and `self_consistency` (majority vote) from memory.

**Debugging:** An agent's tool-assisted self-correction loop keeps "fixing" an already-correct answer into a wrong one. What does that suggest about where the actual bug lives — in the verification tool, or somewhere else?

**Architecture:** Design a reasoning-strategy selection policy: given an incoming task, how would your system decide automatically between plain answering, reflection, tool-assisted verification, and self-consistency, based on what you learned in this part's project?

---

**Next:** [Part XXXII — Context Engineering](part32-context-engineering.md), where you'll formalize what actually goes into an agent's context window on every call — a concern that's been implicit since Part III's discussion of the context window, and becomes critical as your agents grow longer-running.
