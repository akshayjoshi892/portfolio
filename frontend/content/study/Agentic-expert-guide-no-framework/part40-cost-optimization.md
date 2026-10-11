# Part XL — Cost Optimization

## Before you start

**What you need already:** [Part III Stage 16](part03-llm-fundamentals.md) (prompt caching), [Part IV Stage 6](part04-llm-application-engineering.md) (cost tracking), [Part XXXIX](part39-observability.md) (real usage data to actually optimize against).

**What this part gives you:** Turning the cost numbers you've been tracking since Part IV into deliberate, evidence-based optimization decisions — and one specific, high-leverage technique (routing easy requests to cheap models) that requires getting a subtle detail right to avoid defeating its own purpose.

**Time:** 2 days.

**When you finish, you'll be able to:** Build a cost-aware router that measurably cuts spend without hurting quality, and correctly identify when a "cost optimization" would actually make costs worse.

---

## Stage 1 — Where the money actually goes

You've tracked cost per call since [Part IV](part04-llm-application-engineering.md). Now look at it in aggregate, across a real workload, and notice the shape:

```python
def cost_breakdown(call_records: list) -> dict:
    total_input_cost = sum(r.input_tokens / 1_000_000 * PRICES[r.model]["input"] for r in call_records)
    total_output_cost = sum(r.output_tokens / 1_000_000 * PRICES[r.model]["output"] for r in call_records)
    return {
        "input_cost": total_input_cost,
        "output_cost": total_output_cost,
        "total": total_input_cost + total_output_cost,
        "output_fraction": total_output_cost / (total_input_cost + total_output_cost),
    }
```

Run this against a real batch of your own recorded calls. On most real workloads, you'll find `output_fraction` is disproportionately large relative to the actual token *counts* — because output tokens are typically priced several times higher than input tokens (recall Part III), a modest amount of generated text can dominate the bill even when your prompts are much longer in raw token count. This single fact is why "cap the output length" tends to be a surprisingly effective first lever, before reaching for anything more elaborate.

**Check yourself before Stage 2:** If output tokens cost 4x input tokens, and a request has 2000 input tokens and 200 output tokens, roughly what fraction of that request's cost is from the output?

---

## Stage 2 — Not every request needs your best model

Here's the idea behind **model routing**: a request asking "is this email spam or not" doesn't need the same model as one asking "design a database migration strategy." Sending both to your most capable (and most expensive) model wastes money on the easy ones without any quality benefit.

```python
from enum import Enum


class TaskComplexity(Enum):
    TRIVIAL = "trivial"     # classification, extraction, simple formatting
    MODERATE = "moderate"   # standard Q&A, summarization
    COMPLEX = "complex"      # multi-step reasoning, code generation


MODEL_TIERS = {
    TaskComplexity.TRIVIAL: {"model": "small-model", "input_price": 0.15, "output_price": 0.60},
    TaskComplexity.MODERATE: {"model": "mid-model", "input_price": 3.00, "output_price": 15.00},
    TaskComplexity.COMPLEX: {"model": "large-model", "input_price": 15.00, "output_price": 75.00},
}
```

The prices here span two orders of magnitude between tiers — this is real, not exaggerated. Routing a genuinely trivial task to the large-model tier by default, out of habit or caution, can be a 100x cost difference for zero quality gain.

**Check yourself before Stage 3:** Why might routing every request to the largest model "just to be safe" actually be the wrong default, not just an expensive one?

---

## Stage 3 — The subtle trap: the classifier itself must be cheap

Here's where a naive implementation of routing defeats its own purpose. If you use your most capable, most expensive model just to *decide* which tier a request belongs to, you've added a full expensive call on top of whatever tier you route to — the "savings" from routing easy requests to a cheap model get eaten, or worse, by the cost of deciding.

```python
def classify_complexity(task: str) -> TaskComplexity:
    """This classification step itself must be near-free — a cheap model
    call, or even simple heuristics — or the whole routing scheme loses
    money instead of saving it."""
    if len(task.split()) < 10 and "?" in task:
        return TaskComplexity.TRIVIAL
    if any(kw in task.lower() for kw in ["plan", "design", "debug", "multi-step"]):
        return TaskComplexity.COMPLEX
    return TaskComplexity.MODERATE
```

This version uses cheap heuristics — no model call at all. A more sophisticated version might use your smallest, cheapest model to classify, but the principle holds either way: **the classifier's cost must be small relative to the savings it enables**, or you're not optimizing, you're just adding overhead.

```python
def route_by_cost_and_complexity(task: str, input_tokens: int, expected_output_tokens: int) -> dict:
    complexity = classify_complexity(task)
    tier = MODEL_TIERS[complexity]
    cost = (input_tokens / 1_000_000 * tier["input_price"]) + \
           (expected_output_tokens / 1_000_000 * tier["output_price"])
    return {"model": tier["model"], "estimated_cost_usd": cost}
```

**Check yourself before Stage 4:** Why would using your most expensive model to classify task complexity potentially make a routing system cost *more* than not routing at all?

---

## Stage 4 — The other big lever, already built: prompt caching, measured

You built the mechanics of prompt caching in [Part III Stage 16](part03-llm-fundamentals.md) — stable content first, variable content last. Here's confirming, with real measurement, that it's actually saving you money in a real system:

```python
async def measure_caching_savings(client, stable_prefix: str, varying_suffixes: list[str]) -> dict:
    costs_without_cache_discipline = []
    costs_with_cache_discipline = []

    for suffix in varying_suffixes:
        # BAD: varying content mixed into the prefix breaks the cache every time
        bad_prompt = f"{suffix}\n{stable_prefix}"
        result_bad = await client.ask_with_usage(bad_prompt)
        costs_without_cache_discipline.append(result_bad.cost)

        # GOOD: stable prefix first, unchanged every call
        good_prompt = f"{stable_prefix}\n{suffix}"
        result_good = await client.ask_with_usage(good_prompt)
        costs_with_cache_discipline.append(result_good.cost)

    return {
        "avg_cost_bad_ordering": sum(costs_without_cache_discipline) / len(costs_without_cache_discipline),
        "avg_cost_good_ordering": sum(costs_with_cache_discipline) / len(costs_with_cache_discipline),
    }
```

Run this against a real provider with a genuinely long, stable system prompt (1000+ tokens) and confirm the ordering actually matters for your specific setup — not every provider/model combination caches automatically or the same way, so this is worth verifying rather than assuming.

**Check yourself before the project:** Why does putting varying content *before* the stable prefix break caching, even though the stable content is still present somewhere in the request?

---

## Stage 5 — Cost budgets, enforced, not just monitored

You built per-run cost enforcement in [Part XXVIII](part28-agent-harness.md)'s `CostController`. Worth restating here as this part's final principle: **monitoring cost tells you what happened. Enforcing a budget prevents what shouldn't happen.** A dashboard showing "we spent $400 more than expected yesterday" is diagnostic, after the fact. A `CostController.charge()` that raises `BudgetExceededError` mid-run stops the overspend while it's happening — and given how easily an unbounded agent loop (Part VII) can burn through budget quickly, enforcement matters more than monitoring alone for the worst-case scenario.

**Check yourself before the project:** Why is enforcing a budget mid-run a stronger protection than a dashboard alert that fires after the fact?

---

# The Project: A Cost-Aware Router, Measured Against a Real Workload

## What you're building

A routing layer in front of your `LLMClient`, tested against a realistic mixed-complexity workload, reporting real savings compared to an always-use-the-best-model baseline.

```bash
$ python cost_router.py --workload realistic_tasks.json
```

```
100 tasks processed.

Baseline (always large-model): $4.82
Routed (complexity-based):     $1.31

Savings: 72.8%
Quality check (10 spot-checked routed tasks vs. baseline): 9/10 equivalent quality
  1 case where routing to a smaller model gave a noticeably worse answer -> flagged for review
```

## Why this project

This is where cost tracking, which you've had since Part IV, becomes an actual lever you pull deliberately — and the quality spot-check stage forces you to confront the real trade-off routing makes, rather than just trusting the savings number in isolation.

## Build it in stages

---

### Stage A — Build a realistic mixed-complexity workload

**Goal:** 100 real or realistic task descriptions spanning trivial, moderate, and complex, in a JSON file.

**Do:**
1. Write or gather 100 varied tasks — classification/extraction tasks (trivial), standard Q&A/summarization (moderate), multi-step reasoning/planning (complex)
2. Manually tag each with your own honest assessment of its true complexity, for later comparison against your classifier's decisions

**You're done when:** You have a realistic, varied workload with your own ground-truth complexity labels.

---

### Stage B — Baseline: always the largest model

**Goal:** Run all 100 tasks through your most capable model, recording real cost per task.

**Do:**
1. Run the full workload
2. Sum total cost
3. Save the outputs — you'll need them for the quality comparison later

**You're done when:** You have a real total baseline cost figure.

---

### Stage C — Routed version, with a cheap classifier

**Goal:** `route_by_cost_and_complexity`, run against the same workload.

**Do:**
1. Implement the classifier and routing logic
2. Run the full workload through the router
3. Sum total cost, compare against your classifier's decisions and your own ground-truth labels from Stage A — how often did the classifier agree with your own assessment?

**You're done when:** You have a real total routed cost figure and a real classifier-accuracy number against your own labels.

---

### Stage D — The quality check that matters most

**Goal:** Confirm the savings didn't come at an unacceptable quality cost.

**Do:**
1. For 10 tasks, compare the baseline (large-model) output against the routed output, blind if possible
2. Score whether they're equivalent, or whether routing produced a meaningfully worse answer
3. For any case where routing hurt quality, investigate: was it a misclassification (routed to too cheap a tier) or a genuine limitation of the smaller model even for a correctly-classified task?

**You're done when:** You have an honest accounting — not just "we saved 70%" but "we saved 70% and here's exactly what that cost us in quality, on how many cases, and why."

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Routing saves less than expected | Classifier's own cost isn't actually near-zero, or misclassifying too many tasks into the expensive tier | Measure the classifier's own cost contribution separately; tighten its heuristics or prompt |
| Routing "saves money" but quality check reveals real regressions | Complexity boundaries too aggressive — routing genuinely moderate/complex tasks to the trivial tier | Adjust classification thresholds based on your Stage D findings, and re-measure |
| Classifier disagrees with your own labels on many tasks | Heuristics or prompt too simplistic for your actual task distribution | Look at specific disagreements and refine the classification logic accordingly |

## Going further (optional)

1. **A confidence-based escalation.** If the trivial-tier model's response looks low-confidence or hedged, automatically re-route to a higher tier rather than accepting a poor answer — a safety net against classifier mistakes.
2. **Combine with prompt caching.** Measure total cost with routing *and* proper cache-friendly prompt structuring together, and report the combined savings versus either alone.
3. **A cost dashboard over time**, using your Part XXXIX tracing data, showing cost trends and routing-tier distribution across real production days, not just a one-off batch test.

---

## Checkpoint

**Knowledge:**
1. Why does output token pricing tend to dominate cost more than raw output token *count* alone would suggest?
2. Why must a routing classifier itself be cheap, or the whole scheme can lose money?
3. Why is enforcing a budget mid-run a stronger protection than monitoring cost after the fact?

**Implementation:** Build `classify_complexity` (cheap heuristics), `route_by_cost_and_complexity`, and a cost-comparison harness from memory.

**Debugging:** A prompt-caching setup shows no measurable cost reduction despite a stable, unchanged system prompt. What's the most likely cause, given what you know from Part III?

**Architecture:** Design a cost-budget hierarchy — per-request, per-session, per-tenant-per-day — with graceful degradation (not a hard failure) when a budget tier is exhausted.

---

**Next:** [Part XLI — Performance Engineering](part41-performance-engineering.md), where latency — not cost — becomes the thing you're deliberately optimizing, using the same real, measured discipline.
