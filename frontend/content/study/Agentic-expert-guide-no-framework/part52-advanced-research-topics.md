# Part LII — Advanced Research Topics

## Before you start

**What you need already:** [Part XXXVIII](part38-agent-evaluation.md) (evaluation methodology — you'll need it to judge any of this critically), [Part XXXI](part31-reasoning.md) (the distinction between verified and unverified claims — the exact skill this part asks you to apply to research claims themselves).

**What this part gives you:** A deliberate shift in posture — from "build this, it's established practice" to "understand this, know its limits, and don't deploy it uncritically." Everything here is genuinely less settled than Parts I–LI, and pretending otherwise would be dishonest.

**Time:** 1 day — this is reading and critical evaluation, not building production code.

---

## Stage 1 — Why this part is different, and how to read it

Everything through [Part LI](part51-agent-protocols-interoperability.md) taught mature, production-proven techniques — patterns you could build, test, and trust because the engineering community has stress-tested them. This part covers ideas that are genuinely still being worked out: promising, actively researched, but not yet something you should deploy with the same confidence as, say, atomic checkpointing.

The skill this part is actually teaching isn't any single technique — it's **applying [Part XXXVIII](part38-agent-evaluation.md)'s calibration discipline to research claims themselves.** A paper reporting an impressive benchmark result is exactly like an uncalibrated LLM judge: it might be measuring something real, or it might be measuring something narrower than the claim suggests. Read critically, the same way you learned to calibrate a judge against your own scoring before trusting it.

**Check yourself before Stage 2:** Restate, in your own words, why this part asks you to apply Part XXXVIII's evaluation-calibration mindset to research papers themselves.

---

## Stage 2 — A survey, with an honest maturity assessment for each

| Topic | Core idea | Maturity, honestly |
|---|---|---|
| Long-horizon autonomous agents | Agents operating for extended periods with minimal human checkpoints | Experimental — reliability degrades significantly over long horizons without strong verification (recall Part XXXI) built in throughout |
| Continual/lifelong learning | Agents improving from ongoing experience without full retraining | Research — "catastrophic forgetting" (new learning erasing old capability) remains largely unsolved |
| Agent societies / emergent coordination | Large populations of agents with coordination behavior that wasn't explicitly programmed | Research, mostly studied in simulation — hard to predict or control, risky to deploy |
| Self-improving agents | Agents that adjust their own prompts/strategies based on feedback | Early, but with a mature slice — see Stage 3 |
| World models | Learned internal models of environment dynamics, used for planning | Mature in narrow domains (robotics, games), nascent for general agentic tasks |
| Agentic reinforcement learning | Training via RL over multi-step tool-use trajectories | Active research direction, behind some current reasoning-model training |
| Agent benchmarking | Standardized capability comparisons | Active, but benchmarks saturate and can be gamed — apply Part XXXVIII's calibration skepticism directly |

**Check yourself before Stage 3:** For any two rows in this table, explain what "not yet production-mature" specifically means for that topic — what could go wrong if you deployed it with full confidence today?

---

## Stage 3 — The one idea mature enough to actually try, with the right scope

Not everything here is purely theoretical. "Self-improving agents" has a genuinely usable, bounded slice: **prompt-level optimization, guided by an external evaluation score you already know how to build** — as opposed to the much riskier, unbounded idea of an agent modifying its own code or weights autonomously.

```python
from dataclasses import dataclass


@dataclass
class PromptVariant:
    text: str
    score: float | None = None


async def mutate_prompt(base: str, mutation_hint: str, client) -> str:
    """An LLM proposes a prompt variant based on an observed failure pattern —
    this is prompt-level search, guided by an EXTERNAL score (Part V's eval
    harness), not autonomous self-modification of code or weights."""
    prompt = f"Improve this prompt to address this issue: {mutation_hint}\n\nCurrent prompt:\n{base}"
    result, _ = await client.ask(prompt)
    return result


async def self_improving_prompt_search(base_prompt: str, eval_suite_fn, mutation_hints: list[str],
                                        client, max_rounds: int = 3) -> PromptVariant:
    best = PromptVariant(text=base_prompt, score=await eval_suite_fn(base_prompt))

    for round_num in range(max_rounds):
        candidate_text = await mutate_prompt(best.text, mutation_hints[round_num % len(mutation_hints)], client)
        candidate = PromptVariant(text=candidate_text, score=await eval_suite_fn(candidate_text))
        print(f"round {round_num}: candidate={candidate.score:.3f} best={best.score:.3f}")
        if candidate.score > best.score:
            best = candidate

    return best
```

Notice what makes this the *responsible* version of "self-improvement," worth contrasting explicitly against the unbounded idea some research explores: it's bounded (`max_rounds`, exactly [Part VII](part07-what-is-an-agent.md)'s discipline again), scoped to prompt text only (not code, not model weights), and verified against `eval_suite_fn` — your own real evaluation harness from [Part V](part05-prompt-engineering.md), not the model's own unverified claim that a change is "better." This is genuinely usable today, precisely because it borrows every safety property from mechanisms you've already built and trust.

**Check yourself before the project:** What specifically makes `self_improving_prompt_search` safe to actually use, when unbounded autonomous self-modification would not be?

---

# The Exercise: Run the Bounded Self-Improvement Loop, and Critique a Real Claim

## What you're doing

Two parts, both about applying critical, evidence-based thinking rather than building new production infrastructure.

## Part 1 — Run the bounded self-improvement loop

**Do:**
1. Implement `self_improving_prompt_search` against a real prompt and your Part V evaluation harness
2. Run it for 3 rounds on a real task where you have a genuine, measurable weakness to improve
3. Report, honestly, whether it actually improved the score — and if it did, whether the improvement generalizes or just overfit to your specific test cases (recall [Part V Stage 2](part05-prompt-engineering.md)'s overfitting warning)

**You're done when:** You have a real before/after score from your own evaluation harness, and an honest assessment of whether the improvement is genuine or an artifact of a small test set.

## Part 2 — Critique a real research claim

**Do:**
1. Find one recent paper or benchmark result related to agentic AI (an agent benchmark leaderboard entry is a fine choice)
2. Apply the same calibration skepticism from [Part XXXVIII Stage 4](part38-agent-evaluation.md): what exactly is being measured? What's the test distribution? Would this result predict real-world reliability on a task *you* care about, or is it narrower than the headline claim suggests?
3. Write a short, honest critique — not dismissive, not credulous — identifying specifically what the result does and does not tell you

**You're done when:** You can articulate, in writing, the gap (if any) between what a benchmark claims and what it actually demonstrated, using the exact evaluation-literacy skills from Part XXXVIII.

---

## Checkpoint

**Knowledge:**
1. Why is applying Part XXXVIII's calibration mindset to research claims a genuinely transferable skill from earlier in this curriculum, not a new concept?
2. What specifically distinguishes the "mature slice" of self-improving agents from the unbounded, research-stage version?

**Research:** Complete both parts of the exercise above.

---

**Next:** [Part LIII — Project-Based Learning](part53-project-ladder.md), where you'll consolidate everything from Parts I–LII into a structured progression of 50 projects, in order.
