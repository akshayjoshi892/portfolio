# Part XXVII — Multi-Agent Systems

## Before you start

**What you need already:** [Part VIII](part08-agent-architectures.md) (Supervisor, Reflection), [Part XXVI](part26-agent-orchestration.md) (fan-out/fan-in, DAGs). This part combines both to build real multi-agent teams with feedback loops.

**What this part gives you:** A working "research team" agent — researcher, analyst, writer, reviewer — with a bounded review-and-revise cycle, plus the shared-state pattern for when agents need to coordinate without one central conductor telling everyone what to do.

**Time:** 3 days.

**When you finish, you'll be able to:** Build a multi-agent pipeline with quality gating, and know when a Supervisor-style hand-off is the right coordination mechanism versus a shared workspace multiple agents read and write concurrently.

---

## Stage 1 — Why split into multiple agents at all

Before building anything, be honest about the trade-off. A single agent with one giant system prompt and every tool available can, in principle, do everything a multi-agent team does. So why split it up?

Because a single agent's prompt and tool set grow unwieldy as scope increases — a system prompt trying to be simultaneously "a meticulous fact-checker" and "a concise technical writer" pulls in two directions, and the model has to hold both roles at once on every single call. **Specialization** — narrow role, narrow tools, narrow context per agent — tends to make each individual agent more reliable, at the real cost of coordination overhead between them.

This means multi-agent systems aren't automatically better. Use one when a task genuinely decomposes into distinct specialties with different tools or verification needs — not because it "sounds more sophisticated."

**Check yourself before Stage 2:** Give a concrete reason a single "do everything" agent might perform worse on a complex task than several narrowly-specialized agents, even though the single agent has access to everything.

---

## Stage 2 — Build the research team: sequential hand-off with a review gate

This extends Part VIII's Supervisor with something new: a **bounded loop between two specific stages** — writer and reviewer — where the reviewer can send work back for revision, not just approve or reject once.

```python
from __future__ import annotations
import asyncio


async def researcher(topic: str) -> str:
    await asyncio.sleep(0.1)
    return f"Raw findings on {topic}: [fact1, fact2, fact3]"


async def analyst(findings: str) -> str:
    await asyncio.sleep(0.1)
    return f"Analysis of ({findings}): key trend identified"


async def writer(analysis: str, revision_note: str = "") -> str:
    await asyncio.sleep(0.1)
    suffix = f" (incorporating: {revision_note})" if revision_note else ""
    return f"Draft report based on: {analysis}{suffix}"


async def reviewer(draft: str) -> tuple[bool, str]:
    await asyncio.sleep(0.1)
    # a real version uses an LLM call checking the draft against explicit criteria
    approved = "trend" in draft
    feedback = "looks good" if approved else "missing key trend analysis"
    return approved, feedback


async def run_research_team(topic: str, max_review_cycles: int = 2) -> str:
    findings = await researcher(topic)
    analysis = await analyst(findings)
    draft = await writer(analysis)

    for cycle in range(max_review_cycles):
        approved, feedback = await reviewer(draft)
        print(f"review cycle {cycle}: approved={approved} feedback={feedback!r}")
        if approved:
            return draft
        draft = await writer(analysis, revision_note=feedback)

    return f"[unapproved after {max_review_cycles} cycles] {draft}"


print(asyncio.run(run_research_team("renewable energy storage")))
```

Notice the `max_review_cycles` bound — this is exactly [Part VII](part07-what-is-an-agent.md)'s "every loop needs a hard stop" discipline, applied here to a *multi-agent* loop instead of a single agent's tool-calling loop. The risk is identical: without a bound, a reviewer that never approves (a bug in its criteria, or a genuinely unsatisfiable request) sends the writer into an infinite revise cycle, burning cost with no path to completion.

**Check yourself before Stage 3:** Why does the review-and-revise loop need the exact same bounding discipline as the single-agent tool-calling loop from Part VII, even though it's coordinating whole agents rather than individual tool calls?

---

## Stage 3 — A second team, with a fix-cycle instead of a review-cycle

The same bounded-loop pattern applies to a genuinely different team shape: code, tested, fixed, retested.

```python
async def coder(task: str) -> str:
    return f"code implementing: {task}"


async def tester(code: str) -> tuple[bool, str]:
    passed = "bug" not in code   # stand-in for actually running a test suite
    return passed, "all tests passed" if passed else "test failure: found 'bug' marker"


async def run_swe_team(task: str, max_fix_cycles: int = 3) -> str:
    code = await coder(task)

    for cycle in range(max_fix_cycles):
        passed, msg = await tester(code)
        print(f"cycle {cycle}: {msg}")
        if passed:
            return code
        code = await coder(f"{task} (fix needed: {msg})")

    return f"[unresolved after {max_fix_cycles} cycles] {code}"
```

Same shape as Stage 2's review loop — generate, check, feed back and retry if not good, bounded. The specific roles differ (coder/tester instead of writer/reviewer), but the underlying discipline is identical, which is worth noticing: **once you've built one bounded multi-agent feedback loop correctly, you've built the pattern for all of them.**

**Check yourself before Stage 4:** What's structurally identical between `run_research_team` and `run_swe_team`, despite their different roles?

---

## Stage 4 — When there's no natural "conductor": the blackboard pattern

Both teams so far have an implicit conductor — a function (`run_research_team`, `run_swe_team`) explicitly deciding the sequence. Sometimes that's the wrong shape: you want several agents contributing to a shared pool of information, without one of them being "in charge" of sequencing the others.

```python
import asyncio
from dataclasses import dataclass, field


@dataclass
class Blackboard:
    entries: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def write(self, key: str, value: str) -> None:
        async with self.lock:
            self.entries[key] = value

    async def read(self, key: str) -> str | None:
        async with self.lock:
            return self.entries.get(key)
```

Why the `Lock`? This is the exact same concern from [Part IX Stage 4](part09-agent-state.md) — two agents writing to the shared board at the same moment can interleave their writes incorrectly without one. Recall [Part I Stage 6](part01-prerequisites.md)'s `async with sem:` pattern — `async with self.lock:` works the same way: only one coroutine can be inside the locked block at a time, and it's automatically released even if the code inside raises an exception.

```python
async def contributor_a(board: Blackboard):
    await asyncio.sleep(0.1)
    await board.write("market_data", "growth rate: 12% YoY")


async def contributor_b(board: Blackboard):
    await asyncio.sleep(0.15)
    await board.write("competitor_data", "3 major competitors identified")


async def synthesizer(board: Blackboard) -> str:
    while await board.read("market_data") is None or await board.read("competitor_data") is None:
        await asyncio.sleep(0.05)   # wait for both contributors to finish writing
    return f"Synthesis: {await board.read('market_data')}, {await board.read('competitor_data')}"


async def run_blackboard_team() -> str:
    board = Blackboard()
    await asyncio.gather(contributor_a(board), contributor_b(board))
    return await synthesizer(board)


print(asyncio.run(run_blackboard_team()))
```

Notice: `contributor_a` and `contributor_b` don't know about each other at all — no explicit sequencing, no direct message passing between them. They just each write their piece to a shared space, and `synthesizer` reads whatever's there once it's ready. This decouples the contributors from each other completely, at the cost of needing explicit coordination logic (the polling loop in `synthesizer`) to know when the shared state is actually ready to use.

**Check yourself before the project:** What does the blackboard pattern decouple that a Supervisor-style hand-off doesn't? What does it cost you in exchange?

---

# The Project: A Bounded Multi-Agent Team With Human Approval

## What you're building

Project 3 from the plan's original spec — an autonomous development team (coder, tester, reviewer) with a human-approval gate before any change is considered final, and a full audit log of every fix cycle's attempt.

```bash
$ python swe_team.py "implement a function that validates email addresses"
```

```
[coder] wrote implementation
[tester] cycle 0: test failure: missing edge case for empty string
[coder] revised implementation
[tester] cycle 1: all tests passed
[reviewer] approved: code quality is good, style is consistent

PENDING HUMAN APPROVAL — review the diff below before merging:
--- implementation ---
def validate_email(s): ...

Approve? [y/n]:
```

## Why this project

This is the direct precursor to [Part XLIV](part44-code-agents.md)'s full code agent, and it's your first hands-on build of the human-approval gate you'll formalize properly in [Part XXXIV](part34-human-in-the-loop.md).

## Build it in stages

---

### Stage A — Coder/tester fix cycle, for real

**Goal:** `run_swe_team` from Stage 3, with real LLM calls for `coder` and a real (even if simple) test execution for `tester`.

**Do:**
1. Implement `coder` with a real LLM call producing actual code
2. Implement `tester` by actually running the generated code against a small test suite you write (use `subprocess` to run `pytest`, similar to what you'll formalize in Part XLIV)
3. Confirm the fix cycle actually improves broken code across attempts on a real, deliberately tricky task

**You're done when:** You've watched a real fix cycle correct a genuine bug across 2+ attempts, bounded by `max_fix_cycles`.

---

### Stage B — Add a reviewer stage

**Goal:** A third agent, `reviewer`, checking the passing code for quality (not just correctness) before it's considered done.

**Do:**
1. Implement `reviewer` with an LLM call checking style/clarity, separate from the tester's correctness check
2. Insert it after the fix cycle succeeds, before finalizing
3. Confirm it can request one more revision cycle if quality is poor even though tests pass

**You're done when:** You have a genuine 3-stage pipeline — code, test-and-fix, review — with each stage checking something the others don't.

---

### Stage C — Full audit log

**Goal:** Every attempt, every test result, every review verdict logged with enough detail to reconstruct exactly what happened.

**Do:**
1. Log each coder attempt's full diff from the previous attempt
2. Log each tester result and each reviewer verdict, with timestamps
3. Write it all to a structured log file (JSON lines is a good format)

**You're done when:** You could hand your log file to someone who never watched the run and they could reconstruct the full sequence of attempts and decisions.

---

### Stage D — Human approval gate

**Goal:** Nothing gets "finalized" without an explicit human confirmation.

**Do:**
1. After the pipeline produces a passing, reviewed result, print the final diff and prompt for confirmation before doing anything with it (writing to a real file, committing, etc.)
2. Confirm declining the prompt correctly aborts, without applying the change

**You're done when:** You've run the full pipeline end to end, including a real approve and a real decline, and confirmed the decline path leaves nothing changed.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Fix cycle never converges within `max_fix_cycles` on a genuinely simple task | Coder isn't actually receiving the tester's specific failure message, just retrying blindly | Confirm the failure feedback is concretely included in the next attempt's prompt |
| Reviewer approves obviously bad code | Review criteria too vague or not actually checked against real code | Tighten the reviewer's prompt with specific, checkable criteria |
| Audit log missing intermediate attempts | Only logging the final successful attempt, not every attempt along the way | Log inside the loop, on every iteration, not just at the end |

## Going further (optional)

1. **Parallel independent checks.** If your reviewer checks style and a separate security-scan step checks for common vulnerabilities, run both concurrently (Part XXVI's fan-out) rather than sequentially, since they don't depend on each other.
2. **A blackboard variant.** Rebuild the same team using the blackboard pattern instead of explicit sequencing — coder, tester, and reviewer all read/write a shared board — and compare the resulting code's clarity against the explicit-sequencing version.
3. **Cost tracking across the whole team.** Sum token usage across every agent and every cycle, and report total cost per completed task — this is a direct extension of Part VII's per-run cost tracking, now summed across multiple agents.

---

## Checkpoint

**Knowledge:**
1. Give a genuine reason multi-agent decomposition can outperform a single large agent, and a genuine reason it can be needless complexity.
2. Why does a review-and-revise loop need the same bounding discipline as a single-agent tool-calling loop?
3. What does the blackboard pattern decouple, and what does that cost?

**Implementation:** Build a bounded fix-cycle multi-agent pipeline (generate, check, feed back, retry) from memory.

**Debugging:** Two agents sharing a blackboard produce contradictory results downstream. How would you determine whether this is a race condition (missing lock) or a genuine, real disagreement that should be surfaced rather than silently resolved?

**Architecture:** Design spawn limits (max depth, max concurrent sub-agents, cost ceiling) for a system where agents are allowed to dynamically create sub-agents to help with parts of their task.

---

**Next:** [Part XXVIII — Agent Harness](part28-agent-harness.md), where everything you've built since Part VI — tool execution, state, cost tracking, error handling — gets assembled into one reusable production runtime.
