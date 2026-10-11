# Part IX — Agent State

## Before you start

**What you need already:** [Part VII](part07-what-is-an-agent.md) (the `AgentState` object you've been mutating in a loop), [Part I](part01-prerequisites.md) (async, and the Docker/Postgres setup from its final stage — you'll need Postgres running for this part's project).

**What this part gives you:** Right now, every agent you've built keeps its progress in memory, in a Python variable. If the process crashes — power outage, a bug elsewhere in your code, someone restarting the server — everything is gone, including work the agent already paid for and completed. This part fixes that.

**Time:** 3–4 days.

**When you finish, you'll be able to:** Build an agent that can be killed mid-run and resume exactly where it left off, without redoing finished work and without corrupting its own progress if it's killed at the worst possible moment — mid-write.

---

## Stage 1 — Watch the problem happen

Don't take the problem on faith. Cause it.

```python
import time

progress = {"step": 0, "findings": []}

for i in range(5):
    print(f"working on step {i}...")
    time.sleep(1)
    progress["findings"].append(f"result from step {i}")
    progress["step"] = i + 1
    print(f"  done. progress: {progress}")
```

Run it. While it's printing "working on step 2...", hit Ctrl+C.

Everything is gone. `progress` lived only in a Python variable, in memory, in a process that no longer exists. Steps 0 and 1 — real, completed work, possibly work that cost real API calls and real money — vanished the instant the process died. If this were a 40-step research agent that had been running for 10 minutes, you'd lose all 10 minutes and have to start completely over.

This is what "agent state" is really about: **not losing work you already paid for, when something interrupts you** — and something always eventually does, at a scale where interruptions are inevitable, not exceptional.

**Check yourself before Stage 2:** What specifically was lost when you hit Ctrl+C? Why did it disappear so completely?

---

## Stage 2 — Save it, badly, and learn why "badly" bites you

The obvious fix: write progress to a file after every step, so it survives the process dying.

```python
import json
import time

PROGRESS_FILE = "progress.json"


def save(progress: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def load() -> dict:
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"step": 0, "findings": []}


progress = load()
print(f"starting from step {progress['step']}")

for i in range(progress["step"], 5):
    print(f"working on step {i}...")
    time.sleep(1)
    progress["findings"].append(f"result from step {i}")
    progress["step"] = i + 1
    save(progress)
    print(f"  saved. progress: {progress}")
```

Run it, Ctrl+C partway through, run it again. It resumes from where it left off. Problem solved?

Not quite. There's a subtle failure hiding in `save()`, and it's worth understanding exactly why before you trust this pattern with anything real.

### The torn-write problem

`json.dump(progress, f)` doesn't write the whole file atomically — it writes bytes to the file as it serializes the dictionary, piece by piece. If the process is killed at the *exact* moment it's in the middle of writing — not before, not after, but during — you end up with a file that contains half of the old JSON structure and half is simply missing. That file is not valid JSON. Your next `load()` call crashes trying to parse it.

This isn't a paranoid hypothetical. It's rare, but at real scale — thousands of agent runs a day, each writing a checkpoint after every step — "rare" events happen regularly. A checkpoint system that can corrupt itself under this specific, unlucky timing is not a checkpoint system you can rely on.

**Check yourself before Stage 3:** Why can a process being killed *during* a file write leave the file in a broken state, even though the write started as a complete, valid JSON dump?

---

## Stage 3 — The fix: write to a temp file, then swap

The standard fix is simple once you see it: never write directly to the file you're going to read from. Write the *new* content to a different, temporary file first — fully, with nothing depending on it yet — and only once that's completely finished, atomically rename it over the real file.

```python
from pathlib import Path
import json


def save_atomic(progress: dict, path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(progress))   # fully written, nobody's reading it yet
    tmp_path.replace(path)                       # atomic on virtually every real filesystem
```

The key operation is `.replace()`. On essentially every filesystem you'll deploy to, renaming a file over another file is **atomic** — meaning, from the operating system's point of view, it either happens completely or not at all. There is no possible moment where the target file is half-old, half-new. If the process dies before `.replace()` runs, the *original* file is still sitting there, completely untouched and completely valid — you just lose the update that was in progress, not the previously-saved state. If it dies after, the new file is completely in place.

Either way — nothing is ever corrupted. You might lose the most recent unsaved step, but you never lose *and never corrupt* everything that came before it.

```python
def load(path: Path) -> dict:
    if not path.exists():
        return {"step": 0, "findings": []}
    return json.loads(path.read_text())


progress_path = Path("progress.json")
progress = load(progress_path)

for i in range(progress["step"], 5):
    progress["findings"].append(f"result from step {i}")
    progress["step"] = i + 1
    save_atomic(progress, progress_path)
```

**Try it:** Add `import time; time.sleep(3)` right before `tmp_path.replace(path)` in `save_atomic`, then Ctrl+C the program while it's in that sleep. Check `progress.json` afterward — it should still contain the *previous* valid state, untouched, not a broken half-write. That's the guarantee in action.

**Check yourself before Stage 4:** Why is `.replace()` safe against a mid-write crash when a direct `json.dump()` to the target file isn't? What exactly can you lose with this pattern, and what can you never lose?

---

## Stage 4 — A second danger: two writers at once

Files work for one agent running on one machine. What happens when two copies of your program — maybe two workers, maybe a crashed process that got restarted while an old instance was somehow still running — try to save progress for the *same* agent run at the same time?

```
Worker A reads progress: {"step": 3, ...}
Worker B reads progress: {"step": 3, ...}    ← both read the same starting point
Worker A does step 4, saves: {"step": 4, ...}
Worker B does step 4 AGAIN (it never saw A's update), saves: {"step": 4, ...}
```

Worker B just redid work Worker A already finished, and depending on what step 4 actually *does*, "redoing" it might not be harmless — if it sent an email or charged a payment, you now have a duplicate. This is called a **lost update**: two writers each think they're making the next change, neither knows about the other, and one of their updates effectively vanishes or gets duplicated.

### The fix: a version number, checked before every write

```python
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class VersionedProgress:
    step: int = 0
    findings: list[str] = field(default_factory=list)
    version: int = 1


class ConflictError(Exception):
    """Someone else updated this since we last read it."""


def save_versioned(progress: VersionedProgress, path: Path, expected_version: int) -> None:
    current = load_versioned(path)
    if current.version != expected_version:
        raise ConflictError(
            f"expected version {expected_version}, found {current.version} — "
            f"someone else wrote an update since we read this"
        )

    progress.version = expected_version + 1
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(progress.__dict__))
    tmp_path.replace(path)


def load_versioned(path: Path) -> VersionedProgress:
    if not path.exists():
        return VersionedProgress()
    return VersionedProgress(**json.loads(path.read_text()))
```

Now the pattern is: read the current state *and its version number*, do your work, and when you go to save, check that the version is still what you expected. If someone else already wrote a newer version while you were working, your save fails loudly with `ConflictError` instead of silently overwriting or duplicating their work. This is called **optimistic concurrency** — you optimistically assume no one else is writing at the same time, but you *check* that assumption before committing, rather than either ignoring the risk entirely or paying the cost of locking on every single operation.

**Check yourself before Stage 5:** What is a "lost update," concretely? What does checking the version number before saving actually prevent?

---

## Stage 5 — Moving to a real database

Files with atomic writes and version checks work, and they're worth understanding because the *concepts* — atomicity, version conflicts — are the same everywhere. But for anything beyond a single-machine toy, you want a real database doing this for you, because it already handles atomicity and concurrent access correctly, tested by millions of production systems, instead of you reimplementing it by hand.

You already have Postgres running from [Part I](part01-prerequisites.md). Here's the same pattern, properly:

```sql
CREATE TABLE agent_checkpoints (
    run_id TEXT PRIMARY KEY,
    state JSONB NOT NULL,
    version INT NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```python
import asyncpg
import json


async def save_checkpoint(pool: asyncpg.Pool, run_id: str, state: dict, expected_version: int) -> bool:
    """Returns True if the save succeeded, False if someone else updated it first."""
    result = await pool.execute(
        """
        UPDATE agent_checkpoints
        SET state = $2, version = version + 1, updated_at = now()
        WHERE run_id = $1 AND version = $3
        """,
        run_id, json.dumps(state), expected_version,
    )
    # asyncpg reports how many rows were actually changed, e.g. "UPDATE 1" or "UPDATE 0"
    return result == "UPDATE 1"


async def load_checkpoint(pool: asyncpg.Pool, run_id: str) -> tuple[dict, int] | None:
    row = await pool.fetchrow(
        "SELECT state, version FROM agent_checkpoints WHERE run_id = $1", run_id
    )
    if row is None:
        return None
    return json.loads(row["state"]), row["version"]


async def create_checkpoint(pool: asyncpg.Pool, run_id: str, initial_state: dict) -> None:
    await pool.execute(
        "INSERT INTO agent_checkpoints (run_id, state) VALUES ($1, $2) "
        "ON CONFLICT (run_id) DO NOTHING",
        run_id, json.dumps(initial_state),
    )
```

Look at the `WHERE run_id = $1 AND version = $3` clause in `save_checkpoint`. That single line is doing everything the manual version-check did in Stage 4, except Postgres guarantees it atomically — no other process can sneak a write in between your check and your update, because the database checks and updates as one indivisible operation. This is strictly stronger than what the file-based version could guarantee on its own.

**Check yourself before Stage 6:** What does the `WHERE version = $3` clause actually accomplish? Why is doing this inside the database query itself safer than checking the version in Python and then issuing a separate update?

---

## Stage 6 — Resuming a real agent correctly

Now put this together with Part VII's agent loop. The important design question: checkpoint *how often*?

Too rarely (say, only at the very end) and you're back to Stage 1's problem — a crash mid-run loses everything. Too often (say, mid-way through a single tool call) and you risk saving a state that represents "we were in the process of doing something" rather than "we cleanly finished a step" — which makes resuming ambiguous.

The right granularity: **checkpoint after every complete step**, never mid-step.

```python
from dataclasses import dataclass, field, asdict


@dataclass
class AgentState:
    run_id: str
    goal: str
    messages: list[dict] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    final_answer: str | None = None


async def run_or_resume_agent(pool: asyncpg.Pool, run_id: str, goal: str, max_steps: int = 8) -> AgentState:
    loaded = await load_checkpoint(pool, run_id)

    if loaded is None:
        state = AgentState(run_id=run_id, goal=goal)
        await create_checkpoint(pool, run_id, asdict(state))
        version = 1
        print(f"starting new run {run_id}")
    else:
        raw_state, version = loaded
        state = AgentState(**raw_state)
        print(f"resuming run {run_id} from step {state.step_count}")

    while not state.done and state.step_count < max_steps:
        # ... call the model, execute a tool, exactly as in Part VII ...
        # for illustration, a placeholder step:
        state.messages.append({"role": "assistant", "content": f"completed step {state.step_count}"})
        state.step_count += 1

        # checkpoint immediately after the step is FULLY complete, not before
        saved = await save_checkpoint(pool, run_id, asdict(state), version)
        if not saved:
            raise ConflictError(f"another process updated run {run_id} concurrently")
        version += 1

    if not state.done:
        state.final_answer = "STOPPED: max_steps reached"
        state.done = True
        await save_checkpoint(pool, run_id, asdict(state), version)

    return state
```

Notice the checkpoint happens *after* `state.step_count += 1` and *after* the message is appended — the state being saved always represents a fully-completed step, never a half-finished one. If the process dies right after this checkpoint, resuming means starting cleanly at the *next* step. If it dies right before, resuming redoes the step that was in progress — which is safe here because appending a message and incrementing a counter is harmless to redo. It would *not* be safe to redo if that step had sent an email; you'll deal with that specific problem — making a step safe to redo — in [Part XXXIII](part33-tool-engineering.md).

**Check yourself before the project:** Why checkpoint after every complete step rather than only at the very end, or continuously during a step? What determines whether it's safe to redo a step that was interrupted mid-way?

---

# The Project: A Crash-Proof Research Agent

## What you're building

Take your ReAct agent from Part VII and make it fully resumable — kill it mid-run, restart the program, and watch it pick up exactly where it left off, with proof that no completed work was lost or duplicated.

```bash
$ python resumable_agent.py "Research three facts about the Eiffel Tower" --run-id research-1
```

```
starting new run research-1
[step 1] tool_call: search(query="Eiffel Tower height")
[step 1] checkpoint saved (version 2)
[step 2] tool_call: search(query="Eiffel Tower construction year")
^C
interrupted — progress saved through step 2
```

```bash
$ python resumable_agent.py "Research three facts about the Eiffel Tower" --run-id research-1
```

```
resuming run research-1 from step 2
[step 3] tool_call: search(query="Eiffel Tower visitors per year")
[step 3] checkpoint saved (version 4)
[step 4] final answer.

ANSWER: The Eiffel Tower is 330m tall, completed in 1889, and receives about
7 million visitors annually.
```

## Why this project

This exact pattern — checkpoint after every step, version-checked writes, clean resume — is what every production agent framework does under the hood, including LangGraph's checkpointer, which you'll meet in [Part XX](part20-langgraph.md). Building it by hand here means that when you use the framework version later, you'll know exactly what it's doing for you and exactly what could go wrong if it's misconfigured.

## Before you start

- Postgres running from Part I's setup
- Your working agent loop from Part VII
- `pip install asyncpg`

## Build it in stages

---

### Stage A — The checkpoint table and basic save/load

**Goal:** `save_checkpoint` and `load_checkpoint`, working against real Postgres.

**Do:**
1. Create the `agent_checkpoints` table
2. Implement `create_checkpoint`, `save_checkpoint`, `load_checkpoint` exactly as in Stage 5
3. Write a tiny test script: create a checkpoint, load it, modify it, save it, load it again, confirm the changes stuck and the version incremented

**You're done when:** You can watch a row in `agent_checkpoints` change across saves, with the version number incrementing by exactly 1 each time.

---

### Stage B — Wire it into the real agent loop

**Goal:** Your Part VII agent checkpoints after every step and can resume.

**Do:**
1. Add a `--run-id` argument
2. At startup, try loading an existing checkpoint for that ID; if found, resume from it; if not, start fresh
3. After every complete step (tool call + result appended, or final answer set), save a checkpoint

**You're done when:** You can run the agent, let it finish normally, and confirm the final checkpoint shows `done: true`.

---

### Stage C — Prove resumption actually works

**Goal:** Not just "it should resume" — verify it, adversarially.

**Do:**
1. Start a run on a task that takes at least 4–5 steps
2. Hit Ctrl+C after step 2 finishes (watch the printed step numbers to time it)
3. Re-run with the same `--run-id`
4. Confirm: it resumes at step 3, not step 1 — and the final answer is correct and complete

**You're done when:** You've watched this happen for real, at least twice, with the interrupt at a different step each time.

**Now do the harder version:** interrupt it *while* a tool call is in flight, not cleanly between steps (this is trickier to time — try Ctrl+C right after you see the "tool_call:" print but before "checkpoint saved"). Confirm it doesn't corrupt the checkpoint — it should resume from the *last successfully saved* step, safely redoing the interrupted one.

---

### Stage D — Simulate two workers racing

**Goal:** Prove the version check actually prevents a lost update, not just trust that it should.

**Do:**
1. Write a small script that loads the same checkpoint twice (simulating two workers reading at the same moment)
2. Have "worker A" save an update successfully
3. Have "worker B" try to save its own update, using the *version it originally loaded* (which is now stale)
4. Confirm worker B's save fails with a clear conflict signal, rather than silently overwriting worker A's completed work

**You're done when:** You've watched a real conflict get detected and rejected, not just read about the concept.

---

### Stage E — Clean up finished runs

**Goal:** Don't let the checkpoints table grow forever with completed runs you'll never resume.

**Do:**
1. Add a simple cleanup query: delete checkpoints where `done = true` and `updated_at` is older than some threshold (e.g., 7 days)
2. Run it manually and confirm it only removes finished runs, never in-progress ones

**You're done when:** You understand this as a real operational concern, not just an afterthought — production systems accumulate this kind of state forever unless something prunes it.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Resumed run redoes the entire task from step 0 | Checkpoint is never actually being loaded at startup, or `run_id` doesn't match between runs | Print the loaded state right after `load_checkpoint` to confirm it's finding the right row |
| Occasional `ConflictError` even with only one process running | Saving with a stale `version` variable — e.g., reusing an old version number after a retry | Always re-read the version you actually need to increment from, don't cache it across an error/retry path |
| Checkpoint shows a step as "done" that clearly didn't finish (e.g., missing its result) | Checkpointing *before* the step's result is actually appended to state | Move the `save_checkpoint` call to strictly after all of the step's effects are applied to `state` |
| Postgres connection errors under concurrent testing | Not using a connection pool, or pool size too small for your test's concurrency | Use `asyncpg.create_pool` with a reasonable `max_size`, reuse the pool rather than opening new connections |

## Going further (optional)

1. **A `--replay` mode** that loads a finished checkpoint and prints the full step-by-step history — useful for understanding exactly what an agent did after the fact, without re-running it. This is a small taste of the "time travel" debugging you'll see properly in [Part XX](part20-langgraph.md).
2. **Compaction.** If a run has 100+ steps and a huge `messages` list, loading and saving the whole thing every step gets slow and expensive. Sketch (don't need to fully build) how you'd summarize older messages into a compact form while keeping the checkpoint valid — you'll build this properly once you've covered memory in [Part X](part10-memory.md).
3. **Distributed lock instead of optimistic versioning.** Using the Redis pattern from Part I (`SET NX EX`), acquire a lock on `run_id` before starting work on it, so two workers can never even *attempt* to process the same run concurrently, rather than detecting the conflict after the fact. Compare: what does locking prevent that optimistic versioning doesn't, and what does it cost?

---

## Checkpoint

**Knowledge:**
1. Why can a naive `json.dump()` to a file leave that file corrupted if the process is killed mid-write, and how does the temp-file-then-rename pattern fix it?
2. What is a "lost update," and what specifically does a version-check-before-write prevent?
3. Why must a checkpoint always represent a *fully completed* step, never a step in progress?
4. What's the difference between what you can lose with atomic checkpointing (acceptable) and what you can never lose (guaranteed)?

**Implementation:** Build `save_checkpoint`/`load_checkpoint` against Postgres, with version-checked writes, from memory.

**Debugging:** A resumed agent run duplicated a tool call that clearly already completed successfully before a crash. Walk through exactly where in the checkpoint timing this bug most likely lives.

**Architecture:** You're running 5 agent workers that pull tasks from a shared queue and each maintain their own checkpoints in the same Postgres table. Sketch how you'd prevent two workers from ever picking up and running the *same* `run_id` at the same time — not just detecting the conflict after the fact, but preventing it from starting.

---

**Next:** [Part X — Memory](part10-memory.md), where you'll extend this same checkpointing discipline to information that needs to persist *beyond* a single run — what an agent remembers about a user across many separate conversations.
