# Part XXVIII — Agent Harness

## Before you start

**What you need already:** Nearly everything so far — [Part VI](part06-structured-outputs-tool-calling.md) (tool execution, circuit breakers), [Part VII](part07-what-is-an-agent.md) (the loop, step/cost bounds), [Part IX](part09-agent-state.md) (checkpointing).

**What this part gives you:** Every part since VI has built one *piece* of what a production agent needs, in isolation. This part is where they finally get assembled into one coherent thing — the difference between "code that demonstrates a concept" and "code you'd actually run in production."

**Time:** 3 days.

**When you finish, you'll be able to:** Explain exactly why a bare agent loop isn't production-safe on its own, and build a harness that wraps it with every safety mechanism you've built so far, composed cleanly rather than tangled together.

---

## Stage 1 — What a bare loop is missing, made concrete

Go back to your Part VII agent loop. It works. Now imagine it running as a real service, handling real user requests, for real months. List, honestly, everything that could go wrong that the bare loop doesn't handle:

- A tool hangs forever — nothing catches it (Part VI fixed this, but is it wired in here?)
- A user's request costs way more than expected — nothing enforces a budget per-run
- The process crashes mid-run — nothing resumes it (Part IX fixed this, but is it wired in here?)
- A tool call fails 5 times in a row against a struggling downstream service — nothing stops hammering it
- Nobody can tell what the agent actually did on a given run, after the fact
- Every user gets access to every tool, regardless of what they should be allowed to do

Every one of these has a fix you've already built, in a different part, in isolation. The problem is: **none of them are wired together.** A harness is exactly that wiring — one place where all these concerns compose cleanly around the core loop, instead of being scattered fixes you'd have to remember to apply every time you write a new agent.

**Check yourself before Stage 2:** For each bullet above, name the Part where you already built the fix for it.

---

## Stage 2 — Compose two concerns first: cost and permissions

Start small — two concerns, cleanly separated, each as its own class with one job.

```python
from dataclasses import dataclass


class BudgetExceededError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


@dataclass
class CostController:
    max_tokens: int
    tokens_used: int = 0

    def charge(self, tokens: int) -> None:
        self.tokens_used += tokens
        if self.tokens_used > self.max_tokens:
            raise BudgetExceededError(f"budget exceeded: {self.tokens_used}/{self.max_tokens}")


@dataclass
class PolicyEngine:
    allowed_tools: set[str]

    def check(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise PermissionDeniedError(f"tool not permitted for this session: {tool_name}")
```

This is directly Part VII's `max_tokens` check and a new permission concept, each pulled out into its own small, single-purpose class instead of inline `if` statements buried in the loop. Why does this separation matter, beyond tidiness? Because a harness that has these as distinct, swappable components can give *different sessions different budgets and different tool permissions* — a paying customer's session might get a higher token budget and access to more tools than a free-tier session, and that's just a different `CostController`/`PolicyEngine` instance, not a rewrite of the loop itself.

**Check yourself before Stage 3:** Why does pulling cost and permission checks into their own classes, rather than inline `if` statements in the loop, make it easier to give different sessions different limits?

---

## Stage 3 — Add tracing: knowing what happened, after the fact

```python
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tracer:
    events: list[dict] = field(default_factory=list)

    def log(self, event_type: str, **data: Any) -> None:
        self.events.append({"type": event_type, "ts": time.time(), **data})
```

This is deliberately minimal for now — a structured list of timestamped events. It's the direct seed of [Part XXXIX](part39-observability.md)'s full tracing system, but even this small version answers a question a bare loop can't: after a run finishes (or fails), what actually happened, in order? Without it, debugging a production incident means guessing based on the final output alone.

**Check yourself before Stage 4:** What can you learn from a run's `Tracer.events` list that you couldn't learn from just looking at its final output?

---

## Stage 4 — Assemble the harness: every concern, composed around the loop

```python
import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Harness:
    cost: CostController
    policy: PolicyEngine
    tracer: Tracer
    tools: dict[str, Callable[[dict], Any]]
    max_steps: int = 10

    async def run(self, goal: str, decide_fn: Callable) -> str:
        self.tracer.log("run_start", goal=goal)
        step = 0

        while step < self.max_steps:
            decision = decide_fn(step, goal)
            self.cost.charge(decision.get("token_estimate", 50))

            if decision["type"] == "final":
                self.tracer.log("run_end", status="done")
                return decision["content"]

            tool_name = decision["tool"]
            self.policy.check(tool_name)   # raises PermissionDeniedError if not allowed

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.tools[tool_name], decision["args"]), timeout=5.0
                )
                self.tracer.log("tool_call", tool=tool_name, status="ok")
            except asyncio.TimeoutError:
                self.tracer.log("tool_call", tool=tool_name, status="timeout")
                result = {"error": "tool timed out"}

            step += 1

        self.tracer.log("run_end", status="max_steps_reached")
        return "terminated: max_steps reached"
```

Walk through this once, mapping every piece to a Part you've already built: `self.cost.charge(...)` is Part VII's budget check, now a reusable component. `self.policy.check(...)` is a new permission layer. The `asyncio.wait_for(...)` timeout wrapping tool execution is Part VI's timeout pattern. `self.tracer.log(...)` calls are Stage 3's tracing, placed at every meaningful transition. The `while step < self.max_steps` bound is Part VII's step limit.

Nothing here is a new *idea*. What's new is that they're all composed together, in one place, applied uniformly to every tool call and every step — rather than each concern living in a separate file you'd have to remember to import and wire in correctly every time you build a new agent.

**Check yourself before the project:** Pick any three lines in `Harness.run` and identify exactly which earlier Part's mechanism each one is reusing.

---

## Stage 5 — What's still missing, and why that's honest, not a flaw

This harness composes cost, permissions, tool execution with timeout, and tracing. It does **not** yet include: crash-safe checkpointing (Part IX — you'd add a `StateManager` component following that part's atomic-write pattern), a real memory system (Part X), human-approval gating (Part XXXIV), or circuit breakers per tool (Part VI Stage 6, which you'd wire into the `tools` dict).

This isn't an oversight — it's honest about scope. A production harness genuinely needs all of these, and you now have every one of them built, in isolation, ready to compose in exactly the same way `CostController` and `PolicyEngine` were composed above. The project below asks you to add at least one more.

**Check yourself before the project:** Pick one component not yet included in `Harness` (checkpointing, memory, circuit breakers). Sketch, in a sentence, how you'd add it as a new field on the `Harness` dataclass, following the same pattern as `cost` and `policy`.

---

# The Project: A Complete Harness, Used for Two Different Agents

## What you're building

Extend `Harness` with real checkpointing (reusing Part IX) and a real circuit breaker per tool (reusing Part VI), then use the *same* harness class to run two genuinely different agents — proving the composition actually generalizes rather than being hardcoded to one use case.

```bash
$ python harness_demo.py --agent research --goal "population of Tokyo"
```

```
[research agent] run_start
[research agent] tool_call: search -> ok
[research agent] run_end: done (3 steps, 420 tokens, $0.0006)
```

```bash
$ python harness_demo.py --agent research --goal "population of Tokyo" --resume-crash-test
```

```
[research agent] resuming from checkpoint at step 2
[research agent] run_end: done
```

## Why this project

This harness becomes the runtime every later agent in this curriculum can plug into — [Part XXX](part30-planning.md)'s planner, [Part XXXIV](part34-human-in-the-loop.md)'s approval-gated agent, [Part XLIV](part44-code-agents.md)'s code agent. Get the composition right once, here, and you reuse it rather than rebuilding safety mechanisms from scratch every time.

## Build it in stages

---

### Stage A — The base harness, working with a real agent

**Goal:** `Harness` from Stage 4, running a real tool-using agent (reuse your Part VI/VII agent) instead of a fake `decide_fn`.

**Do:**
1. Wire in a real `LLMClient` call as the decision function
2. Wire in 2 real tools
3. Run a real task end to end, confirm tracing captures every step correctly

**You're done when:** A real task completes, and `tracer.events` gives you a complete, accurate record of what happened.

---

### Stage B — Add checkpointing

**Goal:** A `StateManager` component (reusing Part IX's atomic-write pattern) added to `Harness`, checkpointing after every step.

**Do:**
1. Add a `state_manager` field to `Harness`
2. Checkpoint the run's progress after every completed step
3. Add a `resume` capability: if a checkpoint exists for a given run ID, load it and continue instead of starting fresh

**You're done when:** You can kill the process mid-run and re-run with the same run ID, watching it resume rather than restart — the same test discipline from Part IX Stage 6, now inside the harness.

---

### Stage C — Add circuit breakers per tool

**Goal:** Each tool in `Harness.tools` gets its own circuit breaker (reusing Part VI's `CircuitBreaker`), tripping independently.

**Do:**
1. Wrap tool execution with a per-tool circuit breaker check
2. Make one tool always fail (temporarily) and confirm its breaker trips after the threshold, while other tools remain fully usable
3. Confirm the tripped tool is excluded from what's offered to the model until its cooldown passes

**You're done when:** You've watched one tool's breaker trip independently of the others, without affecting the rest of the harness's operation.

---

### Stage D — Run two different agents through the same harness

**Goal:** Prove the harness generalizes — not hardcoded to one specific agent's needs.

**Do:**
1. Build a second, genuinely different agent (different tools, different goal type — e.g., a simple data-lookup agent versus your research agent)
2. Run both through the *same* `Harness` class, just instantiated with different `tools`, `cost`, and `policy` configurations
3. Confirm both work correctly, with independent budgets and independent tool access

**You're done when:** You have two working agents sharing one harness implementation, each correctly isolated from the other's configuration.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Checkpointing doesn't actually let a run resume correctly | Checkpoint captures the loop's `step` counter but not enough context (partial results, messages) to actually continue meaningfully | Checkpoint everything the loop needs to pick back up, not just a step number |
| Circuit breaker for one tool trips the whole harness | Breaker state shared globally instead of per-tool | Confirm each tool has its own independent `CircuitBreaker` instance, not one shared across all tools |
| Second agent in Stage D can't be run without editing `Harness`'s code | Some concern (tools, cost limit) was hardcoded into `Harness.run` instead of passed in via the dataclass fields | Move anything agent-specific into a field the harness accepts, not a hardcoded value |

## Going further (optional)

1. **Per-session policy configuration.** Build a small function that returns a different `PolicyEngine` and `CostController` based on a "user tier" (free vs. paid), and confirm the same harness correctly enforces different limits for different sessions.
2. **A `Harness.summary()` method** reporting total cost, steps used, tools called, and any breaker trips for a completed run — a human-readable digest of the trace log.
3. **Wire in real memory** (Part X), so the harness can retrieve relevant past context for a user before starting a new run, and store new memories after it finishes.

---

## Checkpoint

**Knowledge:**
1. List every concern this part's `Harness` composes, and for each, name the Part where you originally built it in isolation.
2. Why does separating cost/policy into their own classes make per-session configuration easier than inline checks in the loop?
3. What's still missing from this part's harness, and why is that an honest scoping choice rather than an oversight?

**Implementation:** Build `Harness` — cost, policy, tool execution with timeout, tracing — from memory, composing pieces you built in earlier parts.

**Debugging:** A production agent run exceeded its cost budget before being stopped. Given the harness's structure, exactly where would you add a check to prevent this from recurring?

**Architecture:** Design a harness configuration for a multi-tenant platform where each tenant has different tool permissions and cost budgets, all sharing one running harness implementation.

---

**Next:** [Part XXIX — Build Agents From Scratch](part29-build-agents-from-scratch.md), a deliberate pause to consolidate everything you've built without a framework, before Part XXX moves forward into planning.
