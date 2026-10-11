# Part L — Advanced Orchestration

## Before you start

**What you need already:** [Part XXVI](part26-agent-orchestration.md) (DAGs with a fixed, known-upfront structure), [Part IX](part09-agent-state.md) (checkpointing), [Part XX](part20-langgraph.md) (event-driven, framework-provided checkpointing you already used).

**What this part gives you:** What happens when the DAG's structure *itself* isn't known until runtime — because a step's output determines how many further steps are needed — plus the general form of event-driven orchestration for work triggered externally, not by a single request.

**Time:** 2 days.

**When you finish, you'll be able to:** Build a DAG that grows its own structure as it runs, with the same crash-safe resumption guarantee as a fixed DAG, and know when event-driven orchestration fits better than a request/response flow.

---

## Stage 1 — Where a fixed DAG breaks down

Your Part XXVI DAG assumed you know the full node structure before execution starts. That's often true — but consider a "decompose this research question" node whose *output* is a variable-length list of sub-questions, each of which needs its own research node. You don't know how many sub-question nodes to create until the decompose node has actually run.

```python
sub_questions = await decompose_node()   # returns, say, 3 items THIS time, maybe 5 next time
# now you need 3 (or 5) new "research" nodes, added to the graph AFTER it started running
```

A fixed DAG, built entirely upfront, structurally can't express this — the graph needs to be able to grow while it's executing.

**Check yourself before Stage 2:** Why can't Part XXVI's `topological_order`, computed once before execution starts, handle a node whose output determines how many further nodes are needed?

---

## Stage 2 — A DAG that can add nodes to itself, mid-run

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DynamicNode:
    name: str
    fn: callable
    depends_on: list[str] = field(default_factory=list)


class DynamicDAG:
    def __init__(self):
        self.nodes: dict[str, DynamicNode] = {}

    def add_node(self, name: str, fn: callable, depends_on: list[str] | None = None) -> None:
        self.nodes[name] = DynamicNode(name, fn, depends_on or [])

    def expand_based_on_result(self, parent_name: str, sub_items: list) -> None:
        """A node's own execution can cause new nodes to be added to the
        graph — the mechanism a fixed, upfront DAG structurally cannot have."""
        for i, item in enumerate(sub_items):
            self.add_node(f"{parent_name}_sub_{i}",
                          lambda it=item: process_sub_item(it),
                          depends_on=[parent_name])
```

The key structural difference from Part XXVI: `add_node` can be called *during* execution, not just before it starts. A "decompose" node's completion handler calls `expand_based_on_result`, which adds exactly as many new nodes as the decomposition actually produced — the graph's final shape emerges from running it, rather than being fully known in advance.

**Check yourself before Stage 3:** What's the one new capability `DynamicDAG` has that `Part XXVI`'s static DAG doesn't?

---

## Stage 3 — Checkpointing a graph whose shape is still changing

Here's a genuinely interesting question: can you still get crash-safe resumption (Part IX's core guarantee) for a DAG whose structure isn't fixed? Yes — the trick is checkpointing at **node granularity**, recording which nodes have completed, regardless of how many nodes ultimately exist.

```python
import json
from pathlib import Path


class DurableDAGRunner:
    def __init__(self, dag: DynamicDAG, state_path: Path):
        self.dag = dag
        self.state_path = state_path
        self.completed: dict[str, object] = self._load()

    def _load(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def _save(self) -> None:
        # exactly Part IX's atomic write pattern
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.completed, default=str))
        tmp.replace(self.state_path)

    def run(self) -> dict:
        changed = True
        while changed:
            changed = False
            for name, node in list(self.dag.nodes.items()):   # list(...) — nodes may grow mid-loop
                if name in self.completed:
                    continue
                if all(d in self.completed for d in node.depends_on):
                    result = node.fn()
                    self.completed[name] = result
                    self._save()   # checkpoint after EVERY node, not just at fixed points
                    changed = True
        return self.completed
```

Two details worth noticing carefully. `list(self.dag.nodes.items())` — wrapping in `list()` — matters because `expand_based_on_result` might add new nodes to `self.dag.nodes` *while this loop is iterating over it*; without the `list()` copy, that would be a runtime error (mutating a dict while iterating it). And the `while changed:` outer loop keeps re-scanning as long as *something* completed in the last pass — necessary because newly-added nodes need their own chance to become ready once their dependencies are met, potentially in a later pass than the one that added them.

Calling `DurableDAGRunner(dag, path).run()` again after a crash reloads `self.completed` and skips everything already finished — the exact same atomic-write checkpoint discipline from Part IX, now applied at the granularity of individual, possibly-dynamically-added graph nodes rather than whole agent steps.

**Check yourself before Stage 4:** Why does wrapping `self.dag.nodes.items()` in `list(...)` matter specifically for a *dynamic* DAG, when it wouldn't have mattered for Part XXVI's fixed one?

---

## Stage 4 — Event-driven orchestration: a different trigger, not a different mechanism

One more orchestration shape worth knowing, genuinely different in *when* work starts, not in how it's structured once it does. Everything so far assumes a request arrives and you process it. Some systems need the opposite: work is triggered by an external event — a new support ticket arriving, a file being uploaded — with no single "request" initiating it.

```python
import asyncio
from collections import defaultdict


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: callable) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event_type: str, payload: dict) -> None:
        for handler in self._handlers[event_type]:
            asyncio.create_task(handler(payload))   # fire concurrently, don't block the publisher
```

```python
async def on_document_uploaded(payload: dict) -> None:
    print(f"triggering ingestion pipeline for {payload['doc_id']}")
    # reuse Part XII's incremental ingestion pipeline here


bus = EventBus()
bus.subscribe("document_uploaded", on_document_uploaded)
```

This decouples "something happened" from "who handles it" — the publisher of `document_uploaded` doesn't need to know or care what (if anything) is subscribed to react to it. This fits naturally with long-running, externally-triggered systems: a support-ticket triage agent, an ingestion pipeline reacting to new files, anything without a single synchronous request/response shape.

### One trap worth knowing about `asyncio.create_task`

If `on_document_uploaded` raises an exception, `asyncio.create_task` doesn't propagate it anywhere visible by default — the exception is silently swallowed unless you explicitly attach error handling. This is a real, easy-to-miss production bug: an event handler can start failing on every event, silently, with nothing surfacing the problem unless you deliberately guard against it.

```python
def log_task_exception(task: asyncio.Task) -> None:
    if task.exception() is not None:
        print(f"event handler failed silently: {task.exception()}")


async def publish_safe(bus: EventBus, event_type: str, payload: dict) -> None:
    for handler in bus._handlers[event_type]:
        task = asyncio.create_task(handler(payload))
        task.add_done_callback(log_task_exception)
```

**Check yourself before the project:** Why does `asyncio.create_task`'s default exception-swallowing behavior make event-driven orchestration specifically prone to silent, hard-to-notice failures?

---

# The Project: A Dynamic DAG With Durable, Node-Granular Checkpointing

## What you're building

A task whose true sub-task count is only known after an initial "decompose" node runs — with crash-safe resumption at the node level, bounded expansion, and an event-driven trigger wired in.

```bash
$ python dynamic_dag.py "research renewable energy storage"
```

```
[decompose] produced 4 sub-questions (this run's actual count — varies)
added 4 dynamic nodes: decompose_sub_0..3
[decompose_sub_0] completed, checkpoint saved
[decompose_sub_1] completed, checkpoint saved
^C
```

```bash
$ python dynamic_dag.py --resume
```

```
resumed. 2/4 sub-nodes already completed, skipping.
[decompose_sub_2] completed, checkpoint saved
[decompose_sub_3] completed, checkpoint saved
[merge] completed
done.
```

## Why this project

This directly extends both Part IX's crash-safe resumption and Part XXVI's DAG execution into the one case neither fully covered on its own: a graph whose shape isn't known until it's already running.

## Build it in stages

---

### Stage A — A real decompose-and-expand DAG

**Goal:** `DynamicDAG` with a real decompose node (an LLM call) that produces a variable number of sub-nodes.

**Do:**
1. Implement `DynamicDAG.expand_based_on_result`
2. Build a real decompose node using an LLM call producing a genuinely variable-length list of sub-questions
3. Run it on 3 different real goals, confirm the number of generated sub-nodes actually varies

**You're done when:** You've watched the graph's structure genuinely differ across runs, driven by the decompose node's real output.

---

### Stage B — Bound the expansion

**Goal:** A hard cap on total node count, so a runaway decomposition (a bug, or an unusually complex goal) can't grow the graph unboundedly.

**Do:**
1. Add a maximum total-node check inside `expand_based_on_result`, refusing to add more nodes beyond the cap
2. Test with a goal likely to produce an unusually large decomposition, confirm the cap holds

**You're done when:** You've confirmed the graph can't grow past your chosen limit, with a clear signal when it's capped rather than silent truncation.

---

### Stage C — Durable, node-granular checkpointing, tested with a real interrupt

**Goal:** The exact crash-and-resume behavior from the example output.

**Do:**
1. Implement `DurableDAGRunner`
2. Run it on a real dynamic DAG, interrupt (Ctrl+C) after some but not all dynamically-added nodes have completed
3. Resume, confirm completed nodes are correctly skipped and only remaining ones execute

**You're done when:** You've watched a real interrupt-and-resume cycle correctly preserve partial progress on a graph whose full structure wasn't even known until mid-run.

---

### Stage D — Wire in an event trigger

**Goal:** The dynamic DAG runs in response to a published event, not a direct function call.

**Do:**
1. Implement `EventBus` with the exception-safe `publish_safe` variant
2. Subscribe your dynamic DAG's execution as a handler for a `"research_requested"` event
3. Publish an event, confirm the DAG runs correctly
4. Deliberately make a handler raise an exception, confirm `publish_safe`'s done-callback surfaces it rather than silently swallowing it

**You're done when:** You've confirmed both the event-driven trigger works and that handler exceptions are surfaced, not silently lost.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Runtime error mutating a dict during iteration | Iterating `self.dag.nodes.items()` directly instead of over a `list()` copy, while `expand_based_on_result` adds nodes mid-loop | Wrap the iteration in `list(...)` as shown in Stage 3 |
| Resumed run re-adds already-expanded sub-nodes, duplicating them | Decompose node's expansion logic re-runs on resume instead of being skipped like any other completed node | Ensure the decompose node itself is checked against `self.completed` before re-running, same as any other node |
| Event handler silently stops working with no visible error | Default `asyncio.create_task` exception-swallowing | Use the `add_done_callback` pattern from Stage 4, or an equivalent, on every event-triggered task |

## Going further (optional)

1. **Nested dynamic expansion** — a sub-node itself capable of triggering further expansion, and confirm the bound from Stage B still holds transitively, not just at the top level.
2. **Compare against Temporal or a similar durable-workflow engine's approach** (read about it, even without building it) — what would a purpose-built workflow engine give you here that your hand-built `DurableDAGRunner` doesn't, at very high scale/volume?
3. **A backlog dashboard** showing, for a long-running dynamic DAG, how many nodes are pending vs. completed vs. capped-out — useful for a genuinely large, ongoing research task.

---

## Checkpoint

**Knowledge:**
1. Why can't a fixed, upfront DAG handle a node whose output determines how many further nodes are needed?
2. Why does `DurableDAGRunner` need to iterate over a `list()` copy of its nodes, specifically for a dynamic DAG?
3. Why does `asyncio.create_task`'s default behavior make event-driven systems specifically prone to silent failures?

**Implementation:** Build `DynamicDAG` with bounded expansion and `DurableDAGRunner` with node-granular checkpointing, from memory.

**Debugging:** An event-driven pipeline silently stops processing new events after an unrelated code change elsewhere in the system. Given what you know about `asyncio.create_task`'s exception handling, what's the most likely cause?

**Architecture:** Describe a scenario where you'd combine a traditional, durable workflow engine (like Temporal) with an LLM-driven agent framework — what would each one be responsible for, and why wouldn't either one alone be sufficient?

---

**Next:** [Part LI — Agent Protocols and Interoperability](part51-agent-protocols-interoperability.md), where agents built on different frameworks and even different organizations need to discover and delegate to each other.
