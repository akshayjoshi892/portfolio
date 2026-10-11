# Part XXVI — Agent Orchestration

## Before you start

**What you need already:** [Part I](part01-prerequisites.md) (`asyncio.gather`, concurrency limits), [Part VIII](part08-agent-architectures.md) (the Supervisor pattern, and its noted weakness: sequential execution even when work is independent).

**What this part gives you:** A close look at the specific weakness Part VIII flagged and left for later — a Supervisor running workers one at a time, even when they don't actually depend on each other. This part fixes that, and adds the general vocabulary (fan-out/fan-in, DAGs) for structuring work that isn't a simple sequence.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Run independent pieces of agent work concurrently instead of sequentially, and correctly sequence work that has real dependencies between steps — including detecting when your dependency graph has an impossible cycle.

---

## Stage 1 — Recall the exact problem you left open

Go back to [Part VIII Stage 5](part08-agent-architectures.md). Your Supervisor ran `researcher`, then `writer`, one after another — even though, in that specific example, there wasn't actually a reason `writer` couldn't start doing *some* preparatory work while `researcher` was still running, if their jobs were genuinely independent. In that example they weren't (writer needed researcher's output), so sequential was correct. But imagine instead a `fact_checker` worker whose job doesn't depend on `writer` finishing — it could run at the same time as `writer`, not after it.

A Supervisor that always runs workers one at a time has no way to express "these two are independent, run them together." This part builds that.

**Check yourself before Stage 2:** In your own words, when is it *correct* to run two agent tasks sequentially, and when is running them sequentially just wasted time?

---

## Stage 2 — Fan-out, fan-in: the pattern you already know, applied here

You built this exact mechanism in [Part I Stage 3](part01-prerequisites.md) — `asyncio.gather` to run several things at once. Applying it to independent agent sub-tasks is the same idea, just with agent calls instead of API calls.

```python
import asyncio
from dataclasses import dataclass


@dataclass
class SubResult:
    source: str
    content: str
    confidence: float


async def sub_agent(source: str, query: str) -> SubResult:
    # stand-in for a real agent call — a specialist researching one angle
    await asyncio.sleep(0.3)
    return SubResult(source=source, content=f"[{source} finding on {query}]", confidence=0.8)


async def fan_out_fan_in(query: str, sources: list[str]) -> str:
    results = await asyncio.gather(*(sub_agent(s, query) for s in sources))   # FAN OUT: all at once
    kept = [r for r in results if r.confidence >= 0.5]                        # FAN IN: merge/filter
    return "\n".join(f"({r.source}) {r.content}" for r in kept)


import time
start = time.monotonic()
result = asyncio.run(fan_out_fan_in("market trends", ["news_agent", "reports_agent", "social_agent"]))
print(f"took {time.monotonic() - start:.2f}s")
```

```
took 0.30s
```

Three sub-agents, each taking 0.3 seconds, and the total is 0.3 seconds — not 0.9. This is the exact speedup you measured in Part I when you first ran multiple API calls concurrently, now applied to independent sub-agent work. The term **fan-out** describes launching several independent tasks at once; **fan-in** describes collecting and combining their results afterward. Together, they describe exactly the pattern your Part VIII Supervisor was missing.

**Check yourself before Stage 3:** Why does `fan_out_fan_in` take roughly 0.3 seconds total, rather than 0.9 (the sum of all three sub-agent calls)?

---

## Stage 3 — When tasks have real dependencies: the DAG

Fan-out/fan-in handles the "these are all independent" case. Real workflows often have a mix — some steps independent, some genuinely dependent on others' output. You need a structure general enough to express both.

A **DAG** (Directed Acyclic Graph) is exactly that: a set of tasks, each declaring what it depends on, with no cycles (nothing can depend, even indirectly, on itself).

```python
from dataclasses import dataclass, field


@dataclass
class DAGNode:
    name: str
    fn: callable
    depends_on: list[str] = field(default_factory=list)
```

Before running anything, you need to know a valid *order* to run the nodes in — one where every node runs only after everything it depends on has already finished. This is called a **topological sort**, and it needs to detect cycles, because a cycle means there's no valid order at all.

```python
def topological_order(nodes: dict[str, DAGNode]) -> list[str]:
    visited, order = set(), []

    def visit(name: str, in_progress: set[str]):
        if name in visited:
            return
        if name in in_progress:
            raise ValueError(f"cycle detected: {name} depends on itself, directly or indirectly")
        in_progress.add(name)
        for dep in nodes[name].depends_on:
            visit(dep, in_progress)
        in_progress.discard(name)
        visited.add(name)
        order.append(name)

    for name in nodes:
        visit(name, set())
    return order
```

The `in_progress` set is the cycle detector: if you're currently in the middle of visiting a node's dependencies, and one of those dependencies leads back to a node you're *already* visiting, that's a cycle — an impossible-to-satisfy dependency ("A needs B, B needs A") that must be caught before you try to execute anything, not discovered mid-run.

**Try it:** Build a small `nodes` dict with `A depends_on B` and `B depends_on A`, and confirm `topological_order` raises a clear error instead of looping forever or crashing unpredictably.

**Check yourself before Stage 4:** Why must cycle detection happen *before* execution starts, rather than being discovered as a runtime error partway through?

---

## Stage 4 — Executing the DAG: dependencies met means ready to run

```python
async def run_dag(nodes: dict[str, DAGNode]) -> dict[str, object]:
    order = topological_order(nodes)   # validates no cycles, before doing any work
    results: dict[str, object] = {}

    for name in order:
        node = nodes[name]
        dep_results = {dep: results[dep] for dep in node.depends_on}
        if asyncio.iscoroutinefunction(node.fn):
            results[name] = await node.fn(**dep_results)
        else:
            results[name] = node.fn(**dep_results)

    return results
```

This version runs strictly in topological order — one node at a time, even if two nodes at the same "level" (like `research_a` and `research_b`, both feeding into `merge`, but not depending on each other) could technically run concurrently. That's a real limitation worth naming: proper fan-out *within* a DAG (running all currently-ready nodes concurrently, not just running them in a valid sequential order) is a genuine further improvement — Stage 2's `asyncio.gather` pattern, applied to whichever nodes have all their dependencies satisfied at each point in the loop, rather than the whole DAG at once. You'll get the chance to build this properly in this part's project.

**Check yourself before the project:** In `run_dag` as written, if `research_a` and `research_b` are both ready to run (no unmet dependencies) at the same point in the loop, does the current code run them concurrently or one after another? What would you change to fix that?

---

# The Project: A DAG Orchestrator With Real Concurrency

## What you're building

A DAG executor for a 6-node workflow — some nodes dependent, some independent — that runs everything ready to run *concurrently*, not just in a valid sequential order, with cycle detection and partial-failure handling.

```bash
$ python dag_orchestrator.py
```

```
execution plan (levels):
  level 0 (run concurrently): research_a, research_b, research_c
  level 1 (run concurrently): analyze_a, analyze_b
  level 2: merge

running level 0... (3 nodes concurrently)
running level 1... (2 nodes concurrently)
running level 2... (1 node)

total time: 1.2s (vs. 3.6s if run fully sequentially)
```

## Why this project

This directly fixes the weakness left open since Part VIII, and gives you a general-purpose orchestration tool you'll reuse conceptually in [Part L](part50-advanced-orchestration.md) when the DAG structure itself becomes dynamic rather than fixed upfront.

## Build it in stages

---

### Stage A — Topological sort with cycle detection

**Goal:** `topological_order`, correctly ordering a real 6-node dependency graph, and correctly rejecting a deliberately cyclic one.

**Do:**
1. Implement it as shown
2. Build a 6-node DAG by hand with a realistic dependency structure (some nodes depending on 2+ others, some independent)
3. Confirm the returned order respects every dependency
4. Build a deliberately cyclic graph and confirm it's rejected with a clear error, before any node runs

**You're done when:** Both the valid ordering and the cycle-rejection work correctly on real test graphs.

---

### Stage B — Level-based execution (the real fix)

**Goal:** Instead of running nodes strictly one at a time in topological order, group nodes into "levels" — everything with no unmet dependencies at a given point — and run each entire level concurrently with `asyncio.gather`.

**Do:**
1. Write a function that computes execution levels: level 0 is every node with no dependencies; level 1 is every node whose dependencies are all in level 0 or earlier; and so on
2. Execute level by level, using `asyncio.gather` *within* each level
3. Confirm results still flow correctly to dependent nodes in later levels

**You're done when:** Independent nodes at the same level genuinely run concurrently (verify with timing, as in the example output), while dependent nodes still correctly wait for their prerequisites.

---

### Stage C — Partial failure handling

**Goal:** If one node fails, its direct and indirect dependents are marked as skipped — not silently run on missing/broken input, and not crashing the entire orchestrator.

**Do:**
1. Make one node in your test DAG deliberately fail (raise an exception)
2. Confirm nodes that don't depend on it still complete successfully
3. Confirm nodes that *do* depend on it (directly or transitively) are marked `"skipped"` with a clear reason, rather than either crashing or running anyway with missing data

**You're done when:** A single node's failure produces a clean, informative partial result — some nodes succeeded, some skipped, none crashed the whole run or silently used bad data.

---

### Stage D — Timeout per node

**Goal:** One slow node can't hang the whole orchestration.

**Do:**
1. Wrap each node's execution in a timeout (reusing the pattern from Part VI)
2. Test with one deliberately slow node and confirm it times out cleanly without blocking nodes in other, independent branches of the DAG

**You're done when:** A single slow node's timeout doesn't delay unrelated concurrent work in the same level.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| "Concurrent" level still takes the sum of all nodes' time, not the max | Nodes within a level are being awaited one at a time in a loop instead of passed to `asyncio.gather` together | Confirm you're using `gather(*(...))`, not a `for` loop with individual `await`s |
| A node runs with missing/incomplete data after a dependency failed | No skip-propagation logic — only the directly failed node is marked, not its dependents | Walk the dependency graph forward from any failed node and mark everything downstream as skipped |
| Cycle detection doesn't fire on an actual cycle | `in_progress` set not being properly discarded/tracked across recursive calls | Trace through the recursion by hand on a small cyclic example to find where the tracking breaks |

## Going further (optional)

1. **A visual execution plan.** Print the levels and their nodes before running anything, as in the example output — useful for reviewing a complex DAG's structure before committing to running it.
2. **Retry within a level.** Combine this with Part I's retry/backoff pattern — a node that fails should retry with backoff before being marked as failed and propagating skips downstream.
3. **A dynamic DAG.** Add a node whose *output* determines what additional nodes get added to the graph at runtime — a small preview of [Part L](part50-advanced-orchestration.md)'s dynamic DAG treatment.

---

## Checkpoint

**Knowledge:**
1. What's the difference between "fan-out/fan-in" and a "DAG," and when do you need the more general DAG structure?
2. Why must cycle detection happen before execution starts?
3. Why does level-based execution outperform strict topological-order execution for a DAG with independent branches?

**Implementation:** Build `topological_order` with cycle detection and level-based concurrent execution, from memory.

**Debugging:** A DAG-based pipeline occasionally produces a stale result for one specific node. Walk through how you'd determine whether the issue is in dependency resolution, level computation, or something else.

**Architecture:** Design orchestration for a workflow with 12 steps, where 4 are independent research tasks, 3 depend on all 4 research tasks finishing, and 1 final step depends on all 3. Sketch the level structure.

---

**Next:** [Part XXVII — Multi-Agent Systems](part27-multi-agent-systems.md), where multiple genuinely distinct agents — not just parallel function calls — coordinate, communicate, and occasionally disagree.
