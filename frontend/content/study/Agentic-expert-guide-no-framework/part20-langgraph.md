# Part XX — LangGraph

## Before you start

**What you need already:** [Part IX](part09-agent-state.md) (you hand-built atomic, versioned checkpointing), [Part VII](part07-what-is-an-agent.md) (bounded agent loops), [Part VIII](part08-agent-architectures.md) (the different loop shapes — router, supervisor, reflection).

**What this part gives you:** LangGraph turns the agent shapes from Part VIII and the checkpointing discipline from Part IX into a single, unified abstraction: a graph, with built-in save/resume. Once you see the mapping, you'll recognize this isn't new capability — it's your own hand-built mechanisms, packaged and made declarative.

**Time:** 3–4 days.

**When you finish, you'll be able to:** Express any of Part VIII's architectures as an explicit graph, get crash-safe resumption without hand-rolling Part IX's checkpoint logic, and add human-approval pauses without building the durable-storage machinery from Part XXXIV yourself.

---

## Stage 1 — Why a graph, when you already have a loop

Go back to Part VII's agent loop. It's a `while` loop with `if/elif` branches deciding what happens next. That works, but as the branching logic grows — multiple tools, a reflection step, a router deciding between sub-agents — the `if/elif` chain gets harder to read, and harder to visualize as a whole.

LangGraph's proposal: instead of burying your control flow inside `if/elif` statements, make it **explicit and structural**. Define a set of named steps (**nodes**) and the allowed transitions between them (**edges**), as data — not as code buried inside conditionals. The framework then executes that graph for you, and because the structure is explicit rather than implicit in your control flow, the framework can *also* checkpoint after every node automatically, without you writing `save_checkpoint()` calls by hand at every step like you did in Part IX.

```
Your Part VII loop:                LangGraph's version:

while not done:                    graph.add_node("decide", ...)
    if step == 0:                  graph.add_node("act", ...)
        ...                        graph.add_edge("decide", "act")
    elif condition:                graph.add_conditional_edges("act", route_fn, {...})
        ...
    step += 1                      app = graph.compile(checkpointer=...)
```

**Check yourself before Stage 2:** What's the practical difference between control flow expressed as `if/elif` statements versus control flow expressed as named nodes and explicit edges?

---

## Stage 2 — State, and the one detail that will bite you if you skip it

Every LangGraph graph operates on a shared **state** object — structurally, this is exactly your `AgentState` dataclass from Part VII, just declared as a `TypedDict`.

```python
from typing import TypedDict


class AgentState(TypedDict):
    messages: list[str]
    step_count: int
    goal: str
    done: bool
```

Here's the part worth being careful about. Each node returns only the *fields it's updating* — not the whole state. LangGraph then merges that partial update into the existing state. For most fields, "merge" means "overwrite" — a node returning `{"step_count": 3}` replaces the old `step_count` with `3`. That's usually what you want.

But for a field like `messages` — a list you want to *append* to, not replace — plain overwrite is wrong, and it's an easy mistake to make without realizing it:

```python
def researcher_node(state: AgentState) -> dict:
    return {"messages": ["researched the topic"]}   # DANGER: this REPLACES messages, doesn't append!
```

If `messages` already had 5 entries and this node runs, you'd end up with `messages = ["researched the topic"]` — the previous 5 entries silently gone. This is a real, common bug, and it's silent — nothing errors, your history just quietly loses data.

The fix is a **reducer**: telling LangGraph explicitly how to combine a node's partial update with the existing value, instead of defaulting to overwrite.

```python
from typing import Annotated
from operator import add


class AgentState(TypedDict):
    messages: Annotated[list[str], add]   # reducer: append (list + list = concatenation)
    step_count: int                        # no reducer specified: default is overwrite
    goal: str
    done: bool
```

With `Annotated[list[str], add]`, a node returning `{"messages": ["researched the topic"]}` now gets *appended* to the existing list (`add` on two lists concatenates them), rather than replacing it.

**Check yourself before Stage 3:** If you forget to add a reducer to a list-typed state field, what happens to prior entries when a node updates that field? Why is this bug dangerous specifically because it's silent?

---

## Stage 3 — Build a graph: a router, recognized from Part VIII

```python
from langgraph.graph import StateGraph, END


def researcher_node(state: AgentState) -> dict:
    return {"messages": [f"researched: {state['goal']}"], "step_count": state["step_count"] + 1}


def writer_node(state: AgentState) -> dict:
    return {"messages": ["draft written"], "step_count": state["step_count"] + 1, "done": True}


def route(state: AgentState) -> str:
    if state["step_count"] >= 3:
        return "end"
    if "researched" in state["messages"][-1]:
        return "writer"
    return "researcher"


graph = StateGraph(AgentState)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.set_entry_point("researcher")
graph.add_conditional_edges("researcher", route, {"writer": "writer", "researcher": "researcher", "end": END})
graph.add_edge("writer", END)

app = graph.compile()

result = app.invoke({"messages": [], "step_count": 0, "goal": "renewable energy", "done": False})
print(result)
```

Compare this directly to Part VIII Stage 5's Supervisor code. `researcher_node` and `writer_node` are your `Worker` functions. `route` is your `supervisor_decide_next` function. `add_conditional_edges` is your `while` loop's branching logic, made explicit and declarative instead of imperative. Nothing here is a new *idea* — you're looking at your own Supervisor pattern, expressed in the framework's syntax.

**Check yourself before Stage 4:** Map every piece of this graph back to the equivalent piece in your hand-built Supervisor from Part VIII.

---

## Stage 4 — Checkpointing, without writing `save_checkpoint` yourself

This is where the framework earns its keep the most directly. Recall everything you built by hand in [Part IX](part09-agent-state.md): atomic writes, version numbers, a `CheckpointStore` class, careful placement of the save call after every completed step. LangGraph gives you this as one line:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke(
    {"messages": [], "step_count": 0, "goal": "renewable energy", "done": False},
    config=config,
)
```

Every node execution is now automatically checkpointed — no code you wrote yourself. `thread_id` is the equivalent of your `run_id` from Part IX; it's how LangGraph knows which run's state to load and save.

To resume a crashed run, you don't reconstruct anything by hand — you call `invoke` again with `None` as the input and the same `thread_id`:

```python
result = app.invoke(None, config=config)   # resumes from wherever it left off
```

This is precisely the `run_or_resume_agent` function you built in Part IX Stage 6, now provided by the framework. It's not doing anything conceptually different from what you built — atomic checkpoint writes after every node, keyed by an ID, loaded on resume — but you no longer have to write or maintain that machinery yourself.

**Check yourself before Stage 5:** What exactly does `checkpointer=checkpointer` buy you here, in terms of the Part IX code you would otherwise have to write by hand?

---

## Stage 5 — Human approval, without building `ApprovalStore` yourself

Recall [Part XXXIV](part34-human-in-the-loop.md) (which you haven't fully built yet, but the idea is intuitive): before a risky action, pause and wait for a human to approve. LangGraph gives you this directly, because it already has checkpointing wired in — pausing is just "stop before this node, and don't resume until told to."

```python
app_with_pause = graph.compile(checkpointer=checkpointer, interrupt_before=["writer"])

# First call runs the researcher node, then STOPS before writer:
app_with_pause.invoke(
    {"messages": [], "step_count": 0, "goal": "x", "done": False}, config=config
)

# ... your application shows the pending state to a human, gets approval ...

# Resuming picks up exactly where it paused:
app_with_pause.invoke(None, config=config)
```

This works because the checkpoint after `researcher` already durably exists — pausing before `writer` doesn't lose anything, and resuming is the same mechanism as recovering from a crash, just triggered deliberately instead of by failure. You'll build the full version of this — durable pending-approval state, edit-aware resume — by hand in Part XXXIV. Seeing it work here first, for free, will make that part's manual version much easier to appreciate for what it's actually doing.

**Check yourself before Stage 6:** Why does `interrupt_before` work correctly without any extra code, given that checkpointing is already in place?

---

## Stage 6 — Time travel: a capability your hand-built version doesn't have yet

Because every node's checkpoint is saved with an identifiable position in the sequence, LangGraph can do something your Part IX `CheckpointStore` can't easily do out of the box: replay execution from *any* prior checkpoint, not just the most recent one.

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.next, snapshot.values.get("step_count"))
```

This lets you inspect exactly what the state looked like at each step of a past run — genuinely useful for debugging "why did the agent make this specific decision at step 4" without having to re-run the whole thing from scratch. This is a capability that falls out naturally once every step is checkpointed with a stable ordering — it's the same underlying data your Part IX version stores, just with a richer query interface on top of it.

**Check yourself before the project:** Why does storing every step's checkpoint (rather than just the latest one) enable this kind of historical replay?

---

# The Project: Rebuild One of Your Part VIII Architectures in LangGraph

## What you're building

Pick one architecture you built by hand in Part VIII (ReAct, Plan-and-Execute, Supervisor, or Reflection) and rebuild it as a LangGraph graph, with checkpointing and a human-approval pause, then compare against your original.

```bash
$ python langgraph_supervisor.py "renewable energy trends" --thread-id run-1
```

```
[researcher] researched: renewable energy trends
[writer] draft written
done. checkpoint saved at every step.
```

```bash
# simulate a crash by killing the process mid-run, then:
$ python langgraph_supervisor.py --resume --thread-id run-1
```

```
resumed from checkpoint. continuing from writer node.
done.
```

## Why this project

You'll directly experience the trade-off from Part XIX's Stage 5: less code to write for checkpointing and pausing, at the cost of debugging through the framework's abstraction when something's wrong — which you're now equipped for, because you built the underlying mechanism by hand first.

## Build it in stages

---

### Stage A — Rebuild your chosen architecture as a graph

**Goal:** One of your Part VIII architectures, expressed as nodes and edges instead of a `while` loop.

**Do:**
1. Define the `TypedDict` state, with reducers where needed (a `messages` list is the most likely candidate)
2. Convert each of your architecture's "steps" into a node function
3. Convert your branching logic into a conditional edge function
4. Compile and run it, confirming the output matches your original hand-built version

**You're done when:** The LangGraph version produces equivalent results to your Part VIII version on the same test inputs.

---

### Stage B — Add checkpointing

**Goal:** Crash-safe resumption, without writing any of your own `save_checkpoint` code.

**Do:**
1. Add a `SqliteSaver` checkpointer
2. Run the graph with a `thread_id`
3. Kill the process mid-run (deliberately, at a point you can predict — add a `time.sleep` you can interrupt during, similar to Part IX's testing approach)
4. Re-invoke with the same `thread_id` and `None` as input, confirm it resumes correctly rather than restarting

**You're done when:** You've watched a real interrupt-and-resume cycle work, using the same testing discipline from Part IX Stage 6, but with the framework doing the checkpointing.

---

### Stage C — Add a human-approval pause

**Goal:** `interrupt_before` on your riskiest node.

**Do:**
1. Identify which node in your graph would be "risky" in a real system (writing output, taking an action)
2. Add `interrupt_before=["that_node"]` to your compile call
3. Confirm the graph correctly stops before that node and only proceeds after you explicitly resume it

**You're done when:** You've watched the pause-and-resume cycle work for a deliberate approval gate, not just a crash-recovery scenario.

---

### Stage D — Compare against your hand-built version

**Goal:** An honest accounting of the trade-off, from direct experience.

**Do:**
1. Count how many lines of your Part IX checkpointing code you did *not* have to write for this version
2. Deliberately introduce a bug (e.g., forget a reducer on `messages`) and time how long it takes you to notice and diagnose it
3. Write a short, honest comparison: what got easier, what got harder to debug

**You're done when:** You have a specific, evidence-based opinion — not a generic one — about when you'd reach for LangGraph versus your own hand-built harness for a given task.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| `messages` list loses earlier entries after a node runs | Missing reducer — default behavior is overwrite, not append | Add `Annotated[list, add]` to any state field a node should be appending to rather than replacing |
| Conditional edge raises a `KeyError` | The routing function returned a string that doesn't match any key in the edge mapping dict | Print the routing function's return value and cross-check against your `add_conditional_edges` mapping |
| Resume re-runs a node that already completed before the crash | Interrupted mid-node rather than between nodes, or checkpointer not actually persisting (e.g., using `:memory:` instead of a real file) | Confirm you're using a persistent (file-based) `SqliteSaver`, not an in-memory one, for a test that needs to survive a process restart |
| Graph runs forever | No recursion limit hit, or a conditional edge creates an unintended cycle | Check your routing function's logic for a path that never reaches `END` |

## Going further (optional)

1. **Subgraphs.** Take a multi-agent system from your Part XXVII work (once you've built it) and express each agent as its own compiled graph, used as a node within a parent graph. Confirm state passes correctly between the levels.
2. **Time travel debugging.** Deliberately introduce a subtle bug that only manifests after several steps, then use `get_state_history` to inspect exactly which step's state first went wrong, rather than adding print statements and re-running from scratch.
3. **Compare recursion limits.** Set an artificially low recursion limit and confirm the graph fails safely (a clear error) rather than looping — the LangGraph equivalent of the `max_steps` discipline from Part VII.

---

## Checkpoint

**Knowledge:**
1. What problem does a "reducer" solve, and what happens if you omit one on a list-typed state field?
2. What does `checkpointer=...` give you, in terms of the Part IX code it replaces?
3. Why does `interrupt_before` work correctly with no extra code, once checkpointing is already configured?

**Implementation:** Build a router graph with a conditional edge and a reducer-equipped state field, from memory.

**Debugging:** A resumed LangGraph run re-executes a node that clearly already completed successfully before a crash. What checkpointer configuration issue would you check first?

**Architecture:** Design a LangGraph-based system with a Supervisor routing to 3 worker subgraphs, a human-approval pause before any write-capable action, and Postgres-backed (not SQLite) persistence for production use.

---

**Next:** [Part XXI — CrewAI](part21-crewai.md), where you'll see a genuinely different abstraction — role-based agents instead of an explicit graph — for the same underlying multi-agent coordination problem.
