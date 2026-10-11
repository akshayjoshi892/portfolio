# Part XXII — LangChain vs. LangGraph vs. CrewAI

## Before you start

**What you need already:** [Part XIX](part19-langchain.md), [Part XX](part20-langgraph.md), [Part XXI](part21-crewai.md) — you've built the same kinds of tasks in all three. This part is short by design: it's meant to consolidate what you already discovered by building, not introduce new material.

**What this part gives you:** A single reference table and decision guide, backed by your own hands-on experience rather than someone else's opinion.

**Time:** Half a day.

---

## Stage 1 — The comparison, stated as what you already found

| Dimension | LangChain | LangGraph | CrewAI |
|---|---|---|---|
| Core abstraction | `Runnable` — a composable pipeline step | Explicit state graph — nodes, edges, typed state | `Agent` + `Task` + `Crew` — roles and dependencies |
| Control flow | Linear/parallel composition (`|`, `RunnableParallel`) | Arbitrary — including cycles and conditionals | Sequential (`context` dependencies) or manager-delegated |
| State | Threaded through chain input/output manually | Explicit, typed, reducer-merged | Implicit — via task dependencies, optional memory flag |
| Persistence | None built in | Built-in checkpointers (what you used in Part XX) | Limited — memory persists, full execution state does not |
| Human-in-the-loop | Manual | Built-in (`interrupt_before`, what you used in Part XX) | Guardrails — validation-triggered retry, not a true pause |
| Debugging | LangSmith tracing via callbacks | Same tracing + time-travel via checkpoints | Verbose logging, less granular |
| Authoring speed for a fixed pipeline | Fast | Slower (must define graph explicitly) | Fastest (roles + dependencies read naturally) |
| Control for complex/cyclic flows | Limited | Best — this is what it's built for | Limited |

Every row of this table should be something you can verify from your own Parts XIX–XXI builds, not just take on faith.

**Check yourself before Stage 2:** For each row, can you point to a specific piece of code you wrote in Parts XIX–XXI that demonstrates it?

---

## Stage 2 — The decision, as a question sequence

```
Does the task need real cycles or complex conditional branching
(a loop that revisits earlier steps, multiple decision points)?
  YES → LangGraph
  NO  → continue

Does the task decompose naturally into distinct "roles" with a
mostly-linear or simply-delegated flow?
  YES → CrewAI (if authoring speed matters more than fine control)
  NO  → continue

Is this simple, composable pipeline plumbing — no agents,
just prompt/model/parser/retriever chained together?
  YES → LangChain alone
```

None of these are mutually exclusive in a real system. It's completely normal to use LangChain's `Runnable`s *inside* LangGraph nodes (you were already halfway there in Part XX, since your node functions could just as easily call a LangChain chain internally), or to use CrewAI for the authoring-speed win on a subsystem while implementing a more complex parent flow in LangGraph.

**Check yourself:** For each of your three Part XIX–XXI projects, which branch of this decision tree would you have followed if you'd asked the question first, before building? Does it match what you actually built?

---

# The Project: Justify a Real Choice, in Writing

## What you're building

Not code this time — a short written decision document for a real system, the kind you'd actually hand to a team before starting implementation.

## Do

Pick (or invent) a real system with **two genuinely different needs at once** — for example, an enterprise assistant that needs a fixed, auditable compliance-review workflow for one part of its job, and dynamically-delegated open-ended research for another part.

Write, in your own words:
1. Which framework(s) you'd use for which part of the system, and why — referencing specific things you observed building with each in Parts XIX–XXI, not generic claims
2. What you'd lose by choosing differently
3. Whether you'd combine frameworks (e.g., CrewAI for the role-authoring layer, LangGraph for a complex sub-flow) or keep it to one

**You're done when:** Someone who hasn't read this curriculum could read your document and understand not just *what* you'd choose, but *why*, grounded in real trade-offs rather than a stated preference.

---

## Checkpoint

**Knowledge:** Recite the decision-tree from Stage 2 without looking, and justify each branch from your own experience.

**Architecture:** Complete the written project above.

---

**Next:** [Part XXIII — MCP](part23-mcp.md), where you'll meet a standard protocol for tool discovery that works *independently* of which agent framework you choose — solving a different problem than any of the three frameworks in this part.
