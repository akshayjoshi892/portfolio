# Part XXIX — Consolidation: What You've Built Without a Framework

## Before you start

**What you need already:** Parts VI–XXVIII. This part introduces no new mechanisms — it's a deliberate checkpoint before moving forward, making sure the from-scratch foundation is solid before you build more sophisticated things on top of it.

**What this part gives you:** A moment to confirm, honestly, that you can reproduce the core mechanisms from memory — because everything from [Part XXX](part30-planning.md) onward assumes this fluency without re-explaining it.

**Time:** 1 day — this is a review, not new material.

---

## Stage 1 — The inventory

Here's everything you've built by hand since Part VI, with the Part where you built it. For each one, the real test isn't "have I read this" — it's "can I write it again, right now, without looking."

| # | Mechanism | Built in | Reused by |
|---|---|---|---|
| 1 | Provider-agnostic LLM client with retry/fallback | [Part IV](part04-llm-application-engineering.md) | Every later part |
| 2 | Schema-validated tool-calling loop | [Part VI](part06-structured-outputs-tool-calling.md) | Parts VII, XXV, XXVIII |
| 3 | Bounded ReAct agent loop | [Part VII](part07-what-is-an-agent.md) | Part VIII, XXVIII |
| 4 | Long-term memory with conflict resolution and decay | [Part X](part10-memory.md) | Part XXVIII (optional extension) |
| 5 | Dependency-graph DAG execution | [Part XXVI](part26-agent-orchestration.md) | Part L |
| 6 | Router (classify-and-dispatch) | [Part VIII](part08-agent-architectures.md) | Part XXV |
| 7 | Bounded reflection/revision loop | [Part VIII](part08-agent-architectures.md), [XXVII](part27-multi-agent-systems.md) | — |
| 8 | Atomic, versioned checkpointing | [Part IX](part09-agent-state.md) | Part XX, XXVIII, XXXIV |
| 9 | Multi-agent coordination (Supervisor, blackboard) | [Part XXVII](part27-multi-agent-systems.md) | — |
| 10 | An MCP-equivalent tool protocol | [Part XXIII](part23-mcp.md) | Part XXV |
| 11 | A composed production harness | [Part XXVIII](part28-agent-harness.md) | Everything from here forward |

**Check yourself before Stage 2:** Without looking at any earlier part, pick three items from this table at random and write one sentence describing exactly what each one does and why it exists.

---

## Stage 2 — Why this checkpoint exists now, specifically

You've also, by this point, seen two frameworks (LangChain, LangGraph) implement several of these same mechanisms — Runnables for composition, checkpointers for #8, `bind_tools` for #2. That ordering was deliberate: building the mechanism by hand *before* seeing the framework's version means the framework's abstraction is recognizable, not mysterious. Someone who learns LangGraph's checkpointer *first*, without ever building #8 by hand, has no way to reason about what happens when it misbehaves — they only know the API surface, not the underlying guarantee it's providing.

This is the direct, practical payoff of Rule 4 from the very beginning of this curriculum: build before abstracting. You're at the halfway point of the curriculum now, and everything from here — planning ([Part XXX](part30-planning.md)), reasoning ([Part XXXI](part31-reasoning.md)), security ([Part XXXV](part35-security.md)) — will keep building *directly* on this table's mechanisms, assuming you can reach for them without re-deriving them from scratch each time.

**Check yourself before the exercise:** Pick one item from the table that you've also seen a framework implement (checkpointing, tool calling). In one or two sentences, state what the framework's version gives you "for free," and what it costs you in exchange — using the specific comparisons you made in Parts XIX–XXI.

---

# The Exercise: A Timed Self-Test

## What you're doing

Not a coding project this time — a timed, closed-book self-assessment. Set a timer. No looking at earlier parts until you're done.

## Do this

**Part 1 (30 minutes):** From memory, write working code for three of these, your choice:
- A retry-with-backoff wrapper around an async call
- A schema-validated tool-calling loop with structured error return
- An atomic, versioned checkpoint save/load pair
- A bounded reflection loop (generate, critique, revise, capped)

**Part 2 (15 minutes):** For each of the three you chose, write one sentence stating the single most important failure mode it protects against, and what would happen without that protection.

**Part 3 (15 minutes):** Compare what you wrote against the actual code in the relevant Part. For anything that doesn't match — not just syntax, but a missing safety check or a forgotten edge case — go back and re-read that Part's relevant stage before continuing to Part XXX.

## You're done when

You can complete Part 1 without needing to look anything up, and Part 3 reveals no significant gaps — only minor syntax differences, not missing concepts. If Part 3 reveals a real gap (you forgot the version check in checkpointing, or the timeout wrapping on tool calls), that's valuable information now, before it becomes a bug in a system you build on top of it later. Treat it as a signal to revisit, not a failure.

---

## Checkpoint

This part's checkpoint *is* the exercise above. If you completed it honestly and found real gaps, go fix them before proceeding — [Part XXX](part30-planning.md) assumes every mechanism in this table is second nature.

---

**Next:** [Part XXX — Planning](part30-planning.md), where the Plan-and-Execute architecture from Part VIII gets a real fix for the plan-staleness weakness that part deliberately left open.
