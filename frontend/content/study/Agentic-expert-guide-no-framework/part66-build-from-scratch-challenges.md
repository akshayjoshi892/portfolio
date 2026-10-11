# Part LXVI — "Build From Scratch" Challenges

## What this page is for, in plain terms

This is Part XXIX's self-test exercise, generalized into a permanent, repeatable practice you can return to any time you feel your fluency slipping — for instance, months after finishing the curriculum, before starting a big new project, as a warm-up. Twelve challenges, each one thing you already built once in an earlier part, now attempted cold, with no reference material open.

**How to run one, concretely, using Challenge 7 (Checkpointing) as the example:** close every tab related to Part IX. Set a timer for 20 minutes. Write, from memory, atomic checkpoint save/load with version-checked writes — the temp-file-then-rename pattern, the `WHERE version = $3` style conflict check. When you're done (or when the timer runs out, whichever comes first), open Part IX and diff your version against the original. A close match with only minor syntax differences means the mechanism is genuinely internalized. A version missing the atomic-write step, or missing the version check entirely, is telling you something real and specific: go re-read that exact stage of Part IX before trusting yourself with checkpointing in a real system.

Complete these **without** LangChain, LangGraph, or CrewAI. Each maps to where the underlying mechanism was taught; the challenge is building it unaided, then comparing against the framework's abstraction ([Part LXVII](part67-framework-abstraction-mapping.md)).

## Challenge 1 — Tool-Calling Loop
Build a loop that offers tool schemas to a model, parses and validates a tool call, executes it, and feeds the structured result back — with malformed-argument and timeout handling. *Reference mechanism*: [Part VI](part06-structured-outputs-tool-calling.md).

## Challenge 2 — ReAct Agent
Build a Thought/Action/Observation loop with a bounded step count and explicit termination. *Reference*: [Part VIII §2](part08-agent-architectures.md).

## Challenge 3 — Memory
Build a long-term memory store with embedding-based retrieval, decay, and conflict-aware writes. *Reference*: [Part X](part10-memory.md).

## Challenge 4 — Planner
Build a dependency-graph planner with replanning on step failure, bounded by a max-replan count. *Reference*: [Part XXX](part30-planning.md).

## Challenge 5 — Router
Build a classifier-based dispatcher routing requests to distinct handlers, with a fallback for unclassifiable input. *Reference*: [Part VIII §1](part08-agent-architectures.md) architecture #4.

## Challenge 6 — State Machine
Build an explicit state/transition system with a validated transition table (reject invalid transitions rather than silently applying them). *Reference*: [Part IX](part09-agent-state.md).

## Challenge 7 — Checkpointing
Build atomic, versioned checkpoint persistence supporting crash-and-resume without data loss or duplication. *Reference*: [Part IX §4](part09-agent-state.md).

## Challenge 8 — Agent Harness
Build a harness composing cost control, tool policy/permissions, timeout-bound execution, and tracing around a core loop. *Reference*: [Part XXVIII](part28-agent-harness.md).

## Challenge 9 — Multi-Agent Communication
Build a message-passing or blackboard-based communication layer between at least 3 independent agent processes, with proper locking on shared state. *Reference*: [Part XXVII](part27-multi-agent-systems.md).

## Challenge 10 — An MCP-Like Tool Protocol
Build a minimal client/server protocol over stdio or HTTP supporting capability negotiation, tool discovery, and structured invocation/error responses — without using the official MCP SDK. *Reference*: [Part XXIII](part23-mcp.md).

## Challenge 11 — Evaluation Framework
Build trajectory evaluation, tool-argument accuracy scoring, and a calibrated LLM-as-judge pipeline. *Reference*: [Part XXXVIII](part38-agent-evaluation.md).

## Challenge 12 — Agent Tracing
Build a nested-span tracer producing a causally-ordered trace tree with per-span timing and status. *Reference*: [Part XXXIX §3](part39-observability.md).

## How Frameworks Simplify These (Preview of Part LXVII)

| Challenge | What LangGraph gives you "for free" once you understand the manual version |
|---|---|
| Checkpointing (7) | Built-in `SqliteSaver`/`PostgresSaver` checkpointers with automatic per-node persistence |
| State machine (6) | Declarative graph/conditional-edge definition instead of hand-written transition validation |
| Multi-agent communication (9) | Subgraphs sharing a typed state schema instead of hand-rolled message passing |
| Agent harness (8) | Partial — LangGraph gives you state/checkpointing/interrupts; you still assemble cost control, policy, and tracing yourself unless using additional tooling |

| Challenge | What CrewAI gives you "for free" |
|---|---|
| Router (5) | Manager-agent delegation in `Process.hierarchical`, at the cost of determinism |
| Multi-agent communication (9) | Task `context` dependencies handle sequential data flow automatically |

| Challenge | What the official MCP SDK gives you "for free" |
|---|---|
| MCP-like protocol (10) | A fully specified, versioned, capability-negotiated protocol with a maintained client/server SDK, instead of an ad hoc equivalent that only your own systems understand |

## The Point of This Exercise
After completing all 12, you should be able to look at any framework's source code or documentation and immediately map its abstractions back to the mechanism you built by hand — this is the concrete test of Rule 4 (build before abstracting) and Rule 9 (avoid vendor lock-in): you now understand what you're trading away (control, transparency) for what you're gaining (development speed, maintained infrastructure) when you adopt a framework.
