# Part LXIX — Expert Final Examination

## What this page is for, in plain terms

This is the one comprehensive test across the entire curriculum — not a per-part checkpoint, but the full spread: conceptual questions, implementation questions, debugging scenarios, architecture problems, and security scenarios, pulled from everything you've built since Part I. Treat it the way you'd treat a real technical interview loop: write actual answers, out loud or on paper, before checking yourself against the source material — silently reading a question and thinking "yeah, I basically know that" is the least reliable form of self-assessment there is, because it doesn't force you to produce anything a listener could evaluate.

**A worked example of the standard to hold yourself to:** take debugging scenario #2, "a resumed run duplicates a tool call that already executed before a crash." A shallow answer is "check the checkpointing code." A passing answer names the exact mechanism: check whether the checkpoint was saved *after* the step's side effect completed (not before, and not mid-step) per [Part IX §6](part09-agent-state.md), and separately, whether that specific tool call is idempotent or protected by an idempotency key per [Part XXXIII §3](part33-tool-engineering.md) — because either gap, independently, could produce this exact symptom, and a real fix requires knowing which one you're actually facing.

No answers are provided here by design — self-assess using the [Expert Checkpoints](part56-expert-checkpoints.md) method, or have a peer/mentor review your responses. A separate answer key is intentionally not included in this document; work through the material to produce your own defensible answers, then verify against primary sources and your own working implementations.

## Conceptual Questions (Sample of the required 100+ — extend using every Part's Knowledge checkpoint)
1. Why does prefill parallelize while decode does not, and what latency consequence follows?
2. Why is schema-valid JSON not the same guarantee as business-rule-valid data?
3. Why can longer chain-of-thought reduce accuracy on some tasks?
4. What distinguishes a workflow with an LLM step from a true agent?
5. Why must tool permission scope never be expandable by content the model reads?
6. Why does context engineering often matter more than prompt wording in complex agents?
7. Why is idempotency the deciding factor for whether a tool call can be safely retried?
8. What problem does a state-graph reducer solve?
9. Why are embeddings from two different models not comparable?
10. Why do ANN indexes trade recall for speed, and what parameter controls that in HNSW?

*(Continue this pattern by compiling the Knowledge checkpoint question from every Part I–LII — that alone yields well over 100 conceptual questions spanning the full curriculum.)*

## Implementation Questions (Sample of the required 50+)
1. Build an async LLM client with retry, backoff, and multi-provider fallback.
2. Build a tool-calling loop with schema validation and structured error feedback.
3. Build a bounded ReAct agent loop from scratch.
4. Build an atomic, versioned checkpoint store.
5. Build a long-term memory system with decay and conflict-aware writes.
6. Build a dependency-graph planner with bounded replanning.
7. Build a minimal MCP client/server pair.
8. Build a multi-server MCP tool registry with collision detection.
9. Build an HNSW-backed vector store with metadata filtering.
10. Build a calibrated LLM-as-judge evaluation harness.

*(Continue using every Part's Implementation checkpoint as a source — this yields the required depth across all 52+ substantive Parts.)*

## Debugging Scenarios (Sample of the required 50+)
1. A production agent exceeded its cost budget before being killed — where would you add the missing check?
2. A resumed run duplicates a tool call that already executed before a crash — what's the fix?
3. A RAG answer cites a source that doesn't support its claim — how do you isolate chunking vs. retrieval vs. generation as the cause?
4. Two concurrent processes both believe they hold a session lock — what would you check?
5. A filtered Qdrant query is 50x slower than unfiltered — what's the first thing to check?
6. An agent's self-correction loop "fixes" a correct answer into a wrong one — what does this suggest about the critique step?
7. A chaos test passes but a production incident shows a duplicated side effect — what coverage gap does this reveal?
8. A rolling deployment causes a brief 5xx spike — what health-check misconfiguration is most likely?
9. A distributed task was processed twice despite an idempotency key — what's the likely root cause?
10. An event-driven pipeline silently stops processing after an unrelated code change — what does `asyncio.create_task`'s exception-swallowing suggest?

*(Continue using every Part's Debugging checkpoint.)*

## Architecture Problems (Sample of the required 30+)
Draw from [Part LXV](part65-expert-system-design-questions.md)'s 100 system-design questions — select at least 30 spanning RAG, agents, MCP, reliability/security, and multi-agent categories.

## Security Scenarios (Sample of the required 20+)
1. A retrieved document contains "ignore previous instructions" — trace the defense-in-depth layers that should stop it.
2. An MCP server's tool description claims a destructive tool is harmless — how do you detect and contain this?
3. A browser agent encounters a page with hidden injected text in `aria-label` attributes — what's the correct handling?
4. An agent with an accidentally over-scoped tool set deletes data it shouldn't have had access to — what's the root cause class?
5. A compromised third-party MCP server returns a plausible-looking but malicious tool result — what containment layer should stop real damage?

*(Extend to 20 using [Part XXXV](part35-security.md)'s threat taxonomy.)*

## RAG Problems (Sample of the required 20+)
Draw from [Part LXV §"RAG and Retrieval"](part65-expert-system-design-questions.md) (questions 1–15) plus 5 more of your own devising covering chunking, embeddings, and evaluation.

## Multi-Agent Problems (Sample of the required 20+)
Draw from [Part LXV §"Multi-Agent and Advanced Architecture"](part65-expert-system-design-questions.md).

## MCP Problems (Sample of the required 20+)
Draw from [Part LXV §"MCP and Interoperability"](part65-expert-system-design-questions.md).

## LangGraph Problems (Sample of the required 20+)
1. What breaks if you omit a reducer on a list-typed state field?
2. How does `interrupt_before` interact with the checkpointer to enable human approval?
3. What causes a `KeyError` from a conditional edge, and how do you prevent it?
4. How would you implement time-travel debugging to inspect a specific historical decision?
5. How do subgraphs enable multi-agent composition, and what state-sharing pitfalls exist?

*(Extend to 20 using [Part XX](part20-langgraph.md)'s full content.)*

## Production Incidents (Sample of the required 20+)
Draw from the [Failure Mode Catalog](part57-failure-driven-learning.md) — for each of its ~25 rows, write a realistic incident narrative and practice the symptom → root cause → diagnosis → fix → prevention format independently before checking against the catalog's own entry.

## How to Use This Examination
Attempt every question in writing (not just "I could probably answer that") before consulting the referenced Parts. Where you cannot produce a confident, specific answer, that is a direct signal of which Part needs review — treat this exam as a diagnostic instrument for the [Expert Checkpoints](part56-expert-checkpoints.md) gating method, not a one-time test to pass and forget.
