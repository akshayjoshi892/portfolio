# Agentic AI Systems Mastery — 42-Day Expert Curriculum

**MCP · LangChain 1.0 · LangGraph 1.0 · Qdrant · CrewAI · LlamaIndex**

*Compiled September 2026 — pinned to current stable APIs: LangChain/LangGraph 1.0 (GA), MCP spec `2026-07-28`, CrewAI 1.x (Flows), LlamaIndex Workflows 1.0, Qdrant 1.x with FastEmbed/miniCOIL hybrid search.*

---

## How to use this document

You already have a running start: a self-built 7-day LangChain/LangGraph foundations course, hands-on LangGraph multi-agent pipelines (the CATIA MCP aerospace system, the stock-research agent), and deep ML fundamentals (transformers from scratch, tokenization, PyTorch internals). This curriculum does **not** re-teach LCEL basics or what a vector embedding is. It assumes that and spends the time on what actually separates "used LangChain in a tutorial" from "expert": production agent architecture, protocol-level MCP understanding, the specific mechanics of five frameworks side by side, and the judgment to pick the right one for a given job.

Structure: **6 weeks (42 days) + 3 buffer/polish days = 45 days**, matching your target. Each day is a **2–3 hour concept block** followed by a **2–4 hour hands-on build**. Skipping the build and only reading the concept is the single most common way this kind of plan fails — the projects are not optional decoration, they are where the mastery actually forms. Do not move to the next day until the previous day's project runs.

Every week ends with an **integration checkpoint**: a short exercise that forces you to combine that week's tool with at least one previous week's tool, so knowledge doesn't stay siloed.

```
Week 1 (Days 1–7)   Agentic Coding Foundations + Model Context Protocol (MCP)
Week 2 (Days 8–14)  LangChain 1.0 / LangGraph 1.0 — Expert Patterns
Week 3 (Days 15–21) Qdrant & Vector Search Mastery
Week 4 (Days 22–28) LlamaIndex — Data Framework & Agent Workflows
Week 5 (Days 29–35) CrewAI — Role-Based Multi-Agent Systems
Week 6 (Days 36–42) Capstone: one production-grade system using all five
Days 43–45          Buffer, hardening, portfolio packaging
```

### Environment setup (Day 0 — do this before Day 1)

- Python 3.11+ (LangChain 1.0 requires 3.10+; use 3.11 for best library support). Use `uv` for env/dependency management — it is materially faster than pip/poetry for the churn this curriculum involves.
- Docker (for local Qdrant, and later for MCP server containers).
- An LLM provider key (Anthropic or OpenAI). Optionally a second provider to practice provider-agnostic code — LangChain 1.0's "standard content blocks" make this easy to test.
- Editor: VS Code or Cursor with the Python + Docker extensions. Install the MCP Inspector (`npx @modelcontextprotocol/inspector`) now — you'll use it from Day 3 onward.
- Create one working repo now: `agentic-mastery/` with subfolders `week1/` … `week6/`, `capstone/`. Commit daily — the commit log becomes part of your portfolio evidence.
- Baseline install:

```bash
uv venv && source .venv/bin/activate
uv pip install "langchain>=1.0" "langgraph>=1.0" "langchain-anthropic" \
  "mcp[cli]" qdrant-client "fastembed" \
  "llama-index>=0.13" "llama-index-workflows" \
  "crewai>=1.0" "crewai-tools" \
  langsmith python-dotenv
```

---

## Week 1 — Agentic Coding Foundations + Model Context Protocol (Days 1–7)

The goal of this week is to understand agents and MCP **below** the framework layer. Every framework in weeks 2–5 is a wrapper around the concepts below. If you learn LangGraph before you understand raw tool-calling loops and the MCP wire protocol, you'll be able to use the framework but not debug it, extend it, or explain a design decision in an interview. Do this week with the raw Anthropic/OpenAI SDK and the raw `mcp` SDK — no LangChain yet.

### Day 1 — Agent architectures and the tool-use loop, from first principles

**Concept.** An "agent" is nothing mystical: it's a loop around an LLM call where the model's output can request an action, the action executes in your code, and the result is fed back as new context, repeated until the model decides it's done. Three canonical architectures, know the tradeoffs of each:

- **ReAct (Reason + Act)** — the model interleaves a reasoning trace with tool calls, one action at a time, observing the result before deciding the next action. Best for tasks where each step depends on the previous result (research, debugging, file exploration). Weakness: slow (fully sequential), and reasoning traces can drift/hallucinate justification for a bad next step.
- **Plan-and-Execute** — the model produces a full plan up front (a list of subtasks), then a (possibly separate, cheaper) executor works through the plan, only replanning if a step fails. Best for tasks that decompose cleanly (multi-file refactors, report generation). Weakness: brittle if the world changes mid-plan or the plan was wrong at step 1.
- **Reflection / self-critique** — the model produces an output, a second pass (same or different model) critiques it against criteria, and the first model revises. Best bolted onto either of the above for quality-sensitive output (code review, writing). Costs 2–3x the tokens for meaningfully better output on tasks where "good" is checkable.

At the API level, tool-calling is: you send a `tools` array of JSON Schema definitions alongside messages; the model returns either text or one/more `tool_use` blocks with a name and JSON input; you execute the named function locally, and return a `tool_result` block keyed to the tool call id in the next turn. `tool_choice` lets you force a specific tool, force *any* tool, or let the model decide (`auto`) — parallel tool calls (multiple `tool_use` blocks in one turn) are the default in current Claude and GPT models and change how you must structure your executor (it must handle a batch, not just one call, per turn).

**Project.** Implement a ReAct loop **by hand** (no LangChain) against the Anthropic SDK: a `run_agent(query, tools, max_steps)` function that (1) calls the model with your tool schemas, (2) executes any `tool_use` blocks, (3) appends `tool_result` blocks, (4) loops until the model stops requesting tools or `max_steps` is hit. Give it 3 tools: a calculator, a (mocked) web search, and a file reader scoped to a sandbox directory. Then implement Plan-and-Execute for the same tool set and compare behavior on a task like "find the largest file in this directory, then compute what percentage of total directory size it represents." Write down (in a `NOTES.md`) which architecture handled it better and why.

### Day 2 — Tool design, schemas, and context engineering

**Concept.** Most agent failures are tool-design failures, not model failures. Rules that separate amateur tool design from production-grade:

- **Narrow, verb-named tools** beat one mega-tool with a `mode` parameter — the model picks tools better when names and descriptions are unambiguous and single-purpose.
- **Strict JSON Schema** (`additionalProperties: false`, enums instead of free strings where the value set is closed, required fields marked) reduces malformed calls dramatically.
- **Return structured, truncated, and self-describing results.** A tool that dumps 50KB of raw JSON back into context burns your token budget and buries the signal. Summarize, paginate, and include a `next_cursor`-style continuation pattern for large results.
- **Idempotency and side-effect classification.** Mark tools as read-only vs mutating in your own metadata; only mutating tools should ever need human-in-the-loop confirmation (Week 2 covers LangGraph's built-in support for this).
- **Error messages are prompts.** When a tool fails, the string you return becomes part of the model's context — write it so the model can self-correct ("file not found: did you mean `report_v2.csv`?") instead of a bare stack trace.
- **Context engineering** is the 2026 term of art replacing "prompt engineering" for agent systems: deliberately managing what's *in the context window at each step* — system prompt structure, which past turns/tool results are still relevant, when to summarize vs drop vs retrieve. This matters more than clever prompting once loops run more than ~10 turns.

**Project.** Rebuild your Day 1 tool set as a small library with: Pydantic models for every tool's input schema (strict validation, no `additionalProperties`), a standard `ToolResult` wrapper (`ok: bool, data, error, truncated: bool`), and a results-truncation policy (cap any tool result at ~2000 tokens with a "show more" continuation token). Deliberately feed it 5 malformed/edge-case inputs (missing file, division by zero, a search query that returns 500 results) and confirm the agent self-corrects instead of crashing the loop.

### Day 3 — MCP fundamentals: why it exists, the architecture, your first server

**Concept.** Before MCP, every agent framework wired tools to LLMs with bespoke, incompatible glue — an N×M integration problem (N agent frameworks × M tools/data sources). **MCP (Model Context Protocol)**, an open standard originated by Anthropic and now broadly adopted (OpenAI, Google, Microsoft all ship MCP support), is the common interface: it turns N×M into N+M. Think of it as "USB-C for AI applications" — one plug, many devices.

Architecture: a **Host** (the AI application — Claude Desktop, Claude Code, your own agent) contains one or more **Clients**, each maintaining a 1:1 connection to one **Server**. A server exposes some combination of three primitives:

- **Tools** — model-callable functions with JSON Schema, exactly like the tool-calling you did on Day 1, but now served over a protocol instead of hardcoded in your process.
- **Resources** — addressable, read-only data (a file, a DB row, an API response) the *host application* decides to attach to context — the model doesn't call these, the app does, which is a key distinction from tools.
- **Prompts** — reusable, parameterized prompt templates a server exposes, surfaced to the user as e.g. slash commands.

Transport is JSON-RPC 2.0 over either **stdio** (the server is a local subprocess — simplest, used for local dev tools) or **Streamable HTTP** (the server is a network endpoint — needed for anything shared/remote).

**What changed in the current spec (`2026-07-28`) — know this, it's recent and matters if you build anything meant to run at scale:** earlier MCP versions required a stateful `initialize`/`initialized` handshake and a session ID pinned to one server instance. The 2026-07-28 spec moves to a **stateless architecture**: every request carries its own protocol version and capabilities in `_meta`, so any request can land on any server replica behind a plain round-robin load balancer with no shared session store. This is the single biggest operational change if you plan to run an MCP server behind more than one instance (which you will, later, for the CATIA-MCP work).

**Project.** Using the official Python SDK's `FastMCP` class, build your first MCP server exposing 3 tools (reuse your Day 2 tool library — this is the point: the same tools, now protocol-served) and one resource (a read-only config/status resource). Run it over stdio, then again over Streamable HTTP. Connect and exercise it with the **MCP Inspector** (`npx @modelcontextprotocol/inspector`) — call each tool, read the resource, and read the raw JSON-RPC traffic in the Inspector's network tab so you can see the actual wire format, not just the SDK's abstraction over it.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day3-tools")

@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the sandboxed workspace."""
    ...

@mcp.resource("config://status")
def status() -> str:
    """Read-only server status resource."""
    return "ok"

if __name__ == "__main__":
    mcp.run(transport="stdio")  # switch to "streamable-http" for the network variant
```

### Day 4 — MCP deep dive: Sampling, Prompts, multi-round-trip interaction, and the security model

**Concept.** Three more pieces you need before calling yourself MCP-fluent:

- **Sampling** — an inversion of the usual direction: the *server* can ask the connecting *client's* LLM to generate something (with the human able to approve/deny), useful when a server needs LLM reasoning but shouldn't hold its own API key or make model choices for the host app.
- **Multi Round-Trip Requests (MRTR)** — the 2026-07-28 spec's replacement for the older "elicitation" pattern that needed a held-open bidirectional stream. Now, when a tool needs more input mid-execution, the server responds with `resultType: "input_required"` plus a description of what's needed; the client collects it (e.g. by asking the user) and retries the *same* call with `inputResponses` attached. This lets you build interactive, human-in-the-loop tools (e.g., "confirm before deleting") on top of a fully stateless transport — directly relevant to the criticality-tiered human review gates pattern you're already using in the CATIA-MCP system.
- **Security model** — OAuth-based auth for remote servers, with the current spec adding RFC 9207 issuer validation (prevents a class of authorization-server mix-up attack), an `application_type` parameter that fixes localhost-redirect handling for CLI/desktop clients, and **Client ID Metadata Documents (CIMD)** formally replacing Dynamic Client Registration. If you expose an MCP server beyond localhost, this is not optional reading.

**Project.** Extend your Day 3 server with a tool that requires mid-call human confirmation using the MRTR pattern (e.g., a `delete_file` tool that returns `input_required` asking for an explicit "yes", and only executes on the retried call carrying the confirmation). Add a second tool that uses **sampling** to ask the client's model to summarize a file before returning it, instead of returning the raw content. Verify both flows in the Inspector.

### Day 5 — MCP clients and multi-server orchestration

**Concept.** A host application typically talks to *multiple* MCP servers simultaneously (e.g., a filesystem server + a GitHub server + your custom domain server), aggregating all their tools into one namespaced tool list for the model. Building a client means: spawning/connecting to each server, calling `list_tools`/`list_resources`/`list_prompts`, merging results (watch for name collisions — namespace by server), and routing each model tool-call to the right server connection. In production this is also where you add per-server timeouts, retries, and circuit breaking — a hung MCP server should not hang your whole agent.

**Project.** Build an MCP **client** (using the SDK's `ClientSession`) that connects to *two* servers at once: your own Day 4 server, and a public/reference MCP server (e.g., the official filesystem server, or GitHub's MCP server if you have a token). Merge their tool lists with a `server_name.tool_name` namespacing scheme, wire it into your Day 1 hand-rolled ReAct loop (replacing the hardcoded tool list with the merged MCP tool list), and run a task that needs a tool from *each* server in the same session — e.g., "read this local file, then check if there's an open GitHub issue mentioning its filename."

### Day 6 — Guardrails, memory, and observability for agentic systems

**Concept.** Three cross-cutting concerns that apply regardless of which framework you eventually pick:

- **Guardrails** — input validation (reject/sanitize before the model ever sees untrusted content — critical for prompt-injection defense when tool results include external content), output validation (schema-check structured outputs before acting on them), and action guardrails (a policy layer that can block a tool call outright based on rules, independent of what the model "wants").
- **Memory** — short-term is just the message window; long-term needs a strategy: rolling summarization (compress old turns into a summary block), retrieval-based (embed and store past interactions, pull back relevant ones — this is where Week 3's Qdrant work plugs in), or structured extraction (pull durable facts into a schema/DB rather than keeping raw transcript). Know which failure mode each solves and which it doesn't: summarization loses detail, pure retrieval loses temporal/causal ordering.
- **Observability** — agent systems fail in ways unit tests don't catch (a reasonable-looking tool call sequence that's still wrong), so tracing every step (prompt sent, tool called, result returned, tokens/cost/latency per step) is not optional in production. MCP's 2026-07-28 spec added first-class support for exposing this via OpenTelemetry-compatible tracing, and LlamaIndex's Workflows framework (Week 4) ships OTel integration out of the box — this is a convergence worth noticing.

**Project.** Add a lightweight tracing layer to your Day 5 client/agent (a simple structured logger is fine — JSON lines with `{step, tool, input, output, latency_ms, tokens}` — or wire up OpenTelemetry if you want the real thing) and a guardrail layer that blocks any tool call matching a denylist pattern (e.g., path traversal attempts in the file tools) *before* execution, logging the block as its own event. Deliberately try to get your own agent to do something the guardrail should catch, and confirm it's caught.

### Day 7 — Week 1 integration checkpoint: a framework-free agentic coding assistant

**Project (no new concept — this is the proof-of-understanding day).** Build a small but complete CLI "agentic coding assistant" using only what you've built this week: your hand-rolled agent loop (pick ReAct or Plan-and-Execute, justify the choice in `NOTES.md`), your MCP servers as the entire tool layer (file read/write/search, plus at least one external server), your guardrail and memory layers, and your tracing. Task it with something real: "add a docstring to every public function in this file that's missing one, then run the test suite and report failures." The point of this checkpoint is that from Week 2 onward you will let LangGraph/CrewAI/LlamaIndex manage the loop, memory, and orchestration for you — and you should only accept that convenience because you've now built the thing they're automating, and you'll know exactly what each framework is buying you and what it's hiding.

**Week 1 self-check before moving on:** Can you explain, without looking anything up, the difference between a Tool and a Resource in MCP and why that distinction exists? Can you explain why the 2026-07-28 spec removed the initialize handshake? If either answer is shaky, spend an extra half-day here — everything in Weeks 2–6 builds on this.


---

## Week 2 — LangChain 1.0 / LangGraph 1.0: Expert Patterns (Days 8–14)

You've already done LCEL, basic RAG, basic agents, and a multi-agent capstone in your own 7-day course. This week is deliberately **not** a repeat — it starts where that course stopped, at the 1.0 GA APIs (released 2026, replacing the 0.x patterns you learned), and pushes into the production patterns that separate a demo from a system: durable execution, human-in-the-loop gates, multi-agent topologies, and evaluation.

### Day 8 — LangChain 1.0: what actually changed, and `create_agent`

**Concept.** LangChain and LangGraph both hit GA 1.0 in 2026 — the first stable major release in this space, after adoption at companies like Uber and LinkedIn drove the API to settle. Know the concrete changes, because your existing course was almost certainly written against pre-1.0 patterns:

- **`create_agent`** (now in `langchain.agents`) is the new standard entry point for building an agent quickly against *any* model provider — it replaces `create_react_agent` (deprecated) and much of what used to live in `langgraph.prebuilt` (also deprecated). Under the hood, agents built with `create_agent` run on the **LangGraph runtime** — there is no more "LangChain agent vs. LangGraph agent" split; LangChain's high-level API is now a thin, ergonomic layer over LangGraph, and you can drop down to raw LangGraph whenever `create_agent` isn't flexible enough. This is the most important structural fact to internalize: you are not choosing between LangChain and LangGraph, you are choosing how much abstraction you want on top of the same runtime.
- **Middleware system** — the mechanism for injecting behavior at each step of the agent loop (before/after model call, before/after tool call) without hand-editing the loop. Built-ins include human-in-the-loop approval, message summarization for context management, and PII redaction. This is Day 9's topic in depth.
- **Standard content blocks** — a provider-agnostic content-block spec so reasoning traces, citations, and other modern model output types look the same regardless of provider, which matters the moment you test against more than one model.
- **Scope trim** — legacy chains/utilities you may remember from older tutorials moved to `langchain-classic`. If you see an import error for something an older doc mentions, that's why.
- Python 3.10+ is now required.

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5"),
    tools=[read_file, search_web],       # plain Python functions or @tool-decorated
    system_prompt="You are a precise research assistant.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarize report.pdf"}]})
```

**Project.** Take one non-trivial agent from your existing 7-day course (the RAG agent or an agent-with-tools example) and **migrate it to `create_agent`**. Diff the line count and structure against the old version in your `NOTES.md` — you should find the new version is shorter and that several things you previously hand-wired (the loop itself) simply disappear. Then connect it to your Week 1 MCP server using the `langchain-mcp-adapters` package, so its tools come from MCP instead of local Python functions — this is the first moment the two halves of this curriculum touch.

### Day 9 — Middleware: human-in-the-loop, summarization, and writing your own

**Concept.** Middleware wraps the agent loop at defined hook points, and is composable — you attach a list of them to `create_agent`. The three built-ins worth knowing cold:

- **Human-in-the-loop middleware** — pauses execution before specific tool calls (you configure which tools require approval) and surfaces the pending call for a human decision (approve / edit args / reject), resuming the graph afterward. This is built on the same `interrupt` primitive you'll use directly in LangGraph on Day 11.
- **Summarization middleware** — automatically compresses older messages once the context grows past a threshold, so long-running agents don't blow their context window or their cost. Configurable trigger (token count) and summarization prompt.
- **PII redaction middleware** — scans messages for sensitive patterns and redacts before they hit the model or get logged, a compliance-relevant default for anything handling user data.

Writing custom middleware means implementing hooks like `before_model`, `after_model`, `before_tool_call` — this is where you'd put cost tracking, custom retry/backoff on tool failures, or domain-specific validation (e.g., in the CATIA-MCP context: a middleware that blocks any CAD-mutating tool call above a criticality tier without an explicit human approval — a direct upgrade path from ad hoc gating logic to a reusable, composable layer).

**Project.** Attach human-in-the-loop middleware to your Day 8 agent for any tool that writes/deletes, test both the approve and reject paths. Then write one custom middleware from scratch — a cost/token tracker that accumulates spend across the whole run and raises (or triggers a warning tool call) if a budget threshold is crossed. Chain three middlewares together (HITL + summarization + your custom one) and confirm ordering matters by deliberately misordering them once and observing the difference.

### Day 10 — LangGraph core mental model at expert depth: state, reducers, `Send`, `Command`

**Concept.** You've built graphs before; this is about the pieces that make graphs *scale*:

- **State and reducers** — your `TypedDict` state schema controls not just what fields exist but *how updates merge* via `Annotated[list, add]`-style reducers. Getting reducers wrong is the #1 source of "why did my list get overwritten instead of appended" bugs in non-trivial graphs.
- **`Command`** — a return type from a node that combines a state update *and* a routing decision (which node to go to next) in one object, replacing separate conditional-edge functions when the routing logic is naturally colocated with the node's own logic.
- **`Send`** — the primitive for dynamic parallel fan-out: a node can return a list of `Send("worker_node", {...})` objects to spawn N parallel invocations of a subgraph/node over data whose size isn't known until runtime (classic map-reduce inside a graph), which then fan back in via a reducer on the shared state.
- **Subgraphs as nodes** — any compiled graph can be used as a node in a larger graph, which is how you build hierarchy without flattening everything into one giant state machine.

**Project.** Build a parallel research graph: given a topic, an initial node breaks it into N subtopics (N decided at runtime by the model, not hardcoded), fans out with `Send` to a "research subtopic" node running in parallel for each, then fans back in to a synthesis node that reduces all N results into one report. Verify actual parallelism (not just concurrency in appearance) by timing it against a sequential version with the same subtopic count.

### Day 11 — Durable execution: checkpointing, persistence, and human-in-the-loop that survives a restart

**Concept.** A **checkpointer** persists graph state after every super-step to a backend (in-memory for dev, SQLite/Postgres for anything real). This buys you three things at once: **resumability** (a crashed or killed process picks up exactly where it left off, keyed by `thread_id`), **time travel** (you can rewind to any prior checkpoint and re-run from there — invaluable for debugging a multi-step agent that went wrong at step 6 of 12), and true **human-in-the-loop** via the `interrupt()` function — a node can call `interrupt(payload)` to pause the graph and surface `payload` to a human; the graph genuinely stops (no polling loop, no held-open process) until you call `graph.invoke(Command(resume=value), config)` with the human's answer, at which point execution resumes from exactly that point with `value` bound in.

This is the piece that turns the "criticality-tiered human review gates" you're already using into something that can safely pause for hours or days (an engineer reviewing an aerospace part change) without holding a process open the whole time — the state is durably persisted, not sitting in memory waiting.

**Project.** Take your Day 10 graph and add a `PostgresSaver` (or `SqliteSaver` if you want zero extra infra) checkpointer. Add an `interrupt()` call before the synthesis node that requires a human to approve the set of subtopic results before synthesis proceeds. Kill the process (`kill -9`, not a graceful exit) mid-run after the interrupt fires but before you respond, restart it, and confirm you can still resume the exact same thread and complete the run. Then demonstrate time travel: fetch the state history for a completed thread and re-invoke from an earlier checkpoint with a modified input.

### Day 12 — Multi-agent architectures: supervisor, swarm, and network topologies

**Concept.** Three topologies, know when each fits:

- **Supervisor** — one central agent routes each turn to the right specialist subagent (each a subgraph or `create_agent` instance) and synthesizes their outputs; centralizes control, easiest to reason about and constrain, but the supervisor is a bottleneck and single point of failure for routing quality.
- **Swarm / handoff** — agents hand off control to each other directly (agent A decides agent B should take over and transfers context), no central router; more flexible and scales better to many peer specialists, but harder to constrain and debug because control flow isn't centralized.
- **Network** — any agent can call any other agent as a tool; most flexible, least structured, hardest to guarantee termination or bound cost — use only when the task genuinely doesn't decompose into a hierarchy.

For most production systems, **supervisor** is the right default (it's also what the `langgraph-supervisor` prebuilt package implements), reaching for swarm only when you have genuinely peer, interchangeable specialists (e.g., several equally-capable domain experts where routing needs to be dynamic and bidirectional).

**Project.** Build a supervisor system with three specialist subagents (e.g., a code-search agent, a documentation-writer agent, and a test-runner agent — or reuse the CATIA domain: a design-generation agent, a compliance-check agent, a structural-validation agent) where the supervisor routes each user request to the right specialist(s), potentially calling more than one in sequence, and synthesizes a final answer. Add the Week 1 human-in-the-loop gate so any specialist whose output would mutate a real artifact requires approval before the supervisor accepts it.

### Day 13 — Streaming, LangSmith evaluation, and deployment considerations

**Concept.** Three production concerns:

- **Streaming modes** — LangGraph supports streaming `values` (full state after each step), `updates` (just the delta), `messages` (token-level LLM output, for chat UIs), and `custom` (your own events emitted from inside a node, e.g., progress updates from a long tool call). Picking the right mode is a UX decision as much as a technical one — a chat UI wants `messages`, a progress dashboard wants `custom` or `updates`.
- **LangSmith** — tracing (every run, every step, viewable as a tree), dataset creation from real traces (turn production failures into regression test cases), and automated evaluation (LLM-as-judge or exact-match graders run against a dataset on every change, so you catch regressions before deploying a prompt or model swap).
- **Deployment** — whether you self-host (a checkpointer-backed graph behind FastAPI/your own server) or use LangGraph Platform, the operational unit is the same: a graph + a checkpointer + a way to route `thread_id`s to the right conversation/task. Know your options rather than defaulting to whichever one a tutorial used.

**Project.** Add `messages`-mode streaming to your Day 12 supervisor system and build a minimal CLI that renders tokens as they arrive. Separately, pull 10 real interactions from your testing into a LangSmith dataset, write an LLM-as-judge evaluator that scores whether the supervisor routed to the correct specialist, and run it — then deliberately break the supervisor's routing prompt and confirm the eval catches the regression.

### Day 14 — Week 2 integration checkpoint

**Project.** Combine everything: a durable (checkpointed), streaming, supervisor-based multi-agent LangGraph system whose specialist agents get their tools from your **Week 1 MCP servers** (not local Python functions), with human-in-the-loop middleware gating any mutating action, and a LangSmith eval suite covering at least the routing decision and one output-quality check. This is the artifact you should be able to describe end-to-end in an interview as "a production agent architecture," not "a LangGraph demo."


---

## Week 3 — Qdrant & Vector Search Mastery (Days 15–21)

You already understand embeddings and similarity at the ML-internals level (you've implemented transformers from scratch). This week skips "what is a vector" entirely and goes straight to how a real vector database is built and operated: the ANN algorithm, filtering, hybrid search, quantization, and how it plugs into the agent systems from Weeks 1–2.

### Day 15 — HNSW internals and Qdrant's storage architecture

**Concept.** Qdrant's core index is **HNSW (Hierarchical Navigable Small World)** — a multi-layer graph where the top layer is sparse (long-range links, fast coarse navigation) and each lower layer is denser, ending in the full graph at layer 0 containing every point. Search descends from the top layer greedily, refining at each level, until layer 0 gives the final nearest-neighbor candidates. Three parameters control the speed/recall/memory tradeoff and you must be able to reason about all three, not just tune them by trial and error:

- **`m`** — max connections per node per layer. Higher `m` → better recall, more memory (each point is more expensive to store), slower inserts.
- **`ef_construct`** — how wide the candidate list is *while building* the graph. Higher → better graph quality (and thus better recall at query time for a given `ef_search`), slower indexing.
- **`ef_search`** — how wide the candidate list is *at query time*. This is the knob you tune live/per-query for a recall/latency tradeoff without rebuilding anything — raise it when recall matters more than latency for a specific query.

Qdrant's storage model: a **collection** holds **points**, each point = an id + one or more named **vectors** + a JSON **payload**. Internally, points are split across **segments** (immutable + a small mutable one), each with its own HNSW index and payload index, merged/optimized in the background — this is why Qdrant can accept writes and serve reads concurrently without a global lock. Distance metrics (cosine, dot product, Euclidean) are a collection-level, per-named-vector setting — pick cosine for normalized embedding models (the default for nearly everything you'll use) unless the model card says otherwise.

**Project.** Run Qdrant via Docker (`docker run -p 6333:6333 qdrant/qdrant`). Create a collection with 100k+ synthetic or real embedded documents (reuse a corpus from your ML studies — even your 300 ML interview questions, embedded, works fine as a first dataset). Insert them, then run the same query set at three `ef_search` values (e.g., 32, 128, 512) and plot latency vs. recall@10 (compute recall against a brute-force exact search baseline). Separately, rebuild the collection with `m=8` vs `m=48` and observe the memory footprint difference (`GET /collections/{name}`). Write your findings — this is the kind of benchmark you should be able to produce on demand for any production vector-search decision.

### Day 16 — Payload filtering and the Query API

**Concept.** Real search is almost never pure vector similarity — it's similarity **plus constraints** ("similar documents, but only from 2026, only status=published"). Qdrant's filtering is not a post-filter bolted onto ANN results (which would silently reduce your effective `top_k`); it implements **filterable HNSW**, pushing the filter into the graph traversal itself so you get correctly-ranked, correctly-counted results even under a highly selective filter. To get this benefit, payload fields you filter on must be **indexed** (`create_payload_index`) — an unindexed filter still works but falls back to a slower scan.

Filter grammar: `must` (AND), `should` (OR), `must_not` (NOT), composable and nestable, with condition types for exact match, range, geo-radius/polygon, full-text match, and nested-object filters. The modern **Query API** (`query_points`) unifies what used to be separate search/recommend/scroll calls into one interface, and adds **grouping** (e.g., "top 3 results, but no more than 1 per document — dedupe by `doc_id`") and a **recommend** mode (find points similar to a set of positive examples and dissimilar to negatives — directly useful for "more like this, but not like that" agent memory retrieval).

**Project.** Take a realistic multi-field dataset (your stock-research or investing notes are a good fit here: ticker, sector, date, sentiment score, plus the embedded text) and build filtered search: similarity + `sector == X` + `date range` + `must_not: excluded tickers`. Benchmark query latency with and without a payload index on the filtered field at realistic selectivity (filter that keeps ~1% of points) to see the index's actual effect. Then implement grouping (max N results per ticker) and the recommend API (given 3 "liked" documents and 1 "disliked," retrieve more like the former).

### Day 17 — Hybrid search: dense + sparse, and fusion

**Concept.** Dense embeddings are excellent at semantic/conceptual similarity but weak on exact term/keyword matches (a product SKU, an acronym, a rare proper noun) — sparse retrieval (classic BM25, or learned sparse models) is the opposite. **Hybrid search** runs both and fuses the rankings. Qdrant stores sparse vectors as a first-class named vector type alongside dense ones on the same point (no separate keyword index system to keep in sync), and ships **FastEmbed**, a lightweight local embedding library, with both dense models and sparse options including **miniCOIL** — Qdrant's own compact neural sparse model tuned specifically to preserve exact-term signal that pure dense embeddings lose, positioned as a modern replacement for classic BM25/SPLADE in this stack. Fusion is typically **Reciprocal Rank Fusion (RRF)** (combine by rank position, not raw score — avoids the score-scale-mismatch problem between dense cosine scores and sparse BM25-style scores) or Qdrant's DBSF; the Query API supports fusion natively via a `prefetch` + fusion query rather than you hand-merging two result lists.

**Project.** Rebuild your Day 16 collection with both a dense vector (FastEmbed dense model) and a miniCOIL sparse vector per point. Run three searches against the same query set — dense-only, sparse-only, and RRF-fused hybrid — and manually inspect at least 10 queries where hybrid outperforms dense-only (look specifically for queries containing an exact term: a ticker symbol, an acronym, a proper noun) to build real intuition for *why* hybrid wins there, not just that it does.

### Day 18 — Quantization, scaling, and multitenancy

**Concept.** At real scale, full-precision (float32) vectors become a memory bottleneck before they become a compute bottleneck. Qdrant's quantization options, know when each applies:

- **Scalar quantization** (float32 → int8) — ~4x memory reduction, small recall loss, cheap to apply, good default first step.
- **Binary quantization** (→ 1 bit/dimension) — ~32x memory reduction and very fast Hamming-distance comparison, but meaningful recall loss unless paired with **oversampling + rescoring** (retrieve more candidates than needed using the cheap quantized index, then rescore the top candidates against the original full-precision vectors) — this combination is what makes binary quantization production-viable rather than just a memory trick.
- **Product quantization** — a middle ground, better compression than scalar at somewhat more recall cost and complexity than either extreme.

For scaling beyond one node: **sharding** distributes points across nodes for write/read throughput, **replication** gives availability. For **multitenancy** (many customers/users sharing infrastructure), the standard pattern is one collection with a `tenant_id` payload field plus a payload index on it, not N separate collections — this keeps operational overhead flat as tenant count grows, at the cost of needing the filter (and its index) to be correct and mandatory on every query.

**Project.** Take your Day 15 100k-point collection, apply binary quantization with oversampling+rescoring, and benchmark memory footprint, query latency, and recall@10 against the unquantized baseline. Then restructure it as a multitenant collection (partition your existing data into 5 synthetic "tenants" via a payload field), add the payload index, and verify (a) queries are correctly isolated per tenant and (b) query latency doesn't degrade meaningfully as you add tenants at fixed total data volume.

### Day 19 — Qdrant inside a real RAG pipeline: chunking, multi-vector, and reindexing safely

**Concept.** Retrieval quality is bottlenecked by chunking more often than by the vector DB — a chunk boundary that splits a fact from its context guarantees a retrieval miss regardless of index quality. Compare, hands-on rather than just by reading about it: fixed-size chunking, recursive/structure-aware chunking (split on headers/paragraphs first, fall back to size limits), and semantic chunking (split where embedding similarity between adjacent sentences drops). Beyond single-vector-per-chunk, Qdrant supports **multi-vector points** — storing several vectors per point (e.g., a ColBERT-style late-interaction model producing one vector per token, compared via max-sim at query time) for higher-fidelity retrieval at higher storage/compute cost, and **matryoshka embeddings** (models trained so a *truncated* prefix of the vector is still a valid, if lower-fidelity, embedding — lets you search cheaply with a short vector and rescore with the full one). Operationally: use **collection aliases** so you can build a new collection version, fully validate it, and atomically swap the alias to point production traffic at it with zero downtime — never reindex in place if you can avoid it. Snapshots give you point-in-time backup/restore.

**Project.** Build a document RAG pipeline over a real corpus (your ML reference docs are perfect — you have many) with three chunking strategies implemented side by side, evaluated on a hand-written set of 15 questions with known correct source chunks (measure recall@k per strategy, not vibes). Then implement the alias-swap pattern: build v2 of the collection with your best chunking strategy under a new name, validate it against your eval set, and swap the alias — confirm the running "production" query path never saw an error or an empty result during the swap.

### Day 20 — Qdrant as agent memory and semantic caching

**Concept.** Ties directly back to Week 1 Day 6: long-term agent memory is, concretely, a Qdrant collection of past interactions/facts, embedded and retrieved by relevance to the current turn rather than kept as raw transcript. Two patterns worth building: **episodic memory** (store summarized past sessions/turns, retrieve the K most relevant to the current query, inject into context) and **semantic caching** (before calling the LLM, check if a sufficiently-similar past query+response pair exists in Qdrant above a similarity threshold, and return the cached response — a real cost/latency lever for agents that see repetitive query patterns, with the threshold tuning being the entire game: too loose and you serve wrong answers, too tight and the cache never hits).

**Project.** Build a Qdrant-backed long-term memory module and wire it into your **Week 2 supervisor system** as a retrieval step before the supervisor routes each new request (inject the top-K relevant past interactions into context). Separately, implement semantic caching in front of one of your agent's LLM calls, instrument hit rate and measured cost savings, and find the similarity threshold empirically by testing paraphrased-but-answerable-the-same queries against near-miss queries that should *not* hit the cache — document where you set the threshold and why.

### Day 21 — Week 3 integration checkpoint

**Project.** Wrap your Day 19 hybrid-search RAG pipeline as an **MCP server** (connecting straight back to Week 1) exposing a `search_knowledge_base` tool, then give it to a specialist agent inside your Week 2 supervisor system, so the supervisor can route knowledge questions to a genuinely production-grade retrieval backend — hybrid search, filtered, quantized, memory-backed — rather than a toy in-memory list. This is the checkpoint where three weeks of separate work become one system.


---

## Week 4 — LlamaIndex: Data Framework & Agentic Workflows (Days 22–28)

LlamaIndex's identity is different from LangChain/LangGraph's: it started as, and remains strongest at, the **data layer** for LLM applications — ingestion, indexing, and retrieval — and has grown its own orchestration layer (**Workflows**, now a standalone 1.0 framework) for building agents on top of that data layer. Treat this week as "the RAG specialist framework," and use it to notice where its abstractions are genuinely better-fitted to data-heavy problems than LangChain's, and where they overlap.

### Day 22 — Documents, Nodes, and production ingestion pipelines

**Concept.** LlamaIndex's core data model: a **Document** is a raw source unit (a file, a web page, a DB row); a **Node** is a chunk of a document plus metadata and relationships to neighboring nodes (previous/next, parent/child) — the relationship graph between nodes, not just the chunk content, is what lets LlamaIndex do things like retrieve a small chunk but expand to its parent for synthesis (auto-merging retrieval, Day 24). **Readers** are the connector layer (100+ built-in, covering everything from PDFs to Notion to Slack) that turn a source into Documents. **Node parsers / transformations** turn Documents into Nodes (sentence splitters, semantic splitters, metadata extractors that use an LLM to generate a title/summary/questions-this-chunk-answers for each node — the extracted metadata becomes part of what's embedded and searchable, often a bigger retrieval-quality lever than the chunking strategy itself).

The **`IngestionPipeline`** chains transformations with a **docstore + cache**: rerun the same pipeline over updated source data and only changed documents are reprocessed (content-hash-based dedup), which matters enormously the moment your corpus isn't static — recomputing every embedding on every run is both slow and needlessly expensive.

**Project.** Build an `IngestionPipeline` over a heterogeneous corpus (PDFs + markdown + at least one live web source) with a sentence splitter, a metadata extractor (LLM-generated summary per node), and a persistent docstore-backed cache. Run it once, note the timing; add 2 new documents and modify 1 existing one; rerun and confirm (via logs/instrumentation) that only the 3 affected documents were reprocessed, not the whole corpus.

### Day 23 — Indexing strategies, and Qdrant as the vector store backend

**Concept.** LlamaIndex separates "index type" from "vector store backend" — you choose an index structure for *how* it organizes retrieval, and a storage backend for *where* vectors live. Index types worth knowing precisely:

- **`VectorStoreIndex`** — the default: embed every node, retrieve by similarity. Right choice for most fact-lookup RAG.
- **`SummaryIndex`** — no embeddings; on query, synthesizes across *all* nodes (or a subset) sequentially. Right choice when a question needs the whole corpus considered ("summarize everything about X across all documents"), where similarity-based top-k retrieval would arbitrarily miss relevant nodes.
- **`DocumentSummaryIndex`** — indexes per-document summaries for a first-pass filter, then drills into the relevant document(s) — a middle ground for large multi-document corpora where per-chunk retrieval alone loses document-level context.
- **`PropertyGraphIndex`** (the current graph index, superseding the older `KnowledgeGraphIndex`) — extracts entities and relations into a graph, supports both graph traversal and vector-similarity retrieval over the same data; the right choice when relationships between entities *are* the information need ("what depends on component X"), not just topical similarity.

Swap the storage backend to **Qdrant** (`QdrantVectorStore`) so all of Week 3's work — hybrid search, quantization, filtering — is available underneath any LlamaIndex index, rather than LlamaIndex's simple default in-memory store.

**Project.** Over the same corpus from Day 22, build a `VectorStoreIndex` backed by `QdrantVectorStore` (reusing your Week 3 Qdrant instance) and a `SummaryIndex`. Run both a fact-lookup question ("what does function X do") and a whole-corpus question ("what are the common themes across all these documents") against both indices and compare answer quality — you should be able to show concretely why each index type exists by finding a question the "wrong" index handles worse.

### Day 24 — Query engines, retrievers, and response synthesis

**Concept.** A **retriever** returns nodes; a **query engine** wraps a retriever with a **response synthesis** strategy that determines how retrieved nodes become a final answer: `refine` (iteratively update an answer node-by-node — most thorough, most LLM calls), `compact` (pack as many nodes as fit into one context window per LLM call, refine only across the resulting batches — the usual production default), `tree_summarize` (recursively summarize pairs/groups of nodes bottom-up — good for large result sets that don't fit any single context window). Query **transformations** improve the query itself before retrieval: **HyDE** (Hypothetical Document Embeddings — have the LLM write a hypothetical answer first, embed *that*, since it's often closer in embedding space to real answer documents than the bare question is), **sub-question decomposition** (split a compound question into independently-retrievable sub-questions, answer each, combine), and **multi-step query** (iteratively refine the query based on intermediate retrieval results). A **`RouterQueryEngine`** picks which underlying query engine/index to use per incoming query (e.g., route "summarize everything" to your `SummaryIndex`, route factual lookups to your `VectorStoreIndex`) — and **postprocessors** (rerankers, similarity cutoffs, diversity/MMR) run on retrieved nodes before synthesis, an easy, high-leverage retrieval-quality lever that's often skipped by beginners.

**Project.** Build a `RouterQueryEngine` combining your Day 23 `VectorStoreIndex` and `SummaryIndex`, add a reranker postprocessor (cross-encoder reranking on top of the initial vector retrieval) to the vector path, and run a mixed test set of fact-lookup and whole-corpus questions, confirming the router picks correctly and the reranker measurably improves top-result relevance versus no reranking.

### Day 25 — LlamaIndex Workflows 1.0: event-driven orchestration

**Concept.** **Workflows** is LlamaIndex's current orchestration model, now a standalone package (`llama-index-workflows`, importable independent of the rest of LlamaIndex) rather than something bundled only inside the main library — a deliberate move to make it a general-purpose agentic orchestration framework, not just a LlamaIndex-specific feature. The mental model: a workflow is a set of **steps** (plain async functions decorated with `@step`), each step consumes one event type and emits another; the framework wires steps together implicitly by matching event types, rather than you drawing explicit edges the way LangGraph requires. A typed **`Context`** object carries shared state across steps (1.0 added proper typed state in both the Python and TypeScript versions), and built-in OpenTelemetry instrumentation gives you tracing without extra wiring.

This is a genuinely different mental model from LangGraph's explicit graph, and it's worth being deliberate about the comparison rather than treating them as interchangeable: **LangGraph** makes control flow explicit and inspectable (you can look at the graph and see every possible path) at the cost of more upfront structure; **Workflows** makes control flow implicit via event-type matching, which is faster to write and extend (add a new step, wire it to existing events, done — no edge surgery) but harder to statically see the full set of possible paths through the system just by reading the step list.

```python
from workflows import Workflow, step, Context
from workflows.events import StartEvent, StopEvent, Event

class RetrievedEvent(Event):
    nodes: list

class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievedEvent:
        nodes = query_engine.retrieve(ev.query)
        return RetrievedEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrievedEvent) -> StopEvent:
        answer = synthesizer.synthesize(ev.nodes)
        return StopEvent(result=answer)
```

**Project.** Build a **corrective-RAG** workflow: a `retrieve` step, a `grade_relevance` step that scores whether retrieved nodes actually answer the query (LLM-as-grader), a conditional path where low-relevance results trigger a `requery` step (rewriting the query, e.g. via HyDE from Day 24) and loop back to retrieve, and a final `synthesize` step. Implement this as pure event-driven steps (no explicit graph edges) and confirm the loop (retrieve → grade → requery → retrieve again) actually works via events alone.

### Day 26 — Agents and multi-agent systems on top of Workflows

**Concept.** LlamaIndex's current agent abstractions (`FunctionAgent` for a single tool-using agent, and `AgentWorkflow` for orchestrating several agents together) are themselves built on top of the Workflows engine from Day 25 — so everything you learned about steps/events/Context underlies multi-agent orchestration here too, just as everything in LangChain's `create_agent` sits on LangGraph. `AgentWorkflow` supports explicit **handoffs** between named agents (one agent's output can be routed to a named other agent, similar in spirit to LangGraph's swarm pattern from Week 2 Day 12) and structured output validation on agent responses.

**Project.** Build a 2-agent `AgentWorkflow`: a research agent (tools: your Day 24 query engine as a retrieval tool) that hands off to a writer agent (no retrieval tools, just synthesis) once it has gathered enough information, with structured output (a Pydantic schema for the final report) enforced on the writer agent's response. Compare this side-by-side with the LangGraph supervisor system you built in Week 2 Day 12 — same conceptual task, two frameworks — and write down in `NOTES.md` three concrete differences in how much code each required and how debuggable each felt when you deliberately broke something.

### Day 27 — Observability and evaluation in LlamaIndex

**Concept.** LlamaIndex ships its own **evaluation modules**, purpose-built for RAG quality rather than general agent evaluation: **faithfulness** (does the answer's claims actually follow from the retrieved context, or is the model adding unsupported content — the RAG-specific hallucination check), **relevancy** (is the retrieved context actually relevant to the query — a retrieval-quality check independent of the final answer), and **correctness** (does the answer match a reference/ground-truth answer, when you have one). These are narrower and more RAG-specific than LangSmith's general-purpose eval framework from Week 2 — use LlamaIndex's evaluators when the question is specifically about retrieval/grounding quality. The optional `llama-index-instrumentation` package gives OpenTelemetry-compatible tracing, which is worth wiring up if you're already sending LangGraph/MCP traces to the same collector, so you get one unified trace view across frameworks rather than three separate dashboards.

**Project.** Build an evaluation suite over your Day 25 corrective-RAG workflow: a test set of 15–20 questions with known-good reference answers, scored for faithfulness, relevancy, and correctness. Run it against your workflow with the `grade_relevance`/requery loop enabled, then disabled, and confirm the eval numbers actually show the loop improving relevancy (if they don't, that's a real finding — investigate whether your grading step or your requery rewrite is the weak link).

### Day 28 — Week 4 integration checkpoint: LangGraph vs. LlamaIndex Workflows, and a shared decision framework

**Project.** You now have the same conceptual agent (corrective/agentic RAG) built in both LangGraph (Week 2/3) and LlamaIndex Workflows (this week). Write a short internal decision doc — genuinely useful for future project choices, not busywork — covering: when you'd default to LangGraph (complex, highly-stateful, needs explicit inspectable control flow, heavy human-in-the-loop and durable execution needs) versus LlamaIndex Workflows (data/RAG-heavy, benefits from the built-in data connectors and evaluators, event-driven extension is more natural than graph surgery) versus just LlamaIndex's higher-level query engines with no custom orchestration at all (when the task is "good RAG," full stop, with no multi-step agent behavior needed). Then wrap your best-performing corrective-RAG workflow as an MCP tool and give it to your Week 2 supervisor system as an alternative to the raw Qdrant search tool from Week 3 — you should now have two retrieval-backed tools available to compare inside the same agent system.


---

## Week 5 — CrewAI: Role-Based Multi-Agent Systems (Days 29–35)

CrewAI's philosophy is the most distinct of the three orchestration frameworks in this curriculum: instead of an explicit state graph (LangGraph) or implicit event wiring (LlamaIndex Workflows), it's **role-playing multi-agent-first** — you describe agents by role/goal/backstory and tasks by description/expected-output, and the framework handles the loop. This week is about learning it on its own terms, then folding it into the same comparison framework you built in Week 4.

### Day 29 — Agents, Tasks, Crews, and Process

**Concept.** The four core abstractions:

- **Agent** — defined by `role`, `goal`, and `backstory` (yes, a persona — this isn't just flavor text, it measurably shapes output style and the agent's own judgment about what's in-scope), plus `tools`, an LLM, and `allow_delegation` (whether this agent can hand sub-work to other crew members).
- **Task** — a `description` and `expected_output` (a plain-language spec of what "done" looks like — this is doing real work analogous to a JSON schema in tool design, just for the agent's *output* rather than a tool's *input*), assigned to one agent, optionally with `context` (other tasks whose outputs feed this one) and `output_pydantic`/`output_json` for structured results.
- **Crew** — the collection of agents + tasks + a **Process**.
- **Process** — `sequential` (tasks run in the order defined, each getting prior tasks' outputs as context) or `hierarchical` (a manager agent — either one you define or an auto-provisioned manager LLM — dynamically decides task order and delegation, closer to LangGraph's supervisor pattern but with the manager itself LLM-driven rather than code-driven).

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(role="Research Analyst", goal="Find accurate, current facts",
                    backstory="Meticulous, cites sources, flags uncertainty.",
                    tools=[search_tool])
writer = Agent(role="Technical Writer", goal="Produce clear, structured reports",
               backstory="Writes for engineers, no fluff.")

research_task = Task(description="Research {topic}", expected_output="Bullet facts with sources.",
                      agent=researcher)
write_task = Task(description="Write a report from the research", expected_output="A 500-word report.",
                   agent=writer, context=[research_task])

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], process=Process.sequential)
result = crew.kickoff(inputs={"topic": "MCP 2026-07-28 spec changes"})
```

**Project.** Build a 3-agent sequential crew (e.g., researcher → analyst → writer, or map to your CATIA domain: requirements-interpreter → design-generator → compliance-reviewer) and deliberately test how much the `backstory` field actually changes output — run the identical task with a terse, neutral backstory versus a detailed, opinionated one, and compare outputs side by side. This is the fastest way to build real intuition for how much of CrewAI's behavior is "prompt engineering wearing a framework's clothes," which is worth understanding precisely rather than either dismissing or overtrusting.

### Day 30 — Tools, delegation, and connecting your MCP servers

**Concept.** `crewai-tools` ships a large built-in library (web search, file I/O, code execution, RAG tools) and supports the same custom-tool patterns you've now built twice (subclass `BaseTool`, or use the `@tool` decorator) — and, importantly for this curriculum, CrewAI supports consuming **MCP servers directly as tool sources**, so your Week 1 MCP servers plug in here exactly as they did for LangChain and LlamaIndex. **Delegation** (`allow_delegation=True`) lets an agent that gets stuck or judges another agent better-suited hand off a (sub-)task mid-run — this is CrewAI's version of the swarm/handoff pattern from Week 2, but decided by the delegating agent's own LLM judgment at runtime rather than a topology you hardcode.

**Project.** Rebuild your Day 29 crew's tools as MCP-served tools (point it at your Week 1 and Week 3 MCP servers instead of local functions/built-ins), and enable delegation on your researcher agent. Design a task deliberately ambiguous about which specialist should handle a sub-part, and observe whether/how the delegation actually fires — log every delegation decision so you can see the agent's own reasoning about *why* it handed off.

### Day 31 — Hierarchical process, task dependencies, and structured outputs

**Concept.** In `Process.hierarchical`, a manager (auto-provisioned, or an `Agent` you define with `allow_delegation` implicitly true) reads the full task list and dynamically assigns/sequences work across the crew — useful when you don't know up front which agent should do which task, versus sequential where you do. Task `context` chains (task B declares task A as context) is how outputs flow between tasks without you manually threading state — analogous to LangGraph's state object, but implicit per-task rather than one shared schema. For anything downstream that needs to *parse* an agent's output programmatically (not just read it), always set `output_pydantic` (or `output_json`) on the task — treat an unstructured task output the same way you'd treat an untyped API response: fine for a human, unsafe for code.

**Project.** Rebuild Day 30's crew as `Process.hierarchical` with an explicit manager agent (not the auto-provisioned one, so you can inspect its own role/goal), and add `output_pydantic` to every task whose result feeds into another task or into your own code afterward. Run the same ambiguous-ownership task from Day 30 again and compare how the *hierarchical* manager's delegation compares to the *sequential+allow_delegation* version — write down which felt more predictable and which produced better output, and why you think that is.

### Day 32 — CrewAI Flows: deterministic control meets crews as building blocks

**Concept.** **Flows** is CrewAI's answer to "role-playing crews are great for open-ended collaborative work, but sometimes you need deterministic, code-level control flow" — it sits architecturally in almost exactly the space LangGraph and LlamaIndex Workflows occupy, which makes this the moment all three orchestration philosophies are visible side by side in one curriculum. A Flow is Python methods decorated `@start` (entry points), `@listen` (runs when a named prior method completes, receiving its return value), and `@router` (branches to different next steps based on a condition) — with typed **state** (a Pydantic model) shared across the flow, persisted the same way you'd want for durable execution. Crews are invoked *from inside* Flow steps as one callable unit — so the pattern is: use deterministic Flow logic for control decisions and orchestration, and delegate to a Crew wherever you actually want open-ended, role-based collaborative reasoning.

```python
from crewai.flow.flow import Flow, start, listen, router

class ReportState(BaseModel):
    topic: str = ""
    needs_deep_research: bool = False

class ReportFlow(Flow[ReportState]):
    @start()
    def get_topic(self):
        self.state.needs_deep_research = assess_complexity(self.state.topic)

    @router(get_topic)
    def route(self):
        return "deep" if self.state.needs_deep_research else "quick"

    @listen("deep")
    def run_deep_crew(self):
        return deep_research_crew.kickoff(inputs={"topic": self.state.topic})

    @listen("quick")
    def run_quick_crew(self):
        return quick_summary_crew.kickoff(inputs={"topic": self.state.topic})
```

**Project.** Build a Flow that routes between two different crews based on a runtime condition (e.g., a complexity classifier deciding between your Day 31 full hierarchical crew for hard requests and a single-agent fast path for simple ones), with typed Flow state tracking the decision and both crews' outputs. Then write the three-way comparison you're now equipped for: the same conditional-routing behavior expressed as a LangGraph conditional edge (Week 2), a LlamaIndex Workflow router step (Week 4), and this CrewAI Flow — put the three code snippets side by side in `NOTES.md` and note where each felt most natural to write.

### Day 33 — Memory, knowledge sources, and guardrails in CrewAI

**Concept.** CrewAI agents can be given `memory=True` for built-in short-term (current run), long-term (persisted across `kickoff()` calls, backed by local storage by default — swap in Qdrant for production), and entity memory (tracked facts about specific entities mentioned across the conversation). **Knowledge sources** are CrewAI's built-in RAG layer — attach a knowledge source (files, strings, or a custom retriever) directly to an agent or crew and it's automatically retrieved into context relevant to the current task, without you hand-building the retrieval step (though you can still point it at your own Qdrant-backed retriever for full control, tying Week 3 in directly). Task/agent-level **guardrails** (a validation function that can reject an output and force a retry with feedback) are CrewAI's version of the output-validation guardrail concept from Week 1 Day 6.

**Project.** Add persistent long-term memory to your Day 32 flow's crews (backed by Qdrant rather than the local default — reuse Week 3's setup) and confirm memory actually persists and gets used across two separate `kickoff()` calls in different processes. Add a knowledge source backed by your Week 4 corpus, and add one guardrail on the final report task that rejects outputs failing a structural check (e.g., missing a required section), confirming the retry-with-feedback loop actually fires and fixes the output on the second attempt.

### Day 34 — Observability, testing, and deploying a crew

**Concept.** Treat a crew like any other production service: instrument it (CrewAI integrates with standard LLM observability tooling — wire it into the same trace collector you've been using since Week 1/2 if possible, for one unified view rather than a fourth separate dashboard), test it (unit-test individual tools and guardrail functions in isolation the normal way; test full-crew behavior with a small eval set the same way you built LangSmith/LlamaIndex eval suites, since "does this multi-agent system produce a correct/good output" is fundamentally the same question regardless of framework), and deploy it (wrap `crew.kickoff()`/`flow.kickoff()` behind a FastAPI endpoint keyed by a run/session id — the same operational shape as the LangGraph deployment pattern from Week 2, just without a graph-native checkpointer, so you're responsible for your own state persistence if a run needs to survive a restart). Be honest with yourself about cost: role-playing multi-agent crews tend to make **more LLM calls** than an equivalent single-supervisor LangGraph system for the same task, because each agent's persona/reasoning is its own call — this is a real tradeoff against CrewAI's ease-of-authorship, not a hypothetical one, and worth actually measuring rather than assuming.

**Project.** Instrument your Day 33 system end to end, write a 10-case eval set scoring final report quality, and deploy it behind a minimal FastAPI endpoint. Then run a direct cost/latency comparison: the same task through this CrewAI system versus your Week 2 LangGraph supervisor system versus your Week 4 LlamaIndex AgentWorkflow — same task, same underlying model, measure total tokens, wall-clock latency, and number of LLM calls for each. Put the numbers in `NOTES.md`; this is the concrete evidence behind the qualitative comparison you'll finalize on Day 35.

### Day 35 — Week 5 integration checkpoint: the three-framework decision matrix

**Project.** Finalize the comparison you've been building evidence for all week (and since Week 4): a decision matrix across **LangGraph, LlamaIndex Workflows, and CrewAI** covering control-flow style (explicit graph / implicit events / role-based-plus-explicit-flow), best-fit task shape, human-in-the-loop and durable-execution maturity, typical cost profile (calls per task), and your own measured numbers from Day 34. This artifact — not any single framework's syntax — is the actual expert-level output of weeks 2, 4, and 5: the judgment to pick the right tool for a new problem, backed by your own benchmarks rather than someone else's blog post opinion.


---

## Week 6 — Capstone: One Production-Grade System Using All Five (Days 36–42)

The capstone is not "use every tool because they're on the syllabus." Each tool earns its place in the architecture for a specific reason, stated up front — that's what makes this an expert-level exercise rather than a kitchen-sink demo. Two suggested capstone directions, either genuinely worth building given your existing projects (pick one, or substitute your own domain if a better one occurs to you once you start designing):

- **(A) Enterprise Agentic Design-Review Assistant** — a generalized, portfolio-safe version of the CATIA-MCP work: an assistant that ingests engineering/compliance documents, answers questions against them, drafts a design-change proposal, runs it through a criticality-tiered review, and only auto-approves low-risk changes.
- **(B) Agentic Equity Research Analyst** — building on the stock-research work: ingests filings/news, retrieves relevant historical context, produces a research note with an explicit confidence/risk section, and requires human approval before any output is marked "actionable."

**Suggested architecture (adapt to whichever direction you pick), with the rationale for each tool spelled out — copy this reasoning pattern for any future system, it's the actual transferable skill:**

- **MCP** — the tool/data access layer. Every external capability (document store, search, a domain API, a notification/alerting sink) is an MCP server, chosen specifically because it decouples the orchestration layer from tool implementation — you can swap or add a tool source without touching agent code, and the same servers are reusable from a completely different orchestrator later.
- **LangGraph** — the top-level orchestrator, chosen specifically for durable execution and first-class human-in-the-loop: this system has a real approval gate that must survive being paused for hours, which is precisely LangGraph's strongest guarantee among the three orchestration options you now know.
- **LlamaIndex** — the ingestion and RAG layer, chosen specifically for its ingestion pipeline (incremental reprocessing as documents change) and its RAG-specific evaluators (faithfulness matters enormously when the output might be "actionable").
- **Qdrant** — the vector store underneath LlamaIndex (hybrid search for exact-term matches like part numbers/tickers) *and* the backing store for the system's long-term memory (past decisions, past research notes) — one piece of infrastructure serving two roles, which is itself a real architectural decision worth noting explicitly rather than standing up two separate stores.
- **CrewAI** — used narrowly, for exactly one genuinely open-ended, collaborative-reasoning subtask inside the larger graph (e.g., a "red team" crew that role-plays multiple review perspectives before the human sees the proposal, or a multi-analyst-perspective research crew) — invoked as one node in the LangGraph graph, not as the top-level orchestrator, because this system's overall control flow needs to be the explicit, inspectable, durable kind.

### Day 36 — Architecture and design document

**Project.** Before writing code: a one-to-two-page design doc. Requirements (what does the system actually need to do, precisely — vague requirements produce vague architecture). A system diagram showing every MCP server, the LangGraph graph's nodes/edges, where Qdrant sits, where the CrewAI crew is invoked from, and where the human-in-the-loop gate(s) are. An explicit list of what happens on failure at each stage (an MCP server times out — then what; a CrewAI crew produces an unusable output — then what). Get this doc to a state you'd be comfortable presenting in a design review — this is a real professional skill this curriculum has otherwise only exercised implicitly.

### Day 37 — Build the MCP tool/data layer

**Project.** Stand up every MCP server the design calls for (reuse Week 1's servers where they fit, build new ones where they don't), each independently testable via the MCP Inspector before it's wired into anything else. Include your Week 3/4 Qdrant hybrid-search and LlamaIndex RAG capabilities as MCP tools here, exactly as you did in the Week 3/4 checkpoints — this capstone week is substantially about *integrating* prior checkpoints into one coherent system, not building everything from zero.

### Day 38 — Build the LangGraph orchestration layer

**Project.** The top-level durable, checkpointed LangGraph graph: routing logic, the human-in-the-loop approval gate(s) via `interrupt()`, and a node that invokes your (not-yet-built, stub it for now) CrewAI crew as a sub-step. Get the full skeleton running end-to-end with stubbed-out pieces before deepening any one part — a thin, fully-connected system beats a deep, disconnected one at this stage.

### Day 39 — Build the ingestion + hybrid RAG knowledge base

**Project.** The full LlamaIndex `IngestionPipeline` over your real (or realistic synthetic) domain corpus, feeding a Qdrant collection with hybrid dense+sparse vectors, wrapped as the MCP tool from Day 37. Run your Day 27-style evaluation suite (faithfulness/relevancy) against it specifically in this capstone's domain, since a generic eval from Week 4 doesn't guarantee this corpus retrieves well.

### Day 40 — Build the CrewAI subtask and wire it in

**Project.** Replace Day 38's stub with a real CrewAI crew (or Flow, if the subtask needs internal routing) for the one genuinely open-ended reasoning subtask you identified in the design doc. Confirm the LangGraph node correctly invokes it, receives a structured (`output_pydantic`) result, and that a bad/malformed crew output is handled gracefully by the graph rather than crashing the run.

### Day 41 — Observability, evaluation, and guardrails, end to end

**Project.** One unified trace view across the whole stack (every MCP call, every LangGraph node, the CrewAI crew's internal steps, every Qdrant query) — this is the day you find out whether your Week 1–5 observability choices actually compose, or whether you have four incompatible logging formats that need reconciling. Build an end-to-end eval set (not per-component — full task-in, task-out) and a guardrail review: walk through every mutating/high-stakes action in the system and confirm each one is genuinely gated, not just assumed to be.

### Day 42 — Deployment and demo

**Project.** Containerize the system (each MCP server, the orchestrator, Qdrant) with Docker Compose at minimum; deploy the orchestrator behind FastAPI with a real `thread_id`-per-conversation model; load-test it lightly (even 10 concurrent runs is enough to surface a shared-state bug); write a README a stranger could use to run it. Record a short demo (screen recording or GIF) showing a full run including the human-approval gate actually pausing and resuming. This is the artifact you show in an interview or a portfolio — treat Day 42's polish level accordingly.

---

## Days 43–45 — Buffer, Hardening, and Portfolio Packaging

Don't skip these — a system that "works on the happy path" is not the same as a system you can defend under questioning, and the difference is exactly what these three days close.

**Day 43 — Adversarial bug bash.** Deliberately attack your own capstone: prompt-injection attempts via document content the system ingests, malformed/adversarial MCP tool inputs, a CrewAI crew given a task designed to produce garbage output, killing the process at every stage you can think of to test durability. Fix everything you find; log what you found and fixed — this list is itself evidence of rigor worth having ready to discuss.

**Day 44 — Documentation and portfolio packaging.** A clean architecture diagram, a README that explains *why* each framework was chosen (not just how to run it — reuse your Day 36 rationale), a short written case study (problem → architecture → tradeoffs → results, including your Week 5 cost/latency comparison numbers), and the demo recording from Day 42 embedded or linked. This is what goes on your portfolio site and gets referenced in interviews and Upwork proposals.

**Day 45 — Self-assessment and what's next.** Score yourself honestly against the checklist below. For anything you can't answer confidently, that's your next week's work, not a gap to paper over. Given your existing interest in OSS contribution (Browser Use, Khoj, Langfuse, Onyx, mcp-use) as a career accelerant, this is also the natural moment to pick one and make a first real contribution — you now have the protocol-level MCP fluency and multi-framework judgment to read an unfamiliar agentic codebase and contribute meaningfully, which was very likely not true 45 days ago.


---

## Appendix A — Framework decision matrix (fill in your own Week 5 numbers)

| Dimension | LangGraph | LlamaIndex Workflows | CrewAI |
|---|---|---|---|
| Control-flow model | Explicit graph (nodes/edges, `Command`, `Send`) | Implicit, event-type-driven steps | Role-based crews; explicit `Flow` for deterministic control |
| Best-fit task shape | Complex, stateful, needs inspectable control flow | Data/RAG-heavy pipelines, incremental extension | Open-ended collaborative reasoning; fast to author |
| Human-in-the-loop | First-class (`interrupt`/`Command(resume=...)`) | Possible, less built-in | Guardrails (retry-with-feedback), less native pause/resume |
| Durable execution | First-class (checkpointers, time travel) | Possible via Context persistence, less turnkey | Not native; you own persistence |
| Native data/RAG tooling | Via integrations | Best-in-class (readers, ingestion, evaluators) | Built-in "knowledge sources," simpler |
| Typical cost profile | Lower calls/task (single supervisor common) | Comparable to LangGraph for similar topology | Higher calls/task (persona-per-agent reasoning) |
| Learning curve | Moderate–steep (explicit state design) | Moderate (event model is a new mental model) | Shallow to start, steeper for hierarchical/Flows |

## Appendix B — MCP primitives cheat sheet

| Primitive | Who invokes it | Purpose |
|---|---|---|
| Tool | The model (via the host) | Model-callable function, JSON Schema in/out |
| Resource | The host application | Read-only data attached to context by the app, not the model |
| Prompt | The user (surfaced as e.g. a slash command) | Reusable, parameterized prompt template |
| Sampling | The server, of the client's LLM | Server requests LLM generation without holding its own model access |
| MRTR (`input_required`) | Either, mid-tool-call | Interactive, human-input-required flows over a stateless transport |

## Appendix C — Qdrant quantization decision table

| Method | Memory reduction | Recall impact | When to use |
|---|---|---|---|
| None (float32) | 1x | none | Small collections, max accuracy needed |
| Scalar (int8) | ~4x | small | Default first optimization for most collections |
| Product | moderate–high | moderate | Middle ground when scalar isn't enough but binary is too lossy |
| Binary + oversample/rescore | ~32x | small (with rescoring) | Very large collections where memory is the binding constraint |

## Appendix D — Glossary

- **Agent** — a loop around an LLM where model output can trigger actions whose results feed back as context.
- **ANN (Approximate Nearest Neighbor)** — search that trades a small amount of recall for large speed gains versus exact search; HNSW is Qdrant's ANN algorithm.
- **Checkpointer** — a LangGraph component that persists graph state after each step, enabling resumability and time travel.
- **Context engineering** — deliberate management of what occupies the model's context window at each step of a running agent.
- **Fusion (RRF)** — combining two ranked result lists (e.g., dense + sparse search) by rank position rather than raw score.
- **HNSW** — Hierarchical Navigable Small World graph, the ANN index structure underlying Qdrant.
- **Middleware (LangChain 1.0)** — composable hooks injected into the agent loop (before/after model or tool call).
- **MRTR** — Multi Round-Trip Requests, the MCP 2026-07-28 pattern for interactive, mid-call human input over a stateless transport.
- **Node (LlamaIndex)** — a chunk of a Document plus metadata and relationships to neighboring chunks.
- **Node/Edge (LangGraph)** — a step in a state graph / the transition between steps.
- **Reducer** — the function controlling how a state update merges into existing LangGraph state.
- **Sparse vector** — a high-dimensional, mostly-zero vector representing keyword/term-level signal (BM25, SPLADE, miniCOIL), complementary to dense embeddings.
- **Stateless MCP** — the 2026-07-28 architecture where every request carries its own protocol context, removing the need for session affinity.
- **Workflow step/event (LlamaIndex)** — a function consuming one event type and emitting another; steps wire together implicitly by type-matching.

## Appendix E — Primary references (go here before any secondary blog post)

- MCP specification: `https://modelcontextprotocol.io/specification/2026-07-28`
- LangChain / LangGraph docs: `https://docs.langchain.com`
- Qdrant documentation: `https://qdrant.tech/documentation/`
- LlamaIndex documentation and Workflows: `https://docs.llamaindex.ai`, `https://www.llamaindex.ai/workflows`
- CrewAI documentation: `https://docs.crewai.com`

## Appendix F — Expert-level self-assessment (Day 45)

Answer each without notes. Anything shaky is your next week's syllabus, not a footnote:

1. Explain why MCP separates Tools from Resources, and why the 2026-07-28 spec moved to a stateless architecture — what operational problem does it solve?
2. Given a new multi-step task, articulate in under a minute why you'd reach for LangGraph vs. LlamaIndex Workflows vs. CrewAI — with a reason tied to the task's actual shape, not framework familiarity.
3. Explain the recall/latency/memory tradeoff of `ef_search`, `m`, and quantization in Qdrant well enough to size an index for a stated dataset size and latency budget.
4. Design a human-in-the-loop gate that survives a process restart, and explain what specifically makes that possible (name the mechanism).
5. Given an agent that's hallucinating tool arguments, name the three most likely root causes in tool/schema design before touching the prompt.
6. Explain what a reranker or postprocessor buys you that raw vector similarity doesn't, with a concrete failure case it fixes.
7. You have a working capstone. Could you defend every architectural choice in it to a skeptical senior engineer, including the ones where you picked the "wrong-looking" tool for a good reason? If not, that gap — not new syllabus — is the highest-leverage thing to close next.

