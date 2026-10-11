# Part LXVII — Framework Abstraction Mapping

## What this page is for, in plain terms

This is the page you open when a framework does something you don't understand, and you need to translate it back into terms you actually trust — the mechanisms you built by hand in Parts VI–XXVIII. Every row below is a single concept traced through four columns: the raw idea, then how LangChain expresses it, then LangGraph, then CrewAI — so you can look up any framework's behavior and immediately map it to code you've already written and debugged yourself.

**A concrete example of using this under real pressure:** a LangGraph agent in production is behaving strangely — messages seem to be getting overwritten instead of accumulating. You don't remember LangGraph's exact reducer syntax off the top of your head. Look up the "State" row below: it tells you LangGraph's state uses "a typed StateGraph schema with reducers," which immediately reminds you this is the same idea as your own hand-rolled state-merging logic, just requiring an explicit `Annotated[list, add]` declaration per field — and that's exactly the kind of thing that's easy to forget on one specific field. You now know precisely what to check, in under a minute, instead of guessing at unfamiliar framework internals.

For each concept: underlying idea → raw implementation (this curriculum) → LangChain → LangGraph → CrewAI.

## Tools

```text
Concept: model requests execution of a defined capability
  v
Raw: JSON-Schema-validated function dispatch (Part VI)
  v
LangChain: @tool decorator + model.bind_tools([...])
  v
LangGraph: same @tool objects, invoked from within a node function
  v
CrewAI: @tool decorator, attached to an Agent's tools list
```

## State

```text
Concept: information persisting across execution steps
  v
Raw: dataclass + atomic-write checkpoint store (Part IX)
  v
LangChain: no first-class state abstraction — state is whatever you thread through chain inputs/outputs manually
  v
LangGraph: typed StateGraph schema with reducers, automatic checkpointing
  v
CrewAI: implicit, via Task `context=[...]` dependencies + optional Crew memory
```

## Memory

```text
Concept: state deliberately retained beyond one run
  v
Raw: embedding-indexed store with decay/consolidation (Part X)
  v
LangChain: memory classes wrapping a vector store or message history
  v
LangGraph: persisted state via a checkpointer keyed by thread_id, or an external memory store called from a node
  v
CrewAI: built-in short-term/entity/long-term memory flags on Crew(memory=True)
```

## Workflows / Orchestration

```text
Concept: control flow across multiple steps/agents
  v
Raw: DAG executor with topological sort + cycle detection (Part XXVI)
  v
LangChain: Runnable composition (|, RunnableParallel) — linear/parallel only, no cycles
  v
LangGraph: explicit graph with conditional edges, supports cycles natively
  v
CrewAI: Process.sequential (task context order) or Process.hierarchical (manager delegation)
```

## Agents

```text
Concept: LLM-controlled loop choosing actions dynamically
  v
Raw: while-loop with decide/act/observe and bounded termination (Part VII)
  v
LangChain: AgentExecutor wrapping a tool-calling loop
  v
LangGraph: any node (or subgraph) implementing the loop, with the graph controlling transitions between agents
  v
CrewAI: Agent objects executing Tasks, optionally with allow_delegation
```

## Multi-Agent

```text
Concept: multiple specialized agents coordinating
  v
Raw: message-passing/blackboard with explicit locking (Part XXVII)
  v
LangChain: manual composition of multiple AgentExecutors — no first-class primitive
  v
LangGraph: subgraphs as nodes within a parent graph, sharing/passing typed state
  v
CrewAI: Crew(agents=[...], tasks=[...]) — the framework's primary use case
```

## Persistence

```text
Concept: durable, resumable execution across process restarts
  v
Raw: atomic-write, versioned checkpoint store (Part IX)
  v
LangChain: no built-in mechanism — must be added manually
  v
LangGraph: first-class checkpointers (SQLite/Postgres/Redis-backed)
  v
CrewAI: limited — memory persistence exists, but not full execution-state checkpointing at LangGraph's granularity
```

## Human-in-the-Loop

```text
Concept: pause execution for human approval/edit
  v
Raw: durable PendingAction store with edit-aware resume (Part XXXIV)
  v
LangChain: manual — no first-class interrupt mechanism
  v
LangGraph: interrupt_before / interrupt_after with checkpointer-backed resume
  v
CrewAI: guardrail functions that can reject/request revision, less native "pause and wait for a human" semantics
```

## Multi-Agent Evaluation

```text
Concept: measuring trajectory/outcome quality across a run
  v
Raw: trajectory + tool-argument-accuracy + calibrated LLM-judge harness (Part XXXVIII)
  v
LangChain: LangSmith evaluation tooling (separate product, integrates via callbacks/tracing)
  v
LangGraph: same LangSmith integration, plus time-travel debugging via checkpoints
  v
CrewAI: less built-in evaluation tooling — typically paired with an external harness like the one in Part XXXVIII
```

## Observability

```text
Concept: logs/traces/metrics for understanding runtime behavior
  v
Raw: nested-span Tracer with causal trace trees (Part XXXIX)
  v
LangChain: BaseCallbackHandler hooks, LangSmith integration
  v
LangGraph: same callback system, plus per-checkpoint state inspection
  v
CrewAI: verbose logging; less granular step-level span tracing out of the box
```

## Reading This Table
No row implies one framework is strictly "better" — each trades explicitness/control for authoring speed differently (see [Part XXII](part22-langchain-vs-langgraph-vs-crewai.md)'s full comparison). The purpose of this mapping is diagnostic: when a framework behaves unexpectedly, trace its abstraction back to the "Raw" row and reason about the underlying mechanism directly, rather than treating the framework as a black box.
