# Part LVIII — Architecture Pattern Library

## What this page is for, in plain terms

This is a deck of index cards, one per architecture, meant to be read *during design*, not during learning. You already learned each of these deeply — with working code — in Part VIII or Part XXVI. This page exists for the moment weeks or months later when you're starting a new project and thinking "what shape does this task actually need?" — a quick lookup, not a re-teaching.

**How to use it, concretely:** say you're designing a system where three specialist bots (billing, technical, returns) need to jointly draft a response to one complicated ticket, but none of their tasks depend on the others finishing first. Scan the cards below for "independent" and "concurrent" — you land on Parallel Fan-Out/Fan-In. Its "Suitable" line confirms this is the right shape, and its "Limitations" line ("merge/reconciliation logic complexity") immediately warns you about the one hard part you'll need to design carefully: how do you combine three drafts into one coherent reply? That's the kind of five-second lookup this page is built for.

A reusable catalog. Most patterns were built in depth earlier ([Part VIII](part08-agent-architectures.md), [Part XXVI](part26-agent-orchestration.md)); this is the reference-card version for quick lookup during design.

## Router
- **Problem**: Dispatch heterogeneous requests to the right specialist handler.
- **State model**: Stateless per-request classification.
- **Implementation**: Classifier (cheap model or rules) → dict dispatch.
- **Benefits**: Simple, low latency, easy to extend with new routes.
- **Limitations**: Misclassification routes to the wrong handler silently.
- **Failure modes**: Ambiguous inputs; classifier drift over time.
- **Suitable**: Clear-cut intent categories. **Unsuitable**: Overlapping/ambiguous intents needing nuanced judgment.

## Planner-Executor
- **Problem**: Multi-step tasks with a knowable structure.
- **State model**: A `Plan` object with step statuses ([Part XXX](part30-planning.md)).
- **Implementation**: Generate plan → execute steps respecting dependencies → replan on failure.
- **Benefits**: Predictable, debuggable step sequence; parallelizable independent steps.
- **Limitations**: Plan can go stale if execution reveals wrong assumptions.
- **Failure modes**: No replanning bound → infinite fallback generation.
- **Suitable**: Well-understood multi-step tasks. **Unsuitable**: Highly uncertain, reactive environments.

## Supervisor-Worker
- **Problem**: Coordinate heterogeneous specialist agents.
- **State model**: Supervisor holds delegation state; workers are often stateless per-task.
- **Implementation**: Central decision function dispatches to workers, aggregates results ([Part VIII §4](part08-agent-architectures.md)).
- **Benefits**: Clear central control point, easy to audit.
- **Limitations**: Serialization bottleneck if workers could run in parallel.
- **Failure modes**: Supervisor single point of failure.
- **Suitable**: Sequential/dependent specialist tasks. **Unsuitable**: Fully independent parallel subtasks (use Parallel Fan-Out instead).

## Sequential Pipeline
- **Problem**: Well-understood linear processing stages.
- **State model**: Output of stage N is input to stage N+1.
- **Implementation**: Function composition / `Runnable` piping ([Part XIX](part19-langchain.md)).
- **Benefits**: Simple, predictable, easy to test stage-by-stage.
- **Limitations**: No adaptability if one stage's output is poor.
- **Suitable**: Linear, well-understood transformations. **Unsuitable**: Tasks needing conditional branching.

## Parallel Fan-Out/Fan-In
- **Problem**: Independent subtasks that can run concurrently.
- **State model**: Ephemeral per-branch state, merged at the end.
- **Implementation**: `asyncio.gather` + merge function ([Part XXVI §3](part26-agent-orchestration.md)).
- **Benefits**: Minimizes wall-clock latency for independent work.
- **Limitations**: Merge/reconciliation logic complexity; slowest branch bounds total latency without a timeout.
- **Suitable**: Independent research/analysis subtasks. **Unsuitable**: Tasks with real dependencies between branches.

## Reflection
- **Problem**: Output quality is the bottleneck.
- **State model**: Draft + critique history.
- **Implementation**: Generate → critique → revise, bounded iterations ([Part VIII §1](part08-agent-architectures.md) architecture #7).
- **Benefits**: Iterative quality improvement.
- **Limitations**: Can rationalize a wrong initial answer without external verification ([Part XXXI §5](part31-reasoning.md)).
- **Suitable**: Writing/quality-sensitive generation. **Unsuitable**: Tasks with a cheap deterministic verifier (use tool-assisted verification instead).

## Critic / Evaluator
- **Problem**: Quality gating before output is finalized.
- **State model**: Candidate output + score.
- **Implementation**: Separate model/prompt scoring the primary agent's output.
- **Benefits**: Decouples generation from evaluation logic.
- **Limitations**: Critic itself can be miscalibrated ([Part XXXVIII §5](part38-agent-evaluation.md)).
- **Suitable**: Any pipeline needing a quality gate before a costly downstream action.

## Human Approval
- **Problem**: High-risk actions need a human checkpoint.
- **State model**: `PendingAction` durably persisted ([Part XXXIV §3](part34-human-in-the-loop.md)).
- **Implementation**: Interrupt-and-resume with edit-aware execution.
- **Benefits**: Hard safety boundary for irreversible/high-risk actions.
- **Limitations**: Approval fatigue if over-applied.
- **Suitable**: High-risk, side-effecting actions. **Unsuitable**: High-volume, low-risk actions (gate by risk classification, not blanket).

## Hierarchical Agents
- **Problem**: Large task decomposition needing multiple coordination levels.
- **State model**: Nested supervisor state trees.
- **Implementation**: Supervisors of supervisors.
- **Benefits**: Scales decomposition depth.
- **Limitations**: Coordination overhead and latency stacking; hardest to debug.
- **Suitable**: Enterprise-scale multi-domain orchestration. **Unsuitable**: Anything a flat Router/Supervisor could handle — avoid needless hierarchy.

## State Machine
- **Problem**: Predictable, auditable workflows with explicit states.
- **State model**: Enum-typed state + explicit transition table ([Part IX](part09-agent-state.md)).
- **Implementation**: Transition function validated against allowed edges.
- **Benefits**: Highly auditable, compliance-friendly.
- **Limitations**: Rigid — hard to handle unanticipated states.
- **Suitable**: Regulated/compliance-heavy workflows. **Unsuitable**: Open-ended exploratory tasks.

## DAG
- **Problem**: Arbitrary dependency structure among steps.
- **State model**: Per-node completion status ([Part XXVI §4](part26-agent-orchestration.md)).
- **Implementation**: Topological execution with cycle detection.
- **Benefits**: Maximizes safe parallelism given real dependencies.
- **Limitations**: Requires upfront or dynamic graph construction correctness.
- **Suitable**: Complex, partially-parallel workflows.

## Event-Driven
- **Problem**: Reactive, externally-triggered, long-running systems.
- **State model**: Event bus + per-handler state ([Part L §3](part50-advanced-orchestration.md)).
- **Implementation**: Pub/sub with async handlers.
- **Benefits**: Decoupled producers/consumers, scales to many triggers.
- **Limitations**: Harder to reason about ordering/race conditions.
- **Suitable**: Long-running, externally-triggered work. **Unsuitable**: Single bounded request/response flows.

## Retrieval-Agent
- **Problem**: Grounding agent decisions/answers in external knowledge.
- **State model**: Query → retrieved context, ephemeral per-turn.
- **Implementation**: RAG pipeline ([Part XI](part11-rag-fundamentals.md)) invoked as a tool or a fixed pre-step.
- **Suitable**: Any task needing up-to-date/proprietary knowledge grounding.

## Tool-Agent
- **Problem**: Taking real-world action via external capabilities.
- **Implementation**: Tool-calling loop with contracts ([Part VI](part06-structured-outputs-tool-calling.md)/[XXXIII](part33-tool-engineering.md)).
- **Suitable**: Any task requiring live data or side effects.

## MCP-Agent
- **Problem**: Dynamic, standardized tool/resource discovery across many providers.
- **Implementation**: Multi-server client registry ([Part XXV](part25-agent-plus-mcp.md)).
- **Suitable**: Systems needing to integrate many external tool providers without bespoke glue code.

## Hybrid Workflow-Agent
- **Problem**: Some steps need deterministic guarantees, others need model judgment.
- **Implementation**: Fixed pipeline stages with an agentic sub-step embedded at a specific, bounded point ([Part VIII §1](part08-agent-architectures.md) architecture #16).
- **Suitable**: Production pipelines needing some autonomy without full model-driven control flow risk.

## Distributed Agent
- **Problem**: Scaling agent execution across many workers/machines.
- **Implementation**: Task queue + idempotent workers + durable state ([Part XLIX](part49-distributed-agents.md)).
- **Suitable**: High-volume production systems needing horizontal scale and fault tolerance.
