# Part LIX — Glossary

## What this page is for, in plain terms

Two honest uses for this page: (1) you hit an unfamiliar term mid-sentence in some part of this curriculum and want a 10-second definition without losing your place, and (2) you're reading something *outside* this curriculum — a blog post, a paper, a vendor's docs — that assumes you already know a term this curriculum taught. Every entry deliberately includes a "common misconception" line, because the wrong-but-plausible-sounding version of a definition is usually more damaging than not knowing the term at all — it lets you confidently misuse it.

**A worked example of why the misconception line matters:** if you only remembered "idempotency = doing something twice has the same effect as once," you might assume *any* retry is automatically safe. The glossary entry's misconception line calls this out directly — retrying is only safe for operations that actually *are* idempotent, which most side-effecting tools (like sending an email) are not by default. That's the difference this page is trying to catch.

Format per term: definition, related concepts, example, common misconception.

**Agent** — An LLM-driven loop that perceives, decides, acts, and observes, with the model controlling the control flow. *Related*: agent loop, ReAct. *Example*: a ReAct agent answering a multi-hop question. *Misconception*: any LLM app with a function call is "an agent" — a single tool-calling turn without a loop is not.

**Agent harness** — The runtime infrastructure (state, memory, tools, policy, retries, tracing) surrounding an agent loop. *Related*: Part XXVIII. *Misconception*: frameworks like LangGraph *are* the harness — really, they implement pieces of it.

**Agentic RAG** — RAG where an agent decides whether/what/how many times to retrieve, rather than a fixed retrieve-then-generate pipeline. *Related*: Part XVII. *Misconception*: any RAG system using an agent framework is "agentic RAG" — the defining trait is retrieval being a *decision*, not a fixed step.

**ANN (Approximate Nearest Neighbor)** — Search that trades small recall loss for large speedup vs. brute-force k-NN. *Related*: HNSW, IVF. *Misconception*: ANN indexes always find the true nearest neighbor — they approximate.

**Attention** — A mechanism weighting each token's representation by relevance to every other token. *Related*: self-attention, multi-head attention. *Misconception*: attention "looks at" text the way a human reads — it's a learned weighted average, not literal focus.

**Blackboard architecture** — Agents read/write a shared workspace rather than passing messages directly. *Related*: multi-agent systems. *Misconception*: this avoids race conditions automatically — it doesn't; writes still need locking.

**Chain-of-thought (CoT)** — Prompting a model to produce intermediate reasoning before a final answer. *Related*: self-consistency, reflection. *Misconception*: longer CoT always improves accuracy — it can rationalize a wrong initial guess.

**Checkpointing** — Persisting agent/workflow state durably so execution can resume after a crash. *Related*: durable execution, idempotency. *Misconception*: checkpointing alone guarantees no duplicate side effects — it doesn't without idempotent step design.

**Circuit breaker** — A pattern that stops calling a repeatedly-failing dependency for a cooldown period. *Related*: reliability. *Misconception*: it's the same as a retry — retries attempt recovery per-call, circuit breakers stop attempting entirely for a window.

**Context engineering** — Deciding what content populates an LLM's context window, as distinct from prompt wording. *Related*: Part XXXII. *Misconception*: it's the same as prompt engineering — prompt engineering is about *phrasing*, context engineering is about *content selection*.

**Context window** — The maximum tokens (input+output) a model can process per request. *Related*: KV cache, lost-in-the-middle. *Misconception*: a bigger context window means you should always use more of it — recall degrades with irrelevant stuffing.

**Corrective RAG (CRAG)** — RAG that grades retrieval quality and falls back to an alternate source when retrieval is judged poor. *Related*: Part XVII. 

**Cosine similarity** — A similarity metric based on the angle between two vectors, ignoring magnitude. *Related*: embeddings, dot product. *Misconception*: cosine similarity and dot product always give the same ranking — only true for normalized vectors.

**DAG (Directed Acyclic Graph)** — A dependency structure with no cycles, used for orchestration. *Related*: topological sort. *Misconception*: any workflow diagram is a DAG — a cycle (e.g., a retry loop) makes it not acyclic.

**Dead-letter queue** — A store for tasks that exhausted retries, for later inspection/replay. *Related*: reliability.

**Embedding** — A dense/sparse vector representation of data such that similarity approximates semantic similarity. *Related*: cosine similarity, vector database. *Misconception*: embeddings from different models are comparable — they are not.

**Fine-tuning** — Updating model weights on task-specific data. *Related*: LoRA, RAG. *Misconception*: fine-tuning is the right tool for injecting fresh factual knowledge — RAG usually is.

**Function calling / tool calling** — A protocol letting a model request execution of a defined function with typed arguments. *Related*: JSON Schema, MCP.

**Groundedness / faithfulness** — Whether every claim in a generated answer is supported by the provided context. *Related*: RAG evaluation. *Misconception*: a grounded answer is automatically correct — the context itself could be wrong.

**HNSW (Hierarchical Navigable Small World)** — A graph-based ANN index. *Related*: vector database.

**Human-in-the-loop (HITL)** — Deliberate checkpoints requiring human approval/edit/rejection before an agent action proceeds. *Related*: Part XXXIV.

**Hybrid search** — Combining dense (semantic) and sparse (keyword) retrieval, fused via a method like RRF. *Related*: reciprocal rank fusion.

**Hallucination** — Confident but false model output, a consequence of probabilistic generation without built-in verification. *Related*: groundedness, RAG.

**Idempotency** — A property where repeating an operation has the same effect as performing it once. *Related*: retries, distributed systems. *Misconception*: retrying any tool call is safe — only safe for idempotent tools.

**Instruction hierarchy** — The priority ordering (system > user > tool/retrieved content) a well-trained model uses to resolve conflicting instructions. *Related*: prompt injection defense.

**KV cache** — Cached key/value attention states from prior tokens, avoiding recomputation during autoregressive decoding. *Related*: prefill/decode.

**LLM-as-judge** — Using an LLM call to score another output against a rubric. *Related*: evaluation. *Misconception*: an LLM judge is inherently reliable — it must be calibrated against human judgment.

**LoRA (Low-Rank Adaptation)** — A parameter-efficient fine-tuning method injecting small trainable matrices into a frozen base model. *Related*: QLoRA.

**Memory (agent)** — State deliberately retained beyond a single execution run. *Related*: Part X. *Misconception*: memory and state (Part IX) are the same thing — state is broader and includes purely ephemeral execution state.

**MCP (Model Context Protocol)** — A standardized protocol for a host application to discover and invoke tools/resources/prompts from a server. *Related*: Part XXIII.

**Multi-hop retrieval** — Retrieval requiring multiple chained steps, each informed by the previous, to answer a question spanning multiple facts/documents.

**Optimistic concurrency** — Detecting write conflicts via a version check rather than locking upfront. *Related*: state management.

**Plan-and-execute** — An architecture separating an upfront planning phase from step execution. *Related*: Part VIII, Part XXX.

**Prompt caching** — Reusing cached KV state for a stable prompt prefix across requests to reduce cost/latency.

**Prompt injection** — An attack where untrusted content instructs the model to deviate from its intended behavior. *Related*: indirect injection, instruction hierarchy.

**Quantization** — Reducing numeric precision of model weights to save memory/increase throughput.

**ReAct** — An agent architecture interleaving Thought, Action, and Observation steps. *Related*: Part VIII.

**Reranking** — Re-scoring a broad set of retrieved candidates with a more precise (often cross-encoder) model before final selection. *Related*: advanced RAG.

**RLHF (Reinforcement Learning from Human Feedback)** — Post-training aligning model outputs to human preference judgments. *Related*: DPO, instruction tuning.

**RoPE (Rotary Position Embedding)** — A positional encoding method rotating query/key vectors to encode relative position. *Related*: attention.

**Self-consistency** — Sampling multiple reasoning chains and taking a majority vote on the final answer. *Related*: chain-of-thought.

**Supervisor architecture** — A central agent delegating to and aggregating results from worker agents. *Related*: Part VIII.

**Tokenization** — Converting text into discrete integer token IDs via a vocabulary/merge algorithm (e.g., BPE). *Related*: Part III.

**Trajectory (agent)** — The ordered sequence of {state, decision, action, observation} across an agent run, used for evaluation/debugging. *Related*: Part XXXVIII.

**Vector database** — A storage/retrieval system for high-dimensional vectors with ANN search and metadata filtering. *Related*: Qdrant, HNSW.

*(This glossary grows with the curriculum — treat it as a living reference; add terms as new Parts introduce them.)*
