# Part LXIV — Research Paper Curriculum

## What this page is for, in plain terms

Every technique in this curriculum has a real origin — a paper where someone first proposed and tested the idea. You've been using these ideas (attention, ReAct, RAG, chain-of-thought) with working code the whole time; this page is for when you want to go one level deeper and read the actual source, either out of curiosity or because you want to push past what this curriculum covers into genuinely novel territory.

**How to read a paper you're pointed to here, concretely:** don't start at the introduction and read linearly — that's the slowest way in. Look at the figures first (they usually show the architecture visually, faster than prose), form your own guess at how it works, *then* read the method section specifically to check your guess against the real thing. For the ReAct paper, for instance, you already know the mechanism cold — you built it in [Part VIII §2](part08-agent-architectures.md). Reading the original paper after building it yourself is a very different, faster experience than reading it cold: you're confirming and refining an existing mental model, not building one from scratch under time pressure.

For each paper: problem, key idea, architecture, why it matters, what to understand, implementation exercise. Locate exact citations via arXiv/ACL Anthology/official proceedings rather than relying on titles alone here — verify author/year details from the primary source.

## Transformers and Attention
**"Attention Is All You Need"** (the original Transformer paper)
- **Problem**: RNN-based sequence models don't parallelize well and struggle with long-range dependencies.
- **Key idea**: Replace recurrence entirely with self-attention.
- **Architecture**: Encoder-decoder with multi-head self-attention, positional encoding, residual connections.
- **Why it matters**: Direct ancestor of every LLM in this curriculum.
- **What to understand**: Query/key/value mechanics, why scaling by √d_k, why multi-head.
- **Implementation exercise**: [Part II §8](part02-ml-foundations.md)'s from-scratch self-attention.

## Embeddings and Retrieval
**Foundational dense retrieval / embedding papers** (e.g., work establishing dual-encoder retrieval, and sentence-embedding methods)
- **Problem**: Sparse keyword matching misses semantic similarity.
- **Key idea**: Learn a shared embedding space where semantic similarity ≈ geometric proximity.
- **Why it matters**: Direct foundation of [Part XIV](part14-embeddings.md).
- **Implementation exercise**: [Part XIV §4](part14-embeddings.md)'s similarity comparison harness.

## RAG
**The original Retrieval-Augmented Generation paper**
- **Problem**: Parametric-only models can't incorporate fresh/proprietary knowledge and hallucinate under uncertainty.
- **Key idea**: Condition generation on retrieved passages at inference time.
- **Why it matters**: Direct foundation of [Part XI](part11-rag-fundamentals.md).
- **Implementation exercise**: [Part XI §4](part11-rag-fundamentals.md)'s end-to-end pipeline.

## Tool Use and Agents
**ReAct: Synergizing Reasoning and Acting in Language Models**
- **Problem**: Pure reasoning (CoT) can't interact with the world; pure acting has no deliberation.
- **Key idea**: Interleave reasoning traces with actions and observations.
- **Why it matters**: Direct foundation of [Part VIII §2](part08-agent-architectures.md).
- **Implementation exercise**: The ReAct build in Part VIII.

**Toolformer / tool-use training papers**
- **Problem**: Models don't natively know when/how to call external tools.
- **Key idea**: Train models to insert API calls into their own generation based on utility.
- **Why it matters**: Conceptual ancestor of modern native tool-calling APIs ([Part VI](part06-structured-outputs-tool-calling.md)).

## Reflection and Self-Correction
**Reflexion / self-refine-style papers**
- **Problem**: A single generation pass has no mechanism to catch its own errors.
- **Key idea**: Verbal reinforcement — the model critiques its own trajectory and retries with that critique as additional context.
- **Why it matters**: Foundation of [Part VIII](part08-agent-architectures.md)'s Reflection architecture and [Part XXXI](part31-reasoning.md)'s caution about unverified reflection.
- **Implementation exercise**: [Part XXVII §3](part27-multi-agent-systems.md)'s bounded review-cycle loop.

## Planning
**Planning-and-execution / hierarchical task decomposition papers in the LLM-agent context**
- **Problem**: Purely reactive agents struggle with tasks requiring foresight.
- **Key idea**: Separate an explicit planning phase from execution, optionally hierarchical.
- **Why it matters**: Foundation of [Part XXX](part30-planning.md).

## Multi-Agent Systems
**Multi-agent debate / collaboration papers**
- **Problem**: A single model's output may be miscalibrated or wrong with high confidence.
- **Key idea**: Multiple model instances critique/debate to converge on better answers.
- **Why it matters**: Foundation of the Debate architecture ([Part VIII §1](part08-agent-architectures.md), architecture #10) — also read critically per [Part VIII](part08-agent-architectures.md)'s noted weakness (confident-wrong consensus).

## Memory
**Long-term/episodic memory architecture papers for LLM agents**
- **Problem**: Fixed context windows can't hold a lifetime of interaction history.
- **Key idea**: External memory stores with retrieval, consolidation, and reflection over stored experience.
- **Why it matters**: Foundation of [Part X](part10-memory.md).

## Reasoning
**Chain-of-thought prompting paper**; **self-consistency paper**
- **Problem/Key idea**: See [Part V §7](part05-prompt-engineering.md) and [Part XXXI](part31-reasoning.md) for full treatment, including the critical caveat about when these techniques do and don't help.
- **Implementation exercise**: [Part V §12](part05-prompt-engineering.md)'s empirical CoT-vs-baseline test.

## Evaluation
**LLM-as-judge calibration and agent-benchmark papers**
- **Problem**: Automated evaluation of open-ended agent behavior is hard to trust without validation.
- **Key idea**: Calibrate automated judges against human judgment; design benchmarks resistant to gaming/saturation.
- **Why it matters**: Foundation of [Part XVIII](part18-rag-evaluation.md)/[Part XXXVIII](part38-agent-evaluation.md)'s calibration discipline.

## How to Read These Effectively
For each paper: read the abstract and figures first, form a hypothesis about the architecture, *then* read the method section to check yourself, then implement the smallest possible version of the key idea (as this curriculum's practical projects do throughout) before considering the paper "understood." Passive reading without an implementation exercise does not satisfy this curriculum's Rule 5 (every major concept needs a project).
