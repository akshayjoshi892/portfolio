# Part XXXII — Context Engineering

## Before you start

**What you need already:** [Part III Stage 6](part03-llm-fundamentals.md) (the context window, and "lost in the middle"), [Part X](part10-memory.md) (memory retrieval), [Part V](part05-prompt-engineering.md) (the untrusted-content delimiting pattern).

**What this part gives you:** A name and a discipline for a problem that's been quietly present since Part VII: as an agent runs longer, its conversation history, tool results, and retrieved memories all compete for a limited context window. This part is about deciding, deliberately, what goes in — rather than letting it grow unmanaged until something breaks.

**Time:** 2 days.

**When you finish, you'll be able to:** Build a context assembly system that stays within a token budget by prioritizing what matters, and correctly handles information that's gone stale during a long-running session.

---

## Stage 1 — Watch your own agent's context grow unmanaged

Go back to your Part VII agent loop. Every tool result gets appended to `state.messages`. Nothing ever removes anything. Run a 20-step research task and look at what's actually accumulating:

```python
# after a 20-step run, state.messages might contain:
[
    {"role": "user", "content": "research renewable energy storage"},
    {"role": "tool", "content": "search results: [huge block of text, 800 words]"},
    {"role": "tool", "content": "search results: [another huge block, 600 words]"},
    # ... 18 more tool results, most of them only marginally useful ...
]

total_tokens = sum(count_tokens(m["content"]) for m in state.messages)
print(total_tokens)   # could easily be 15,000+ after 20 steps
```

Every one of those 20 tool results gets sent, in full, on *every single subsequent call* — because `state.messages` is the entire conversation history, resent each time. This has three real costs, all compounding as the run gets longer: you pay for those tokens again and again (Part IV's cost tracking would show this clearly), each call gets slower (Part III's prefill cost scales with input length), and — the subtlest cost — quality can actually degrade, because "lost in the middle" means a fact from step 3 is less reliably recalled by step 20 than it would have been if the context were smaller and more focused.

**Check yourself before Stage 2:** Why does an agent's context cost, latency, and even quality all get worse as an unmanaged conversation history grows, even if every individual tool result was genuinely useful when it arrived?

---

## Stage 2 — Not everything deserves equal priority

The fix starts with an honest observation: not everything in an agent's available context is equally important. A system prompt is almost always essential. A tool result from 15 steps ago, about a sub-question that's already been resolved, usually isn't. Right now, your Part VII loop treats every message identically — append, resend, forever.

```python
from dataclasses import dataclass


@dataclass
class ContextItem:
    source: str        # "system" | "retrieval" | "memory" | "tool_result" | "history"
    content: str
    priority: int       # lower number = more important, evicted last
    token_estimate: int
```

Assigning a `priority` explicitly, per item, is the first real move: it turns "everything is equally in context" into "some things matter more than others, and I've said so, deliberately" — rather than leaving it to accident of insertion order.

**Check yourself before Stage 3:** For an agent mid-way through a research task, rank these by priority, from most to least important: the system prompt, the current question, a tool result from 2 steps ago, a tool result from 15 steps ago that led to a dead end.

---

## Stage 3 — A hard budget, enforced by eviction, not by hoping

Once items have priorities, you can enforce a real token budget: include as much as fits, in priority order, and drop what doesn't fit — deliberately, not by accident of whatever the provider happens to truncate.

```python
def assemble_context(items: list[ContextItem], token_budget: int) -> list[ContextItem]:
    items_by_priority = sorted(items, key=lambda i: i.priority)
    included: list[ContextItem] = []
    used = 0

    for item in items_by_priority:
        if used + item.token_estimate <= token_budget:
            included.append(item)
            used += item.token_estimate
        else:
            print(f"dropped (over budget): {item.source} priority={item.priority} "
                  f"({item.token_estimate} tokens)")

    return included
```

This is deliberately similar in shape to a resource-allocation problem you've seen before — greedily take the most valuable items first, stop when you run out of room. The key improvement over Stage 1's unmanaged growth: **when something has to be dropped, it's the lowest-priority item, chosen on purpose** — not whatever the API happens to silently truncate when you exceed the context window (recall Part III's warning about `finish_reason == "length"`), and not everything equally, which would degrade the whole context rather than sacrificing only the least valuable part.

**Check yourself before Stage 4:** Why is deliberately dropping the lowest-priority item strictly better than letting the provider silently truncate whatever happens to be at the end of an over-length request?

---

## Stage 4 — Position matters too, separately from priority

Here's a subtlety worth not missing: priority determines *what* gets included, but *where* it goes in the final prompt is a separate decision, because of "lost in the middle" from Part III. An item that's important enough to survive the budget cut should also be *placed* somewhere the model reliably attends to — near the start or the end, not buried in the middle of a long context.

```python
def render_context(items: list[ContextItem]) -> str:
    # Order for POSITION in the final prompt — a different concern from
    # the PRIORITY used to decide what to include in Stage 3.
    position_order = {"system": 0, "retrieval": 1, "memory": 1, "tool_result": 2, "history": 3}
    ordered = sorted(items, key=lambda i: position_order.get(i.source, 99))
    return "\n\n".join(f"[{i.source}]\n{i.content}" for i in ordered)
```

Notice `assemble_context` (Stage 3) and `render_context` (this stage) do two genuinely different jobs, run in sequence: first decide *what survives the budget*, by priority; then decide *where each surviving item sits* in the final assembled prompt, by position. Conflating these — using one ordering for both — either puts your most important content in a bad position, or your best-positioned content isn't actually your most important content.

**Check yourself before Stage 5:** Why are "what to include" and "where to place it" genuinely separate decisions, rather than one ordering serving both purposes?

---

## Stage 5 — Compression: fitting more meaning into fewer tokens

Sometimes the answer to a tight budget isn't dropping items — it's shrinking them. A 15-step-old tool result probably doesn't need to be included verbatim; a one-sentence summary of what it established often carries nearly all the useful signal at a fraction of the token cost.

```python
async def compress_history(messages: list[str], max_messages: int, client) -> list[str]:
    if len(messages) <= max_messages:
        return messages

    excess = messages[: len(messages) - max_messages]
    recent = messages[len(messages) - max_messages :]

    summary_prompt = f"""Summarize the key facts and decisions from this
conversation history in 2-3 sentences, preserving anything a later step
might still need to reference:

{chr(10).join(excess)}
"""
    summary, _ = await client.ask(summary_prompt)

    return [f"[summary of earlier conversation]: {summary}"] + recent
```

This is precisely the same compaction discipline you designed for state in [Part IX](part09-agent-state.md) and applied to memory in [Part X](part10-memory.md) — old detail gets consolidated into a compact summary rather than either being kept in full forever or discarded entirely. The trade-off is honest and real: you're trading some fidelity (a summary loses detail a full transcript wouldn't) for a large reduction in token cost. Whether that trade is worth it for your specific agent is measurable, not assumed — you can test whether a held-out question about early-conversation content still gets answered correctly after compaction, exactly as you'd evaluate any other lossy transformation in this curriculum.

**Check yourself before the project:** What's the trade-off compression makes, and how would you actually test whether it's an acceptable one for a given agent?

---

## Stage 6 — Context is also where injected instructions hide

One more thing worth connecting explicitly, because it's easy to think of context engineering purely as a cost/quality problem and forget it's also a security boundary. Every tool result and every retrieved document that enters an agent's context is untrusted input, per [Part V Stage 6](part05-prompt-engineering.md) — and a long-running agent, accumulating many tool results over many steps, has many more opportunities for one of them to carry an injected instruction than a single-turn RAG query does.

```python
def flag_untrusted_content(item: ContextItem) -> ContextItem:
    if item.source in {"tool_result", "retrieval"}:
        item.content = f"<untrusted_data source={item.source!r}>\n{item.content}\n</untrusted_data>"
    return item
```

Apply this to every tool result and retrieved item as part of assembling context — not as an afterthought, but as a standard step in the same pipeline that handles priority and budgeting. This foreshadows [Part XXXV](part35-security.md)'s fuller treatment, but the habit belongs here, in the context assembly code itself, not bolted on separately later.

**Check yourself before the project:** Why does a long-running agent accumulate more injection risk over its lifetime than a single-turn RAG query does?

---

# The Project: A Context Manager for a Long-Running Agent

## What you're building

A context assembly pipeline — priority-based eviction, position-aware rendering, history compaction, and untrusted-content wrapping — wired into a real long-running agent, tested against a 30-turn conversation.

```bash
$ python context_managed_agent.py --conversation-file long_session.json
```

```
turn 25: context budget 4000 tokens
  included: system(120t), retrieval(340t), memory(180t), recent_history(2100t)
  dropped: 3 old tool results (would have added 1400t, over budget)
  compacted: turns 1-15 summarized into 90t (was 2800t)

answer: [correctly references a fact established in turn 3, despite
turns 1-15 having been compacted into a summary]
```

## Why this project

This is the mechanism that keeps a genuinely long-running agent (the kind you'll build in later parts, especially research agents in [Part XLVI](part46-research-agents.md)) from either blowing its context budget or degrading in quality as it accumulates history.

## Build it in stages

---

### Stage A — Priority-based budget enforcement

**Goal:** `assemble_context` from Stage 3, wired into a real agent, with a real token budget.

**Do:**
1. Assign priorities to your agent's context sources (system prompt, current question, retrieved memory, tool results)
2. Implement the budget-enforcing assembly
3. Run a task that generates enough context to exceed a deliberately small test budget, confirm the right items get dropped and it's logged clearly

**You're done when:** You can force an over-budget scenario and watch the correct, lowest-priority items get dropped rather than a provider-side silent truncation.

---

### Stage B — Position-aware rendering

**Goal:** `render_context`, separately ordering what survived Stage A's cut.

**Do:**
1. Implement it as shown
2. Confirm system/retrieval content lands near the start of the final prompt, not scattered based on insertion order

**You're done when:** You can inspect the final assembled prompt and confirm its structure matches your intended positioning, independent of your priority ordering.

---

### Stage C — History compaction, measured

**Goal:** `compress_history`, with an actual accuracy check before/after.

**Do:**
1. Build a 30-turn synthetic conversation with one specific fact established early (turn 3) that a later question will require
2. Run your agent without compaction, confirm it correctly answers a question about that early fact
3. Trigger compaction (set `max_messages` low enough to force it), confirm whether the answer is still correct afterward

**You're done when:** You have a real, measured answer to "does compaction preserve the specific facts my agent needs," not an assumption that it does.

---

### Stage D — Untrusted content wrapping, applied uniformly

**Goal:** Every tool result and retrieved item gets wrapped, automatically, as part of context assembly — not something you have to remember to add per-tool.

**Do:**
1. Apply `flag_untrusted_content` inside your assembly pipeline, to every item from `"tool_result"` or `"retrieval"` sources
2. Confirm system-prompt and current-question content is *not* wrapped (it shouldn't be — it's not untrusted)
3. Run an injection test similar to Part V's, now against a multi-turn agent with several tool results, confirming the wrapping holds across the whole pipeline

**You're done when:** Every untrusted-source item in your final assembled context is wrapped, automatically, by the pipeline itself.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Budget enforcement drops the system prompt before anything else | Priority not actually set lower (more important) for system content | Explicitly assign the system prompt the lowest priority number (most important) |
| Compaction loses a fact the agent needs later | Summary prompt too aggressive, compressing away specific details | Tighten the summarization prompt to explicitly preserve concrete facts, not just general gist |
| Untrusted wrapping applied to the system prompt or current question | Wrapping logic not correctly scoped to only `tool_result`/`retrieval` sources | Double check the source-type filter in `flag_untrusted_content` |

## Going further (optional)

1. **Dynamic priority.** Instead of fixed priorities, have older tool results decay in priority over time (reusing the decay idea from Part X's memory), so recency naturally influences what survives a budget cut, not just source type.
2. **A context budget dashboard** printing, for every turn, exactly what's included, what's dropped, and what's compacted — useful for debugging why an agent's answer seems to be missing information it should have.
3. **Compare compaction thresholds.** Test `max_messages` at several different values on your 30-turn conversation and find the point where the early-fact-recall test starts failing — that's a real, measured limit for your specific agent and task.

---

## Checkpoint

**Knowledge:**
1. Why does an unmanaged, ever-growing context hurt cost, latency, *and* quality, not just cost?
2. Why are "what to include" (priority) and "where to place it" (position) genuinely separate decisions?
3. Why does a long-running agent accumulate more prompt-injection risk over time than a single-turn query?

**Implementation:** Build `assemble_context` with priority-based eviction and position-aware rendering from memory.

**Debugging:** An agent gives an answer based on information that was true early in a long session but has since changed. Is this a context-engineering bug (stale content still in the budget) or a memory bug (Part X)? How would you tell the difference?

**Architecture:** Design a context budget allocation (what fraction of the window goes to system/retrieval/memory/history) for an agent handling both quick single-turn questions and week-long research tasks within the same system.

---

**Next:** [Part XXXIII — Tool Engineering](part33-tool-engineering.md), where you'll formalize the idempotency and risk-classification concepts that have been referenced since Part VI, into a real contract every tool must declare before it's trusted with real side effects.
