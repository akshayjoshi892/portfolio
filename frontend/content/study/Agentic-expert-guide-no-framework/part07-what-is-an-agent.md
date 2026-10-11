# Part VII — What Is an Agent?

## Before you start

**What you need already:** [Part VI](part06-structured-outputs-tool-calling.md) — tool calling, the registry, validation and error recovery. Everything here builds directly on it.

**What this part gives you:** A precise definition of "agent" — precise enough that you can look at any system and say definitively whether it is one — and the actual loop that makes something an agent instead of a fancy function call.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Build a bounded, self-directing loop where the model decides what to do next, tell the difference between an agent and a workflow that merely uses an LLM, and explain exactly why every agent needs a hard stop.

---

## Stage 1 — The confusing part: everyone means something different by "agent"

Before building anything, deal with a real problem: the word "agent" gets used for at least four different things in this industry, and people argue past each other because they're not using the same definition.

| What people call "an agent" | What it actually is |
|---|---|
| A chatbot that answers questions | Not an agent by this curriculum's definition — no tools, no loop, just one response |
| A fixed pipeline: "summarize, then translate, then email" | Not an agent — you wrote the sequence, the LLM just fills in content at each fixed step |
| A single LLM call that can use one tool | Borderline — usually not an agent unless it can *decide* to call the tool again, or call a different one, based on what it learns |
| A loop where the model decides what to do, uses tools, observes results, and decides again | **This is what this curriculum means by "agent."** |

The distinguishing line, stated as sharply as possible:

> **In a workflow, you wrote the control flow, and the LLM fills in content at fixed points. In an agent, the LLM *is* the control flow — it decides which step happens next, using what, and when to stop.**

This isn't pedantry. It matters because the two things have completely different failure modes, completely different testing strategies, and completely different cost profiles. Confusing them leads to building something rigid when you needed flexibility, or something unpredictable and expensive when you needed a fixed pipeline.

**Check yourself before Stage 2:** Is a system that always does "step 1: search, step 2: summarize the results, step 3: email" — in that exact fixed order every time — an agent? Why or why not?

---

## Stage 2 — The loop, stated plainly

Every agent, no matter how sophisticated, runs this cycle:

```
     ┌─────────────────────────────────────┐
     │                                     │
     ▼                                     │
  OBSERVE           ← what's the current state?
     │                (the question, prior results, any new info)
     ▼
  DECIDE            ← the model looks at the state and picks:
     │                a tool to call, or a final answer to give
     ▼
  ACT               ← if it picked a tool: your code runs it
     │                (this is exactly Part VI's ToolRegistry.call)
     ▼
  UPDATE STATE       ← record what happened, so the next OBSERVE sees it
     │
     └──── not done? go back to OBSERVE
     │
  done → return the final answer
```

You already built the ACT step in [Part VI](part06-structured-outputs-tool-calling.md). This part builds the loop *around* it.

**Check yourself before Stage 3:** Name the four steps of the loop in order. Which one did you already build in Part VI?

---

## Stage 3 — Build the smallest possible version

Start with a fake decision-maker, so the loop's *mechanics* are visible before a real model adds unpredictability.

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AgentState:
    goal: str
    messages: list[dict] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    final_answer: str | None = None


def fake_decide(state: AgentState) -> dict:
    """Stands in for a real model call. Returns a tool call or a final answer."""
    if state.step_count == 0:
        return {"type": "tool_call", "name": "search", "args": {"query": state.goal}}
    return {"type": "final", "content": f"Based on my search, here's the answer to: {state.goal}"}


def fake_search_tool(args: dict) -> str:
    return f"3 relevant results found for {args['query']!r}"


def run_agent(goal: str, max_steps: int = 6) -> AgentState:
    state = AgentState(goal=goal)
    state.messages.append({"role": "user", "content": goal})

    while not state.done and state.step_count < max_steps:
        decision = fake_decide(state)

        if decision["type"] == "tool_call":
            result = fake_search_tool(decision["args"])
            state.messages.append({"role": "assistant", "tool_call": decision})
            state.messages.append({"role": "tool", "content": result})
        elif decision["type"] == "final":
            state.final_answer = decision["content"]
            state.done = True

        state.step_count += 1

    if not state.done:
        state.final_answer = "STOPPED: reached max_steps without finishing"
        state.done = True

    return state


result = run_agent("What is the capital of France?")
for msg in result.messages:
    print(msg)
print("FINAL:", result.final_answer)
print("steps used:", result.step_count)
```

```
{'role': 'user', 'content': 'What is the capital of France?'}
{'role': 'assistant', 'tool_call': {'type': 'tool_call', 'name': 'search', ...}}
{'role': 'tool', 'content': "3 relevant results found for 'What is the capital of France?'"}
FINAL: Based on my search, here's the answer to: What is the capital of France?
steps used: 2
```

Every piece maps directly to Stage 2's diagram: `state.messages` is OBSERVE's input, `fake_decide` is DECIDE, `fake_search_tool` is ACT, and appending to `state.messages` is UPDATE STATE. The `while` loop is the cycle.

**Check yourself before Stage 4:** In `run_agent`, which line corresponds to OBSERVE? Which to ACT?

---

## Stage 4 — The line that matters most in the whole function

Look again at this part of `run_agent`:

```python
while not state.done and state.step_count < max_steps:
```

Delete `state.step_count < max_steps` in your head. What happens if `fake_decide` has a bug and never returns `{"type": "final", ...}`?

The loop runs forever. Not crashes — *runs forever*, calling tools, burning API calls and money, with nothing to stop it. This isn't a hypothetical: it's the single most common serious bug in agent systems, and it's caused real financial damage to real companies who shipped an agent without this exact check.

### Why "trust the model to know when to stop" isn't enough

You might think: "the model will just say it's done when it's done." Sometimes it will. But models are not perfectly reliable at this, especially when:

- A tool keeps returning ambiguous or unhelpful results, and the model keeps trying variations
- The task genuinely doesn't have a clean stopping point
- There's a subtle bug in how you're parsing the model's "I'm done" signal, so your code never actually sees it

None of these are exotic edge cases. They happen in normal operation. `max_steps` is not a workaround for a bad model — it's a required safety boundary for *any* system where a piece of it can behave unpredictably, exactly like the timeout you added to tool calls in [Part VI Stage 6](part06-structured-outputs-tool-calling.md).

### The second boundary: cost, not just steps

Steps aren't the only thing that can run away. A single step can be expensive too — a huge prompt, a verbose response. Bound cost directly, not just step count:

```python
@dataclass
class AgentState:
    goal: str
    messages: list[dict] = field(default_factory=list)
    step_count: int = 0
    tokens_used: int = 0
    max_steps: int = 6
    max_tokens: int = 5000
    done: bool = False
    final_answer: str | None = None
    stop_reason: str | None = None


def run_agent(goal: str) -> AgentState:
    state = AgentState(goal=goal)
    state.messages.append({"role": "user", "content": goal})

    while not state.done:
        if state.step_count >= state.max_steps:
            state.stop_reason = "max_steps reached"
            break
        if state.tokens_used >= state.max_tokens:
            state.stop_reason = "max_tokens reached"
            break

        decision = fake_decide(state)
        state.tokens_used += 150  # a real call would report actual usage

        if decision["type"] == "tool_call":
            result = fake_search_tool(decision["args"])
            state.messages.append({"role": "tool", "content": result})
        elif decision["type"] == "final":
            state.final_answer = decision["content"]
            state.done = True

        state.step_count += 1

    if not state.done:
        state.final_answer = f"STOPPED: {state.stop_reason}"
        state.done = True

    return state
```

Notice `stop_reason` is now tracked explicitly and separately from `final_answer`. This matters for the same reason your tool errors in Part VI came back as structured data instead of a crash: when something stops the agent early, you want to know *why*, not just that it stopped. That distinction — "finished normally" vs. "hit a budget limit" — becomes essential once you're debugging real production runs in [Part XXXIX](part39-observability.md).

**Try breaking it:** Remove the `max_steps` check entirely and make `fake_decide` always return a tool call, never a final answer. Run it (with a `print` inside the loop so you can watch, and be ready to hit Ctrl+C). Watch it genuinely never stop. This is worth doing once, deliberately, so the danger is concrete rather than theoretical.

**Check yourself before Stage 5:** Why isn't "the model will say when it's done" a sufficient stopping condition on its own? What's the difference between `max_steps` and `max_tokens` as limits — why do you need both?

---

## Stage 5 — Connecting to a real model and real tools

Time to replace `fake_decide` with your actual `LLMClient` and `ToolRegistry` from Parts IV and VI.

```python
import asyncio
from llm_client import LLMClient   # from Part IV's project
from tool_registry import ToolRegistry   # from Part VI's project


async def run_real_agent(client: LLMClient, registry: ToolRegistry, goal: str) -> AgentState:
    state = AgentState(goal=goal)
    state.messages.append({"role": "user", "content": goal})

    while not state.done:
        if state.step_count >= state.max_steps:
            state.stop_reason = "max_steps reached"
            break

        # Send the conversation so far, plus available tool schemas, to the model.
        # Your provider's API returns either a tool_call or a plain text final answer.
        response = await client.ask_with_tools(state.messages, registry.available_schemas())

        if response.tool_call:
            outcome = await registry.call(response.tool_call.name, response.tool_call.arguments)
            state.messages.append({"role": "assistant", "tool_call": response.tool_call})
            state.messages.append({"role": "tool", "content": outcome})
        else:
            state.final_answer = response.text
            state.done = True

        state.step_count += 1

    if not state.done:
        state.final_answer = f"STOPPED: {state.stop_reason}"
        state.done = True

    return state
```

`client.ask_with_tools` isn't something you built yet — it's an extension of the `LLMClient` from Part IV that adds tool schemas to the request and parses either a tool call or plain text back out. Building it is the first stage of this part's project.

**Check yourself before the project:** What changed between the fake version and this one? What stayed exactly the same?

---

# The Project: A Real, Bounded, Tool-Using Agent

## What you're building

A command-line agent that answers questions using real tools, with a hard step and cost limit, that never hangs and never runs away.

```bash
$ python agent.py "What's the population of the capital of Japan, doubled?"
```

```
[step 1] deciding...
[step 1] tool_call: search(query="capital of Japan")
[step 1] tool result: "The capital of Japan is Tokyo."

[step 2] deciding...
[step 2] tool_call: search(query="population of Tokyo")
[step 2] tool result: "Tokyo's population is approximately 14 million."

[step 3] deciding...
[step 3] tool_call: calculator(expression="14000000 * 2")
[step 3] tool result: 28000000

[step 4] deciding...
[step 4] final answer.

ANSWER: Tokyo's population, doubled, is approximately 28 million.

Used 4/8 steps, ~620 tokens, $0.0009
```

## Why this project

This is the actual shape of every agent for the rest of this curriculum — [Part VIII](part08-agent-architectures.md)'s variations, [Part XXVIII](part28-agent-harness.md)'s production harness, all of it extend *this* loop. Get the bounding and the state tracking right here, and you won't have to relearn it under pressure later.

## Before you start

- Your working `LLMClient` from Part IV
- Your working `ToolRegistry` with at least 2 tools from Part VI
- Check your provider's docs for how tool-calling responses are actually shaped — you'll need to extend `LLMClient` to send tool schemas and parse a tool-call-or-text response

## Build it in stages

---

### Stage A — Extend the client for tool calling

**Goal:** `LLMClient` can send tool schemas and correctly tell you whether the model wants to call a tool or is giving a final answer.

**Do:**
1. Add a method (`ask_with_tools` or similar) that includes tool schemas in the request
2. Parse the response: does it contain a tool call, or plain text?
3. Return something that clearly distinguishes the two cases — don't make calling code guess by checking if a field is `None`

**You're done when:** You can send a question with a calculator tool available, and reliably tell in code whether the model chose to call it or answered directly.

---

### Stage B — The loop, for real, with a low step limit

**Goal:** The full loop from Stage 5, running against real tools, with `max_steps=3` so you can watch it hit the limit easily during testing.

**Do:**
1. Implement `run_real_agent` as in Stage 5
2. Ask a question needing 2 tool calls — confirm it finishes within 3 steps
3. Ask a deliberately hard/ambiguous question likely to need more steps than you've allowed — confirm it stops cleanly with `stop_reason = "max_steps reached"`, not a crash or a hang

**You're done when:** Both cases work, and the second one gives you a clear, honest "I ran out of steps" message instead of pretending to succeed.

---

### Stage C — Track cost and steps together

**Goal:** Both limits from Stage 4 enforced, with real numbers, not estimates.

**Do:**
1. Pull real token usage from each model response (your provider reports this — check the response format)
2. Accumulate it in `state.tokens_used`
3. Set a `max_tokens` limit and confirm the agent stops if it's exceeded, even if `max_steps` hasn't been hit yet
4. Print running token/cost totals as the agent works, not just at the end

**You're done when:** You can watch the running total grow step by step, and confirm the agent respects whichever limit is hit first.

---

### Stage D — Full transparency into what happened

**Goal:** Reproduce the example output exactly — every step's decision and result visible, plus a final summary.

**Do:**
1. Print each step's decision (tool name + arguments, or "final answer") as it happens
2. Print each tool result
3. At the end, print total steps used, total tokens, estimated cost, and the final answer

**You're done when:** Watching your agent run tells a complete story of what it did and why, without you needing to add debug prints afterward to understand a run.

---

### Stage E — Prove the stop conditions actually work

**Goal:** Don't just trust the code — verify it under conditions designed to break it.

**Do:**
1. Set `max_steps=1` and ask a question that obviously needs 2+ tool calls. Confirm it stops cleanly after 1 step with an honest "I didn't finish" message, not a fabricated answer
2. Ask a question with an intentionally broken/unavailable tool (temporarily break one). Confirm the agent doesn't get stuck retrying it forever — it should either try an alternative or give up within the step limit
3. Time a full run. Confirm total wall-clock time is roughly what you'd expect from the number of steps × per-step latency — no surprise hangs

**You're done when:** All three tests behave the way you'd want a production system to behave, not just the way a happy-path demo would.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Agent occasionally never stops | `max_steps` check happens after acting instead of before, or is missing on one code path | Check the limit at the *top* of every loop iteration, before doing any work |
| Agent gives a fabricated answer after hitting `max_steps` | Final-answer fallback text isn't distinguished from a real answer | Make the "stopped early" message unmistakably different, and log `stop_reason` separately from `final_answer` |
| Token count stays at 0 | Not reading real usage from the API response | Print the raw response once and find the actual usage field |
| Agent loops between two tools without progressing | No detection of repeated identical tool calls | Optional but useful: track recent tool calls and treat 2 identical calls in a row as a signal to force a final answer |

## Going further (optional)

1. **Repeated-call detection.** If the agent calls the exact same tool with the exact same arguments twice in a row, that's a strong signal it's stuck, not making progress. Detect this and force early termination with a clear reason, rather than waiting for `max_steps`.
2. **A `--max-cost` flag** in dollars instead of tokens, computed live using your Part IV price table. More intuitive for a human operator to reason about than a raw token count.
3. **Compare step limits.** Run the same 5 questions with `max_steps` at 3, 6, and 12. Note how many actually needed more than 3, and how the total cost scales. This gives you a real, evidence-based sense of what a reasonable default limit looks like — rather than picking a number arbitrarily.

---

## Checkpoint

**Knowledge:**
1. State the sharpest possible distinction between a workflow and an agent.
2. Name the four steps of the agent loop in order.
3. Why isn't "the model will say when it's done" sufficient on its own to stop a loop?
4. Why do you need both a step limit and a token/cost limit — what failure does each one specifically catch that the other might not?

**Implementation:** Write the bounded agent loop — with both limits, and a clearly distinguished `stop_reason` — from memory, no reference.

**Debugging:** A production agent hit its `max_steps` limit on 15% of runs last week, up from 2% the week before. Walk through how you'd investigate: what would you look at first, and what are two or three plausible root causes you'd want to rule out?

**Architecture:** You're building a research agent that might legitimately need 40+ steps for a hard question, but you don't want to allow a runaway loop to hit anywhere near that on an easy question. Describe an approach that doesn't just mean "set `max_steps=40`" — what would make the limit adaptive to the actual difficulty of the task?

---

**Next:** [Part VIII — Agent Architectures](part08-agent-architectures.md), where you'll see that the loop you just built is only one of several ways to structure an agent — and learn when each shape actually fits.
