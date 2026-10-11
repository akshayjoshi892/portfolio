# Part VIII — Agent Architectures

## Before you start

**What you need already:** [Part VII](part07-what-is-an-agent.md) — the bounded agent loop. Everything in this part is a variation on that same loop.

**What this part gives you:** The single loop from Part VII is *one shape*. It's a good default, but it's not the only useful shape, and using it for everything leads to real problems: agents that ramble when a fixed sequence would've been faster and cheaper, or rigid pipelines forced where real judgment was needed. This part teaches you to recognize which shape a problem actually calls for, before you write code.

**Time:** 4–5 days.

**When you finish, you'll be able to:** Look at a task description and know, before writing a line of code, whether it needs ReAct, a fixed plan, a router, a supervisor with sub-agents, or something else — and defend that choice.

---

## Stage 1 — Why one loop shape isn't enough

Go back to Part VII's loop and ask: what does it assume about the task?

It assumes the agent should interleave thinking and acting, one tool call at a time, discovering the right path as it goes. That's a real strength for genuinely uncertain tasks — you don't know in advance how many searches you'll need, or what you'll find.

But now consider a different task: "translate this document, then summarize the translation, then email it to the team." There's no real uncertainty here. You already know the three steps and their order. Running Part VII's loop on this would work — but at every step, the model has to *re-decide* what to do next, including re-deciding things that were never actually in question. That's wasted tokens, wasted latency, and a small but real chance the model decides to do something other than what you actually wanted, because you gave it more freedom than the task needed.

The lesson: **match the shape of your agent to the actual shape of the task's uncertainty.** A task with a known sequence wants a pipeline. A task with real step-by-step discovery wants a loop. A task that's really "pick one of five known paths" wants a router. This part is a tour of the common shapes and when each one fits.

**Check yourself before Stage 2:** Why would using Part VII's discover-as-you-go loop be wasteful for a task whose steps you already know in full?

---

## Stage 2 — ReAct: the loop you already built, named

What you built in Part VII has a name: **ReAct** (Reason + Act). Each cycle is: think about what to do (Thought), do it (Action), see what happened (Observation), repeat.

```
Thought: I need to find the capital of France first.
Action: search(query="capital of France")
Observation: "Paris is the capital of France."
Thought: Now I need Paris's population.
Action: search(query="population of Paris")
Observation: "Paris has about 2.1 million residents."
Thought: I have what I need.
Final Answer: Paris, the capital of France, has about 2.1 million residents.
```

You already know this shape — it's Part VII's loop, just with the "Thought" text made explicit rather than implicit in the model's internal decision. Making the thought explicit has a real benefit: it's directly visible in the transcript, which makes debugging *why* the agent did something much easier — you're reading its reasoning, not guessing at it.

**When to reach for it:** a task where you genuinely don't know the right sequence of tool calls in advance, and the number of steps needed is itself uncertain. Open-ended research questions are the classic case.

**When it's the wrong choice:** a task where the steps are already known. Using ReAct here means paying for "Thought" text that doesn't discover anything new — the model already knew the plan; you just made it re-derive and re-state it at every step.

**Check yourself:** Why does writing out an explicit "Thought" before each action help you debug an agent, compared to Part VII's version where the model's reasoning was implicit?

---

## Stage 3 — Plan-and-Execute: decide the whole path first

Here's the fix for the wasted-uncertainty problem from Stage 1. Separate planning from execution into two distinct phases: figure out the steps once, then run them.

```python
from dataclasses import dataclass, field


@dataclass
class PlanStep:
    id: str
    description: str
    status: str = "pending"       # pending | done | failed
    result: str | None = None


def generate_plan(goal: str) -> list[PlanStep]:
    """A real version would ask the LLM to produce this list. Simulated here
    to show the shape clearly first."""
    return [
        PlanStep(id="find_ceo", description="Find the current CEO of Acme Corp"),
        PlanStep(id="find_founding", description="Find Acme Corp's founding year"),
        PlanStep(id="summarize", description="Combine into one summary sentence"),
    ]


def execute_step(step: PlanStep) -> tuple[bool, str]:
    """A real version calls a tool or the model for each step."""
    fake_results = {
        "find_ceo": (True, "Jane Doe is CEO."),
        "find_founding": (True, "Founded in 1998."),
        "summarize": (True, ""),
    }
    return fake_results[step.id]


def run_plan_and_execute(goal: str) -> str:
    plan = generate_plan(goal)
    print(f"Plan for {goal!r}:")
    for s in plan:
        print(f"  - {s.description}")

    findings = []
    for step in plan[:-1]:      # last step is the summary, handled separately
        success, result = execute_step(step)
        step.status = "done" if success else "failed"
        step.result = result
        findings.append(result)

    return f"Summary: {' '.join(findings)}"


print(run_plan_and_execute("Summarize Acme Corp's leadership and history"))
```

```
Plan for 'Summarize Acme Corp's leadership and history':
  - Find the current CEO of Acme Corp
  - Find Acme Corp's founding year
  - Combine into one summary sentence
Summary: Jane Doe is CEO. Founded in 1998.
```

Notice what's *not* happening here that was happening in ReAct: the model isn't re-deciding "what should I do next?" at every single step. It decided once, upfront, and execution just... executes. Cheaper, faster, and the plan itself is something you can show a human for review *before* any tool calls happen — useful when the actions are risky.

### The real weakness, and it's a serious one

What happens if step 1 reveals something that invalidates step 2? Say `find_ceo` actually returns "No company named Acme Corp was found." The plan above just barrels ahead to `find_founding` anyway, because nothing in this code checks whether earlier steps' results make later steps pointless.

This is **plan staleness** — a plan made with incomplete information can become wrong the moment execution reveals new information, and a naive Plan-and-Execute has no way to notice. The fix is **replanning**: after a step fails or reveals something surprising, regenerate the remaining plan instead of blindly continuing. You'll build this properly in [Part XXX](part30-planning.md) — for now, just recognize the risk so you don't get burned by it in production before you've read that part.

**Check yourself before Stage 4:** What's the actual cost difference between ReAct and Plan-and-Execute for a 3-step task with a known sequence? What's the specific danger of Plan-and-Execute that ReAct doesn't have?

---

## Stage 4 — Router: picking the right specialist, once

Some tasks aren't really "figure out steps" at all — they're "figure out which *one* of several known handlers should take this." A customer support bot deciding between "billing question," "technical issue," and "general inquiry" is a router problem, not a research problem.

```python
def classify_intent(user_message: str) -> str:
    """A real version uses a cheap, fast model call for this classification —
    it doesn't need your most capable (and expensive) model just to sort
    messages into 3 buckets."""
    message_lower = user_message.lower()
    if any(word in message_lower for word in ["refund", "charge", "bill", "payment"]):
        return "billing"
    if any(word in message_lower for word in ["error", "bug", "crash", "broken"]):
        return "technical"
    return "general"


def handle_billing(message: str) -> str:
    return f"[billing team handling]: {message}"


def handle_technical(message: str) -> str:
    return f"[technical team handling]: {message}"


def handle_general(message: str) -> str:
    return f"[general support handling]: {message}"


HANDLERS = {"billing": handle_billing, "technical": handle_technical, "general": handle_general}


def route(user_message: str) -> str:
    intent = classify_intent(user_message)
    print(f"routed to: {intent}")
    return HANDLERS[intent](user_message)


print(route("I was charged twice for my subscription"))
print(route("The app keeps crashing when I open it"))
```

```
routed to: billing
[billing team handling]: I was charged twice for my subscription
routed to: technical
[technical team handling]: The app keeps crashing when I open it
```

**When to reach for it:** the task is really "pick a lane," and each lane has a well-defined handler. The classification step itself is cheap and the routing decision doesn't need to happen more than once.

**The real weakness:** misclassification. If "billing" and "general" overlap in a user's actual message ("I have a question about my bill" — is that billing or general?), a bad router routes confidently to the wrong lane with no way to notice it did anything wrong. Test your router specifically on ambiguous, borderline messages, not just clean examples — this is exactly the kind of thing [Part V](part05-prompt-engineering.md)'s evaluation harness is for.

**Check yourself before Stage 5:** Why would you use a cheap, fast model for the classification step rather than your most capable model? What's the main way a router silently fails?

---

## Stage 5 — Supervisor: one coordinator, several workers

A step beyond routing: sometimes you don't just need to pick *one* handler — you need several specialists to each contribute, with something coordinating the sequence and combining their outputs.

```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class Worker:
    name: str
    handle: Callable[[str], str]


def researcher(task: str) -> str:
    return f"[research findings on: {task}]"


def writer(research: str) -> str:
    return f"[polished write-up based on: {research}]"


WORKERS = {"researcher": Worker("researcher", researcher), "writer": Worker("writer", writer)}


def supervisor_decide_next(completed: dict[str, str]) -> str | None:
    """The 'brain' of the supervisor: what should happen next, given what's
    already been done? A real version uses an LLM call for this decision."""
    if "researcher" not in completed:
        return "researcher"
    if "writer" not in completed:
        return "writer"
    return None   # nothing left to do


def run_supervisor(goal: str) -> dict[str, str]:
    completed: dict[str, str] = {}

    while True:
        next_worker_name = supervisor_decide_next(completed)
        if next_worker_name is None:
            break

        worker = WORKERS[next_worker_name]
        input_data = goal if next_worker_name == "researcher" else completed["researcher"]
        completed[next_worker_name] = worker.handle(input_data)
        print(f"{next_worker_name} produced: {completed[next_worker_name]}")

    return completed


run_supervisor("renewable energy trends")
```

```
researcher produced: [research findings on: renewable energy trends]
writer produced: [polished write-up based on: [research findings on: renewable energy trends]]
```

This is genuinely different from routing. A router picks *one* path and is done. A supervisor orchestrates a *sequence* of specialists, deciding after each one finishes what should happen next — closer to a mini agent loop, except the "actions" are entire sub-agents rather than individual tool calls.

**The real weakness:** the supervisor decides one worker at a time, in sequence, even when two workers' work doesn't actually depend on each other and could run concurrently. If you had a `fact_checker` worker whose job didn't depend on the `writer`'s output, running them one after another wastes time you didn't need to spend. This is exactly the problem [Part XXVI](part26-agent-orchestration.md)'s fan-out/fan-in pattern solves — worth knowing the limitation now so you recognize when you've outgrown a plain sequential Supervisor.

**Check yourself before Stage 6:** How is a Supervisor different from a Router — what can it do that a Router can't? What's the cost of the Supervisor always running workers one at a time?

---

## Stage 6 — Reflection: catching your own mistakes before they ship

Sometimes the bottleneck isn't gathering information — it's output *quality*. You have what you need, but the first draft isn't good enough. Reflection adds a check-and-revise cycle:

```python
def generate_draft(task: str) -> str:
    return f"[first-pass draft addressing: {task}]"


def critique(draft: str, criteria: list[str]) -> tuple[bool, list[str]]:
    """A real version asks the LLM to check the draft against the criteria."""
    # Simulated: the first draft is missing a word count constraint.
    if "word count" not in draft:
        return False, ["missing explicit word count"]
    return True, []


def revise(draft: str, feedback: list[str]) -> str:
    return f"{draft} [revised to address: {', '.join(feedback)}, now includes word count]"


def run_with_reflection(task: str, criteria: list[str], max_revisions: int = 3) -> str:
    draft = generate_draft(task)

    for revision in range(max_revisions):
        passed, feedback = critique(draft, criteria)
        print(f"revision {revision}: passed={passed} feedback={feedback}")
        if passed:
            return draft
        draft = revise(draft, feedback)

    return f"[unresolved after {max_revisions} revisions] {draft}"


print(run_with_reflection("write a summary", criteria=["word count", "cites sources"]))
```

```
revision 0: passed=False feedback=['missing explicit word count']
revision 1: passed=True feedback=[]
[first-pass draft addressing: write a summary] [revised to address: missing explicit word count, now includes word count]
```

Notice the `max_revisions` bound — this is Part VII's stopping-condition discipline, applied here for exactly the same reason. A critique-and-revise loop with no bound is just as capable of running forever as a tool-calling loop with no `max_steps`.

**The real weakness, worth remembering from [Part V Stage 5](part05-prompt-engineering.md):** the critique step can only reliably catch things that are *checkable against stated criteria* — length, required sections, format. It's much less reliable at catching genuine factual errors, because the model critiquing its own factual claim has no more access to the truth than the model that made the claim in the first place. Don't use plain reflection as your only defense against factual mistakes; you need retrieval ([Part XI](part11-rag-fundamentals.md)) or real verification ([Part XXXI](part31-reasoning.md)) for that.

**Check yourself before Stage 7:** What kind of problem does reflection reliably catch? Why is `max_revisions` non-negotiable, using the same reasoning as Part VII's `max_steps`?

---

## Stage 7 — A field guide to the rest

You've now built the four architectures that show up most often, with working code for each. The remaining shapes in the field are mostly *combinations or variations* of these same four ideas, applied at a different scale or with a different trigger. Here's a map so you recognize them when you see them, and know which Part to visit for a full build later.

| Architecture | One-sentence idea | Built from | See it built in |
|---|---|---|---|
| **Critic / Generator-Critic** | Like reflection, but the critic is a separate, differently-prompted "role" from the generator — useful when you want the critique to come from a genuinely different perspective, not just a second pass by the same prompt | Stage 6, split into two roles | [Part XXVII](part27-multi-agent-systems.md) |
| **Debate** | Multiple independent agents argue different positions; a judge decides | Stage 6's critique idea, run in parallel across several agents | [Part VIII catalog below] |
| **Hierarchical** | A Supervisor whose "workers" are themselves Supervisors | Stage 5, nested | [Part XXVII](part27-multi-agent-systems.md) |
| **State machine** | An explicit set of named states and allowed transitions between them — more rigid and auditable than a free-form loop | A constrained version of Part VII's loop | [Part IX](part09-agent-state.md) |
| **Graph-based (e.g. LangGraph)** | A general framework for expressing any of the above as nodes and edges | All of the above, as a shared abstraction | [Part XX](part20-langgraph.md) |
| **Event-driven** | Instead of a request triggering the loop, an external event does (a new support ticket arrives, a file changes) | Stage 2's loop, triggered by a subscription instead of a direct call | [Part L](part50-advanced-orchestration.md) |
| **Parallel fan-out/fan-in** | Several independent agents run at once (not one after another like Stage 5's Supervisor), then results are merged | Stage 5, but with `asyncio.gather` instead of a sequential loop | [Part XXVI](part26-agent-orchestration.md) |
| **Pipeline** | A strict, always-the-same sequence of stages, no branching | The simplest possible version of Stage 3's plan — a plan that's always identical | Used throughout; formalized in [Part XXVI](part26-agent-orchestration.md) |
| **Blackboard** | Multiple agents read and write to one shared workspace instead of passing messages directly to each other | A shared-state variant of Stage 5's coordination | [Part XXVII](part27-multi-agent-systems.md) |

You don't need to memorize this table. You need to recognize, when a real task lands in front of you, which row it resembles — and that comes from having *built* Stages 2–6 yourself, which you just did.

---

## Stage 8 — Choosing, with a real decision process

Here's how to actually decide, for a task in front of you, which architecture fits — as a sequence of questions, not a vibe:

```
Is the number and order of steps known in advance?
  YES → Plan-and-Execute (Stage 3), or even a plain Pipeline if it never branches
  NO  → continue

Is this really "pick one of a few known paths," with no further steps after?
  YES → Router (Stage 4)
  NO  → continue

Does it need multiple distinct specialists contributing, with something coordinating them?
  YES → are their contributions independent of each other?
          YES → Parallel fan-out (Stage 7, built properly in Part XXVI)
          NO  → Supervisor (Stage 5)
  NO  → continue

Is the bottleneck output quality rather than gathering information?
  YES → Reflection (Stage 6)
  NO  → continue

Otherwise: the task has real, undiscovered structure that only reveals
itself step by step → ReAct (Stage 2), which is Part VII's loop
```

This isn't a rigid formula — real systems often combine shapes (a Router that dispatches to a ReAct agent for the "technical" lane and a fixed Pipeline for the "billing" lane is completely normal). But running through these questions *before* writing code will save you from the single most common architecture mistake: defaulting to a free-form ReAct loop for everything, because it's the one you know, even when the task's structure was fully known in advance and a Pipeline would have been faster, cheaper, and more predictable.

**Check yourself before the project:** Walk through the decision process above for: (a) "translate this document into 3 languages," (b) "answer open-ended questions about our internal wiki," (c) "decide whether an incoming ticket is a bug report or a feature request." What architecture does each land on, and why?

---

# The Project: Same Task, Three Architectures, Measured

## What you're building

The same real task — "research a company and produce a 3-sentence summary" — implemented three different ways: ReAct, Plan-and-Execute, and Supervisor. Then you measure all three honestly and report which actually won, and on what axis.

```bash
$ python compare_architectures.py "Acme Corporation"
```

```
=== ReAct ===
6 tool calls, 1,840 tokens, 4.2s
"Acme Corporation, founded in 1998 and led by CEO Jane Doe, is a..."

=== Plan-and-Execute ===
3 tool calls, 980 tokens, 2.1s
"Acme Corporation, founded in 1998 and led by CEO Jane Doe, is a..."

=== Supervisor (researcher -> writer) ===
2 tool calls, 1,240 tokens, 3.0s
"Acme Corporation, a company founded in 1998 under CEO Jane Doe..."

Quality scores (1-5, rated by you): ReAct=4  Plan-and-Execute=4  Supervisor=5
```

## Why this project

Reading about trade-offs is not the same as measuring them. This project forces you to build the same thing three ways and confront real numbers — which is exactly the discipline [Part V](part05-prompt-engineering.md) asked you to apply to chain-of-thought, now applied to architecture choice.

## Build it in stages

---

### Stage A — Pick and fix the task

**Goal:** One real, well-specified task you'll implement three times.

**Do:**
1. Define the task precisely: what input, what output shape, what "correct" means
2. Set up 2 real tools it will need (e.g., a search tool and a fact-lookup tool — mocked HTTP is fine, but keep them consistent across all three implementations)
3. Write down, before building anything, your honest guess about which architecture you expect to win on speed, and which on quality

**You're done when:** You have a task specific enough that "correct" is checkable, and a written prediction to compare against your actual results later.

---

### Stage B — Build the ReAct version

**Goal:** Reuse Part VII's loop directly for this task.

**Do:**
1. Wire your task's tools into the `ToolRegistry`
2. Run the loop, recording: total tool calls, total tokens, wall-clock time, and the final output

**You're done when:** It produces a correct 3-sentence summary and you have real numbers for all four metrics.

---

### Stage C — Build the Plan-and-Execute version

**Goal:** Same task, same tools, but plan first.

**Do:**
1. Add a planning step: one LLM call that produces an ordered list of steps for *this specific task*
2. Execute each step in order, calling tools as needed
3. Record the same four metrics

**You're done when:** It also produces a correct summary, and you can directly compare its metrics against Stage B's.

---

### Stage D — Build the Supervisor version

**Goal:** Split the task into a `researcher` role and a `writer` role, coordinated.

**Do:**
1. Give `researcher` the tools; it gathers facts
2. Give `writer` no tools; it only turns gathered facts into the final 3-sentence summary
3. A supervisor function calls `researcher` then `writer`, in that order
4. Record the same four metrics

**You're done when:** All three versions have produced a summary for the same input, and you have a table of all their numbers side by side.

---

### Stage E — Score quality honestly, then decide

**Goal:** Don't just compare speed and cost — compare whether the outputs are actually *good*.

**Do:**
1. Read all three outputs blind, if you can arrange it (or at least read them without looking at which architecture produced which first)
2. Score each 1–5 on: factual correctness, whether it's actually 3 sentences as specified, and how well-written it reads
3. Write a short, honest paragraph: which architecture actually won, on which axis, and does that match your Stage A prediction?

**You're done when:** You have a real answer, grounded in your own numbers, to "which architecture should I use for this kind of task" — not a guess, and possibly not what you expected going in.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| All three architectures give nearly identical token counts | The task doesn't actually have meaningful step-order uncertainty | Pick a task where the "known steps vs. discover as you go" distinction is real, or note this finding — it's itself informative |
| Plan-and-Execute produces a worse answer than ReAct | The upfront plan didn't get revised when a step revealed unexpected information (Stage 3's plan-staleness risk) | Note this as a concrete example of the weakness described in Stage 3, rather than treating it as a bug to silently fix |
| Supervisor is slower than expected | Researcher and writer run strictly sequentially even though writer only needs researcher's *final* output, not an ongoing exchange | This is expected given Stage 5's stated weakness — it's the reason [Part XXVI](part26-agent-orchestration.md) exists |

## Going further (optional)

1. **Add a fourth version: Router.** If your task actually has 2–3 distinct "kinds" of company (public vs. private, say), build a router that dispatches to a specialized sub-flow for each, and see whether that beats a one-size-fits-all approach.
2. **Run each architecture 5 times** and look at the *variance* in tool-call count and tokens, not just the average. Which architecture is most consistent run to run? Consistency matters for cost predictability in production, separate from raw average cost.
3. **Deliberately trigger Plan-and-Execute's weakness.** Feed it a company name that doesn't exist, and watch whether it blindly continues through steps 2 and 3 anyway. This is your first hands-on encounter with the exact problem [Part XXX](part30-planning.md)'s replanning mechanism fixes.

---

## Checkpoint

**Knowledge:**
1. State the difference between ReAct and Plan-and-Execute in terms of *when* the model decides the sequence of steps.
2. What's the main weakness of Plan-and-Execute, and why doesn't ReAct have the same weakness?
3. What's the main weakness of a Router, and how would you test for it?
4. What kind of problem does Reflection reliably solve, and what kind does it not?
5. Walk through the Stage 8 decision process for a task of your own choosing.

**Implementation:** Build ReAct, Plan-and-Execute, and Supervisor for a simple task from memory, without looking at this file.

**Debugging:** A Supervisor-based system never calls its `writer` worker. List the three most likely causes, in the order you'd check them.

**Architecture:** You're building a system for a hierarchical support desk: tier-1 agents handle common questions using a fixed script, and escalate anything unusual to a tier-2 agent with broader tools and real discretion. Sketch which architecture(s) from this part fit which tier, and why.

---

**Next:** [Part IX — Agent State](part09-agent-state.md), where you'll learn to make an agent's progress survive a crash — because right now, if your process dies mid-run, everything is lost.
