# Part XXI — CrewAI

## Before you start

**What you need already:** [Part VIII](part08-agent-architectures.md) (Supervisor, especially), [Part XX](part20-langgraph.md) (so you can compare two genuinely different abstractions for the same problem).

**What this part gives you:** A different way of expressing multi-agent coordination than LangGraph's explicit graph — one built around describing *roles* rather than wiring *control flow*. Seeing both makes clear that neither is "the" right abstraction; they trade off differently.

**Time:** 1–2 days.

**When you finish, you'll be able to:** Build a role-based multi-agent team quickly, and recognize when that speed comes at the cost of the explicit control you had in LangGraph.

---

## Stage 1 — A different starting question

LangGraph starts from: "what are the states and transitions?" CrewAI starts from a different, more human question: "who's on the team, and what's each person's job?"

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Researcher",
    goal="Find accurate, up-to-date information on the given topic",
    backstory="You are a meticulous researcher who always cites sources.",
)

writer = Agent(
    role="Technical Writer",
    goal="Turn research findings into a clear, concise report",
    backstory="You write for engineers who value precision over flair.",
)
```

`role`, `goal`, and `backstory` all get folded into that agent's system prompt behind the scenes — this is directly the same "role prompting" idea from [Part V](part05-prompt-engineering.md), just given a structured place to live instead of you writing the prompt string by hand.

**Check yourself before Stage 2:** What is CrewAI's `Agent(role=..., goal=..., backstory=...)` actually generating under the hood, in terms of what you built in Part V?

---

## Stage 2 — Tasks and dependency, without an explicit graph

```python
research_task = Task(
    description="Research the current state of {topic}.",
    expected_output="A bulleted list of 5 key findings with sources.",
    agent=researcher,
)

write_task = Task(
    description="Write a 3-paragraph report based on the research findings.",
    expected_output="A polished 3-paragraph report.",
    agent=writer,
    context=[research_task],   # write_task depends on research_task's output
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], process=Process.sequential)
result = crew.kickoff(inputs={"topic": "renewable energy storage"})
```

Notice: there's no explicit edge, no conditional routing function, no `add_node`/`add_edge` calls. The `context=[research_task]` line is doing the entire job that a LangGraph edge would do — it tells CrewAI "don't run `write_task` until `research_task` is done, and feed its output in." `Process.sequential` then just runs the task list in order, respecting those dependencies.

This is genuinely less code than the equivalent LangGraph graph, for exactly this kind of task — a fixed sequence where each step needs the previous one's output. What you traded away: LangGraph would let you express arbitrary conditional branching and cycles explicitly; here, the flow is largely determined by the order of `tasks` and their `context` dependencies, which is more limited but much faster to write for straightforward cases.

**Check yourself before Stage 3:** What does `context=[research_task]` on `write_task` accomplish, and how is it different from an explicit LangGraph edge?

---

## Stage 3 — Hierarchical process: delegation instead of a route function

Recall your Part VIII Supervisor — a `supervisor_decide_next` function explicitly deciding which worker runs next. CrewAI's `Process.hierarchical` gives you something similar, but the decision is made by an LLM-driven manager agent, not by code you write:

```python
manager = Agent(role="Project Manager", goal="Coordinate the team to produce the report", allow_delegation=True)

crew_hierarchical = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_agent=manager,
)
```

This is a genuinely different trade-off than your hand-written `supervisor_decide_next`. Your version is deterministic — same state always produces the same decision, and you can read exactly why. The `manager_agent` version is a live LLM call deciding delegation at runtime — more flexible in the face of unanticipated situations, but less predictable and less debuggable, since "why did the manager delegate to X instead of Y" now requires inspecting a model's reasoning rather than reading a function.

**Check yourself before Stage 4:** What's the specific trade-off between your hand-written `supervisor_decide_next` function (Part VIII) and CrewAI's `manager_agent`-driven delegation?

---

## Stage 4 — Guardrails: reflection, built into the task definition

```python
def validate_report_length(output: str) -> tuple[bool, str]:
    if len(output.split()) > 500:
        return False, "Report exceeds 500 words — please condense."
    return True, output


write_task_guarded = Task(
    description="Write a 3-paragraph report.",
    expected_output="A polished report under 500 words.",
    agent=writer,
    guardrail=validate_report_length,
)
```

This is Part VIII Stage 6's Reflection loop, but the critique-and-retry cycle is built into the task itself rather than something you code by hand — if `validate_report_length` returns `False`, CrewAI automatically re-invokes the agent with the feedback and tries again. Same idea, same limitation from Part VIII: this reliably catches checkable format violations (word count), not genuine factual errors, for exactly the reasons discussed there.

**Check yourself before the project:** What kind of problem would `validate_report_length`-style guardrails reliably catch, and what kind would they not?

---

# The Project: Same Team, Sequential vs. Hierarchical, Measured

## What you're building

A 3-agent research-then-write-then-review crew, built once with `Process.sequential` and once with `Process.hierarchical`, compared on cost, latency, and whether the manager's delegation decisions actually made sense.

```bash
$ python crew_compare.py "quantum computing"
```

```
=== Sequential ===
researcher -> writer -> reviewer (fixed order)
3 tasks, 1,420 tokens, 3.8s

=== Hierarchical ===
manager delegated: researcher -> writer -> reviewer (matches expected order)
3 tasks, 1,890 tokens, 5.1s (manager overhead)
```

## Why this project

Building both versions makes the trade-off from Stage 3 concrete rather than theoretical — you'll see exactly what the manager's flexibility costs in tokens and latency, and whether its delegation choices were actually sensible across multiple different topics.

## Build it in stages

---

### Stage A — Sequential crew

**Goal:** Researcher → Writer → Reviewer, with explicit `context` dependencies.

**Do:**
1. Define all three agents with real roles/goals/backstories
2. Define three tasks with correct `context` dependency chaining
3. Run with `Process.sequential` on 3 different topics

**You're done when:** All three topics produce coherent, correctly-ordered output.

---

### Stage B — Hierarchical crew

**Goal:** The same three agents, coordinated by a manager instead of fixed task order.

**Do:**
1. Add a `manager_agent`
2. Run the same 3 topics with `Process.hierarchical`
3. Log the manager's delegation order for each run

**You're done when:** You can compare, topic by topic, whether the manager's delegation order matched what you'd expect (research first, then writing, then review).

---

### Stage C — Measure and compare

**Goal:** Real cost and latency numbers for both approaches.

**Do:**
1. Track total tokens and wall-clock time for each run
2. Report the average overhead the hierarchical version adds
3. Note any case where the manager's delegation was surprising or suboptimal

**You're done when:** You have real numbers backing a genuine opinion about when the flexibility is worth the cost, for tasks like this one.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| `write_task` runs without the research findings | `context=[research_task]` missing or referencing the wrong task object | Confirm the dependency is set and points at the exact task instance |
| Hierarchical manager never delegates to one agent | Manager's goal too vague, or that agent's `role`/`goal` doesn't clearly signal when it should be used | Sharpen role descriptions — this is the same "specificity matters" lesson from Part V, applied to agent roles |
| Guardrail never triggers even on an obviously-too-long output | Guardrail function has a bug, or isn't actually wired to the task | Test the guardrail function directly, in isolation, against a known-bad input |

## Going further (optional)

1. **Add memory** (`Crew(memory=True)`) and test whether a second `kickoff()` call on a related topic benefits from what was learned in the first — compare against your hand-built memory system from Part X.
2. **A 4-agent crew** with a genuinely non-linear dependency (two tasks that can run in parallel before a final task depends on both) — see how naturally (or not) `context=[...]` with multiple dependencies expresses this compared to LangGraph's explicit fan-in.

---

## Checkpoint

**Knowledge:**
1. What does `Agent(role=..., goal=..., backstory=...)` generate under the hood?
2. What's the trade-off between `Process.sequential`'s `context` dependencies and LangGraph's explicit edges?
3. What's the trade-off between a hand-written `supervisor_decide_next` function and CrewAI's `manager_agent`?

**Implementation:** Build a sequential 3-agent crew with correct task dependencies from memory.

**Debugging:** A hierarchical crew's manager never delegates to one specific agent, no matter the topic. What would you check first in that agent's definition?

**Architecture:** For a task with a fixed, well-understood sequence of specialist steps, would you reach for CrewAI's sequential process, LangGraph's explicit graph, or your own hand-built Supervisor from Part VIII? Justify your choice.

---

**Next:** [Part XXII — LangChain vs. LangGraph vs. CrewAI](part22-langchain-vs-langgraph-vs-crewai.md), a direct, factual comparison now that you've built with all three.
