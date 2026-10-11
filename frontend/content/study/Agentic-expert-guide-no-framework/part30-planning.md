# Part XXX — Planning

## Before you start

**What you need already:** [Part VIII Stage 3](part08-agent-architectures.md) — you built Plan-and-Execute and explicitly identified its weakness: a plan made with incomplete information can go stale the moment execution reveals something unexpected, and the naive version has no way to notice or recover.

**What this part gives you:** The actual fix for that weakness — replanning — plus dependency-aware planning for tasks where steps aren't purely sequential.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Build a planner that detects when a step's result invalidates the rest of the plan, and repairs the plan instead of blindly continuing or giving up entirely.

---

## Stage 1 — Watch the exact failure Part VIII warned about

Recreate it, deliberately, so it's concrete rather than a remembered warning.

```python
from dataclasses import dataclass, field


@dataclass
class PlanStep:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str | None = None


def naive_plan(goal: str) -> list[PlanStep]:
    return [
        PlanStep(id="find_company", description="Confirm the company exists"),
        PlanStep(id="find_ceo", description="Find the company's CEO", depends_on=["find_company"]),
        PlanStep(id="find_founding", description="Find the founding year", depends_on=["find_company"]),
        PlanStep(id="summarize", description="Combine into a summary", depends_on=["find_ceo", "find_founding"]),
    ]


def naive_execute_step(step: PlanStep) -> tuple[bool, str]:
    if step.id == "find_company":
        return False, "No company found matching that name."   # the plan's premise was wrong
    if step.id == "find_ceo":
        return True, "Jane Doe"     # runs anyway, on a false premise
    if step.id == "find_founding":
        return True, "1998"          # runs anyway, on a false premise
    return True, "Summary generated"


def run_naive_plan(goal: str) -> str:
    plan = naive_plan(goal)
    for step in plan:
        success, result = naive_execute_step(step)
        step.status = "done" if success else "failed"
        step.result = result
        print(f"{step.id}: success={success} result={result!r}")
    return "plan completed (but was it actually valid?)"


print(run_naive_plan("Summarize a made-up company"))
```

```
find_company: success=False result='No company found matching that name.'
find_ceo: success=True result='Jane Doe'
find_founding: success=True result='1998'
summarize: success=True result='Summary generated'
```

Look at this output carefully. `find_company` genuinely failed — the entire premise of the task was wrong. But nothing here stops `find_ceo` and `find_founding` from running anyway, on top of a false premise, producing a confident-looking but entirely fabricated summary at the end. This is exactly the plan-staleness risk Part VIII flagged: **the naive planner has no mechanism to notice that an early failure invalidates everything downstream.**

**Check yourself before Stage 2:** Why does `naive_execute_step` letting `find_ceo` and `find_founding` "succeed" on a nonexistent company produce output that's worse than an honest failure?

---

## Stage 2 — Dependency-aware execution: only run what's actually ready

Before fixing the staleness problem, fix a related one: the naive version runs steps in a fixed list order, even though `find_ceo` and `find_founding` don't actually depend on each other — only both depend on `find_company`. A better executor only runs a step once everything it depends on has genuinely succeeded.

```python
def ready_steps(plan: list[PlanStep]) -> list[PlanStep]:
    plan_by_id = {s.id: s for s in plan}
    return [
        s for s in plan
        if s.status == "pending"
        and all(plan_by_id[dep].status == "done" for dep in s.depends_on)
    ]
```

This is the same dependency-readiness idea from [Part XXVI](part26-agent-orchestration.md)'s DAG execution — a step is ready only once its dependencies have *actually completed successfully*, not just been attempted. Notice this already partially fixes the staleness problem just by being stricter: if `find_company` fails (status becomes `"failed"`, not `"done"`), then `find_ceo` and `find_founding` are never `ready_steps()`, because their dependency never reached `"done"`.

**Check yourself before Stage 3:** Why does checking `plan_by_id[dep].status == "done"` (rather than just "was attempted") prevent downstream steps from running on a failed premise?

---

## Stage 3 — Replanning: don't just stop, adapt

Stage 2 stops the plan from blindly continuing on a false premise — but "stop entirely" isn't always the right response either. Sometimes a failure calls for trying a different approach, not giving up. This is **replanning**: when a step fails, generate a *new* step (or steps) to try instead, and rewire anything that was depending on the failed step to depend on the new one instead.

```python
def replan_on_failure(plan: list[PlanStep], failed_step: PlanStep) -> PlanStep:
    """Returns the new fallback step. Rewires dependents to point at it instead."""
    fallback = PlanStep(
        id=f"{failed_step.id}_fallback",
        description=f"Try an alternate approach for: {failed_step.description}",
        depends_on=list(failed_step.depends_on),
    )
    plan.append(fallback)

    for step in plan:
        if failed_step.id in step.depends_on:
            step.depends_on.remove(failed_step.id)
            step.depends_on.append(fallback.id)

    failed_step.status = "done"   # superseded — no longer blocks anything
    failed_step.result = "superseded by fallback"

    return fallback
```

Trace through what this does: it creates a new step describing an alternate approach, adds it to the plan, and — critically — finds every step that was depending on the failed one and **rewires their dependency to point at the new fallback instead**. It then marks the failed step as `"done"` (not `"failed"`) specifically so it stops blocking `ready_steps()` — its role is over, superseded by the fallback, which is now what everything downstream actually depends on.

**Check yourself before Stage 4:** Why mark the failed step as `"done"` (superseded) rather than leaving it `"failed"`, given how `ready_steps()` checks dependency status?

---

## Stage 4 — Bound replanning, for the same reason you bound everything else

You should be able to predict this by now: replanning without a limit is exactly as dangerous as any other unbounded loop in this curriculum. A step that keeps failing even after a fallback — maybe the underlying tool is genuinely broken, maybe the task is genuinely impossible — would generate fallback after fallback forever without a cap.

```python
def run_plan_with_replanning(plan: list[PlanStep], execute_fn, max_replans: int = 2) -> str:
    replans_used = 0

    while any(s.status == "pending" for s in plan):
        current_ready = ready_steps(plan)
        if not current_ready:
            break   # nothing ready and nothing pending left to become ready — done or stuck

        for step in current_ready:
            success, result = execute_fn(step)
            if success:
                step.status = "done"
                step.result = result
                print(f"[done] {step.id}: {result}")
            else:
                print(f"[failed] {step.id}: {result}")
                if replans_used < max_replans:
                    replan_on_failure(plan, step)
                    replans_used += 1
                else:
                    step.status = "failed"   # out of replans — accept the failure honestly

    summary_step = next((s for s in plan if s.id == "summarize"), None)
    if summary_step and summary_step.status == "done":
        return summary_step.result
    return f"plan incomplete after {replans_used} replan(s) — could not reach a valid summary"


def execute_fn(step: PlanStep) -> tuple[bool, str]:
    if step.id == "find_company":
        return False, "No company found matching that name."
    if "fallback" in step.id:
        return False, "Alternate approach also failed."   # even the fallback doesn't work
    return True, f"result for {step.id}"


plan = naive_plan("Summarize a made-up company")
print(run_plan_with_replanning(plan, execute_fn))
```

```
[failed] find_company: No company found matching that name.
[failed] find_company_fallback: Alternate approach also failed.
plan incomplete after 2 replan(s) — could not reach a valid summary
```

Compare this final line to Stage 1's output. Instead of confidently producing a fabricated summary built on a false premise, the system now **honestly reports that it couldn't complete the task** — exactly the same "honest decline" principle from [Part XI Stage 5](part11-rag-fundamentals.md)'s RAG relevance threshold, now applied to planning. A wrong confident answer is worse than an honest failure; this is true everywhere in this curriculum, and it's true here too.

**Check yourself before the project:** Compare the final outputs of Stage 1 (naive) and Stage 4 (bounded replanning) on the exact same failing scenario. Why is Stage 4's output strictly more useful to a real user, even though it "failed" while Stage 1 technically "succeeded"?

---

# The Project: An Autonomous Task Planner With Replanning

## What you're building

A full dependency-graph planner — generation, dependency-aware execution, bounded replanning, and checkpointing (reusing Part IX) — tested against tasks specifically designed to trigger replanning.

```bash
$ python planner.py "Research and summarize NonExistent Corp's history"
```

```
plan generated: 4 steps
[ready] find_company
[failed] find_company: no matching company found
[replan] generated fallback: find_company_fallback
[ready] find_company_fallback
[failed] find_company_fallback: alternate approach also failed
[replan budget exhausted]

RESULT: plan incomplete — could not verify the company exists.
No fabricated summary was produced.
```

## Why this project

Plan-and-Execute is only production-safe with replanning — this project is what turns Part VIII's flagged weakness into something you've actually fixed and tested against real failure scenarios, not just read about.

## Build it in stages

---

### Stage A — Real plan generation

**Goal:** `generate_plan(goal)` using a real LLM call to produce a genuine dependency structure, not a hardcoded list.

**Do:**
1. Write a prompt asking the model to decompose a goal into steps with explicit dependencies
2. Parse the response into `PlanStep` objects with correctly populated `depends_on` fields
3. Validate the result has no cycles, reusing `topological_order`'s cycle check from Part XXVI

**You're done when:** You can hand it 3 different real goals and get back sensible, correctly-structured plans each time.

---

### Stage B — Dependency-aware execution with real tools

**Goal:** `ready_steps` and the execution loop, wired to real tools instead of fake `execute_fn` results.

**Do:**
1. Wire in a real tool (search, calculator, whatever fits your test goals) for step execution
2. Run a plan end to end on a goal that should succeed cleanly
3. Confirm steps run in correct dependency order, and independent steps could run concurrently (reuse Part XXVI's level-based execution here if you want the full benefit)

**You're done when:** A real, multi-step plan executes correctly, respecting dependencies.

---

### Stage C — Trigger replanning deliberately

**Goal:** A test scenario specifically designed to fail at an early step, forcing replanning.

**Do:**
1. Construct a goal where an early step will genuinely fail (e.g., asking about something that doesn't exist)
2. Confirm the failure triggers `replan_on_failure` correctly, and that dependents get rewired to the fallback
3. Confirm the bounded replan count is respected — test both a case where the fallback succeeds and one where it also fails

**You're done when:** You've watched both outcomes — successful recovery via replanning, and an honest failure after exhausting replan attempts — happen for real.

---

### Stage D — Checkpoint the plan's execution

**Goal:** A crashed mid-plan run can resume without redoing completed steps or losing track of replans already used.

**Do:**
1. Reuse Part IX's checkpoint pattern, saving the full plan state (including any replan-generated fallback steps) after every step
2. Kill the process mid-plan, resume with the same run ID
3. Confirm it picks up exactly where it left off, including correctly remembering how many replans have already been used

**You're done when:** A resumed run doesn't repeat completed steps, doesn't lose its replan budget tracking, and correctly continues toward the same goal.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Downstream steps run despite an upstream failure | `ready_steps` checking for `"pending"` dependencies incorrectly, or not checking status strictly enough | Confirm the exact check is `status == "done"`, not merely "not pending" |
| Replanning loops indefinitely | `max_replans` not actually enforced, or reset incorrectly on each failure | Confirm `replans_used` persists correctly across the whole run, not reset per-step |
| A plan with real cycles slips through undetected | Cycle check from Part XXVI not actually applied to LLM-generated plans | Always validate a generated plan for cycles before executing any of it |
| Resumed plan re-runs already-completed steps | Checkpoint doesn't capture step statuses correctly, or a "done" step gets re-added to `ready_steps` on reload | Confirm the full `PlanStep` list, with statuses, round-trips correctly through your checkpoint serialization |

## Going further (optional)

1. **Concurrent execution within `ready_steps`.** Combine this part's dependency logic with Part XXVI's level-based `asyncio.gather` execution, so independent ready steps run concurrently rather than one at a time.
2. **A "give up gracefully" message that's actually useful** — instead of just "plan incomplete," have the final failure message explain, in plain language, what was tried and why it didn't work, using the plan's own execution log.
3. **Resource-constrained planning.** Add a cost estimate per step and a total plan budget; if a generated plan would exceed budget, have the planner produce a cheaper alternative before execution even starts.

---

## Checkpoint

**Knowledge:**
1. Why does the naive planner from Stage 1 produce a worse outcome than an honest failure, even though every individual step in its output "succeeded"?
2. Why must a failed step be marked `"done"` (superseded), not left `"failed"`, after replanning creates a fallback?
3. Why does replanning need the same bounding discipline as every other loop in this curriculum?

**Implementation:** Build `ready_steps`, `replan_on_failure`, and the bounded execution loop from memory.

**Debugging:** A plan execution gets stuck with several steps still `"pending"` and none of them ever becoming `"ready"`. What dependency-graph bug does this most likely indicate?

**Architecture:** Design a planner for a task where individual step outcomes are only probabilistically known in advance (e.g., "there's a 70% chance this data source has what we need"). How would the plan structure need to change to handle that uncertainty, beyond simple success/fail replanning?

---

**Next:** [Part XXXI — Reasoning](part31-reasoning.md), where you'll formalize the tool-assisted verification idea that's been implicit since Part V's chain-of-thought caution — and settle, with evidence, when longer reasoning actually helps.
