# Part XXXIV — Human-in-the-Loop

## Before you start

**What you need already:** [Part IX](part09-agent-state.md) (atomic checkpointing), [Part XX Stage 5](part20-langgraph.md) (you saw `interrupt_before` work "for free" because checkpointing was already in place), [Part XXXIII](part33-tool-engineering.md) (the `requires_human_approval` flag).

**What this part gives you:** The real, durable pause-and-resume mechanism that `interrupt_before` was built on — you build it yourself here, so you understand exactly what LangGraph automated for you.

**Time:** 2 days.

**When you finish, you'll be able to:** Build a real approval gate that survives a process restart while waiting for a human, and correctly executes an *edited* version of a proposed action rather than blindly re-running the original.

---

## Stage 1 — Why "just ask before doing it" isn't enough

The naive version of human approval looks trivial:

```python
def naive_approval(tool_name: str, args: dict) -> bool:
    response = input(f"Approve {tool_name}({args})? [y/n]: ")
    return response.strip().lower() == "y"
```

This works for a quick script running in your terminal. It falls apart the moment the approval needs to happen across a real gap — a manager reviewing pending actions the next morning, a mobile approval sent via notification, or simply a process that needs to keep running (or get restarted) while waiting for a human to respond. `input()` blocks the entire process; if it crashes or restarts while waiting, the proposed action and all context about it are gone.

This is exactly the same shape of problem [Part IX](part09-agent-state.md) solved for agent progress: something needs to survive an interruption. Here, the "something" is a pending decision, not agent state — but the fix is the same discipline.

**Check yourself before Stage 2:** Why does `input()` fail as an approval mechanism for anything beyond a quick local script?

---

## Stage 2 — A durable pending-action record

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


@dataclass
class PendingAction:
    run_id: str
    tool_name: str
    args: dict
    risk: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    edited_args: dict | None = None
    rejection_reason: str | None = None
```

This is deliberately structured just like the `AgentState` from Part IX — a record of everything needed to pick up exactly where things left off, now specifically about a decision awaiting a human, rather than an agent's step-by-step progress.

```python
class ApprovalStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, action: PendingAction) -> None:
        data = {
            "run_id": action.run_id, "tool_name": action.tool_name, "args": action.args,
            "risk": action.risk, "status": action.status.value,
            "edited_args": action.edited_args, "rejection_reason": action.rejection_reason,
        }
        path = self._path(action.run_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)   # atomic write — the exact Part IX pattern

    def load(self, run_id: str) -> PendingAction | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return PendingAction(
            run_id=data["run_id"], tool_name=data["tool_name"], args=data["args"],
            risk=data["risk"], status=ApprovalStatus(data["status"]),
            edited_args=data.get("edited_args"), rejection_reason=data.get("rejection_reason"),
        )
```

Notice the `save` method uses exactly the atomic temp-file-then-rename pattern from [Part IX Stage 3](part09-agent-state.md). Same reason: if the process crashes mid-write to this file, the previous valid state must survive intact, not get corrupted — and a pending approval that a human might be about to act on is exactly the kind of data you can't afford to corrupt.

**Check yourself before Stage 3:** Why does `ApprovalStore.save` need the exact same atomic-write pattern as your Part IX checkpoint store?

---

## Stage 3 — Propose, decide, resume — as three genuinely separate steps

The key design decision: proposing an action, a human deciding on it, and actually executing it are three separate operations, potentially separated by a long real-world gap (minutes, hours, overnight), and potentially running as three separate process invocations.

```python
RISK_REQUIRES_APPROVAL = {"high"}


def propose_action(store: ApprovalStore, run_id: str, tool_name: str, args: dict, risk: str) -> str:
    if risk not in RISK_REQUIRES_APPROVAL:
        return "auto_approved"   # low-risk tools, from Part XXXIII's classification, skip this entirely
    action = PendingAction(run_id=run_id, tool_name=tool_name, args=args, risk=risk)
    store.save(action)
    return "pending_approval"


def human_decide(store: ApprovalStore, run_id: str, decision: ApprovalStatus,
                  edited_args: dict | None = None, reason: str | None = None) -> None:
    action = store.load(run_id)
    if action is None:
        raise ValueError(f"no pending action for run {run_id}")
    action.status = decision
    action.edited_args = edited_args
    action.rejection_reason = reason
    store.save(action)
```

`propose_action` only durably persists something if the action's risk actually requires approval — reusing exactly the `RiskLevel`-driven decision from [Part XXXIII](part33-tool-engineering.md), so low-risk tools never even touch this machinery. `human_decide` is deliberately a completely separate function — in a real system, it would be called from a different process entirely, likely hours later, triggered by a human clicking "approve" in some UI, with zero shared state with whatever process originally called `propose_action` beyond the durably-saved `PendingAction` itself.

**Check yourself before Stage 4:** Why must `propose_action` and `human_decide` be safely callable from entirely separate process invocations, potentially far apart in time?

---

## Stage 4 — The detail that actually matters: resuming with the *edited* version

Here's the subtlety that's easy to get wrong, and important enough to be the reason this part exists rather than a two-line "ask before doing it" function. A human reviewing a proposed action might not just approve or reject it — they might want to **edit** it first (change a dollar amount, fix a recipient's email address) and then approve that edited version.

```python
def resume_after_approval(store: ApprovalStore, run_id: str, execute_tool_fn) -> dict:
    action = store.load(run_id)
    if action is None or action.status == ApprovalStatus.PENDING:
        raise RuntimeError("action still pending — cannot resume yet")

    if action.status == ApprovalStatus.REJECTED:
        return {"executed": False, "reason": action.rejection_reason}

    # THE CRITICAL LINE: use the edited args if this was an edit, not the
    # original proposed args. Getting this backwards silently discards the
    # human's correction and executes what they explicitly changed away from.
    final_args = action.edited_args if action.status == ApprovalStatus.EDITED else action.args

    return {"executed": True, "tool": action.tool_name, "args": execute_tool_fn(action.tool_name, final_args)}
```

If this logic used `action.args` unconditionally — ignoring `edited_args` even when the status is `EDITED` — a human's explicit correction would be silently discarded, and the *original*, uncorrected action would execute instead. That's a genuinely dangerous bug: a human specifically changed the dollar amount because the original was wrong, approved their correction, and the system executed the wrong amount anyway. This is exactly why the field exists and exactly why it must be checked correctly.

**Try it:** Propose an action, call `human_decide` with `ApprovalStatus.EDITED` and different `edited_args`, then resume — confirm the executed arguments are the *edited* ones, not the original proposal. Then deliberately introduce the bug (always use `action.args`) and watch the edited correction get silently ignored. Fix it back.

**Check yourself before the project:** What specifically goes wrong if `resume_after_approval` always executes `action.args`, ignoring `edited_args` even when the status is `EDITED`?

---

# The Project: An Agent That Requires Human Approval Before Dangerous Actions

## What you're building

Wire the full approval flow into your Part XXVIII harness — any tool flagged high-risk by Part XXXIII's `requires_human_approval` pauses durably, survives a restart while waiting, and correctly resumes with any human edits applied.

```bash
$ python approval_agent.py "delete inactive customer records older than 2 years"
```

```
[step 1] tool_call: query_table(table="customers", where_column="status", where_value="inactive")
[step 1] found 47 matching records

[step 2] tool_call: delete_records(table="customers", where="status=inactive AND age>2y")
[step 2] HIGH RISK — requires approval. Paused. run_id=run-a1b2c3
```

```bash
# process can be killed and restarted here — the pending approval survives
$ python approve.py run-a1b2c3 --edit 'where=status=inactive AND age>3y' --approve
```

```
approved with edits.
```

```bash
$ python approval_agent.py --resume run-a1b2c3
```

```
[step 2] resuming with EDITED args: where="status=inactive AND age>3y"
[step 2] executed: deleted 12 records (not the original 47 — the human tightened the filter)

done.
```

## Why this project

This is your first fully hand-built human-in-the-loop system, and it directly demystifies what `interrupt_before` was doing for you automatically back in Part XX — you now understand every piece of the mechanism it wraps.

## Build it in stages

---

### Stage A — `ApprovalStore`, tested for durability

**Goal:** Save, load, and atomic-write correctness, exactly as in Part IX's checkpoint testing discipline.

**Do:**
1. Implement `ApprovalStore` as shown
2. Save a pending action, kill your test script mid-save (add a deliberate `time.sleep` before `.replace()`, interrupt during it), confirm the file remains in its previous valid state — never corrupted

**You're done when:** You've watched the atomic-write guarantee hold under a deliberately timed interruption, the same test discipline from Part IX Stage 3.

---

### Stage B — Wire `propose_action` into your harness

**Goal:** A high-risk tool call in your Part XXVIII harness pauses instead of executing immediately.

**Do:**
1. Use `requires_human_approval` (Part XXXIII) to decide whether a given tool call needs this flow
2. On a high-risk call, save a `PendingAction` and stop the agent run, reporting the `run_id`
3. Confirm a low-risk tool call skips this entirely and executes immediately, as before

**You're done when:** You can trigger both paths — a low-risk call executing immediately, and a high-risk call pausing with a `run_id` you can act on later.

---

### Stage C — A separate approval CLI

**Goal:** `human_decide`, callable as its own separate command, completely decoupled from the agent process.

**Do:**
1. Build a small CLI script accepting a `run_id` and a decision (approve, reject with reason, or approve with edited args)
2. Run it against a pending action created by a *different, already-finished* process invocation of your agent script
3. Confirm the decision is correctly persisted

**You're done when:** You've approved (and separately, rejected) a pending action from a genuinely separate process invocation, proving the decoupling actually works, not just in theory.

---

### Stage D — Resume, with the edit-aware logic

**Goal:** `resume_after_approval`, correctly using edited args when present.

**Do:**
1. Implement it exactly as shown, with the critical `edited_args` check
2. Test the full cycle from the example output: propose, approve-with-edits via the separate CLI, resume, and confirm the *edited* arguments were what actually executed
3. Also test the reject path: propose, reject via the CLI, resume, confirm nothing executes and the rejection reason is reported

**You're done when:** All three outcomes — approved-as-is, approved-with-edits, rejected — resume correctly, with edits genuinely taking effect when present.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Resumed action executes with the original args despite being edited | The critical bug from Stage 4 — `resume_after_approval` not checking `status == EDITED` before choosing which args to use | Trace through the exact conditional and confirm it matches Stage 4's logic |
| Pending action disappears after a process restart | `ApprovalStore` writing to a temp path but not actually calling `.replace()`, or writing to the wrong directory | Confirm the final file actually exists on disk after `save()`, independent of any in-memory state |
| Approval CLI can't find the pending action | `run_id` mismatch between the agent process and the approval CLI invocation | Print and double-check the exact `run_id` string is passed identically in both places |

## Going further (optional)

1. **An expiration/escalation timeout.** If a pending approval sits unresolved for more than some threshold, have a separate process detect this and notify a different approver, rather than leaving it silently stuck forever.
2. **Multi-level approval.** For an especially high-risk action, require two separate approvals before resuming — extend `PendingAction` to track multiple required approvers.
3. **A simple web UI** (even a minimal Flask page) replacing the CLI for the approval step — the underlying `ApprovalStore` mechanism doesn't need to change at all, only how a human interacts with it.

---

## Checkpoint

**Knowledge:**
1. Why does a durable, file-or-database-backed `PendingAction` succeed where a simple `input()` prompt fails?
2. Why must proposing an action and deciding on it be safely separable across different process invocations?
3. What specifically goes wrong if resumption always uses the original proposed arguments, ignoring any edits?

**Implementation:** Build `ApprovalStore`, `propose_action`, `human_decide`, and the edit-aware `resume_after_approval` from memory.

**Debugging:** A human edits a proposed action's arguments through your approval UI, but the executed action still used the original, unedited arguments. What's the bug, and exactly where would you look for it?

**Architecture:** Design an approval system for a platform handling hundreds of daily agent actions, where only a small fraction are genuinely high-risk. How would you avoid approval fatigue — humans starting to rubber-stamp requests without real review — while still gating the truly risky ones?

---

**Next:** [Part XXXV — Security](part35-security.md), where the untrusted-content handling from Part V, the sandboxing from Part XXIV, and the permission scoping from this part all come together into a systematic threat model.
