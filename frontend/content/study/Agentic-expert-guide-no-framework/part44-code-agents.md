# Part XLIV — Code Agents

## Before you start

**What you need already:** [Part XXVII Stage 3](part27-multi-agent-systems.md) (the coder/tester fix cycle — you already built a version of this), [Part XXXI](part31-reasoning.md) (tool-assisted verification — running real tests is the definitive example of this), [Part XXIV](part24-mcp-server-development.md) (sandboxing).

**What this part gives you:** A real coding agent — repository understanding, planning a change, sandboxed modification, and a verification loop that trusts actual test execution over the model's own claim that its code works.

**Time:** 3 days.

**When you finish, you'll be able to:** Build an agent that finds the right place to make a change in a real codebase, and verifies its own work against reality rather than its own say-so.

---

## Stage 1 — Recall exactly what you already built

[Part XXVII Stage 3](part27-multi-agent-systems.md)'s `run_swe_team`: coder writes code, tester checks it, failures feed back into another coder attempt, bounded. That's the core loop of a code agent. This part fills in the two pieces that were stubbed out there: **how do you actually find the right place in a real codebase to make a change**, and **how do you run real tests instead of a fake string check**.

**Check yourself before Stage 2:** In Part XXVII's `run_swe_team`, what was `tester` actually checking, and why was that a stand-in rather than a real verification?

---

## Stage 2 — Finding the right place: symbol extraction, not character-count chunking

You already know, from [Part XIII](part13-chunking.md), that fixed-size chunking regularly splits meaningful units (a paragraph, a table row) in half. Code has exactly the same problem, but worse — a function split in half by a character-count chunker isn't just awkwardly divided, it's syntactically broken. Code needs chunking that respects its actual structure: functions and classes, not arbitrary character counts.

```python
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Symbol:
    name: str
    kind: str   # "function" | "class"
    file: str
    line: int


def extract_symbols(file_path: Path) -> list[Symbol]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.append(Symbol(node.name, "function", str(file_path), node.lineno))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(node.name, "class", str(file_path), node.lineno))
    return symbols
```

Python's built-in `ast` module parses source code into its real syntactic structure — the same tree structure a Python interpreter itself uses to understand your code. `ast.walk` traverses that tree, and checking `isinstance(node, ast.FunctionDef)` finds every function definition *exactly*, by the language's own grammar — never accidentally splitting one in half, because it's working with the code's actual structure, not counting characters.

```python
def build_repo_index(root: Path) -> dict[str, list[Symbol]]:
    index: dict[str, list[Symbol]] = {}
    for py_file in root.rglob("*.py"):
        try:
            symbols = extract_symbols(py_file)
        except SyntaxError:
            continue
        for s in symbols:
            index.setdefault(s.name, []).append(s)
    return index
```

This gives you the code-specific equivalent of the chunking discipline from Part XIII: a retrievable index of *coherent, complete units* (whole functions and classes), never fragments — the direct analogue of never splitting a fact mid-sentence, now applied to "never splitting a function mid-body."

**Check yourself before Stage 3:** Why does AST-based extraction guarantee a function is never split, in a way that character-count chunking structurally cannot guarantee?

---

## Stage 3 — The critical distinction: the model's claim vs. reality

Here's the single most important idea in this whole part, and it's a direct, concrete instance of [Part XXXI](part31-reasoning.md)'s tool-assisted verification. When a code agent finishes and says "I fixed the bug, the tests pass" — that sentence is generated text. It is not evidence. The only evidence that the code actually works is **actually running the tests and reading the real exit code.**

```python
import subprocess


def apply_patch(file_path: str, new_content: str) -> None:
    Path(file_path).write_text(new_content, encoding="utf-8")


def run_tests(test_path: str = "tests/") -> tuple[bool, str]:
    result = subprocess.run(
        ["pytest", test_path, "-x", "--tb=short"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0, result.stdout + result.stderr
```

`result.returncode == 0` is the ground truth — a real, deterministic signal from actually executing the code, completely independent of anything the model claims about its own output. This is exactly [Part XXXI](part31-reasoning.md)'s calculator-verification idea, applied at a larger scale: instead of trusting a claimed arithmetic result, you compute the real one; here, instead of trusting a claimed "tests pass," you actually run them and read the real exit code.

**Check yourself before Stage 4:** Why is a model's own statement "the tests pass" not evidence of anything, in the same way a model's claimed arithmetic answer wasn't trustworthy on its own in Part XXXI?

---

## Stage 4 — The full loop, bounded exactly like every other loop in this curriculum

```python
from dataclasses import dataclass


@dataclass
class PatchAttempt:
    file_path: str
    test_result: str | None = None


def code_agent_loop(file_path: str, generate_fix_fn, max_attempts: int = 4) -> PatchAttempt:
    attempt_num = 0
    last_failure = None

    while attempt_num < max_attempts:
        new_content = generate_fix_fn(file_path, last_failure)
        apply_patch(file_path, new_content)
        passed, output = run_tests()

        if passed:
            return PatchAttempt(file_path=file_path, test_result="pass")

        last_failure = output   # the REAL test output feeds the next attempt
        attempt_num += 1

    return PatchAttempt(file_path=file_path, test_result=f"failed after {max_attempts} attempts")
```

Same bounded-loop shape you've built repeatedly since Part VII — this time, the "check" step is real subprocess execution, not the model's opinion, and the feedback fed into the next attempt (`last_failure`) is the actual captured test output, giving the model concrete, real information about what specifically went wrong rather than a vague "try again."

**Check yourself before Stage 5:** Why does feeding the *real* test output (`output`) into the next attempt matter, rather than just telling the model "that attempt failed, try again"?

---

## Stage 5 — This absolutely must run sandboxed, and never on the developer's real working tree directly

One safety note that isn't optional. `apply_patch` in Stage 3 writes directly to a real file path with no isolation. In a real system, this loop must run inside a container or an isolated git worktree — never directly against a developer's actual working directory — for the same reason [Part XXIV](part24-mcp-server-development.md)'s filesystem tools were sandboxed: a model-generated patch, or a model-generated *test* that itself has side effects (deleting files, making network calls), executing with the same privileges as the developer running the agent is a real, serious risk, not a theoretical one.

```python
import subprocess


def create_review_branch(base: str, branch_name: str) -> None:
    subprocess.run(["git", "checkout", "-b", branch_name, base], check=True)


def generate_diff() -> str:
    result = subprocess.run(["git", "diff"], capture_output=True, text=True)
    return result.stdout
```

Working on a disposable branch inside a container means the worst-case outcome of a bad or malicious patch is confined to something you can simply discard — not something that touched the developer's real, unbacked-up working files.

**Check yourself before the project:** Why is running `apply_patch` directly on a developer's real working directory a genuine security concern, not just a style preference?

---

# The Project: A Sandboxed, Verified, Bounded Coding Agent

## What you're building

Combine symbol extraction, the bounded fix loop with real test verification, git-branch isolation, and — connecting back to Part XXXIV — a final human-approval gate before any patch merges anywhere real.

```bash
$ python code_agent.py --repo ./my_project --task "fix the off-by-one bug in paginate()"
```

```
building repo index... found 42 functions across 8 files
located relevant symbol: paginate() in utils/pagination.py:14
created isolated branch: agent-fix-a1b2c3

[attempt 1] generated patch, running tests...
[attempt 1] FAILED: test_paginate_last_page (AssertionError: expected 3 items, got 4)
[attempt 2] generated revised patch, running tests...
[attempt 2] PASSED: all 12 tests green

diff ready for review:
--- utils/pagination.py ---
- return items[start:start+size]
+ return items[start:start+size] if start < len(items) else []

PENDING HUMAN APPROVAL before merge. Approve? [y/n]:
```

## Why this project

This is a direct, concrete composition of nearly everything you've built: Part XIII-style structural chunking (for code), Part XXXI's tool-assisted verification, Part XXIV's sandboxing, Part XXVII's bounded fix cycle, and Part XXXIV's human approval — all in one working system.

## Build it in stages

---

### Stage A — Real repo indexing

**Goal:** `build_repo_index`, run against a real (even small) Python codebase, with correct symbol extraction.

**Do:**
1. Implement `extract_symbols` and `build_repo_index`
2. Run it against a real project (your own past project, or a small open-source one)
3. Given a function name, confirm you can locate its exact file and line

**You're done when:** You can search for any function in your test repo and get back its precise location.

---

### Stage B — Isolated, sandboxed execution

**Goal:** Every patch attempt happens on a disposable git branch, never the real working tree directly.

**Do:**
1. Implement `create_review_branch`
2. Confirm patches are applied and tested only on the isolated branch
3. Confirm you can discard the branch entirely with zero effect on the original code

**You're done when:** You've verified, by actually checking, that a bad or failed patch attempt never touched anything outside the disposable branch.

---

### Stage C — The bounded fix loop, with real tests

**Goal:** `code_agent_loop`, wired to a real LLM generating patches and real `pytest` execution.

**Do:**
1. Set up a small test repo with a deliberate, real bug and a test suite that catches it
2. Run the loop, with a real LLM generating fix attempts based on real test failure output
3. Confirm it converges to a genuinely passing state within `max_attempts`, or honestly reports failure if it doesn't

**You're done when:** You've watched a real bug get fixed through 1+ real fix-and-retest cycles, verified by actual test execution, not by trusting the model's claim.

---

### Stage D — Human approval before merge

**Goal:** A passing patch doesn't automatically merge — it requires explicit human sign-off, reusing Part XXXIV's approval pattern.

**Do:**
1. On a passing test result, generate and display the diff
2. Prompt for approval before merging the branch back
3. Confirm declining leaves the disposable branch untouched and unmerged

**You're done when:** You've run the full cycle end to end, including both an approve and a decline path.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Fix loop "succeeds" but the patch is clearly wrong on manual inspection | Test suite doesn't actually cover the bug being fixed | Strengthen the test suite before trusting the loop's pass/fail signal — the verification is only as good as the tests |
| Loop never converges within `max_attempts` on a genuinely simple bug | Real test failure output isn't actually being fed back into the next generation attempt | Confirm `last_failure` is populated with the real captured output, not a placeholder |
| A test with a real side effect (file write, network call) causes unexpected damage during testing | Running tests without sandboxing at all, not just the patch application | Run the *whole* test execution, not just the patch write, inside your isolated/sandboxed environment |

## Going further (optional)

1. **Static analysis as a pre-test gate** — run a linter before the full test suite, failing fast (and cheaply) on syntax or obvious style errors before paying the cost of a full test run, mirroring [Part XL](part40-cost-optimization.md)'s cheap-classifier-first pattern.
2. **A container, not just a git branch**, for full filesystem and process isolation — if you have Docker set up, run the actual test execution inside a container with no network access, for genuine sandboxing rather than just source-control isolation.
3. **Multiple candidate patches per attempt** — generate 2–3 different fix attempts in parallel (reusing Part XXVI's fan-out) and pick whichever one passes tests first, rather than one attempt at a time sequentially.

---

## Checkpoint

**Knowledge:**
1. Why does AST-based symbol extraction guarantee a function is never split, in a way character-count chunking can't?
2. Why is a model's own claim "the tests pass" not evidence, and what's the actual evidence instead?
3. Why must patch application and test execution both happen in a sandboxed, isolated environment, not the developer's real working tree?

**Implementation:** Build `extract_symbols`, the bounded fix loop with real test verification, and git-branch isolation from memory.

**Debugging:** A code agent reports "all tests pass" but the reported fix clearly doesn't address the actual bug on manual review. What verification gap does this suggest, and where would you look first?

**Architecture:** Design sandboxing and review-gate architecture for a coding agent with write access to a production codebase used by a real team — what layers of isolation and approval would you require before any agent-generated change reaches the real repository?

---

**Next:** [Part XLV — Browser Agents](part45-browser-agents.md), where an agent's environment becomes a real, live website — with all the untrusted-content risk that implies, at a larger scale than any single document.
