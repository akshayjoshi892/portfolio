# Part XXIV — MCP Server Development

## Before you start

**What you need already:** [Part XXIII](part23-mcp.md) — a working MCP server/client pair with pure-calculation tools. This part builds servers with real stakes: filesystem access, database queries, tools that can genuinely damage something if misused.

**What this part gives you:** The specific defensive patterns every real MCP server needs — because unlike the calculator from Part XXIII, these tools touch real resources, and a validation gap here has real consequences.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Build a filesystem tool that can't escape its sandbox and a database tool that can't be used for injection, and explain exactly why each defense works.

---

## Stage 1 — The attack you're defending against: path traversal

Suppose you build a "read a file" tool for an MCP server, meant to let an agent read files within one specific project folder. The naive version:

```python
def read_file_naive(relative_path: str) -> str:
    with open(relative_path) as f:
        return f.read()
```

This looks reasonable. Now watch what a model — confused, manipulated by a prompt injection, or just given a slightly wrong argument — can do with it:

```python
read_file_naive("../../../../etc/passwd")
```

Nothing in `read_file_naive` stops this. `../` means "go up one directory," and there's nothing preventing the model from chaining enough of them together to walk right out of your intended project folder and into anywhere else on the filesystem. This is called **path traversal**, and it's one of the oldest, most common real vulnerabilities in any tool that touches a filesystem — not unique to AI agents at all, but AI agents make it easier to trigger accidentally, because the "attacker" providing the path might just be a confused model rather than a deliberate human attacker.

**Try it, safely:** In a throwaway test directory, confirm `open("../../../../etc/passwd")` (or an equivalent path relevant to your OS) actually does escape your intended folder. Seeing this work is what makes the fix in the next stage feel necessary rather than paranoid.

**Check yourself before Stage 2:** Why doesn't the naive `read_file_naive` function stop a path like `"../../../../etc/passwd"` from working?

---

## Stage 2 — The fix: resolve, then check the result is still inside the sandbox

The fix isn't "detect and block `..`" — that's fragile and easy to bypass with encoding tricks or symlinks. The real fix: resolve the path to its true, absolute location, and check whether that resolved location is still inside your intended directory, *after* resolution, not before.

```python
from pathlib import Path


class SecurityError(Exception):
    pass


SANDBOX_ROOT = Path("./sandbox").resolve()
SANDBOX_ROOT.mkdir(exist_ok=True)


def resolve_safe_path(relative_path: str) -> Path:
    candidate = (SANDBOX_ROOT / relative_path).resolve()
    if not str(candidate).startswith(str(SANDBOX_ROOT)):
        raise SecurityError(f"path escapes sandbox: {relative_path!r}")
    return candidate


def read_file(relative_path: str) -> str:
    path = resolve_safe_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {relative_path}")
    return path.read_text(encoding="utf-8")
```

`.resolve()` is doing the critical work: it collapses every `..` and symlink down to the actual, final, absolute path the operating system would use. Only *after* that collapsing happens do you check whether the result is still inside `SANDBOX_ROOT`. This is safe against `../../../etc/passwd`-style attacks precisely because you're checking the *destination*, not the *route* — no amount of `..` trickery in the input can produce a resolved path that's actually inside the sandbox if the real target isn't.

**Try it again:** Call `resolve_safe_path("../../../../etc/passwd")` against your sandbox and confirm you now get a clean `SecurityError` instead of a successful escape.

**Check yourself before Stage 3:** Why does checking the *resolved, absolute* path work reliably, when checking the raw input string for `..` would not?

---

## Stage 3 — Build the write side, and the danger it adds

```python
def write_file(relative_path: str, content: str) -> str:
    path = resolve_safe_path(relative_path)   # same sandbox check
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {relative_path}"
```

The sandbox check protects against escaping the intended directory, but notice `write_file` silently **overwrites** any existing file at that path with no warning. If a model calls this tool intending to *append* a note but actually overwrites an important existing document, that's real, avoidable data loss — and it happened because the tool's behavior wasn't clearly communicated.

This is exactly the "tool description matters" lesson from [Part XXIII Stage 7](part23-mcp.md), now with real consequences: the tool's docstring/description, which becomes part of what the model sees when deciding how to use it, must clearly state that this operation overwrites. A vague description ("writes a file") invites exactly this kind of accidental destructive use.

**Check yourself before Stage 4:** Why does a tool's description matter for preventing accidental misuse, not just for helping the model choose the right tool?

---

## Stage 4 — A different kind of danger: database tools and injection

Filesystem tools need path validation. Database tools need a related but distinct defense. Here's the naive version, and why it's dangerous:

```python
def query_table_naive(table: str, where_clause: str) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(f"SELECT * FROM {table} WHERE {where_clause}")
    return [dict(zip([c[0] for c in cursor.description], row)) for row in cursor.fetchall()]
```

If a model (again — confused, manipulated, or just wrong) passes `where_clause = "1=1; DROP TABLE users; --"`, and your database driver allows multiple statements, you've just given an LLM the ability to delete an entire table through a natural-language conversation. This is **SQL injection**, applied to an agentic context — the same decades-old vulnerability class, with a new, more unpredictable source of malicious-or-accidental input.

### The fix: never let the model construct SQL structure directly

```python
ALLOWED_TABLES = {"orders", "customers"}
ALLOWED_COLUMNS = {"id", "status", "email"}


def query_table(table: str, where_column: str, where_value: str, limit: int = 20) -> list[dict]:
    import sqlite3

    if table not in ALLOWED_TABLES:
        raise PermissionError(f"table not allowed: {table!r}")
    if where_column not in ALLOWED_COLUMNS:
        raise PermissionError(f"column not allowed: {where_column!r}")

    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    # table/column names come from a fixed allowlist above — never string-formatted
    # from arbitrary input. Only the VALUE is parameterized, safely, by the driver.
    cursor = conn.execute(
        f"SELECT * FROM {table} WHERE {where_column} = ? LIMIT ?",
        (where_value, min(limit, 100)),
    )
    return [dict(row) for row in cursor.fetchall()]
```

Notice two separate defenses doing two separate jobs. **Table and column names** can never be safely parameterized by a database driver — they're structural parts of the query, not values — so they're checked against a fixed allowlist instead, exactly like the character-allowlist check on the calculator's expression in Part VI. **The actual value** being searched for (`where_value`) *is* safely parameterized using `?` placeholders, which the database driver handles correctly — it can never be interpreted as part of the query's structure, no matter what's in it.

This is the same principle from Stage 2, generalized: **never let untrusted input control the *structure* of a dangerous operation — only its *values*, and even then, only through a mechanism (like parameterized queries) specifically designed to keep values from being reinterpreted as structure.**

**Check yourself before Stage 5:** Why can't table and column names be safely parameterized the same way a value like `where_value` can be? What would go wrong if you tried?

---

## Stage 5 — Auditing and rate-limiting: knowing what happened, and slowing down abuse

Two more concerns that matter specifically for tools with real side effects.

```python
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    max_calls: int
    window_s: float
    _calls: list[float] = field(default_factory=list)

    def check(self) -> None:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.window_s]
        if len(self._calls) >= self.max_calls:
            raise PermissionError("rate limit exceeded — too many write operations recently")
        self._calls.append(now)


write_limiter = RateLimiter(max_calls=10, window_s=60)


def audited_write_file(relative_path: str, content: str, actor: str) -> str:
    write_limiter.check()
    result = write_file(relative_path, content)
    print(f"AUDIT: actor={actor} action=write_file path={relative_path} bytes={len(content)}")
    return result
```

The rate limiter protects against a misbehaving or manipulated agent spamming write operations far faster than any legitimate use case would need — a mistake caught by a simple threshold rather than any deep understanding of what "legitimate" looks like. The audit log doesn't prevent anything by itself, but it's the difference between "we have no idea what happened" and "we can see exactly who did what, when" if something later needs investigating — a distinction that matters a great deal once something does go wrong.

**Check yourself before the project:** Why does a rate limiter provide real protection even without understanding whether any specific call is "legitimate" or not?

---

# The Project: A Sandboxed Filesystem and Database MCP Server

## What you're building

Extend your Part XXIII server with a real filesystem tool (path-traversal-proof) and a real database tool (injection-proof), both rate-limited and audited — then deliberately try to break each one.

```bash
$ python secure_server_test.py
```

```
read_file("notes.txt") -> "meeting notes..."
read_file("../../../etc/passwd") -> BLOCKED: path escapes sandbox
write_file("notes.txt", "updated") -> wrote 7 chars
write_file x 11 in 10 seconds -> BLOCKED on 11th: rate limit exceeded

query_table("orders", "status", "shipped") -> [3 rows]
query_table("orders; DROP TABLE orders; --", ...) -> BLOCKED: table not allowed
```

## Why this project

Every real MCP server you build from here on inherits these patterns. This is also where [Part XXXV](part35-security.md)'s security discipline starts becoming concrete rather than abstract — you're building real defenses against real attack classes, not just reading about them.

## Build it in stages

---

### Stage A — Sandboxed filesystem tools

**Goal:** `read_file`, `write_file`, `list_files`, all confined to a sandbox directory.

**Do:**
1. Implement `resolve_safe_path`, `read_file`, `write_file` exactly as shown
2. Add `list_files(relative_dir)` following the same pattern
3. Register all three as MCP tools on your server

**You're done when:** All three work correctly for legitimate paths within the sandbox.

---

### Stage B — Break it, then confirm it's fixed

**Goal:** Prove the path-traversal defense actually works, adversarially.

**Do:**
1. Try at least 3 different traversal attempts (`../`, absolute paths, and — if you're on a system where it's easy to test — a symlink pointing outside the sandbox)
2. Confirm every one is caught with a clear `SecurityError`, not a silent failure or a successful escape

**You're done when:** You've personally watched every attempted escape get blocked.

---

### Stage C — Injection-safe database tool

**Goal:** `query_table` with table/column allowlisting and parameterized values.

**Do:**
1. Create a small SQLite database with 2 tables and some sample rows
2. Implement `query_table` exactly as shown, with a real allowlist
3. Test a legitimate query and confirm correct results

**You're done when:** Real queries against allowed tables/columns work correctly.

---

### Stage D — Attempt injection, confirm it's blocked

**Goal:** Prove the SQL injection defense works.

**Do:**
1. Attempt to pass a table name that isn't in your allowlist — confirm rejection
2. Attempt to pass a `where_value` containing SQL-injection-style content (e.g., `"1' OR '1'='1"`) — confirm it's treated as a literal string value (correctly finding nothing, or exactly what a literal match would find) rather than being interpreted as part of the query structure

**You're done when:** You've confirmed, by testing, that injection attempts are neutralized — not just trusted the parameterization to work.

---

### Stage E — Rate limiting and audit logging

**Goal:** Both wired into your write-capable tools.

**Do:**
1. Add `RateLimiter` to `write_file`
2. Add audit logging to every write-capable tool call
3. Test: call `write_file` rapidly enough to trigger the rate limit, confirm the block, then wait for the window to pass and confirm it works again

**You're done when:** You've watched the rate limit trigger and recover correctly, and you have a real audit log you could hand to someone investigating a past incident.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Path traversal still succeeds on some input | Checking the raw string for `..` instead of resolving and checking the final path | Always resolve first, then check the resolved path's prefix |
| `query_table` rejects legitimate queries | Allowlist too narrow, or comparing table/column names case-sensitively when input might vary in case | Normalize case before checking against the allowlist, or document that it's case-sensitive |
| Rate limiter blocks legitimate rapid use | Threshold set too low for real usage patterns | Base the threshold on actual expected legitimate call frequency, not an arbitrary guess |
| Audit log missing entries for some failure paths | Logging placed only on the success path | Log both successful and blocked/failed attempts — the blocked ones are often the more important record |

## Going further (optional)

1. **A read-only vs. read-write distinction at the server level** — expose two separate tool sets, and think through why a "read-only mode" deployment of this server might be safer for lower-trust callers even before you get to the full permission system in Part XXXV.
2. **Symlink-specific testing**, if your OS makes this easy — create a symlink inside the sandbox pointing outside it, and confirm `resolve_safe_path` still catches the escape (it should, since `.resolve()` follows symlinks before you check the prefix).
3. **A second database tool for writes** (`insert_row`, similarly allowlisted and parameterized) and think through what additional risk a write tool carries compared to `query_table`'s read-only risk.

---

## Checkpoint

**Knowledge:**
1. Why does checking the resolved, absolute path defend against path traversal, when checking the raw input string doesn't?
2. Why can't table/column names be parameterized the same way values can, and what defense do they need instead?
3. What does a rate limiter protect against, without needing to understand whether any individual call is legitimate?

**Implementation:** Build `resolve_safe_path` and the allowlisted `query_table` function from memory.

**Debugging:** A filesystem MCP tool successfully reads a file outside its intended sandbox directory. What's the exact class of bug, and how would you verify a fix actually closes it?

**Architecture:** Design permission tiers (read-only, read-write, admin) for an MCP server exposed to multiple internal teams with different trust levels.

---

**Next:** [Part XXV — Agent + MCP](part25-agent-plus-mcp.md), where an agent connects to multiple MCP servers at once, discovering and routing across all of them dynamically.
