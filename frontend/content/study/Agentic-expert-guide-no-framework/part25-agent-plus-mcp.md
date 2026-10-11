# Part XXV — Agent + MCP

## Before you start

**What you need already:** [Part VII](part07-what-is-an-agent.md) (the bounded agent loop), [Part XXIII](part23-mcp.md)–[XXIV] (a working MCP server/client, with real security-conscious tools).

**What this part gives you:** Connecting your Part VII agent loop to *multiple* MCP servers at once, discovering all their tools dynamically, and routing calls to the right one — plus the trust question that comes with connecting to servers you didn't write.

**Time:** 2 days.

**When you finish, you'll be able to:** Run an agent that discovers and uses tools from several independent MCP servers simultaneously, with proper timeout, error, and trust handling for each.

---

## Stage 1 — What's actually new here

You already know: a tool-calling loop ([Part VI](part06-structured-outputs-tool-calling.md)), and a single MCP client talking to a single server ([Part XXIII](part23-mcp.md)). This part combines them with one new wrinkle: **more than one server, connected at the same time, with tools from all of them offered to the model together as if they were one unified toolset.**

This introduces exactly one genuinely new problem: what happens if two different servers happen to expose a tool with the same name?

**Check yourself before Stage 2:** Why might two independently-developed MCP servers happen to both expose a tool called, say, `search`?

---

## Stage 2 — Connect to several servers, and reject name collisions on sight

```python
from __future__ import annotations
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]


class MultiServerToolRegistry:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tool_to_server: dict[str, str] = {}
        self._exit_stack = AsyncExitStack()

    async def connect_all(self, configs: list[MCPServerConfig]) -> None:
        for cfg in configs:
            params = StdioServerParameters(command=cfg.command, args=cfg.args)
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[cfg.name] = session

            tools = await session.list_tools()
            for t in tools.tools:
                if t.name in self.tool_to_server:
                    raise ValueError(
                        f"tool name collision: {t.name!r} is exposed by both "
                        f"{self.tool_to_server[t.name]!r} and {cfg.name!r}"
                    )
                self.tool_to_server[t.name] = cfg.name

            print(f"connected to {cfg.name}: {[t.name for t in tools.tools]}")

    async def close(self) -> None:
        await self._exit_stack.aclose()
```

Two things worth understanding about the shape of this code. First, `AsyncExitStack` — it's a way to hold onto multiple `async with` contexts (one per server connection) and guarantee all of them get cleanly closed together, in reverse order, even if something fails partway through connecting. Without it, you'd need deeply nested `async with` blocks, one per server, which becomes unmanageable past 2 or 3 servers.

Second, and more important: notice the collision check **raises immediately** rather than trying to be clever about resolving it (like "last server wins" or "prefix with server name"). This is deliberate. A name collision between two independently-developed servers is a genuine configuration problem — you cannot know, automatically, which `search` tool the model actually meant when it says "call search." Silently picking one is worse than failing loudly, because it produces a plausible-looking but potentially wrong tool call with no visible sign anything went wrong.

**Check yourself before Stage 3:** Why is raising an error on a tool name collision better than picking one server automatically ("last one wins")?

---

## Stage 3 — Route a call, with the same error handling discipline as Part VI

```python
    async def call(self, tool_name: str, args: dict, timeout_s: float = 5.0) -> dict:
        server_name = self.tool_to_server.get(tool_name)
        if server_name is None:
            return {"error": f"unknown tool: {tool_name!r}"}

        session = self.sessions[server_name]
        try:
            result = await asyncio.wait_for(session.call_tool(tool_name, args), timeout=timeout_s)
            return {"result": result.content}
        except asyncio.TimeoutError:
            return {"error": f"{tool_name} on {server_name} timed out after {timeout_s}s"}
        except Exception as e:
            return {"error": f"{tool_name} on {server_name} failed: {e}"}
```

Recognize this immediately: it's your `ToolRegistry.call` from [Part VI Stage 7](part06-structured-outputs-tool-calling.md), applied across a network boundary. Same timeout wrapping, same "return structured `{"error": ...}` instead of crashing" discipline. The only difference is `tool_to_server` lookup replaces a single local dictionary. Nothing about the reliability principles changed just because the tool now lives in a different process.

**Check yourself before Stage 4:** What would happen to your agent loop if `call()` here raised an exception directly instead of returning `{"error": ...}`, given what you know from Part VI?

---

## Stage 4 — Wire it into your agent loop

```python
async def run_agent_with_mcp(registry: MultiServerToolRegistry, goal: str, llm_call, max_steps: int = 6):
    available_tool_names = list(registry.tool_to_server.keys())
    print(f"agent has access to: {available_tool_names}")

    step = 0
    messages = [{"role": "user", "content": goal}]

    while step < max_steps:
        decision = await llm_call(messages, available_tool_names)   # your Part VII decision logic

        if decision["type"] == "final":
            return decision["content"]

        outcome = await registry.call(decision["name"], decision["args"])
        messages.append({"role": "tool", "content": outcome})
        step += 1

    return "STOPPED: max_steps reached"
```

This is exactly Part VII's loop. The only thing that changed is where tool execution actually happens — instead of calling a local Python function directly, `registry.call()` routes over the network to whichever server owns that tool. From the agent loop's point of view, nothing is different: it doesn't know or care whether a tool lives in-process or on a remote server, because the interface — schema in, structured result or error out — is identical either way.

**Check yourself before Stage 5:** What had to change in your Part VII agent loop to support MCP-sourced tools? What stayed exactly the same?

---

## Stage 5 — Trust tiers: not every server deserves the same treatment

You connected to a filesystem server and a database server in Part XXIV — both servers you wrote yourself, with sandboxing and allowlisting already built in. But real systems often want to connect to servers written by *other* teams, or even third parties. That changes the risk calculus, and it's worth being explicit about it rather than treating every connected server identically.

```python
TRUST_LEVELS = {
    "internal_filesystem": "sandboxed",     # you wrote it, it's already constrained
    "internal_database": "read-only",       # you wrote it, only exposes safe queries
    "third_party_search": "untrusted",      # you didn't write it, results and descriptions
                                              # should be treated with extra scrutiny
}


def requires_human_approval(tool_name: str, server_name: str) -> bool:
    return TRUST_LEVELS.get(server_name) == "untrusted"
```

This connects directly back to two things you already know: from [Part V Stage 6](part05-prompt-engineering.md), that content from external sources should be delimited as untrusted data before it re-enters the model's context — that applies to tool *results* from an untrusted server exactly as much as to a retrieved document. And from [Part XXIII Stage 7](part23-mcp.md), that a server's tool *descriptions* are themselves untrusted input — a server you don't control could describe a dangerous tool as harmless.

You'll build the full human-approval mechanism this connects to in [Part XXXIV](part34-human-in-the-loop.md). For now, the important habit is: **know, explicitly, which of your connected servers you actually trust, and don't treat all of them the same by default.**

**Check yourself before the project:** Why does connecting to a third-party MCP server introduce risks that connecting to your own Part XXIV servers doesn't?

---

# The Project: An Agent Using Two Independent MCP Servers at Once

## What you're building

An agent that connects to both your Part XXIV filesystem server and database server simultaneously, discovers all their tools, and uses a real LLM to choose and sequence calls across both to complete a task.

```bash
$ python multi_server_agent.py "Find all customers with status='pending' and write a summary to pending_customers.txt"
```

```
connected to filesystem: ['read_file', 'write_file', 'list_files']
connected to database: ['query_table']
agent has access to: ['read_file', 'write_file', 'list_files', 'query_table']

[step 1] query_table(table="customers", where_column="status", where_value="pending")
[step 1] result: 3 rows found

[step 2] write_file(relative_path="pending_customers.txt", content="3 pending customers: ...")
[step 2] result: wrote 142 chars

done. Wrote summary of 3 pending customers to pending_customers.txt
```

## Why this project

This is the realistic shape of a production agent — most real systems need more than one tool source, and this project makes sure you can wire multiple MCP servers together correctly, with proper isolation and error handling for each.

## Build it in stages

---

### Stage A — Connect to both servers at once

**Goal:** `MultiServerToolRegistry` successfully connects to your Part XXIV filesystem and database servers simultaneously.

**Do:**
1. Implement `connect_all` with both server configs
2. Confirm both connect and their tools are correctly merged into one `tool_to_server` mapping
3. Deliberately name a tool in a test third server the same as an existing one, and confirm the collision check catches it

**You're done when:** Both servers connect cleanly, tools are discoverable from both, and the collision detection actually fires when you test it.

---

### Stage B — Real agent loop, real task requiring both servers

**Goal:** The example output above, working for real, with a real LLM choosing the tool sequence.

**Do:**
1. Wire `MultiServerToolRegistry` into your Part VII agent loop
2. Give the model a task that genuinely requires both a database tool and a filesystem tool, in sequence
3. Confirm the model correctly sequences the calls without you hardcoding the order

**You're done when:** A real, non-trivial task involving both servers completes correctly, driven by the model's own tool-selection decisions.

---

### Stage C — Force a server failure mid-task

**Goal:** Confirm one server's failure doesn't take down the whole agent.

**Do:**
1. Kill or disable one server mid-run (or simulate this by making one of its tools always raise an error)
2. Confirm the agent receives a structured error for that specific call, and can still use the *other* server's tools normally
3. Confirm the whole process doesn't crash

**You're done when:** You've watched a partial server failure get handled gracefully, isolated to just the affected tool calls.

---

### Stage D — Trust tiers, applied

**Goal:** Mark one server as lower-trust and confirm your code treats it differently.

**Do:**
1. Add a third, simple mock server (or reuse your Part XXIII calculator server) and mark it `"untrusted"` in your `TRUST_LEVELS` map
2. Implement `requires_human_approval` and confirm it correctly flags calls to the untrusted server's tools
3. (You don't need to build the full approval flow yet — just confirm the flag correctly fires; the full mechanism comes in Part XXXIV)

**You're done when:** You can point to code that treats your trusted and untrusted servers differently, not identically.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Connection to one server hangs the whole startup | Waiting on `connect_all` sequentially with no per-server timeout | Add a timeout around each individual server connection attempt |
| A tool call to server B succeeds even though server A crashed | This is actually correct — confirm it's intentional isolation, not a bug you need to fix | This is the desired behavior from Stage C; verify your understanding matches |
| Cleanup leaves orphaned subprocess servers running after the program exits | `close()` not called, or not called in a `finally` block | Always wrap your agent run in `try/finally` calling `registry.close()` |

## Going further (optional)

1. **A server registry file** — instead of hardcoding `MCPServerConfig` objects in your script, load them from a JSON/YAML config, making it easy to add or remove servers without code changes.
2. **Per-server rate limiting** — apply the `RateLimiter` pattern from Part XXIV per connected server, so one server's usage doesn't count against another's budget.
3. **A health check loop** — periodically ping each connected server and mark it unavailable (skip offering its tools) if it stops responding, rather than only discovering the failure at call time.

---

## Checkpoint

**Knowledge:**
1. Why should a tool name collision between two servers be rejected outright rather than resolved automatically?
2. What changed in your agent loop to support MCP tools, and what stayed the same as Part VII?
3. Why does a third-party MCP server warrant different trust treatment than one you wrote yourself?

**Implementation:** Build `MultiServerToolRegistry.connect_all` and `.call` with collision detection and timeout handling, from memory.

**Debugging:** An agent using 3 MCP servers appears to hang on one specific tool call. How would you isolate which server is unresponsive, without killing the whole process?

**Architecture:** Design a trust-boundary policy distinguishing servers you wrote, vetted third-party servers, and completely unknown servers — including what "requires human approval" should mean for each tier.

---

**Next:** [Part XXVI — Agent Orchestration](part26-agent-orchestration.md), where you'll go beyond a single agent's tool calls to coordinating multiple independent pieces of work — sequential, parallel, and conditional.
