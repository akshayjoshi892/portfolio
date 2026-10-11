# Part XXIII — Model Context Protocol (MCP)

## Before you start

**What you need already:** [Part VI](part06-structured-outputs-tool-calling.md) — your hand-built `ToolRegistry`, schema validation, structured error handling. This part gives that same idea a standardized wire protocol so it can work across process and organization boundaries, not just within one Python program.

**What this part gives you:** An understanding of the actual problem MCP solves (not just "how to use it"), and a working server/client pair you build and run yourself.

**Time:** 2–3 days.

**When you finish, you'll be able to:** Explain precisely why MCP exists as a protocol rather than a library, and build both sides of an MCP connection from scratch.

---

## Stage 1 — The problem: every tool integration was bespoke

Here's the situation MCP fixes. Suppose you've built a great `ToolRegistry` in Part VI. It works great — inside your one Python process. Now imagine your company has 5 different AI applications, built by 5 different teams, each wanting to use a shared database tool. Without a standard, each team writes their own custom integration code to talk to that database from their own application — 5 different, slightly incompatible implementations of essentially the same thing.

Now flip it around: the database team wants to expose their tool to *any* AI application that wants to use it, without writing custom glue code for every single consumer. Without a standard, they'd have to build and maintain 5 separate integrations too.

This is an **N × M problem**: N applications, M tools, and without a shared protocol, you need N×M custom integrations. MCP's fix is the same one used by many protocols before it (HTTP for "how do clients talk to web servers," LSP for "how do editors talk to language servers"): agree on one standard wire format, and every application that speaks it can use every tool that speaks it, with zero custom glue code per pair.

**Check yourself before Stage 2:** Why is this genuinely an N×M problem, and how does a shared protocol reduce it?

---

## Stage 2 — The architecture, mapped to what you already know

```
Host Application (your agent)
      |
   MCP Client   ← lives inside your application
      |
   MCP Protocol (JSON-RPC messages, over stdio or HTTP)
      |
   MCP Server   ← a separate process, exposing tools
      |
  Tools / Resources / Prompts
```

- **Host**: your application — the thing the user actually interacts with.
- **Client**: the part of your application that speaks MCP to a server. You'll build one in this part.
- **Server**: a separate process exposing capabilities over the protocol. You'll build one too.
- **Tools**: this is exactly [Part VI](part06-structured-outputs-tool-calling.md)'s tool-calling mechanism — a name, a schema, an executable function — just exposed over a network protocol instead of living inside your own process.
- **Resources**: a new idea — readable *data* (a file, a database row) addressable by URI, distinct from tools because it represents state to read, not an action to invoke.
- **Prompts**: reusable, parameterized prompt templates a server can offer, similar to [Part V](part05-prompt-engineering.md)'s templates, but shareable across any client that connects.

**Check yourself before Stage 3:** What's the difference between an MCP "tool" and an MCP "resource," and why does that distinction matter for how a client uses each?

---

## Stage 3 — Build a real MCP server

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def evaluate(expression: str) -> float:
    """Evaluate a simple arithmetic expression, e.g. '2 + 2 * 3'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError(f"disallowed characters: {expression!r}")
    return eval(expression, {"__builtins__": {}}, {})


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Look closely: `@mcp.tool()` is doing exactly what your `ToolSpec.to_schema()` did by hand in [Part VI Stage 4](part06-structured-outputs-tool-calling.md) — reading the function's type hints and docstring, generating a JSON Schema, and registering it so it can be discovered and called remotely. The `evaluate` function itself, including the exact same character-allowlist safety check, is identical to the calculator tool you built in Part VI — the *safety discipline* doesn't change just because the tool is now exposed over a protocol instead of called directly.

Save this as `calculator_server.py`. It's now a standalone program, runnable on its own.

**Check yourself before Stage 4:** What does `@mcp.tool()` generate automatically, and how does that compare to what you built by hand with Pydantic in Part VI?

---

## Stage 4 — Build a real MCP client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server_params = StdioServerParameters(command="python", args=["calculator_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()   # handshake — more on this in Stage 5

            tools = await session.list_tools()
            print("discovered tools:", [t.name for t in tools.tools])

            result = await session.call_tool("evaluate", {"expression": "2 + 2 * 3"})
            print("result:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

Run this against your server script. It will start `calculator_server.py` as a subprocess, connect to it over stdio (standard input/output — the same mechanism you'd use to pipe data between two Unix commands), discover its tools, and call one.

Notice `session.list_tools()` — this is genuinely new compared to Part VI's `ToolRegistry`, where your available tools were hardcoded in your own process. Here, the client discovers what's available *dynamically, at connection time*, from a completely separate program it doesn't control the source code of. This dynamic discovery is the actual technical enabler of the N×M fix from Stage 1: the client doesn't need to know in advance what a server offers.

**Check yourself before Stage 5:** Why is `session.list_tools()` — dynamic discovery — essential to solving the N×M problem, in a way that Part VI's hardcoded `ToolRegistry` couldn't?

---

## Stage 5 — The handshake: why `initialize()` exists

You might wonder why `session.initialize()` is a separate call before you can do anything else. It's not boilerplate — it's a real negotiation step.

When client and server first connect, they don't yet know what version of the protocol the other speaks, or which optional features (resources? prompts? streaming?) each supports. `initialize()` is where they exchange this information — each side declares its capabilities, and they agree on what's usable in this specific connection.

This matters practically: it's what lets the protocol evolve over time without breaking older implementations. A newer client talking to an older server can gracefully use only the older server's capabilities, rather than assuming a feature exists and crashing when it doesn't.

**Check yourself before Stage 6:** What problem would you run into if a client just assumed a server supported every possible feature, without a capability-negotiation step first?

---

## Stage 6 — Errors, handled exactly like Part VI, over the wire

```python
result = await session.call_tool("evaluate", {"expression": "not valid math"})
```

If `evaluate` raises a `ValueError` on the server side (from the disallowed-characters check), that error doesn't crash the client's process, and it doesn't crash the server either. It gets serialized as a structured error response and delivered back to the client — the same "structured error fed back, not a crash" pattern from [Part VI Stage 5](part06-structured-outputs-tool-calling.md), just carried across a process boundary using JSON-RPC's standard error format (a code, a message, and optional extra data) instead of an in-process Python exception.

**Check yourself before the project:** Why does an error on the server side need to be *serialized and sent back*, rather than simply raised, given that client and server are separate processes?

---

## Stage 7 — The trust problem MCP introduces

One thing worth flagging clearly before you build anything real: an MCP server is, from your application's point of view, an **external, untrusted input source** — exactly like a retrieved document from [Part V Stage 6](part05-prompt-engineering.md) or [Part XI](part11-rag-fundamentals.md). A malicious or compromised server could describe a destructive tool using an innocent-sounding name and description, or return results crafted to manipulate your agent's next decision.

This isn't a hypothetical worry to defer — it's the direct reason [Part XXV](part25-agent-plus-mcp.md) later builds explicit trust tiers for different MCP servers, and [Part XXXV](part35-security.md) treats "tool poisoning" as a named threat category. For now, the practical takeaway: **only connect to MCP servers you trust, and treat their tool descriptions and results with the same caution you'd apply to any other untrusted input.**

**Check yourself before the project:** Why is an MCP server's tool *description* — not just its results — something you should be cautious about trusting?

---

# The Project: A Working MCP Server and Client, With Real Failure Handling

## What you're building

A complete calculator MCP server with two tools, and a client that discovers and uses them — with every failure mode from Part VI reproduced over the protocol boundary.

```bash
$ python mcp_client.py
```

```
connecting to calculator_server.py...
discovered tools: ['add', 'evaluate']
calling evaluate(expression="2 + 2 * 3")... result: 8
calling evaluate(expression="rm -rf /")... error: disallowed characters
calling unknown_tool(...)... error: tool not found
```

## Why this project

This is the foundation for [Part XXIV](part24-mcp-server-development.md) (building more sophisticated servers) and [Part XXV](part25-agent-plus-mcp.md) (connecting an agent to multiple servers at once). Get the basic mechanics solid here.

## Build it in stages

---

### Stage A — Server with two tools

**Goal:** `calculator_server.py`, with `add` and `evaluate`, running standalone.

**Do:**
1. Implement both tools exactly as in Stage 3
2. Run the server directly (`python calculator_server.py`) and confirm it starts without error and waits for a connection

**You're done when:** The server process runs and doesn't crash on startup.

---

### Stage B — Client with discovery and calls

**Goal:** A client that connects, lists tools, and calls both of them successfully.

**Do:**
1. Implement the client from Stage 4
2. Call both `add` and `evaluate` with valid arguments
3. Print the discovered tool list and both results

**You're done when:** Both tools execute correctly through the full client-server round trip, not called directly in-process.

---

### Stage C — Force every failure mode

**Goal:** Confirm errors are handled gracefully across the protocol boundary, not just within one process.

**Do:**
1. Call `evaluate` with disallowed characters — confirm you get a structured error back, and neither process crashes
2. Call a tool name that doesn't exist — confirm a clear "not found" error
3. Call `evaluate` with missing/malformed arguments — confirm a validation error

**You're done when:** All three failure types produce clean, informative errors on the client side, and the server process is still alive and responsive after each one.

---

### Stage D — A third tool with a different risk profile

**Goal:** Extend the server with one more tool that has real (if limited) side effects, to feel the difference from pure-calculation tools.

**Do:**
1. Add a tool that writes a short note to a local file (in a fixed, safe test directory)
2. Confirm the client can call it and the file is actually written
3. Test what happens if the write target is invalid (bad path) — confirm this fails as a clean structured error too, not a server crash

**You're done when:** You've built and tested a tool with a real side effect, and confirmed the same error-handling discipline holds for it as for the pure-calculation tools.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Client hangs on `session.initialize()` | Server process failed to start, or path to the server script is wrong | Run the server script directly first to confirm it starts cleanly before connecting a client to it |
| `list_tools()` returns an empty list | Tools decorated in a module that isn't actually imported/run by the server entry point | Confirm the `@mcp.tool()`-decorated functions are defined in the same file `mcp.run()` is called from |
| A tool's error crashes the server process instead of returning cleanly | Raising an exception type or with content that the framework can't serialize | Stick to standard exception types with plain string messages, as in Part VI |

## Going further (optional)

1. **A resource.** Add a static resource (e.g., server version info) using `@mcp.resource(...)` and read it from the client, distinguishing this from a tool call in your own understanding.
2. **HTTP transport instead of stdio.** If your MCP library supports it, run the server as an HTTP service instead of a local subprocess, and connect a client over the network — this is the shape a *remote* MCP server (not just a local one) would take.
3. **Timeout the client side.** Wrap `session.call_tool(...)` in a timeout (reusing Part VI's pattern) and test what happens if you deliberately make a tool sleep longer than the timeout.

---

## Checkpoint

**Knowledge:**
1. Why is MCP genuinely an N×M problem, and how does a shared protocol reduce the number of integrations needed?
2. What's the difference between an MCP tool and an MCP resource?
3. Why does `initialize()` need to be a real negotiation step, not just boilerplate?
4. Why should an MCP server's tool descriptions be treated as untrusted input?

**Implementation:** Build a working MCP server and client pair, with tool discovery and structured error handling, from memory.

**Debugging:** A host application reports "tool not found" for a tool you just added to your server. What's the most likely cause, given what you know about how `list_tools()` works?

**Architecture:** Design capability negotiation for an MCP server that must support both an old client version and a new one simultaneously during a rolling deployment.

---

**Next:** [Part XXIV — MCP Server Development](part24-mcp-server-development.md), where you'll build servers with real security stakes — filesystem access, database queries, and the sandboxing discipline that protects against a compromised or careless caller.
