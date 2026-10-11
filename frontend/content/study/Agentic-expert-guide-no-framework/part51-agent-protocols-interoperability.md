# Part LI — Agent Protocols and Interoperability

## Before you start

**What you need already:** [Part XXIII](part23-mcp.md) (MCP's client/server/capability-negotiation pattern), [Part XXXV](part35-security.md) (permission scoping, treating external input as untrusted).

**What this part gives you:** Recognizing that the exact same integration problem MCP solved for *tools* also exists for *agents talking to other agents* — and that the fix, and even the risks, are structurally the same, just one level up.

**Time:** 1–2 days.

**When you finish, you'll be able to:** Build a minimal agent-to-agent communication protocol with capability discovery and cryptographic identity verification, recognizing it as MCP's pattern applied at the agent level.

---

## Stage 1 — The same N×M problem, one level up

Recall [Part XXIII Stage 1](part23-mcp.md)'s N×M problem: without a shared protocol, N applications needing M tools required N×M custom integrations. The exact same shape appears when *agents* (not just tools) need to talk to each other — if your research agent and a partner company's data-analysis agent need to collaborate, and there's no shared protocol, you're back to writing bespoke integration code for every pair of agents that need to interoperate.

**Check yourself before Stage 2:** Restate, in your own words, why agent-to-agent interoperability is the same underlying problem as MCP's tool interoperability, just at a different layer.

---

## Stage 2 — A minimal agent-to-agent protocol, recognized from MCP

```python
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    TASK_ERROR = "task_error"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"


@dataclass
class AgentMessage:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK_REQUEST
    sender_id: str = ""
    recipient_id: str = ""
    payload: dict = field(default_factory=dict)
    correlation_id: str | None = None   # links a response back to its request
```

`CAPABILITY_QUERY`/`CAPABILITY_RESPONSE` is directly `list_tools()` from [Part XXIII](part23-mcp.md), applied to agents instead of tool servers — a way for one agent to discover what another can do before delegating. `correlation_id` is exactly [Part IV Stage 8](part04-llm-application-engineering.md) and [Part XXXIX Stage 2](part39-observability.md)'s correlation concept, applied here to match asynchronous responses back to their originating requests.

```python
class AgentEndpoint:
    def __init__(self, agent_id: str, capabilities: list[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities

    def handle_capability_query(self, msg: AgentMessage) -> AgentMessage:
        return AgentMessage(
            type=MessageType.CAPABILITY_RESPONSE, sender_id=self.agent_id,
            recipient_id=msg.sender_id, correlation_id=msg.message_id,
            payload={"capabilities": self.capabilities},
        )

    def handle_task_request(self, msg: AgentMessage, execute_fn) -> AgentMessage:
        requested = msg.payload.get("capability")
        if requested not in self.capabilities:
            return AgentMessage(
                type=MessageType.TASK_ERROR, sender_id=self.agent_id, recipient_id=msg.sender_id,
                correlation_id=msg.message_id, payload={"error": f"capability not supported: {requested}"},
            )
        try:
            result = execute_fn(msg.payload)
            return AgentMessage(type=MessageType.TASK_RESULT, sender_id=self.agent_id,
                                 recipient_id=msg.sender_id, correlation_id=msg.message_id,
                                 payload={"result": result})
        except Exception as e:
            return AgentMessage(type=MessageType.TASK_ERROR, sender_id=self.agent_id,
                                 recipient_id=msg.sender_id, correlation_id=msg.message_id,
                                 payload={"error": str(e)})
```

`handle_task_request` is, once again, the exact same pattern from every tool-calling error path since [Part VI](part06-structured-outputs-tool-calling.md): an unsupported capability or a raised exception becomes a structured `TASK_ERROR` message, sent back, never an unhandled crash.

**Check yourself before Stage 3:** What earlier Part's mechanism does `handle_capability_query` directly correspond to, and how does it differ only in scope (agents, not tools)?

---

## Stage 3 — The one genuinely new risk: identity, across a boundary you don't control

Here's where agent-to-agent interoperability introduces something MCP (in a single-organization deployment) often doesn't have to worry about as sharply: `sender_id` in `AgentMessage` is, as written, just a string the sender claims about itself. Within one organization's systems, you might reasonably trust that. **Across an organizational boundary — your agent talking to a partner company's agent — trusting a self-reported `sender_id` is exactly the same mistake as trusting a client-supplied `tenant_id` from [Part XLVII Stage 1](part47-production-architecture.md).**

```python
def verify_agent_identity(token: str, trusted_issuers: set[str]) -> dict:
    """Real identity verification — e.g., a signed JWT — not a self-reported
    string in the message payload. This is Part XLVII's server-side identity
    resolution, applied to agent-to-agent communication instead of a client
    request."""
    claims = decode_and_verify_jwt(token)   # a real JWT library, not shown here
    if claims["iss"] not in trusted_issuers:
        raise PermissionError(f"untrusted issuer: {claims['iss']}")
    return claims
```

The principle is identical to Part XLVII: identity must be verified cryptographically, from something the other party cannot forge, before you act on any claim about who they are or what they're authorized to do — whether that "other party" is an HTTP client or another agent entirely.

**Check yourself before the project:** Why is trusting a self-reported `sender_id` in an `AgentMessage` the same category of mistake as trusting a client-supplied `tenant_id`, even though one is "just a string field" and the other is "a whole different system"?

---

# The Project: Two Agents, Talking Across a Network Boundary, With Verified Identity

## What you're building

Two `AgentEndpoint`s running as separate HTTP services, one delegating a task to the other, with capability discovery, correlation, and real identity verification.

```bash
$ python orchestrator.py
```

```
querying capabilities of research-agent-1...
response: ['web_search', 'summarize']

delegating task_request(capability='web_search', query='renewable energy')...
response: TASK_RESULT {result: '...'}

attempting delegation with a FORGED sender_id...
response: TASK_ERROR {error: 'untrusted issuer'} -- rejected correctly
```

## Why this project

This is the direct capstone of the interoperability discipline this curriculum has built repeatedly — from MCP's client/server pattern to Part XLVII's server-side identity resolution — now applied to agent-to-agent delegation specifically.

## Build it in stages

---

### Stage A — Two real HTTP-based agent endpoints

**Goal:** `AgentEndpoint` wrapped in a real FastAPI (or similar) service, running as two separate processes.

**Do:**
1. Implement one agent with a real capability (e.g., web search, reusing Part XLV)
2. Wrap it in an HTTP endpoint accepting `AgentMessage`-shaped requests
3. Confirm you can query its capabilities and delegate a task to it over a real network call, from a separate client script

**You're done when:** A capability query and a task delegation both work correctly over a real HTTP round trip between two separate processes.

---

### Stage B — Correlation, verified

**Goal:** Confirm `correlation_id` correctly links async responses back to requests, even with multiple in-flight requests.

**Do:**
1. Send 3 concurrent task requests to the same agent
2. Confirm each response's `correlation_id` correctly matches its originating request, even though they may complete out of order

**You're done when:** You've verified correct request/response matching under concurrent, out-of-order delivery.

---

### Stage C — Real identity verification

**Goal:** `verify_agent_identity`, using a real JWT library, rejecting a forged identity.

**Do:**
1. Implement real token issuance and verification (a simple self-signed JWT setup is fine for this exercise)
2. Wire it into your endpoint so every incoming message's claimed identity is cryptographically verified, not just read from the payload
3. Attempt to send a message with a forged/invalid token, confirm it's rejected

**You're done when:** You've watched a genuine forged-identity attempt get correctly rejected, and a genuine, correctly-signed identity get correctly accepted.

---

## Common mistakes on this project

| Symptom | Cause | Fix |
|---|---|---|
| Concurrent requests' responses get mismatched | `correlation_id` not actually being checked on the client side, just assuming in-order delivery | Explicitly match responses to pending requests by `correlation_id`, never by arrival order |
| Forged identity is incorrectly accepted | Verification logic checking the payload's self-reported `sender_id` instead of a cryptographically verified token | Confirm identity comes exclusively from verified token claims, never from the message payload's own fields |

## Going further (optional)

1. **Capability versioning** — have an agent's capabilities change between two calls, and confirm a client that cached an old capability list correctly discovers the change on its next query rather than assuming stale information forever.
2. **A third agent**, forming a small delegation chain (A delegates to B, which delegates to C), and confirm identity and correlation both hold correctly across the multi-hop chain.

---

## Checkpoint

**Knowledge:**
1. Why is agent-to-agent interoperability the same N×M problem MCP solved, one layer up?
2. Why is trusting a self-reported `sender_id` in a message payload the same mistake as Part XLVII's client-supplied `tenant_id` trap?

**Implementation:** Build `AgentEndpoint` with capability discovery, correlation, and real cryptographic identity verification, from memory.

**Architecture:** Design an interoperability layer for an enterprise with agents built on three different frameworks that must delegate tasks to each other, applying this part's identity and correlation discipline.

---

**Next:** [Part LII — Advanced Research Topics](part52-advanced-research-topics.md), a deliberate shift from engineering practice to the frontier — distinguishing what's production-mature from what's still experimental.
