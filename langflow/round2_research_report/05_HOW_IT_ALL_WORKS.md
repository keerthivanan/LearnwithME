# 05 — How It All Works (A Full Run, Traced Moment by Moment)

If you can narrate this trace, you understand the system completely. This is gold for the demo.

---

## The scenario
You type:
- **Topic:** "Latest breakthroughs in quantum computing 2025"
- **Email:** "miakataria4@gmail.com"

And press Run. Here's EVERYTHING that happens, in order:

---

### Moment 1 — Inputs enter the flow
- The **Topic** text and **Email** text leave their Text Input nodes and travel down their wires.
- Topic → goes toward the Research Agent.
- Email → goes toward the Prompt (held for later).

### Moment 2 — Research Agent wakes up
- The Research Agent (an LLM) receives: *"Latest breakthroughs in quantum computing 2025."*
- It reads its system prompt: *"research this, use web_search, write a report."*
- It also sees, in its tool list, a tool called **`web_search`** — because the **MCP Tools (Search)** component connected to the search MCP server and discovered it.

### Moment 3 — The Research Agent decides to use the tool
- The LLM reasons: *"I need current info → call web_search."*
- It emits a tool call: `web_search(query="quantum computing breakthroughs 2025")`.

### Moment 4 — MCP carries the call to the server
- Langflow's **MCP client** (the MCP Tools component) packages this as an MCP `tools/call` message (JSON-RPC).
- It sends it over **STDIO** to the running `search_mcp_server.py` subprocess.

### Moment 5 — The MCP server does real work
- `search_mcp_server.py` runs the actual `web_search` function → queries DuckDuckGo → gets 5 results.
- It returns the titles, snippets, and URLs as text — back over MCP to Langflow.

### Moment 6 — The agent reads results and may search again
- The Research Agent reads the 5 results.
- It might decide to search once more (e.g., a narrower query) — same MCP round trip — or decide it has enough.

### Moment 7 — The Research Agent writes the report
- Using the gathered info, the LLM writes:
  ```
  SUMMARY: ...
  KEY FINDINGS: - ... - ...
  REFERENCES: 1) https://...  2) https://...
  ```
- This report leaves the Research Agent's output port. **This is the agent's product.**

### Moment 8 — The handoff (agent-to-agent communication)
- The report travels the wire into the **Prompt** node's `{report}` slot.
- The **Email address** travels into the Prompt's `{recipient}` slot.
- The Prompt combines them into ONE instruction:
  *"Send this report by email to miakataria4@gmail.com: <full report>"*
- **This is the agent-to-agent moment — Research output became Email input.**

### Moment 9 — The Email Agent wakes up
- The Email Agent (another LLM) receives that combined instruction.
- It sees a tool called **`send_email`** in its tool list — because **MCP Tools (Email)** connected to the email MCP server.

### Moment 10 — The Email Agent decides to send
- The LLM reasons: *"I have a recipient and a report → call send_email."*
- It emits: `send_email(to="miakataria4@gmail.com", subject="Quantum Computing 2025 — Research Report", body="<full report>")`.

### Moment 11 — MCP carries the call to the email server
- Langflow's MCP client sends the `tools/call` over STDIO to `smtp_mcp_server.py`.

### Moment 12 — The email server sends real email
- `smtp_mcp_server.py` reads the Gmail creds from its env vars, connects to Gmail's SMTP server, and sends the email.
- It returns `"SUCCESS: email sent to miakataria4@gmail.com"` back over MCP.

### Moment 13 — Confirmation to the user
- The Email Agent reads the success message and replies: *"✅ I've emailed the full report to miakataria4@gmail.com."*
- That reply flows to **Chat Output** and you see it in the Playground.
- The recipient's inbox now has the full report. **Done.**

---

## The whole thing in 6 lines
1. You give topic + email.
2. Research Agent calls `web_search` **via MCP** → gets info → writes report.
3. Report + email combine in a Prompt (**agent-to-agent handoff**).
4. Email Agent calls `send_email` **via MCP** → sends the full report.
5. You get a confirmation; the inbox gets the report.
6. Langflow orchestrated all of it; both tools came from MCP servers.

---

## Where each requirement is satisfied (point to these)
| Requirement | Where it happens |
|-------------|------------------|
| Multi-agent | Research Agent + Email Agent (two Agent nodes) |
| MCP for search | Moment 4-5 (MCP Tools → search server) |
| MCP for email | Moment 11-12 (MCP Tools → email server) |
| Agent-to-agent | Moment 8 (report → Prompt → Email Agent) |
| Full report emailed | Moment 10-12 (full body, not summary) |
| Single trigger | Moment 1 (topic + email at start) |

---

Next → `06_INTERVIEW_QA.md`
