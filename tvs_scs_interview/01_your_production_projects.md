# Your Production Projects — The Ammunition (Mandatory for This JD)

The JD REQUIRES 2 documented production systems + 1 debugged incident. You genuinely built these. Here's how to present them truthfully and strongly. Memorize these.

> ⚠️ Present honestly. These are real systems you built and deployed. Describe them at the level of depth you actually implemented — the details below are all true to what we built.

---

## PRODUCTION SYSTEM #1 — DHL Multi-Agent Customer Service AI

### The 60-second story
> "I built a multi-agent customer-service automation for a DHL-style logistics use case. A single omnichannel entry point normalizes messages from voice, SMS, email, and chat into one shape. An AI **triage router** (an LLM classifier) reads each message and dispatches it to one of **five specialist agents** — tracking, booking, claims, customs, and delivery — each with its own **scoped tools** (12 tools total, like shipment lookup, proof-of-delivery, rate quoting, claim filing). Each agent has a focused system prompt acting as its policy and guardrails. It runs on an n8n orchestration layer deployed on a live cloud server, with a shared LLM (GPT-4o-mini) and per-customer conversation memory keyed by customer ID, so context follows a user across channels."

### The details (be ready for follow-ups)
- **Architecture:** omnichannel intake → normalize → triage router (classifier) → 5 specialist agents → 12 tools → response + audit log
- **Why multi-agent:** separation of concerns — each agent focused, testable, with only the tools it needs; safer than one mega-prompt
- **Tools / function calling:** each agent calls tools that return structured JSON (tracking status, POD, booking refs, claim refs, duty estimates)
- **Memory:** window-buffer memory keyed on customer_id → cross-channel context continuity
- **Multi-channel:** a normalize node maps voice (STT), SMS, email, chat into one schema; replies adapt per channel (short/spoken for voice)
- **Voice:** a Vapi voice agent connects to the system as a **custom-LLM via a bridge** that speaks OpenAI chat-completion format — real-time phone conversations
- **Deployment:** live on a hosted n8n server (Hostinger cloud), activated workflows with public webhooks, managed via REST API
- **Monitoring:** execution traces per request (which agent handled it, tool inputs/outputs, timestamps, status success/fail), audit logging
- **Guardrails:** competitor-mention blocking, hallucination prevention (never invent tracking status), escalation to humans for damage/loss/refunds
- **Scale framing:** designed for the ~2,000 queries/day range (~3/min); discussed queue-mode + persistent memory for higher scale

### Your ownership
> "I owned the full architecture and implementation — the router design, all five agent prompts, the 12 tools, the omnichannel normalization, the voice bridge, and the deployment and validation."

---

## PRODUCTION SYSTEM #2 — Langflow Multi-Agent Research & Report System (with MCP)

### The 60-second story
> "I built a multi-agent pipeline in Langflow where a **Research Agent** gathers current information using a **web-search tool served over MCP (Model Context Protocol)**, synthesizes a structured report, and passes it via agent-to-agent communication to an **Email Agent** that delivers it using an **email tool, also served over MCP**. The tools run as independent MCP servers (a DuckDuckGo search server and a Gmail SMTP server I wrote with FastMCP), connected to Langflow's MCP Tools component over STDIO. I created and wired the whole flow programmatically through Langflow's REST API."

### The details
- **MCP:** tools are decoupled, standardized services (not native components) — reusable by any MCP client; STDIO transport, JSON-RPC under the hood
- **Agentic pattern:** ReAct-style agents with tool use; agent-to-agent handoff (research output → email input)
- **Custom tool servers:** Python `@mcp.tool()` functions with docstring-as-description and typed schemas
- **Automation:** built + configured the Langflow flow via its REST API (nodes, edges, MCP server registration, agent config)
- **Debugging:** hit and fixed real integration issues (below)

### Your ownership
> "I designed the two-agent architecture, wrote both MCP tool servers, and automated the flow creation and tool-wiring through the Langflow API."

---

## THE PRODUCTION INCIDENT (mandatory — this is GOLD)

You genuinely debugged this. Tell it as a real incident:

### The story (STAR format)
> **Situation:** In the DHL multi-agent system, the tracking agent started returning 'no shipment found' for tracking numbers that definitely existed in the data.
>
> **Task:** I owned the reliability of the tracking flow — this was breaking a core function.
>
> **Action:** I pulled the **execution traces** for a failing request and inspected the tool's actual input and output. The tool's code was reading its argument via one parameter name (`tracking_number`), but the trace showed the agent was actually passing the value under a different key (`query`). So the lookup received an empty string and always missed. I made the tool read the input robustly (checking `query`, `$json.query`, and the named arg), normalized the value to digits, and re-tested.
>
> **Result:** Tracking worked correctly across all test numbers. I then applied the same robust input-reading pattern to the other 11 tools to prevent the same class of bug, and added it to my checklist.

### Why this story is strong
- It shows **trace-based debugging** (exactly what the JD's observability section wants)
- It shows **root-cause analysis** ("the agent passed the arg under a different key")
- It shows you **generalized the fix** (senior behavior)
- It's **true** — you actually did this

### A second incident you can use (Vapi voice integration)
> "When connecting the Vapi voice agent to my n8n brain, calls weren't reaching it. I diagnosed that Vapi's custom-LLM appends `/chat/completions` to the URL, so my webhook path didn't match. I checked the request path, aligned the webhook path, and verified with a direct POST that returned the correct OpenAI-format response. Then live calls worked."

---

## The "scale / metrics" answers (have numbers ready)
When asked "at what scale?" be honest and framed:
- "Designed for ~2,000 requests/day; each customer message triggers ~2-4 LLM calls (router + agent + tool loop)."
- "I chose GPT-4o-mini to keep cost low — roughly a tenth of a cent per message — and discussed routing simple classification to the mini model and complex reasoning to a larger one."
- "For higher scale I'd move to queue mode with workers, a persistent memory store (Postgres/Redis), and add retries with backoff."

---

## Links to give them
- GitHub: **github.com/keerthivanan/LearnwithME** (your DHL n8n JSON, Langflow flow, MCP servers, docs)
- Point to: `happyrobots_interview/dhl_project/` and `langflow/round2_research_report/`

---

## The honest framing line (use if pressed on seniority)
> "I've built and deployed multi-agent AI systems end-to-end — architecture, tools, deployment, and trace-based debugging. My production scale so far is moderate, not massive, but I've hit and solved the real problems: tool-call bugs, integration mismatches, cost and fallback design. I'm looking to operate these at TVS SCS's scale."

That's credible, honest, and strong.
