# 07 — MEGA Interview Q&A (50+ Questions)

Every question they could realistically ask, with strong answers. Practice out loud.

---

## 🤖 AGENTIC AI (10)

**1. What is Agentic AI?**
> AI that can autonomously plan, reason, choose and use tools, and take actions to achieve a goal with minimal human intervention. Unlike a plain chatbot that only replies, an agent decides *what to do* and *which tools to use*.

**2. What is an AI agent?**
> An LLM wrapped in a reason-act loop with access to tools. It reads a goal, decides if it needs a tool, calls it, observes the result, and repeats until done.

**3. What is the ReAct pattern?**
> Reason + Act. The agent alternates between reasoning ("I need current data") and acting (calling a tool), using each observation to inform the next step.

**4. What is a multi-agent system?**
> Multiple specialized agents collaborating on a task. In my assignment, a Research Agent gathers info and an Email Agent delivers it — each expert at one job.

**5. Why use multiple agents instead of one?**
> Separation of concerns. Each agent has one clear responsibility and toolset, making the system modular, easier to test, reusable, and scalable. One mega-agent mixing research and email logic is harder to maintain.

**6. How do agents communicate?**
> Agent-to-agent communication: one agent's output becomes the next agent's input. In my flow, the Research Agent's report is passed directly into the Email Agent.

**7. What is orchestration?**
> Coordinating the agents — the order they run, passing data between them, and managing the overall flow. In my project, Langflow is the orchestrator.

**8. What is a tool in agentic AI?**
> An external capability the agent can call — web search, email, a database, an API. The tool has a name, description, and input schema so the LLM knows when and how to use it.

**9. How does an agent decide which tool to use?**
> The LLM reads each tool's name and description, matches it to the user's need, and emits a tool call with arguments. The framework runs the tool and returns the result.

**10. What's the difference between an LLM and an agent?**
> An LLM is text-in/text-out — it can't act in the world. An agent adds tools and a decision loop, so it can fetch live data and take actions.

---

## 🧩 MCP (15) — THE most important section

**11. What is MCP?**
> Model Context Protocol — an open standard from Anthropic that standardizes how AI agents connect to external tools and data. Like USB-C for AI tools: build a tool once, any MCP-compatible agent can use it.

**12. What problem does MCP solve?**
> The M×N integration problem. Without MCP, every AI app needs custom code for every tool (M apps × N tools). MCP makes it M+N — one standard interface.

**13. What is an MCP server?**
> A program that exposes tools (and optionally resources and prompts) over MCP. It advertises each tool's name, description, and input schema, and runs the function when called.

**14. What is an MCP client?**
> The component inside the AI app that speaks MCP and connects to a server. In Langflow, the **MCP Tools component** is the client.

**15. What is the MCP host?**
> The AI application that wants to use tools — in my case, Langflow (and the agents inside it).

**16. How does a tool call flow through MCP?**
> The client asks the server to list tools → the agent decides to call one → the client sends a `tools/call` request (JSON-RPC) with arguments → the server runs the real function → returns the result → the agent reads it.

**17. What transports does MCP use?**
> STDIO (the host runs the server as a local subprocess, talking over stdin/stdout) and HTTP/SSE (the host connects to a remote server over a URL). I use STDIO for local servers.

**18. What's the difference between STDIO and SSE transport?**
> STDIO runs the server locally as a subprocess — simplest, no URL or headers. SSE/HTTP connects to a remote running server over a URL, where headers carry authentication. STDIO for local, SSE for remote.

**19. How does Langflow's MCP Tools component connect to a server?**
> It's the MCP client. You configure it with STDIO (a command Langflow runs) or SSE (a URL). It auto-discovers the server's tools, and you wire those tools into an Agent's Tools input.

**20. Native tool vs MCP tool — difference?**
> A native tool is built into the platform and only that platform can use it. An MCP tool lives in an independent server using a standard protocol — reusable by any agent, any language, hosted anywhere. MCP turns a tool into a portable service.

**21. Why does this assignment require MCP instead of direct API calls?**
> To demonstrate understanding of MCP — the interoperable, production way to give agents tools. Direct API calls or native components are tightly coupled and not reusable; MCP is standardized and portable.

**22. What format do MCP messages use?**
> JSON-RPC 2.0 — a simple "call this method with these params, get a result" message format. The MCP library handles the encoding.

**23. What can an MCP server expose besides tools?**
> Resources (read-only data the AI can fetch, like files or DB rows) and Prompts (reusable prompt templates). For this assignment I only use Tools.

**24. How do you secure an MCP server?**
> For STDIO, it runs locally so the surface is small; keep secrets in environment variables, not in code. For remote/SSE, use HTTPS, auth headers/tokens, and restrict who can reach the URL. Validate tool inputs.

**25. How would you host an MCP server remotely?**
> Run it in SSE/HTTP mode on a cloud VM, or expose a local server through a tunnel like ngrok, then point Langflow's MCP Tools component at that URL (with auth headers). That's the bonus requirement.

---

## 🎨 LANGFLOW (8)

**26. What is Langflow?**
> An open-source, visual low-code framework for building LLM apps, agents, RAG systems, and multi-agent workflows by dragging and connecting components. Built on LangChain.

**27. What is a component in Langflow?**
> A single node that does one job — Chat Input, Prompt, Model, Agent, MCP Tools — with typed input/output ports you wire together.

**28. Langflow vs LangChain?**
> LangChain is the Python code framework; Langflow is the visual layer over it. Langflow is faster for prototyping; LangChain gives full code control.

**29. What is the Playground in Langflow?**
> A built-in chat interface to test your flow live without deploying it.

**30. How do you deploy a Langflow flow?**
> Every flow can be run via API — Langflow generates a Python/JS/curl snippet. You can also publish it as an MCP server, or run Langflow in Docker for production.

**31. What are typed ports?**
> Ports carry specific data types (Message, Data, Tool). Langflow only allows compatible connections, color-coded, preventing wiring mistakes.

**32. How do you add memory in Langflow?**
> Add a Chat Memory component (or enable memory on the agent). It stores conversation history and feeds it back, so the agent remembers context.

**33. How does Langflow connect to an MCP server?**
> Via the MCP Tools component — STDIO command or SSE URL — which discovers tools and connects them to an agent.

---

## 🏗️ THIS ASSIGNMENT / ARCHITECTURE (8)

**34. Walk me through your architecture.**
> The user provides a topic and an email. The Research Agent uses an MCP-connected web_search tool to gather current info and writes a structured report (Summary, Key Findings, References). The report is passed to the Email Agent, which uses an MCP-connected send_email tool to deliver the full report. Langflow orchestrates the flow.

**35. Where is agent-to-agent communication in your flow?**
> The Research Agent's output (the report) flows directly into the Email Agent's input. That wire is the handoff.

**36. Why two agents here?**
> Separation of concerns — one researches, one emails. Modular, testable, and I could swap the Email Agent for a "save to Notion" agent without touching research.

**37. How is the email recipient passed to the Email Agent?**
> It's carried in the text — the Research Agent prepends a "RECIPIENT:" line, and the Email Agent reads it. (Or via a Prompt that combines the report and the email address.)

**38. What does your search MCP server do?**
> It exposes a `web_search` tool that queries DuckDuckGo and returns titles, snippets, and URLs. The Research Agent calls it via MCP.

**39. What does your email MCP server do?**
> It exposes a `send_email` tool that sends mail through Gmail SMTP, with credentials in environment variables and error handling for auth/send failures.

**40. What happens if the search returns no results?**
> The tool returns a clear "NO_RESULTS" message; the agent can rephrase and retry or tell the user it couldn't find information — graceful handling, no crash.

**41. What happens if the email fails to send?**
> The send_email tool catches the error and returns a message (AUTH_ERROR or SEND_ERROR). The Email Agent reports the failure to the user instead of silently failing. In production I'd add retries and logging.

---

## 🐍 PYTHON & APIs (8)

**42. What is an API?**
> An Application Programming Interface — a set of rules for two software systems to talk and exchange data.

**43. What is REST?**
> An architectural style for web APIs using HTTP methods on resources via URLs. Stateless, returns JSON.

**44. Name the main HTTP methods.**
> GET (read), POST (create), PUT/PATCH (update), DELETE (remove).

**45. What is JSON?**
> JavaScript Object Notation — a lightweight, human-readable data format of key-value pairs and arrays, used to exchange data between systems. Example: `{"topic": "AI", "email": "x@y.com"}`.

**46. Explain a class and an object.**
> A class is a blueprint; an object is an instance of it. `class Agent: pass` defines it; `a = Agent()` creates an object.

**47. What is inheritance?**
> A class deriving from another, reusing its behavior. `class ResearchAgent(Agent): ...` inherits from Agent.

**48. What is a decorator in Python? (relevant to your MCP code)**
> A function that wraps another to add behavior. `@mcp.tool()` is a decorator that registers a function as an MCP tool.

**49. How do you handle errors in Python?**
> try/except blocks. My email server uses `try/except` to catch SMTP errors and return a clear message instead of crashing.

---

## 🧠 LLM CONCEPTS (6)

**50. What is prompt engineering?**
> Crafting instructions so the LLM reliably produces the right output. My Research Agent's prompt specifies the exact report sections.

**51. What is tool calling / function calling?**
> The LLM, instead of answering directly, outputs a structured request to call a tool with arguments. The framework runs it and feeds the result back. This is what powers agents.

**52. What is RAG?**
> Retrieval-Augmented Generation — fetch relevant documents and inject them into the prompt so the LLM answers from your data, not just its training. Reduces hallucination.

**53. What is hallucination and how do you reduce it?**
> When the LLM makes up facts. Reduce it with grounding ("answer only from the context/tool results"), RAG, and using tools for real data — which is exactly why the Research Agent searches instead of guessing.

**54. What is temperature?**
> A setting controlling randomness. Low (0–0.2) = consistent/factual; high = creative. For research/extraction I'd keep it low.

**55. Why use gpt-4o-mini here?**
> It's fast and cheap, strong enough for research synthesis and tool calling, keeping cost low for a multi-step multi-agent flow.

---

## 🌟 BONUS / "HOW WOULD YOU IMPROVE IT?" (5)

**56. How would you improve this system?**
> Add memory for follow-up questions, a human-approval step before sending email, retry logic for tools, multiple search sources, PDF report generation, logging/monitoring, and remote-hosted MCP servers.

**57. How would you add memory?**
> Attach a Chat Memory component to the Research Agent (keyed by session) so it remembers the topic and can answer follow-ups without re-searching.

**58. How would you scale this?**
> Host MCP servers remotely (SSE) so they're shared and independently scalable, run Langflow in containers with autoscaling, add a queue for high request volume, and cache repeated searches.

**59. How would you add error handling end-to-end?**
> Tool-level try/except (done), agent instructions to report failures, retries with backoff, a fallback message to the user, and logging every step for debugging.

**60. Why is MCP better than hardcoding the Tavily/Gmail APIs?**
> MCP standardizes and decouples tools from the agent. I can swap or scale tools, reuse them across any MCP client, write them in any language, and host them anywhere — without changing agent logic.

---

## 🎤 30-SECOND INTRO (memorize)
> "I'm Keerthi. I built a multi-agent system in Langflow where a Research Agent gathers information using an MCP-based search tool and generates a structured report, then passes it via agent-to-agent communication to an Email Agent that sends it through an MCP-based email tool. It demonstrates Agentic AI, MCP integration, and workflow orchestration — and I've also built a production multi-agent voice automation, so I think about reliability and tooling in real deployments."

---

Next → `08_SUPPORTING_TECH.md`
