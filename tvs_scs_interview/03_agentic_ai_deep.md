# Agentic AI — Deep (⭐ Top Pillar)

Everything on agents, frameworks, memory, and orchestration. This is the heart of the role.

---

## 1. Agent fundamentals

**Agent = LLM + tools + a reason-act loop.** It decides what to do, calls tools, observes results, repeats until the goal is met.

**ReAct (Reason + Act)** — the core pattern:
```
Thought: I need current data → Action: call search tool → Observation: results
Thought: now I can answer → Answer: ...
```
The LLM reasons in text, emits an action (tool call), reads the observation, loops.

**Other reasoning patterns to name:**
- **Chain-of-Thought (CoT):** "think step by step" — reason before answering
- **ReAct:** reason + act with tools (interleaved)
- **Reflexion / self-correction:** the agent critiques its own output and retries
- **Plan-and-Execute:** a planner makes a multi-step plan, an executor runs each step
- **Tree of Thoughts:** explore multiple reasoning branches

---

## 2. Tool use / Function calling (must know cold)

The LLM doesn't run code — it **outputs a structured request** to call a function; your framework runs it and feeds the result back.

- You define tools with **name + description + JSON schema** for arguments
- The model picks a tool and fills the arguments (structured output)
- **Structured output enforcement:** force valid JSON (OpenAI `response_format`/tools, or a schema validator like Pydantic) so downstream code never breaks
```python
tools = [{"type":"function","function":{
  "name":"get_shipment","description":"Look up a shipment by tracking number",
  "parameters":{"type":"object","properties":{"tracking_number":{"type":"string"}},"required":["tracking_number"]}}}]
```
> **Interview line:** "Function calling turns the LLM into a controller — it emits a validated tool call, the framework executes it, and the result goes back into the context. I always enforce structured output with a schema so consequential actions never get malformed input." (You debugged exactly this — the tool got the arg under the wrong key.)

---

## 3. Memory (the JD explicitly lists short/long/episodic)

| Memory type | What | How |
|-------------|------|-----|
| **Short-term** | Current conversation context | Message history in the prompt (window buffer) |
| **Long-term** | Persistent facts across sessions | Vector store / DB, retrieved by relevance |
| **Episodic** | Specific past events/interactions | Store past episodes, recall similar ones |
| **Working/scratchpad** | The agent's in-task reasoning | Held during the ReAct loop |

- **Window buffer:** keep the last N turns (simple, what you used, keyed by customer_id)
- **Summary memory:** summarize old turns to save tokens
- **Vector memory:** embed and store messages/facts; retrieve relevant ones (long-term)
> Your DHL system used per-customer window memory keyed on customer_id → cross-channel continuity.

---

## 4. Multi-agent systems (your strength)

Multiple specialized agents cooperating. Patterns:
- **Router / supervisor → specialists** (your DHL design): a router classifies and dispatches to the right expert agent
- **Pipeline / sequential** (your Langflow): agent A's output → agent B (research → email)
- **Hierarchical:** a manager agent delegates to sub-agents
- **Debate / collaboration:** agents critique each other

**Why multi-agent > one mega-agent:** separation of concerns — each agent has one job, a focused prompt, and only its tools. Easier to test, safer, cheaper (route simple tasks to cheap models).

**HITL (Human-in-the-Loop):** insert a human approval step at high-stakes points (e.g., before issuing a refund). The agent pauses, a human approves/edits, then it continues. The JD calls this out — mention it for "high-stakes decision points."

---

## 5. The frameworks (know what each is for)

### LangChain
The most popular framework — chains, agents, tools, memory, retrievers, model wrappers. `AgentExecutor` runs the ReAct loop. Great for building RAG + agents fast. Criticized for abstraction bloat; know it anyway.

### LangGraph (⭐ important — the modern choice)
Built by LangChain team. Models agent workflows as a **graph (state machine)**: nodes = steps/agents, edges = transitions, with a shared **state** object. Gives you **cycles, branching, persistence, and human-in-the-loop** — far more control than a linear chain. **Best answer for "how do you build reliable, stateful, multi-agent systems?"**
```
State flows through nodes; conditional edges decide the next node; you can loop, pause for HITL, and checkpoint state.
```

### LlamaIndex
Data-framework focused on **RAG**: ingestion, indexing, retrieval, query engines. Strong for connecting LLMs to your data. Also has agents. Pair it with LangChain/LangGraph.

### AutoGen (Microsoft)
Multi-agent **conversation** framework — agents (e.g., AssistantAgent, UserProxyAgent) talk to each other to solve tasks, with code execution. Good for collaborative/coding agents.

### CrewAI
Higher-level multi-agent: define **agents with roles/goals** + **tasks** + a **crew** (process: sequential or hierarchical). Very readable for role-based teams (researcher, writer, reviewer).

### Semantic Kernel (Microsoft)
SDK for orchestrating LLMs with **plugins/skills**, planners, and memory. Enterprise/.NET-friendly.

**How to answer "which framework?":**
> "I pick based on the job: LangGraph when I need a reliable stateful multi-agent workflow with branching, cycles, and human-in-the-loop; LlamaIndex when RAG/data-indexing is the core; CrewAI for readable role-based teams; AutoGen for conversational/collaborative agents. LangChain ties tools and models together across all of them."

---

## 6. Long-horizon task execution & self-correction (JD phrases)
- **Long-horizon:** tasks needing many steps over time → planning, persistent memory/state, checkpointing (LangGraph)
- **Self-correction:** the agent validates its own output (or a tool/validator does) and retries on failure — reflexion loops, output validators, retry with feedback

---

## 7. Reliability patterns (the senior signal)
- **Max iterations / loop guards** — stop runaway agents
- **Timeouts** on tool calls
- **Fallback model** if the primary fails or is rate-limited
- **Guardrails** on inputs (injection) and outputs (schema, safety)
- **Tracing every tool call** — so you can debug (LangSmith/LangFuse) — see file `06`
- **Idempotent tools** for actions that shouldn't double-fire (don't send two emails)

---

## 8. Common agent failure modes (have answers)
| Failure | Cause | Fix |
|---------|-------|-----|
| Infinite loop | No stop condition | Max iterations + loop detection |
| Wrong tool / bad args | Weak tool descriptions / schema | Better descriptions, enforce schema (your bug!) |
| Hallucinated tool result | Model invents data | Ground on real tool output; never let it guess |
| Runaway cost | Too many LLM calls | Cap iterations, route to cheaper model, cache |
| Context overflow | Too much history | Summary memory, trim, retrieve only relevant |
| Silent failure | No tracing | Structured logging + traces on every step |

---

## The sentence that proves you get agents
> "An agent is an LLM in a reason-act loop with tools and memory. For production I favor LangGraph — modeling the workflow as a state machine gives me cycles, branching, checkpointed state, and human-in-the-loop. I enforce structured tool outputs, cap iterations, add model fallbacks, and trace every tool call so I can root-cause failures — which is exactly how I found a tool-argument mismatch in my multi-agent system."
