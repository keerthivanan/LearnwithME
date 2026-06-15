# 04 — Build It In Langflow (Every Node, Every Wire, With WHY)

Follow this in Langflow (`http://localhost:7860`). Take your time — understand each step.

---

## Before you start
- Langflow is open
- You have your **OpenAI API key** ready (for the agents' brains)
- You have your **Gmail address + app password** ready (for the email server)
- The two MCP server files exist in `mcp_servers/`

---

## The nodes you'll add (8 total)
1. Text Input — **Topic**
2. Text Input — **Recipient Email**
3. **MCP Tools** — Search (connects to search_mcp_server.py)
4. **Agent** — Research Agent
5. **Prompt** — combine report + email (the handoff)
6. **MCP Tools** — Email (connects to smtp_mcp_server.py)
7. **Agent** — Email Agent
8. **Chat Output** — final confirmation

---

## STEP 1 — Create a blank flow
- Click **New Flow** → **Blank Flow**.

## STEP 2 — Add the two inputs
- Drag **Text Input** (from "Inputs") → rename it **"Topic"**.
- Drag another **Text Input** → rename it **"Recipient Email"**.
- *Why two inputs?* The flow needs a topic AND an email at the start. (This still counts as "single user trigger" — both are provided together when you run the flow.)

## STEP 3 — Add the Search MCP Tools (the search tool, via MCP)
- In the component search box (left), type **"MCP"** → drag **"MCP Tools"** onto the canvas.
- Set **Mode/Transport = STDIO**.
- In **Command**, paste:
  ```
  uv run --with mcp --with ddgs python "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers\search_mcp_server.py"
  ```
- Langflow connects and shows the tool **`web_search`**. ✅ (If it shows the tool, your MCP server is working.)
- *Why:* This is the MANDATORY MCP requirement — the search tool comes from an MCP server, not a native component.

## STEP 4 — Add the Research Agent
- Drag an **Agent** component (from "Agents").
- Set its **Model** = OpenAI, model `gpt-4o-mini`, paste your **API key** (or use a Global Variable).
- In its **System Prompt / Instructions**, paste:
  ```
  You are a Research Agent. You are given a TOPIC.
  Use the web_search tool to gather current, factual information about the topic.
  You may search more than once with different queries if needed.
  Then write a STRUCTURED REPORT with exactly these three sections:

  SUMMARY:
  (3-5 sentence overview)

  KEY FINDINGS:
  (5-8 bullet points of the most important facts)

  REFERENCES:
  (the source URLs you used, as a numbered list)

  Output ONLY the report. Do not send any email — that is another agent's job.
  ```
- *Why:* The system prompt makes it RESEARCH and produce the exact report structure the assignment requires.

## STEP 5 — Wire search tool + topic into the Research Agent
- Connect **MCP Tools (Search)** output → Research Agent's **Tools** input.
- Connect **Text Input (Topic)** output → Research Agent's **Input** (the message/task).
- *Why:* Now the agent has the topic AND the ability to search.

## STEP 6 — Add a Prompt to combine report + email (the handoff)
- Drag a **Prompt** component. In its template, type:
  ```
  Send the following research report by email to: {recipient}

  ----- REPORT START -----
  {report}
  ----- REPORT END -----

  Email the FULL report above (all sections) to the recipient. Write a clear subject line.
  ```
  Typing `{recipient}` and `{report}` creates two input ports.
- Connect **Text Input (Recipient Email)** → the `{recipient}` port.
- Connect **Research Agent** output → the `{report}` port.
- *Why:* This is where **agent-to-agent communication** is visible — the Research Agent's output flows into the instruction for the Email Agent. **Point at this wire in the interview.**

## STEP 7 — Add the Email MCP Tools (the email tool, via MCP)
- Drag another **MCP Tools** component. Mode = **STDIO**.
- **Command:**
  ```
  uv run --with mcp python "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers\smtp_mcp_server.py"
  ```
- Add **Environment Variables**:
  ```
  GMAIL_ADDRESS = youremail@gmail.com
  GMAIL_APP_PASSWORD = your16charpassword
  ```
- It should discover the tool **`send_email`**. ✅

## STEP 8 — Add the Email Agent
- Drag another **Agent**. Model = OpenAI gpt-4o-mini + key.
- **System Prompt:**
  ```
  You are an Email Agent. You receive an instruction containing a recipient email
  address and a full research report.
  Use the send_email tool to send the COMPLETE report as the email body.
  Write a clear, professional subject line based on the report's topic.
  Do not shorten or summarize the report — send all sections in full.
  After sending, reply with a short confirmation including the recipient address.
  ```
- Connect **MCP Tools (Email)** → Email Agent's **Tools** input.
- Connect **Prompt** (from step 6) output → Email Agent's **Input**.

## STEP 9 — Add Chat Output
- Drag **Chat Output**. Connect **Email Agent** output → **Chat Output**.

---

## The finished wiring (check yours matches)
```
[Topic]──────────────► [Research Agent] ──► [Prompt {report}] ──► [Email Agent] ──► [Chat Output]
                            ▲                      ▲                    ▲
                   [MCP Search Tools]    [Recipient Email →{recipient}] [MCP Email Tools]
```

---

## STEP 10 — Run it
- Click **Playground** (or **Run**).
- Enter **Topic** (e.g., "Latest breakthroughs in quantum computing 2025") and **Recipient Email**.
- Watch: Research Agent searches (via MCP) → writes report → Prompt combines → Email Agent sends (via MCP) → confirmation.
- **Check the inbox** — the full report should be there. ✅

---

## If something doesn't work
| Problem | Fix |
|---------|-----|
| MCP Tools shows no tools | Test the server in PowerShell (file 03). Check the command path is exact (quotes around the path). |
| `uv` not found by Langflow | Use full path to uv: `(Get-Command uv).Source`, or `uv pip install --system mcp ddgs` then use `python "...path..."` |
| Email not sent | Check Gmail app password + 2FA; read the `send_email` return message (it explains the error) |
| Agent doesn't call the tool | Make sure the MCP Tools output is wired to the Agent's **Tools** input, and the prompt tells it to use the tool |
| Report cut off in email | In the Email Agent prompt, stress "send ALL sections in full, do not summarize" |

---

Next → `05_HOW_IT_ALL_WORKS.md`
