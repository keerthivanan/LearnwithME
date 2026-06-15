# 📋 COPY-PASTE KIT — Build the Flow Fast

Everything you need to paste, in one place. As you build in the UI, copy from here.
(Building this way ALWAYS works — and you'll understand it, which the interview tests.)

---

## 🔑 Things to have ready
- **OpenAI API key:** (the same one from your DHL `.env`)
- **Gmail address:** your gmail
- **Gmail app password:** 16 chars (Google Account → Security → App passwords)

---

## 🧱 The 8 boxes to add (in order)

1. **Chat Input**  (or Text Input) — for the topic + email
2. **MCP Tools** (#1, Search)
3. **Agent** (Research Agent)
4. **Prompt** (combiner)
5. **MCP Tools** (#2, Email)
6. **Agent** (Email Agent)
7. **Chat Output**

---

## 📌 PASTE #1 — Search MCP Tools → "Command" field (STDIO mode)
```
uv run --with mcp --with ddgs python "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers\search_mcp_server.py"
```

## 📌 PASTE #2 — Email MCP Tools → "Command" field (STDIO mode)
```
uv run --with mcp python "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers\smtp_mcp_server.py"
```
**Email MCP Tools → Environment Variables:**
```
GMAIL_ADDRESS = youremail@gmail.com
GMAIL_APP_PASSWORD = your16charapppassword
```

---

## 📌 PASTE #3 — Research Agent → "System Prompt / Instructions"
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

---

## 📌 PASTE #4 — Prompt (combiner) → template text
```
Send the following research report by email to: {recipient}

----- REPORT START -----
{report}
----- REPORT END -----

Email the FULL report above (all sections) to the recipient. Write a clear subject line.
```
(Typing `{recipient}` and `{report}` makes two input dots on the box.)

---

## 📌 PASTE #5 — Email Agent → "System Prompt / Instructions"
```
You are an Email Agent. You receive an instruction containing a recipient email
address and a full research report.
Use the send_email tool to send the COMPLETE report as the email body.
Write a clear, professional subject line based on the report's topic.
Do not shorten or summarize the report — send all sections in full.
After sending, reply with a short confirmation including the recipient address.
```

---

## 🔌 The wiring (draw these lines)
```
Chat Input ───────────────► Research Agent (input)
MCP Tools #1 (Search) ─────► Research Agent (Tools)

Research Agent ───────────► Prompt  ({report} dot)
Chat Input (email part) ──► Prompt  ({recipient} dot)

Prompt ───────────────────► Email Agent (input)
MCP Tools #2 (Email) ─────► Email Agent (Tools)

Email Agent ──────────────► Chat Output
```

---

## ✅ Both Agents → Model settings
- Model provider: **OpenAI**
- Model name: **gpt-4o-mini**
- API key: **(paste your OpenAI key)**

---

## ▶️ Test
Click **Playground** → enter a topic + your email → watch it search, write, and send.
Check your inbox for the full report. 🎉

---

## 💡 After it works
Click the **menu (⋮) → Export** to save YOUR working flow as JSON. THAT exported file
will re-import perfectly (because Langflow made it itself). Keep it as your backup.
