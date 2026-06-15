# 03 — The Two MCP Servers (Code Explained + How to Run)

You have two ready-made MCP servers in the `mcp_servers/` folder. This file explains exactly what they do, line by line, and how to run them.

---

## Server 1 — `search_mcp_server.py` (the Research Agent's tool)

### What it does
Exposes ONE tool: **`web_search(query, max_results)`**. It searches DuckDuckGo (no API key needed) and returns titles, snippets, and URLs.

### The code explained
```python
from mcp.server.fastmcp import FastMCP      # FastMCP = the easy way to build an MCP server
mcp = FastMCP("web-search")                 # create the server, name it "web-search"

@mcp.tool()                                 # this decorator turns the function into an MCP TOOL
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web... """                # ← the docstring becomes the tool DESCRIPTION the LLM reads
    results = DDGS().text(query, ...)        # do the actual search
    return formatted_results                 # return text the agent can read

if __name__ == "__main__":
    mcp.run()                               # start the server over STDIO
```

**Key things to KNOW (for the interview):**
- `@mcp.tool()` is what **exposes** the function as an MCP tool. The function name = tool name. The **docstring = the description** the LLM uses to decide when to call it. The **type hints** (`query: str`) become the input schema.
- `mcp.run()` with no args = **STDIO transport** (Langflow launches it as a subprocess).

### How to test it
```powershell
cd "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers"
uv run --with mcp --with ddgs python search_mcp_server.py
```
It starts and **waits silently** (it's listening for an MCP client). That silence = success. Press **Ctrl+C** to stop.

---

## Server 2 — `smtp_mcp_server.py` (the Email Agent's tool)

### What it does
Exposes ONE tool: **`send_email(to, subject, body)`**. It sends a real email through Gmail's SMTP server.

### Setup needed (one time)
Gmail won't accept your normal password from a script. You need an **App Password**:
1. Google Account → **Security** → turn on **2-Step Verification**
2. Google Account → **Security** → **App passwords** → create one → copy the **16 characters**
3. That's your `GMAIL_APP_PASSWORD` (your gmail is `GMAIL_ADDRESS`)

### The code explained
```python
@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    sender = os.environ.get("GMAIL_ADDRESS")        # read creds from env vars (never hardcode)
    password = os.environ.get("GMAIL_APP_PASSWORD")
    msg = MIMEMultipart()                           # build the email
    msg["To"] = to; msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:   # secure connection to Gmail
        server.login(sender, password)
        server.sendmail(sender, to, msg.as_string())          # send it
    return "SUCCESS: email sent..."
```

**Note the error handling** — it returns clear messages (`AUTH_ERROR`, `SEND_ERROR`) instead of crashing. That covers part of the BONUS "error handling" requirement.

### How to test it
```powershell
$env:GMAIL_ADDRESS="youremail@gmail.com"
$env:GMAIL_APP_PASSWORD="your16charpassword"
cd "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers"
uv run --with mcp python smtp_mcp_server.py
```
Starts silently = success. Ctrl+C to stop.

---

## The exact commands Langflow needs (copy these into the MCP Tools component)

### For the Research Agent's MCP Tools (STDIO):
```
uv run --with mcp --with ddgs python "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers\search_mcp_server.py"
```

### For the Email Agent's MCP Tools (STDIO):
```
Command: uv run --with mcp python "C:\Users\91709\OneDrive\Documents\learning_interview\langflow\round2_research_report\mcp_servers\smtp_mcp_server.py"
Env vars:
   GMAIL_ADDRESS = youremail@gmail.com
   GMAIL_APP_PASSWORD = your16charpassword
```

---

## Why `uv run --with ...`?
- `uv run` runs a Python command in a managed environment.
- `--with mcp --with ddgs` tells uv to **temporarily install** those packages just for this run — so you don't need a separate `pip install`. Clean and reliable.

---

## What you can SAY about these servers
> "Each server uses FastMCP. The `@mcp.tool()` decorator exposes a Python function as an MCP tool — the function name becomes the tool name, the docstring becomes the description the LLM reads, and the type hints define the input schema. `mcp.run()` serves it over STDIO, so Langflow's MCP Tools component launches it as a subprocess and discovers the tool automatically. The search server wraps DuckDuckGo; the email server wraps Gmail SMTP with credentials read from environment variables and proper error handling."

---

Next → `04_BUILD_IN_LANGFLOW.md`
