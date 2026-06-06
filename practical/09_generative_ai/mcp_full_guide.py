"""
MCP — Model Context Protocol
==============================
What it is, why it exists, how it works, and how to build one.

Install: pip install anthropic mcp
"""

# ════════════════════════════════════════════════════════════
# WHAT IS MCP?
# ════════════════════════════════════════════════════════════
#
#  MCP = Model Context Protocol
#  Created by Anthropic (open standard)
#
#  PROBLEM it solves:
#  ─────────────────
#  Claude (LLM) lives in isolation.
#  It cannot:
#    ✗ Read files on your computer
#    ✗ Query your database
#    ✗ Call your internal APIs
#    ✗ Search the web
#    ✗ Run code
#
#  MCP = a standard way to give Claude TOOLS to do these things.
#
#  Think of it like:
#  ─────────────────
#  Claude = a super smart employee
#  MCP    = giving that employee a computer, phone, and access to files
#
#  Without MCP:  Claude only knows what you type to it
#  With MCP:     Claude can read your files, query DB, call APIs
#
# ════════════════════════════════════════════════════════════
# HOW IT WORKS (Simple Flow)
# ════════════════════════════════════════════════════════════
#
#   You (User)
#      ↓  "Summarize all .py files in my project"
#   Claude (LLM)
#      ↓  "I need to read files — let me call the file tool"
#   MCP Server  ← reads your actual filesystem
#      ↓  returns file contents
#   Claude
#      ↓  "Here's a summary of your Python files..."
#   You (User) ✓
#
#  MCP Server = a small program that exposes TOOLS to Claude
#  Claude decides WHEN and HOW to call those tools
#
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
# PART 1 — TOOL USE (Foundation of MCP)
# ════════════════════════════════════════════════════════════
# Before MCP, understand Tool Use — Claude calling functions

import anthropic
import json

client = anthropic.Anthropic()

# Step 1: Define tools (tell Claude what tools exist)
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name e.g. Mumbai, Delhi"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Do math calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '2 + 2' or '10 * 5'"
                }
            },
            "required": ["expression"]
        }
    }
]

# Step 2: Your actual tool functions (the real logic)
def get_weather(city: str, unit: str = "celsius") -> dict:
    # In real app: call a weather API here
    weather_data = {
        "Mumbai": {"temp": 32, "condition": "Humid", "humidity": 85},
        "Delhi":  {"temp": 28, "condition": "Sunny", "humidity": 40},
        "Chennai": {"temp": 35, "condition": "Hot",   "humidity": 90},
    }
    data = weather_data.get(city, {"temp": 25, "condition": "Unknown", "humidity": 60})
    return {
        "city": city,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"],
        "humidity": data["humidity"]
    }

def calculate(expression: str) -> dict:
    try:
        result = eval(expression)  # use safer parser in production!
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_weather":
        result = get_weather(**tool_input)
    elif tool_name == "calculate":
        result = calculate(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)

# Step 3: The tool use loop
def chat_with_tools(user_message: str):
    print(f"\nUser: {user_message}")
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        # Claude wants to use a tool
        if response.stop_reason == "tool_use":
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            # Add Claude's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Run each tool and collect results
            tool_results = []
            for tool_call in tool_calls:
                print(f"  → Claude calling: {tool_call.name}({tool_call.input})")
                result = run_tool(tool_call.name, tool_call.input)
                print(f"  ← Tool returned: {result}")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_call.id,
                    "content":     result
                })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        # Claude is done, return final answer
        elif response.stop_reason == "end_turn":
            final = response.content[0].text
            print(f"Claude: {final}")
            return final

# Test it
# chat_with_tools("What's the weather in Mumbai and what is 15 * 24?")
# Output:
#   → Claude calling: get_weather({'city': 'Mumbai'})
#   ← Tool returned: {"city": "Mumbai", "temperature": 32, ...}
#   → Claude calling: calculate({'expression': '15 * 24'})
#   ← Tool returned: {"expression": "15 * 24", "result": 360}
#   Claude: In Mumbai it's 32°C and humid. 15 × 24 = 360.


# ════════════════════════════════════════════════════════════
# PART 2 — BUILD YOUR OWN MCP SERVER
# ════════════════════════════════════════════════════════════
# MCP Server = exposes tools that Claude can call
# Save this as: my_mcp_server.py  and run it separately

MCP_SERVER_CODE = '''
# my_mcp_server.py
# Run: python my_mcp_server.py
# pip install mcp

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import os, json, asyncio
from pathlib import Path

# Create the MCP server
server = Server("my-tools-server")

# ── TOOL 1: Read a file ───────────────────────────────
@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="read_file",
            description="Read contents of a file from the filesystem",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="list_files",
            description="List all files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path"},
                    "extension": {"type": "string", "description": "Filter by extension e.g. .py"}
                },
                "required": ["directory"]
            }
        ),
        types.Tool(
            name="search_in_files",
            description="Search for a keyword in all files in a folder",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "keyword":   {"type": "string"}
                },
                "required": ["directory", "keyword"]
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):

    if name == "read_file":
        path = arguments["path"]
        if not os.path.exists(path):
            return [types.TextContent(type="text", text=f"Error: {path} not found")]
        with open(path) as f:
            content = f.read()
        return [types.TextContent(type="text", text=content)]

    elif name == "list_files":
        directory = arguments["directory"]
        ext = arguments.get("extension", "")
        files = []
        for f in Path(directory).rglob("*"):
            if f.is_file():
                if not ext or f.suffix == ext:
                    files.append(str(f))
        return [types.TextContent(type="text", text=json.dumps(files, indent=2))]

    elif name == "search_in_files":
        directory = arguments["directory"]
        keyword   = arguments["keyword"]
        results   = []
        for f in Path(directory).rglob("*.py"):
            content = f.read_text(errors="ignore")
            if keyword in content:
                lines = [f"Line {i+1}: {line.strip()}"
                         for i, line in enumerate(content.splitlines())
                         if keyword in line]
                results.append({"file": str(f), "matches": lines})
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

# Run the server
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
'''

print("MCP Server code ready — save as my_mcp_server.py")


# ════════════════════════════════════════════════════════════
# PART 3 — MCP WITH DATABASE TOOLS
# ════════════════════════════════════════════════════════════

MCP_DB_SERVER = '''
# db_mcp_server.py
import sqlite3, json, asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("db-server")
DB_PATH = "company.db"

def query_db(sql: str, params: tuple = ()) -> list:
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql, params)
    columns = [d[0] for d in cursor.description] if cursor.description else []
    rows    = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="run_sql",
            description="Run a SELECT SQL query on the company database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_table_info",
            description="Get list of tables and their columns",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "run_sql":
        sql = arguments["query"]
        # Security: only allow SELECT
        if not sql.strip().upper().startswith("SELECT"):
            return [types.TextContent(type="text", text="Only SELECT queries allowed")]
        results = query_db(sql)
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "get_table_info":
        tables = query_db("SELECT name FROM sqlite_master WHERE type=\'table\'")
        info = {}
        for t in tables:
            table_name = t["name"]
            cols = query_db(f"PRAGMA table_info({table_name})")
            info[table_name] = [c["name"] for c in cols]
        return [types.TextContent(type="text", text=json.dumps(info, indent=2))]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

asyncio.run(main())
'''


# ════════════════════════════════════════════════════════════
# PART 4 — CONNECT MCP SERVER TO CLAUDE (Client Side)
# ════════════════════════════════════════════════════════════

MCP_CLIENT_CODE = '''
# mcp_client.py — connects to MCP server and uses it with Claude
import asyncio
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def chat_with_mcp(user_question: str):
    client = anthropic.Anthropic()

    # Connect to YOUR MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["my_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get all tools from MCP server
            tools_response = await session.list_tools()
            tools = [
                {
                    "name":        t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools_response.tools
            ]

            print(f"Available tools: {[t['name'] for t in tools]}")

            # Chat loop
            messages = [{"role": "user", "content": user_question}]

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    tools=tools,
                    messages=messages
                )

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  → Calling tool: {block.name}")
                            # Actually call the MCP tool!
                            result = await session.call_tool(block.name, block.input)
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     result.content[0].text
                            })

                    messages.append({"role": "user", "content": tool_results})

                else:
                    return response.content[0].text

# Run
async def main():
    answer = await chat_with_mcp(
        "List all Python files in my project and summarize what each one does"
    )
    print(answer)

asyncio.run(main())
'''


# ════════════════════════════════════════════════════════════
# PART 5 — CONFIGURE IN CLAUDE DESKTOP (claude_desktop_config.json)
# ════════════════════════════════════════════════════════════

CLAUDE_DESKTOP_CONFIG = '''
// File: ~/Library/Application Support/Claude/claude_desktop_config.json  (Mac)
// File: %APPDATA%/Claude/claude_desktop_config.json  (Windows)

{
  "mcpServers": {
    "my-file-tools": {
      "command": "python",
      "args": ["/full/path/to/my_mcp_server.py"]
    },
    "my-db-tools": {
      "command": "python",
      "args": ["/full/path/to/db_mcp_server.py"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/folder"]
    }
  }
}

// After saving → restart Claude Desktop
// Now Claude can use your tools directly in the chat!
'''


# ════════════════════════════════════════════════════════════
# PART 6 — REAL WORLD USE CASES
# ════════════════════════════════════════════════════════════

USE_CASES = """
Real World MCP Use Cases:
─────────────────────────

1. CODING ASSISTANT
   Tools: read_file, write_file, run_python, search_code
   → Claude reads your code, fixes bugs, writes new files

2. DATA ANALYSIS
   Tools: run_sql, read_csv, plot_chart, get_stats
   → "Analyze sales data and show me trends" → Claude runs SQL, plots graphs

3. CUSTOMER SUPPORT BOT
   Tools: search_knowledge_base, get_ticket, update_ticket
   → Claude searches your docs, answers customers, updates tickets

4. DEVOPS ASSISTANT
   Tools: check_logs, restart_service, get_metrics, deploy
   → "Why is production slow?" → Claude reads logs, finds issue

5. RESEARCH ASSISTANT
   Tools: search_web, read_pdf, save_notes, summarize
   → Claude searches web, reads papers, saves summaries

6. PERSONAL ASSISTANT
   Tools: read_calendar, send_email, create_task, search_files
   → "Schedule a meeting and send invite" → Claude does it all
"""

print(USE_CASES)


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════

SUMMARY = """
MCP in one picture:
────────────────────

  You:     "Summarize my project code"
      ↓
  Claude:  (thinks) I need to read files
      ↓
  MCP:     reads files from your computer
      ↓
  Claude:  reads content, generates summary
      ↓
  You:     get the answer ✓

Key Concepts:
─────────────
  MCP Server  = you build this → exposes tools (read_file, run_sql, call_api)
  MCP Client  = connects to server → passes tools to Claude
  Tool Use    = Claude decides when to call which tool
  Loop        = Claude calls tools → gets results → calls more or answers

Files to create:
────────────────
  my_mcp_server.py        ← your tools live here
  mcp_client.py           ← connects server + Claude
  claude_desktop_config   ← for Claude Desktop app

Think of MCP as:
────────────────
  API = humans call functions
  MCP = Claude calls functions automatically when it needs them
"""

print(SUMMARY)
