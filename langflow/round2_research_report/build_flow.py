"""
Builds the Round-2 multi-agent flow directly inside the user's Langflow (v1.10)
via the REST API, using the REAL component templates from /api/v1/all so it
imports clean (no broken nodes).

Run:  uv run --with requests python build_flow.py
"""
import json, random, string, copy, requests

BASE = "http://127.0.0.1:7860"
import os
KEY = os.environ.get("LANGFLOW_API_KEY", "sk-REPLACE_WITH_YOUR_LANGFLOW_API_KEY")
H = {"x-api-key": KEY, "Content-Type": "application/json"}

Q = "œ"  # the special quote char Langflow uses in handle strings


def rid(prefix):
    return f"{prefix}-{''.join(random.choices(string.ascii_letters + string.digits, k=5))}"


def enc(d, spaced=True):
    """Encode a handle dict into Langflow's œ-string form."""
    s = json.dumps(d, separators=(", ", ": ") if spaced else (",", ":"))
    return s.replace('"', Q)


print("Fetching component catalog...")
allc = requests.get(f"{BASE}/api/v1/all", headers=H, timeout=120).json()

T_chatin = allc["input_output"]["ChatInput"]
T_chatout = allc["input_output"]["ChatOutput"]
T_agent = allc["models_and_agents"]["Agent"]
T_mcp = allc["models_and_agents"]["MCPTools"]


def make_node(template, comp_type, node_id, x, y):
    node = copy.deepcopy(template)
    return {
        "id": node_id,
        "type": "genericNode",
        "position": {"x": x, "y": y},
        "data": {"id": node_id, "type": comp_type, "node": node},
    }


# --- ids ---
ci = rid("ChatInput")
ra = rid("Agent")        # Research Agent
ea = rid("Agent")        # Email Agent
co = rid("ChatOutput")
ms = rid("MCPTools")     # search
me = rid("MCPTools")     # email

# --- nodes ---
n_ci = make_node(T_chatin, "ChatInput", ci, 100, 300)
n_ra = make_node(T_agent, "Agent", ra, 560, 250)
n_ea = make_node(T_agent, "Agent", ea, 1040, 250)
n_co = make_node(T_chatout, "ChatOutput", co, 1500, 300)
n_ms = make_node(T_mcp, "MCPTools", ms, 560, 650)
n_me = make_node(T_mcp, "MCPTools", me, 1040, 650)

# --- set agent instructions ---
n_ra["data"]["node"]["template"]["system_prompt"]["value"] = (
    "You are a Research Agent. The user message contains a TOPIC and a recipient EMAIL address.\n"
    "1) Use the web_search tool to gather current, factual information about the topic.\n"
    "2) Write a structured report with sections: SUMMARY, KEY FINDINGS (bullets), REFERENCES (URLs).\n"
    "At the very TOP of your output, write one line exactly: 'RECIPIENT: <the email from the user message>'.\n"
    "Then the full report. Do not send any email."
)
n_ea["data"]["node"]["template"]["system_prompt"]["value"] = (
    "You are an Email Agent. Your input starts with a line 'RECIPIENT: <email>' followed by a full research report.\n"
    "Use the send_email tool to send the COMPLETE report (all sections, do not summarize) to that recipient.\n"
    "Write a clear professional subject line based on the topic. After sending, reply with a short confirmation."
)


def msg_edge(src_id, src_type, tgt_id):
    src = {"dataType": src_type, "id": src_id, "name": "response" if src_type == "Agent" else "message", "output_types": ["Message"]}
    tgt = {"fieldName": "input_value", "id": tgt_id, "inputTypes": ["Message"], "type": "str"}
    return {
        "animated": False, "className": "",
        "data": {"sourceHandle": src, "targetHandle": tgt},
        "id": f"xy-edge__{src_id}{enc(src, False)}-{tgt_id}{enc(tgt, False)}",
        "selected": False,
        "source": src_id, "target": tgt_id,
        "sourceHandle": enc(src, True),
        "targetHandle": enc(tgt, True),
    }


edges = [
    msg_edge(ci, "ChatInput", ra),
    msg_edge(ra, "Agent", ea),
    msg_edge(ea, "Agent", co),
]

flow = {
    "name": "Round2 - Research & Email (Multi-Agent + MCP)",
    "description": "Research Agent (web_search via MCP) -> Email Agent (send_email via MCP). Wire the two MCP Tools to each Agent's Tools input and set OpenAI key.",
    "data": {"nodes": [n_ci, n_ra, n_ea, n_co, n_ms, n_me], "edges": edges},
    "is_component": False,
    "endpoint_name": None,
}

print("Creating flow in Langflow...")
r = requests.post(f"{BASE}/api/v1/flows/", headers=H, data=json.dumps(flow), timeout=60)
if r.status_code in (200, 201):
    j = r.json()
    print("SUCCESS! Flow created.")
    print("Flow id:", j.get("id"))
    print("Open:", f"{BASE}/flow/{j.get('id')}")
else:
    print("FAILED:", r.status_code)
    print(r.text[:800])
