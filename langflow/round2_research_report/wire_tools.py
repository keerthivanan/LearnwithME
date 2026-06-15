"""Enable tool-mode on both MCP nodes and add the tool edges:
search MCP -> Research Agent.tools, email MCP -> Email Agent.tools."""
import os, json, requests

BASE = "http://127.0.0.1:7860"
KEY = os.environ.get("LANGFLOW_API_KEY", "sk-REPLACE_WITH_YOUR_LANGFLOW_API_KEY")
FLOW_ID = "75f41c22-8509-4020-bfcb-e1e6d67f432b"
H = {"x-api-key": KEY, "Content-Type": "application/json"}

CAT_OUTPUT = {
    "allows_loop": False, "cache": True, "display_name": "Toolset",
    "group_outputs": False, "hidden": None, "method": "to_toolkit",
    "name": "component_as_tool", "options": None, "required_inputs": None,
    "selected": "Tool", "tool_mode": True, "types": ["Tool"], "value": "__UNDEFINED__",
}


def enc(d, spaced):
    s = json.dumps(d, separators=(", ", ": ") if spaced else (",", ":"))
    return s.replace('"', "œ")  # œ


def tool_edge(mcp_id, agent_id):
    src = {"dataType": "MCPTools", "id": mcp_id, "name": "component_as_tool", "output_types": ["Tool"]}
    tgt = {"fieldName": "tools", "id": agent_id, "inputTypes": ["Tool"], "type": "other"}
    return {
        "animated": False, "className": "",
        "data": {"sourceHandle": src, "targetHandle": tgt},
        "id": f"reactflow__edge-{mcp_id}{enc(src, False)}-{agent_id}{enc(tgt, False)}",
        "selected": False, "source": mcp_id, "target": agent_id,
        "sourceHandle": enc(src, True), "targetHandle": enc(tgt, True),
    }


flow = requests.get(f"{BASE}/api/v1/flows/{FLOW_ID}", headers=H, timeout=30).json()
nodes = flow["data"]["nodes"]
edges = flow["data"]["edges"]

# identify agents
research_agent = email_agent = None
for n in nodes:
    if n["data"]["type"] == "Agent":
        sp = (n["data"]["node"]["template"].get("system_prompt", {}).get("value") or "")
        if "Research Agent" in sp:
            research_agent = n["id"]
        elif "Email Agent" in sp:
            email_agent = n["id"]

# identify MCP nodes (search vs email)
search_mcp = email_mcp = None
for n in nodes:
    if n["data"]["type"] == "MCPTools":
        srv = n["data"]["node"]["template"].get("mcp_server", {}).get("value", {}) or {}
        name = srv.get("name", "")
        # enable tool mode + add component_as_tool output + selected_output
        n["data"]["node"]["tool_mode"] = True
        n["data"]["selected_output"] = "component_as_tool"
        outs = n["data"]["node"]["outputs"]
        if not any(o.get("name") == "component_as_tool" for o in outs):
            outs.append(dict(CAT_OUTPUT))
        if name == "web-search":
            search_mcp = n["id"]
        else:
            email_mcp = n["id"]

print("research_agent:", research_agent, "| email_agent:", email_agent)
print("search_mcp:", search_mcp, "| email_mcp:", email_mcp)

# add edges (avoid duplicates)
existing_ids = {e["id"] for e in edges}
added = 0
if search_mcp and research_agent:
    e = tool_edge(search_mcp, research_agent)
    if e["id"] not in existing_ids:
        edges.append(e); added += 1; print("added search tool edge")
if email_mcp and email_agent:
    e = tool_edge(email_mcp, email_agent)
    if e["id"] not in existing_ids:
        edges.append(e); added += 1; print("added email tool edge")

payload = {"name": flow["name"], "description": flow.get("description", ""),
           "data": flow["data"], "is_component": False, "endpoint_name": flow.get("endpoint_name")}
r = requests.patch(f"{BASE}/api/v1/flows/{FLOW_ID}", headers=H, data=json.dumps(payload), timeout=60)
print("Save status:", r.status_code, "| edges added:", added)
if r.status_code not in (200, 201):
    print(r.text[:600])
