"""Fix the flow: trim API key spaces, set model, rebuild ALL edges correctly
(chain + tool wires). Node ids are known from inspection."""
import os, json, requests

BASE = "http://127.0.0.1:7860"
KEY = os.environ.get("LANGFLOW_API_KEY", "")
FLOW_ID = "75f41c22-8509-4020-bfcb-e1e6d67f432b"
H = {"x-api-key": KEY, "Content-Type": "application/json"}

CHATIN = "ChatInput-4uNeu"
RESEARCH = "Agent-JoorG"
EMAIL = "Agent-KCvq7"
CHATOUT = "ChatOutput-VOzho"
MCP_SEARCH = "MCPTools-U9A1u"
MCP_EMAIL = "MCPTools-6Gqqh"

CAT_OUTPUT = {"allows_loop": False, "cache": True, "display_name": "Toolset",
              "group_outputs": False, "hidden": None, "method": "to_toolkit",
              "name": "component_as_tool", "options": None, "required_inputs": None,
              "selected": "Tool", "tool_mode": True, "types": ["Tool"], "value": "__UNDEFINED__"}


def enc(d, spaced):
    s = json.dumps(d, separators=(", ", ": ") if spaced else (",", ":"))
    return s.replace('"', "œ")


def msg_edge(src, src_type, tgt):
    name = "response" if src_type == "Agent" else "message"
    s = {"dataType": src_type, "id": src, "name": name, "output_types": ["Message"]}
    t = {"fieldName": "input_value", "id": tgt, "inputTypes": ["Message"], "type": "str"}
    return {"animated": False, "className": "", "data": {"sourceHandle": s, "targetHandle": t},
            "id": f"xy-edge__{src}{enc(s, False)}-{tgt}{enc(t, False)}", "selected": False,
            "source": src, "target": tgt, "sourceHandle": enc(s, True), "targetHandle": enc(t, True)}


def tool_edge(src, tgt):
    s = {"dataType": "MCPTools", "id": src, "name": "component_as_tool", "output_types": ["Tool"]}
    t = {"fieldName": "tools", "id": tgt, "inputTypes": ["Tool"], "type": "other"}
    return {"animated": False, "className": "", "data": {"sourceHandle": s, "targetHandle": t},
            "id": f"reactflow__edge-{src}{enc(s, False)}-{tgt}{enc(t, False)}", "selected": False,
            "source": src, "target": tgt, "sourceHandle": enc(s, True), "targetHandle": enc(t, True)}


flow = requests.get(f"{BASE}/api/v1/flows/{FLOW_ID}", headers=H, timeout=30).json()

for n in flow["data"]["nodes"]:
    t = n["data"]["type"]
    tmpl = n["data"]["node"]["template"]
    if t == "Agent":
        if tmpl.get("api_key", {}).get("value"):
            tmpl["api_key"]["value"] = tmpl["api_key"]["value"].strip()  # fix leading space
        if "model" in tmpl:
            tmpl["model"]["value"] = "gpt-4o-mini"
        print("fixed agent", n["id"])
    if t == "MCPTools":
        n["data"]["node"]["tool_mode"] = True
        n["data"]["selected_output"] = "component_as_tool"
        outs = n["data"]["node"]["outputs"]
        if not any(o.get("name") == "component_as_tool" for o in outs):
            outs.append(dict(CAT_OUTPUT))

# rebuild ALL edges correctly
flow["data"]["edges"] = [
    msg_edge(CHATIN, "ChatInput", RESEARCH),
    msg_edge(RESEARCH, "Agent", EMAIL),
    msg_edge(EMAIL, "Agent", CHATOUT),
    tool_edge(MCP_SEARCH, RESEARCH),
    tool_edge(MCP_EMAIL, EMAIL),
]
print("rebuilt edges:", len(flow["data"]["edges"]))

payload = {"name": flow["name"], "description": flow.get("description", ""),
           "data": flow["data"], "is_component": False, "endpoint_name": flow.get("endpoint_name")}
r = requests.patch(f"{BASE}/api/v1/flows/{FLOW_ID}", headers=H, data=json.dumps(payload), timeout=60)
print("Save:", r.status_code)
if r.status_code not in (200, 201):
    print(r.text[:500])
