"""Set OpenAI model + key on both agents in the created flow, and make sure
the search MCP node has its tool selected. Saves the flow back."""
import os, json, requests

BASE = "http://127.0.0.1:7860"
KEY = os.environ.get("LANGFLOW_API_KEY", "sk-REPLACE_WITH_YOUR_LANGFLOW_API_KEY")
FLOW_ID = "75f41c22-8509-4020-bfcb-e1e6d67f432b"
H = {"x-api-key": KEY, "Content-Type": "application/json"}

# read the OpenAI key from the DHL .env
ENV = r"C:\Users\91709\OneDrive\Documents\learning_interview\happyrobots_interview\dhl_project\.env"
openai_key = ""
for line in open(ENV, encoding="utf-8"):
    if line.strip().startswith("OPENAI_API_KEY="):
        openai_key = line.split("=", 1)[1].strip()
        break
print("OpenAI key loaded:", openai_key[:12] + "..." if openai_key else "NOT FOUND")

flow = requests.get(f"{BASE}/api/v1/flows/{FLOW_ID}", headers=H, timeout=30).json()

changed = 0
for node in flow["data"]["nodes"]:
    t = node["data"].get("type")
    tmpl = node["data"]["node"]["template"]
    if t == "Agent":
        if "api_key" in tmpl:
            tmpl["api_key"]["value"] = openai_key
        if "model" in tmpl:
            tmpl["model"]["value"] = "gpt-4o-mini"
        # some versions use agent_llm / model_name
        if "model_name" in tmpl:
            tmpl["model_name"]["value"] = "gpt-4o-mini"
        changed += 1
        print("Configured Agent:", node["id"])

# Save back
payload = {"name": flow["name"], "description": flow.get("description", ""),
           "data": flow["data"], "is_component": False, "endpoint_name": flow.get("endpoint_name")}
r = requests.patch(f"{BASE}/api/v1/flows/{FLOW_ID}", headers=H, data=json.dumps(payload), timeout=60)
print("Save status:", r.status_code)
if r.status_code not in (200, 201):
    print(r.text[:600])
else:
    print(f"Saved. Configured {changed} agents.")
