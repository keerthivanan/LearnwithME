# Components Deep Dive

The components you'll actually use, grouped by category. You don't need all of them — master these.

---

## 📥 INPUTS
| Component | What it does |
|-----------|-------------|
| **Chat Input** | Captures the user's chat message (the start of most flows) |
| **Text Input** | A plain text field for non-chat data |

## 📤 OUTPUTS
| Component | What it does |
|-----------|-------------|
| **Chat Output** | Shows the AI's reply in the Playground chat |
| **Text Output** | Outputs plain text (for non-chat flows) |

---

## 🧠 MODELS (the AI brain)
| Component | What it does |
|-----------|-------------|
| **OpenAI** | Calls GPT-4o, GPT-4o-mini, etc. |
| **Anthropic** | Calls Claude models |
| **Google Generative AI** | Calls Gemini |
| **Ollama** | Runs local open-source models (LLaMA, Mistral) on your machine |
| **Language Model** | A generic model component — pick the provider inside |

**Key fields:** model name, API key, **temperature** (0 = focused, 1 = creative), max tokens.

---

## 📝 PROMPTS
| Component | What it does |
|-----------|-------------|
| **Prompt** | A template with `{variables}`. Each `{variable}` becomes an input port. This is how you give the AI instructions + inject data. |

Example:
```
You are a DHL support agent. Answer using ONLY this context:
{context}

Question: {question}
```
This creates two ports: `context` and `question`.

---

## 🧩 HELPERS / MEMORY
| Component | What it does |
|-----------|-------------|
| **Chat Memory / Message History** | Stores past messages so the AI remembers the conversation |
| **Combine Text** | Merges multiple text pieces into one |
| **Structured Output** | Forces the model to return structured JSON (like Pydantic in your DHL tools) |

---

## 🤖 AGENTS
| Component | What it does |
|-----------|-------------|
| **Agent** | An AI that can **decide to use tools**. You give it tools, it picks which to call. (This is the multi-agent idea from your DHL project.) |

The Agent component is the heart of "agentic" Langflow. Connect tools to it and it reasons about which to use.

---

## 🔧 TOOLS
| Component | What it does |
|-----------|-------------|
| **Search (Tavily/SerpAPI)** | Web search |
| **Calculator** | Math |
| **Python REPL** | Runs Python code |
| **API Request** | Calls any REST API (like your DHL tools calling endpoints) |
| **Custom Component** | Write your own Python tool |

Tools connect into an **Agent's** tool port.

---

## 📚 DATA & RAG (chat with documents)
| Component | What it does |
|-----------|-------------|
| **File** | Load a PDF/TXT/CSV/DOCX |
| **URL** | Load a webpage |
| **Split Text** | Chops documents into chunks |
| **Embeddings (OpenAI/HuggingFace)** | Turns text chunks into vectors (numbers) |
| **Vector Store (Astra DB, Chroma, FAISS, Pinecone)** | Stores & searches vectors |

These build a **RAG pipeline** (next file).

---

## ⚙️ LOGIC
| Component | What it does |
|-----------|-------------|
| **If-Else / Conditional Router** | Branch the flow based on a condition |
| **Listen / Notify** | Pass data between flows |
| **Loop** | Repeat over a list |

---

## 🛠️ Custom Components (the power feature)
Langflow lets you write your **own component in Python**. Click "New Custom Component" and write a class. This is how you extend Langflow for anything it doesn't have built-in — like a custom DHL tracking tool.

```python
from langflow.custom import Component
from langflow.io import MessageTextInput, Output

class DHLTracker(Component):
    display_name = "DHL Tracker"
    inputs = [MessageTextInput(name="tracking_number", display_name="Tracking #")]
    outputs = [Output(name="status", display_name="Status", method="track")]

    def track(self) -> str:
        # your lookup logic here
        return f"Shipment {self.tracking_number} is out for delivery."
```

---

## The 6 components you'll use 90% of the time
1. **Chat Input**
2. **Prompt**
3. **OpenAI / Language Model**
4. **Chat Output**
5. **Agent** (when you need tools)
6. **Vector Store + Embeddings** (when you need RAG)

Learn these well and you can build almost anything.

---

Next → **04_rag_and_agents.md**
