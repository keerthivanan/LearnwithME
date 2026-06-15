# Langflow Interview Q&A

Common questions + strong answers. Practice saying these out loud.

---

**Q: What is Langflow?**
> "Langflow is an open-source, visual low-code platform for building AI applications. It's built on LangChain and gives you a drag-and-drop canvas where you connect components — inputs, prompts, models, agents, tools, vector stores — to build chatbots, RAG systems, and AI agents. You can test flows in a built-in playground and deploy them as APIs or MCP servers."

---

**Q: What's a component in Langflow?**
> "A component is a single node that does one job — like a Chat Input, a Prompt template, an OpenAI model, or a vector store. Each has typed input and output ports, and you wire them together to form a flow. It's the same idea as a node in n8n."

---

**Q: How would you build a RAG system in Langflow?**
> "Two parts. First, ingestion: load documents with the File component, split them into chunks with Split Text, convert chunks to vectors with an Embeddings component, and store them in a Vector Store. Second, retrieval: embed the user's question, search the vector store for the most relevant chunks, inject them into a Prompt as context, and pass that to the model. The prompt instructs the model to answer only from the provided context, which prevents hallucination."

---

**Q: What's the difference between a Model and an Agent in Langflow?**
> "A Model component just generates text from a prompt — it's a one-shot call. An Agent can reason and decide to use tools — like web search, a calculator, or an API request — to gather information or take actions before answering. You use an Agent when the task needs live data or multi-step actions."

---

**Q: How do you handle secrets like API keys?**
> "Langflow has Global Variables. You store the API key once as a global variable, mark it as a secret, and reference it in components instead of pasting it everywhere. That keeps keys out of the flow definition and reusable."

---

**Q: How do you deploy a Langflow flow?**
> "Every flow can be run via API — Langflow generates a code snippet in Python, JavaScript, or curl that calls the flow endpoint. You can also publish a flow as an MCP server so AI assistants can call it as a tool, or run Langflow itself in Docker for production."

---

**Q: Langflow vs LangChain — what's the relationship?**
> "Langflow is built on top of LangChain. LangChain is the Python framework — you write code. Langflow is the visual layer over it — you drag and drop instead of coding, and it generates/uses LangChain under the hood. Langflow is faster for prototyping; LangChain gives you full code control."

---

**Q: Langflow vs n8n?**
> "Both are visual node-based builders, but n8n is for general business automation across hundreds of apps with triggers and schedules, while Langflow is purpose-built for AI — deep support for RAG, embeddings, agents, and prompts. I'd use n8n for the automation and orchestration layer, and Langflow for building the AI brain itself. They complement each other."

---

**Q: How do you add memory to a chatbot?**
> "Add a Chat Memory component (or enable the memory option on the model/agent). It stores the conversation history and feeds past messages back into the model, so the bot remembers context across turns. You can key memory by session or user."

---

**Q: Can you extend Langflow with custom logic?**
> "Yes — Custom Components. You write a Python class with defined inputs and outputs and a method, and it appears on the canvas like any built-in component. That's how you add custom tools or integrations Langflow doesn't ship with."

---

**Q: What are typed ports and why do they matter?**
> "Ports carry specific data types — Message, Text, Data, etc. Langflow only lets you connect compatible types, color-coded so you can see at a glance. It prevents wiring mistakes — you can't accidentally feed the wrong kind of data into a component."

---

## 30-second summary to memorize
> "Langflow is a visual, open-source builder for AI apps, built on LangChain. You drag components — inputs, prompts, models, agents, tools, vector stores — onto a canvas, wire them into a flow, test in the playground, and deploy as an API. It's ideal for prototyping RAG systems and agents fast, without heavy coding."

---

## If asked "have you used it?"
Be honest but show initiative:
> "I've installed Langflow and built flows with it — a basic chatbot and a RAG-style flow. I come from building a production multi-agent system in n8n for a DHL use case, so the node-based concepts transferred directly. Langflow's strength for me is rapid prototyping of the AI layer specifically."

That's honest, and it ties back to your real DHL project — your strongest asset.
