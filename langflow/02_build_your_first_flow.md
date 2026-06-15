# Build Your First Flow — Step by Step

Goal: build a working AI chatbot in Langflow in ~10 minutes.

---

## The fastest way: start from a template

When Langflow opens at `localhost:7860`:
1. Click **"New Flow"** (or **"New Project"**)
2. You'll see **templates** — click **"Basic Prompting"** (or "Simple Agent")
3. Langflow drops a ready-made flow on the canvas 🎉

This template already has: Chat Input → Prompt → Language Model → Chat Output.

---

## Building it from scratch (so you understand it)

If you start with a **blank flow**, here's how to build the chatbot:

### Step 1 — Add Chat Input
- In the left sidebar, find **"Inputs"** → drag **"Chat Input"** onto the canvas.

### Step 2 — Add a Prompt
- Sidebar → **"Prompts"** → drag **"Prompt"** onto the canvas.
- In the Prompt's text box, type:
  ```
  You are a friendly assistant. Answer the user's question clearly.

  User: {user_message}
  ```
- Typing `{user_message}` creates a new input port called `user_message` on the Prompt.

### Step 3 — Add a Language Model
- Sidebar → **"Models"** → drag **"OpenAI"** (or "Language Model").
- Fill in the **Model** (e.g., `gpt-4o-mini`) and your **API Key**.
  - Better: store the key as a Global Variable first, then select it here.

### Step 4 — Add Chat Output
- Sidebar → **"Outputs"** → drag **"Chat Output"**.

### Step 5 — Wire them together
Draw these connections (output port → input port):
```
Chat Input  ──→  Prompt (the {user_message} port)
Prompt      ──→  Language Model (input/prompt)
Language Model ──→ Chat Output
```

### Step 6 — Test it
- Click **"Playground"** (top right)
- Type: *"Explain what Langflow is in one sentence"*
- Watch your bot reply! ✅

**Congratulations — you built your first AI app in Langflow.**

---

## Add MEMORY (make it remember the conversation)

A plain flow forgets everything after each message. To add memory:
1. Sidebar → **"Helpers"** or **"Memory"** → drag **"Chat Memory"** (or "Message History")
2. Connect it so the model receives past messages
3. Now it remembers context across turns — like the shared memory in your DHL project

Many model/agent components have a built-in **"memory"** toggle too — even easier.

---

## Common beginner mistakes (and fixes)

| Problem | Fix |
|---------|-----|
| Wire won't connect | The port types don't match — check colors |
| Model errors | Wrong/missing API key, or out of credits |
| `{variable}` not working | Make sure the curly-brace name matches the connected port |
| Playground shows nothing | Chat Output not connected, or a component errored (red) |
| Nothing happens | A component in the chain has an error — look for a red badge |

---

## What you just learned
- How to add components and wire them
- The core chain: **Input → Prompt → Model → Output**
- How to test in the Playground
- How to add memory

That's a real, working AI chatbot. Next, learn what every component does.

---

Next → **03_components_deep_dive.md**
