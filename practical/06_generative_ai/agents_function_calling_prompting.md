# Agents, Function Calling, Advanced Prompting & LLM Evaluation

> Function calling, ReAct agents, Tree of Thought, benchmarks — all missing until now.

---

## PART 1: FUNCTION CALLING & TOOL USE

### Why LLMs Need Tools

An LLM without tools is like a genius locked in a room with no internet.
It knows a lot but can't:
- Get real-time information
- Do precise math
- Run code
- Search databases
- Call APIs

Function calling gives LLMs the ability to use tools.

### How Function Calling Works (OpenAI Format)

You define tools as JSON schemas. The LLM decides when to call them.

```python
import openai

client = openai.OpenAI()

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris, France"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# First LLM call — model decides to use tool
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"   # let model decide, or "required" to force
)

# Model responds with tool call
tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "get_current_weather"
print(tool_call.function.arguments)  # '{"location": "Paris, France"}'

# Execute the actual function
import json
args = json.loads(tool_call.function.arguments)
weather_result = get_current_weather(**args)   # your actual function

# Second LLM call — model uses tool result
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
    response.choices[0].message,     # model's tool call message
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(weather_result)
    }
]
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)
print(final_response.choices[0].message.content)
# "The current weather in Paris is 18°C and partly cloudy..."
```

### Parallel Tool Calls

Modern LLMs can call multiple tools simultaneously:

```python
# User: "What's the weather in Paris and London?"
# Model: calls get_weather(Paris) AND get_weather(London) in parallel

tool_calls = response.choices[0].message.tool_calls
# tool_calls has 2 items — both weather calls
```

### Anthropic Tool Use Format

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "search_database",
        "description": "Search the product database",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Find products similar to noise cancelling headphones"}]
)

# Check if Claude wants to use a tool
if response.stop_reason == "tool_use":
    tool_use = next(b for b in response.content if b.type == "tool_use")
    print(tool_use.name)    # "search_database"
    print(tool_use.input)   # {"query": "noise cancelling headphones", "max_results": 5}
```

---

## PART 2: REACT AGENTS — REASON + ACT

### The ReAct Framework (Yao et al., 2022)

ReAct = Reason + Act. Interleave reasoning (thought) with actions (tool calls).

```
Standard generation: Input → [Think] → Output (one shot)
ReAct:               Input → [Think] → [Act] → [Observe] → [Think] → [Act] → ... → Output
```

### The ReAct Loop

```
Thought: I need to find the population of Tokyo in 2024.
         Let me search for this.
Action: search["Tokyo population 2024"]
Observation: Tokyo's population in 2024 is approximately 13.96 million
             in the city proper (37.4M in greater metro area)

Thought: I have the population. The user asked for the city proper population.
Answer: The population of Tokyo city proper in 2024 is approximately 13.96 million.
```

### Building a ReAct Agent with LangChain

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

# Tools available to the agent
tools = [
    DuckDuckGoSearchRun(),    # web search
    PythonREPLTool(),         # execute Python code
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ReAct prompt template
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = agent_executor.invoke({
    "input": "What is the square root of the population of Tokyo?"
})
```

The agent will:
1. Think: "I need Tokyo's population first"
2. Act: search for Tokyo population
3. Think: "Now I need the square root of 13,960,000"
4. Act: use Python REPL → `import math; math.sqrt(13960000)`
5. Answer: "The square root of Tokyo's population (~13.96M) is approximately 3,737"

---

## PART 3: LANGGRAPH — STATEFUL AGENTS

### Why LangGraph?

LangChain agents are good for simple ReAct loops.
LangGraph enables **complex, stateful, multi-agent workflows**.

```
LangChain: A → B → C (linear chain)
LangGraph: 
  A → B → C
      ↓       ↑
      D → E → F  (with cycles, conditionals, parallel branches)
```

### Key Concepts

**Graph:** Nodes connected by edges
**Node:** A function (LLM call, tool call, condition check)
**Edge:** Transition between nodes (conditional or unconditional)
**State:** Shared dictionary passed between all nodes

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # accumulate messages
    next_action: str

# Define nodes
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    # extract tool call from last message and execute
    ...

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"    # route to tool node
    return "end"         # we're done

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.add_node("tool", call_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tool": "tool",
    "end": END
})
workflow.add_edge("tool", "agent")    # after tool, go back to agent

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})
```

### Agent Memory Types

| Memory Type | What it stores | Example |
|-------------|---------------|---------|
| **In-context** | Current conversation | Last 10 messages |
| **Episodic** | Past conversation summaries | "User prefers concise answers" |
| **Semantic** | Facts about the user/world | User's name, preferences |
| **Procedural** | How to do things | Custom tool usage patterns |

---

## PART 4: ADVANCED PROMPT ENGINEERING

### Chain-of-Thought Variants

**Zero-Shot CoT:**
```
Q: "If John has 5 apples and gives 2 to Mary, who gives 1 to Tom, how many does Tom have?"
Prompt: "Let's think step by step."
```
Just adding "Let's think step by step" improves reasoning without examples.

**Few-Shot CoT:**
```
Q: [example question]
A: [step-by-step reasoning + answer]

Q: [example question 2]
A: [step-by-step reasoning + answer]

Q: [actual question]
A: [model generates step-by-step reasoning]
```

**Self-Consistency CoT (Wang et al., 2022):**
```
Same question → generate 20 different reasoning chains (temperature=0.7)
→ Take majority vote on final answers

Q: "What is 15% of 340?"
Run 20 times:
  Run 1:  "340 × 0.15 = 51"  → 51
  Run 2:  "340 × 15/100 = 51" → 51
  Run 3:  "10% = 34, 5% = 17, total = 51" → 51
  ...
Majority vote: 51 ✓

Improves accuracy by 5-20% over single CoT.
```

### Tree of Thought (ToT)

**Paper:** Yao et al., Princeton, 2023

**Problem with CoT:** Linear — if you make a wrong step, you're stuck.
**ToT solution:** Explore multiple reasoning paths simultaneously.

```
Standard CoT (linear):
  Question → Step 1 → Step 2 → Step 3 → [wrong] → stuck

Tree of Thought (branching):
  Question →  Branch A: Step 1a → Step 2a → Step 3a → [evaluate: good]
           →  Branch B: Step 1b → [evaluate: bad, prune]
           →  Branch C: Step 1c → Step 2c → [evaluate: very good, continue]
           
Best branch → Final Answer
```

**Implementation:**
```python
def tree_of_thought(question, breadth=3, depth=3):
    thoughts = [question]
    
    for step in range(depth):
        # Generate multiple continuations (breadth)
        candidates = []
        for thought in thoughts:
            for _ in range(breadth):
                continuation = llm.generate(thought + "\nNext step:")
                candidates.append(continuation)
        
        # Evaluate each candidate
        scores = [llm.score(q, c) for c in candidates]
        
        # Keep top-k (beam search)
        thoughts = [c for c, s in sorted(zip(candidates, scores), 
                                          key=lambda x: x[1])[-breadth:]]
    
    # Final answer from best thought
    return llm.answer(max(thoughts, key=lambda t: llm.score(question, t)))
```

**When to use ToT:**
- Complex planning problems
- Creative writing with constraints
- Multi-step math proofs
- NOT needed for simple factual queries (too slow/expensive)

### Prompt Engineering Patterns

**System Prompt Best Practices:**
```
GOOD system prompt:
"You are an expert SQL engineer. Rules:
1. Always write syntactically valid SQL
2. Add comments for complex queries
3. If the request is ambiguous, ask for clarification
4. Never use SELECT * in production queries"

BAD system prompt:
"You are helpful"
```

**Structured Output Forcing:**
```
"Respond ONLY with a JSON object in this exact format:
{
  'sentiment': 'positive' or 'negative' or 'neutral',
  'confidence': float between 0 and 1,
  'reason': string explaining the sentiment
}
Do not include any other text."
```

**Role-Playing for Domain Expertise:**
```
"You are a senior oncologist with 20 years of experience.
Explain this lab result to a patient in simple terms..."
```

---

## PART 5: LLM EVALUATION DEEP DIVE

### LLM-as-Judge (GPT-4 as Evaluator)

Using a strong LLM to evaluate weaker LLM outputs.

```python
def llm_judge(question, answer, rubric):
    prompt = f"""You are an expert evaluator. Score this answer 1-10.

Question: {question}
Answer: {answer}

Rubric:
- Accuracy (1-10): Is the answer factually correct?
- Completeness (1-10): Does it fully address the question?
- Clarity (1-10): Is it well-explained?

Respond with JSON: {{"accuracy": X, "completeness": X, "clarity": X, "reasoning": "..."}}"""
    
    return judge_llm.generate(prompt)
```

### LLM-as-Judge Biases (Critical for Interviews)

| Bias | Description | Mitigation |
|------|-------------|-----------|
| **Position bias** | Judge prefers answer A over B just because it's first | Swap order and average |
| **Verbosity bias** | Judge prefers longer answers even when shorter is better | Instruct to prefer concise |
| **Self-enhancement** | Model judges own outputs as better | Use different model as judge |
| **Sycophancy** | Judge agrees with opinions in the answer | Use chain-of-thought in judge |

**Mitigation: Pairwise Comparison + Swap**
```python
# Compare A vs B
score1 = judge(question, answer_A, answer_B)   # A first
# Swap order
score2 = judge(question, answer_B, answer_A)   # B first
# Average to remove position bias
final_score = average(score1, score2)
```

### MT-Bench

- 80 multi-turn conversations across 8 categories
  (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities)
- Each conversation has 2 turns
- GPT-4 rates 1-10
- Scores: Claude 3.5 Sonnet ~9.0, GPT-4 ~8.99, LLaMA 3 70B ~8.1

### Chatbot Arena — The Gold Standard

**Bradley-Terry model** for Elo ratings:
```
P(A beats B) = exp(R_A) / (exp(R_A) + exp(R_B))

Where R_A, R_B are the Elo ratings
```

- Real users compare two models (blind)
- Vote for better response
- Elo updates based on outcome
- 300K+ human preference votes

**Why this is the most reliable benchmark:**
- Real users, real prompts
- No benchmark contamination
- Measures actual preference, not proxy metrics

### Benchmark Contamination Problem

**The issue:** LLM training data from the internet contains benchmark questions
and answers. The model "cheats" by memorizing.

```
Training data: [Wikipedia, books, StackOverflow, MATH problems...]
                         ↓
MATH benchmark questions also appear in training data
                         ↓
Model scores 95% on MATH — but is it reasoning or memorizing?
```

**Detection methods:**
- n-gram overlap between training data and benchmark
- "Canary strings" embedded in benchmarks
- Dynamic benchmarks that change questions

**Why LiveCodeBench matters:**
Uses competitive programming problems AFTER the model's training cutoff.
Contamination is impossible for new problems.

---

## INTERVIEW BLAST — Agents & Prompting

**"What is function calling in LLMs?"**
> "Function calling lets you define tool schemas (as JSON) that the LLM can decide
> to invoke. The LLM outputs a structured tool call with arguments instead of text.
> Your code executes the function, returns the result, and the LLM uses it to generate
> the final answer. It enables LLMs to access real-time data, run code, query databases,
> and call APIs."

**"What is Tree of Thought and when is it better than CoT?"**
> "CoT generates a single linear reasoning chain — if you go wrong early, you're stuck.
> Tree of Thought explores multiple reasoning branches simultaneously, evaluates each
> branch's quality, and keeps the most promising ones (like beam search). It's better
> for complex planning, puzzle-solving, and multi-step math proofs. But it's much more
> expensive (breadth × depth × evaluation calls), so use it only for hard problems."

**"How do you evaluate LLM outputs in production?"**
> "Three levels: automated metrics (ROUGE/BLEU for specific tasks, perplexity),
> LLM-as-judge (GPT-4 scoring on a rubric with pairwise comparisons to remove position
> bias), and human evaluation (gold standard, but expensive). For production monitoring,
> I use LLM-as-judge on 5-10% of traffic with alerting on score drops. For model
> selection, Chatbot Arena Elo is the most reliable because it's real users with real
> prompts and no contamination."
