"""
LangChain — Definitions + Code + Outputs
==========================================
pip install langchain langchain-anthropic langchain-community langchain-core
pip install langsmith chromadb sentence-transformers
"""

import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    FewShotPromptTemplate, MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)


# ══════════════════════════════════════════════════════
# 1. BASIC LLM CALL
# ══════════════════════════════════════════════════════
# WHAT IS LANGCHAIN?
#   → A framework for building LLM-powered applications
#   → Connects LLMs to: data sources, tools, memory, other models
#   → LCEL (LangChain Expression Language): chain components with | operator
#
# WHAT IS AN LLM?
#   → Large Language Model — takes text in, produces text out
#   → ChatAnthropic wraps Claude API in LangChain's standard interface
#   → .invoke()  → single call (returns AIMessage)
#   → .stream()  → streaming response (tokens as they arrive)
#   → .batch()   → process multiple inputs in parallel
#
# MESSAGE TYPES:
#   → SystemMessage  : instructions for the model ("You are a...")
#   → HumanMessage   : user's message
#   → AIMessage      : model's previous response (for conversation history)

print("=" * 55)
print("1. BASIC LLM CALL")
print("=" * 55)

response = llm.invoke("What is machine learning in one sentence?")
print(f"Response: {response.content}")
# Response: Machine learning is a field where algorithms learn from data...

messages = [
    SystemMessage(content="You are a data science tutor. Be concise."),
    HumanMessage(content="What is gradient descent?"),
]
response = llm.invoke(messages)
print(f"Gradient descent: {response.content}")


# ══════════════════════════════════════════════════════
# 2. PROMPT TEMPLATES
# ══════════════════════════════════════════════════════
# WHAT IS A PROMPT TEMPLATE?
#   → A reusable template with {placeholders} for variables
#   → Separates prompt structure from runtime values
#   → Makes prompts testable, versioned, and reusable
#
# TYPES:
#   → PromptTemplate        : simple string template with {variables}
#   → ChatPromptTemplate    : list of messages (system + human + ai)
#   → FewShotPromptTemplate : includes examples to guide the model
#
# WHAT IS FEW-SHOT PROMPTING?
#   → Show the model a few input/output EXAMPLES before the real question
#   → Model learns the pattern from examples → better answers
#   → "Zero-shot" = no examples, "Few-shot" = 2-5 examples, "Many-shot" = 10+

print("\n" + "=" * 55)
print("2. PROMPT TEMPLATES")
print("=" * 55)

prompt = PromptTemplate(
    input_variables=["topic", "level"],
    template="Explain {topic} to a {level} student in 2 sentences."
)
formatted = prompt.format(topic="neural networks", level="beginner")
print(formatted)
# Explain neural networks to a beginner student in 2 sentences.

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}. Always answer in {language}."),
    ("human",  "Question: {question}"),
])
msgs = chat_prompt.format_messages(
    role="data scientist", language="English", question="What is overfitting?"
)
print(f"Chat prompt messages: {[m.content for m in msgs]}")

# Few-shot prompting
examples = [
    {"input": "2 + 2",   "output": "4"},
    {"input": "10 * 5",  "output": "50"},
    {"input": "100 / 4", "output": "25"},
]
few_shot = FewShotPromptTemplate(
    examples        = examples,
    example_prompt  = PromptTemplate(input_variables=["input", "output"],
                                     template="Input: {input}\nOutput: {output}"),
    prefix          = "Solve these math problems:",
    suffix          = "Input: {question}\nOutput:",
    input_variables = ["question"]
)
print(few_shot.format(question="6 * 7"))
# Input: 2 + 2 \n Output: 4 \n ... \n Input: 6 * 7 \n Output:


# ══════════════════════════════════════════════════════
# 3. CHAINS — LCEL (LangChain Expression Language)
# ══════════════════════════════════════════════════════
# WHAT IS LCEL?
#   → LangChain Expression Language — pipe operator | to chain components
#   → prompt | llm | parser  →  builds a runnable chain
#   → Each component: takes input → transforms it → passes to next
#
# WHAT IS A CHAIN?
#   → A sequence of LLM calls and transformations
#   → Simple chain: prompt → LLM → parse output
#   → Complex chain: retrieve context → prompt → LLM → post-process
#
# OUTPUT PARSERS:
#   → StrOutputParser()  : extract .content string from AIMessage
#   → JsonOutputParser() : parse JSON from LLM response
#
# RunnablePassthrough:
#   → Passes the input through unchanged — useful when you need the
#     original input alongside transformed values in a dict

print("\n" + "=" * 55)
print("3. CHAINS — LCEL")
print("=" * 55)

simple_prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
chain         = simple_prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "transformers"})
print(f"Chain result: {result}")
# Transformers are neural network architectures that use self-attention...

# Sequential chain: summarize → translate
sum_prompt   = ChatPromptTemplate.from_template("Summarize in 5 words: {text}")
trans_prompt = ChatPromptTemplate.from_template("Translate to Hindi: {text}")
sum_chain    = sum_prompt | llm | StrOutputParser()
trans_chain  = trans_prompt | llm | StrOutputParser()

full_chain = (
    {"text": RunnablePassthrough()}
    | sum_chain
    | (lambda x: {"text": x})
    | trans_chain
)
result = full_chain.invoke("Machine learning uses data to find patterns automatically.")
print(f"Translated summary: {result}")

# JSON output parsing
json_prompt = ChatPromptTemplate.from_template(
    "Return a JSON with keys 'name', 'type', 'use_case' for: {model}"
)
json_chain = json_prompt | llm | JsonOutputParser()
result     = json_chain.invoke({"model": "Random Forest"})
print(f"JSON: {result}")
# {'name': 'Random Forest', 'type': 'Ensemble', 'use_case': 'Classification/Regression'}

# Batch — process multiple inputs at once
topics  = [{"topic": "CNN"}, {"topic": "LSTM"}, {"topic": "Transformer"}]
results = chain.batch(topics)
for t, r in zip(topics, results):
    print(f"  {t['topic']}: {r[:60]}...")


# ══════════════════════════════════════════════════════
# 4. MEMORY — Conversation History
# ══════════════════════════════════════════════════════
# WHAT IS MEMORY IN LANGCHAIN?
#   → LLMs are stateless — each call has no memory of previous calls
#   → Memory stores past messages and injects them into the next prompt
#
# TYPES:
#   → ConversationBufferMemory : stores ALL messages (grows unbounded)
#     - Use for short conversations
#   → ConversationSummaryMemory: summarizes old messages to save tokens
#     - Use for long conversations (summary = compressed history)
#
# HOW IT WORKS:
#   → memory.save_context(input, output) → store message pair
#   → memory.load_memory_variables({})   → retrieve stored messages
#   → Inject history into prompt with MessagesPlaceholder

print("\n" + "=" * 55)
print("4. MEMORY")
print("=" * 55)

memory      = ConversationBufferMemory(return_messages=True)
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain_mem = conv_prompt | llm | StrOutputParser()

def chat(message: str, mem) -> str:
    history  = mem.load_memory_variables({})["history"]
    response = chain_mem.invoke({"input": message, "history": history})
    mem.save_context({"input": message}, {"output": response})
    return response

r1 = chat("My name is Keerthi", memory)
r2 = chat("What is my name?", memory)
print(f"Bot: {r2}")   # "Your name is Keerthi"

summary_mem = ConversationSummaryMemory(llm=llm, return_messages=True)
print("Summary memory configured — summarizes old messages to save tokens")


# ══════════════════════════════════════════════════════
# 5. RAG — Retrieval Augmented Generation
# ══════════════════════════════════════════════════════
# WHAT IS RAG?
#   → Retrieval Augmented Generation — combine LLM with your own data
#   → Problem: LLM doesn't know your private/recent documents
#   → Solution: retrieve relevant docs → inject as context → LLM answers
#
# RAG PIPELINE:
#   1. LOAD     : load documents (PDF, TXT, web, database)
#   2. SPLIT    : chunk into pieces (RecursiveCharacterTextSplitter)
#   3. EMBED    : convert chunks to vectors (sentence-transformers)
#   4. STORE    : save vectors in a vector database (Chroma, Pinecone, FAISS)
#   5. RETRIEVE : given query, find top-k similar chunks (cosine similarity)
#   6. GENERATE : pass query + chunks as context to LLM → answer
#
# CHUNK SIZE vs OVERLAP:
#   → chunk_size=200   : max characters per chunk (smaller = more precise)
#   → chunk_overlap=20 : overlap between chunks → no info lost at boundaries
#
# VECTOR DATABASE:
#   → Stores embeddings → fast similarity search
#   → Chroma: simple local vector DB (great for learning/prototyping)
#   → Pinecone, Weaviate: production-scale cloud vector DBs

print("\n" + "=" * 55)
print("5. RAG PIPELINE")
print("=" * 55)

# Step 1: Prepare documents
texts = [
    "Python was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Transformers use self-attention mechanism for NLP tasks.",
    "BERT was introduced by Google in 2018.",
    "GPT stands for Generative Pre-trained Transformer.",
]
docs   = [Document(page_content=t) for t in texts]

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks   = splitter.split_documents(docs)
print(f"Chunks created: {len(chunks)}")

# Step 3: Embed + store in vector DB
embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: Build RAG chain
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If not in context, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Who created Python?",
    "What is BERT?",
    "What is the capital of France?",   # not in our documents → "I don't know"
]
for q in questions:
    ans = rag_chain.invoke(q)
    print(f"  Q: {q}\n  A: {ans}\n")


# ══════════════════════════════════════════════════════
# 6. TOOLS + AGENTS
# ══════════════════════════════════════════════════════
# WHAT IS A TOOL?
#   → A function the agent can call to get information or take action
#   → Examples: calculator, search engine, database query, API call
#   → Tool = name + function + description (LLM reads description to decide when to use)
#
# WHAT IS AN AGENT?
#   → An LLM that DECIDES which tools to call and in what order
#   → Follows the ReAct pattern: Reason → Act → Observe → Repeat
#   → Unlike a chain (fixed steps), agents are DYNAMIC — adapt to the task
#
# ReAct PATTERN:
#   → Thought: "I need to calculate sqrt(144)"
#   → Action:  Calculator("sqrt(144)")
#   → Observation: "12"
#   → Thought: "Now I know the answer is 12"
#   → Final Answer: "sqrt(144) = 12"

print("\n" + "=" * 55)
print("6. TOOLS + AGENTS")
print("=" * 55)

import math

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}},
                      {"sqrt": math.sqrt, "pi": math.pi, "abs": abs})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def search_docs(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    return "\n".join(d.page_content for d in docs[:2])

tools = [
    Tool(name="Calculator",     func=calculator,   description="Evaluate math expressions like '2*3' or 'sqrt(16)'"),
    Tool(name="DocumentSearch", func=search_docs,  description="Search documents about Python, ML, AI, FastAPI"),
]

agent = initialize_agent(
    tools               = tools,
    llm                 = llm,
    agent               = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose             = True,
    handle_parsing_errors = True,
)

result = agent.invoke("What is sqrt(144) + 5?")
print(f"Agent answer: {result['output']}")
# Agent answer: 17

result2 = agent.invoke("Who created Python and what is 10 * 7?")
print(f"Agent answer: {result2['output']}")
# Agent answer: Python was created by Guido van Rossum. 10 * 7 = 70.


# ══════════════════════════════════════════════════════
# 7. LANGSMITH — Observability & Tracing
# ══════════════════════════════════════════════════════
# WHAT IS LANGSMITH?
#   → Observability platform for LangChain — see WHAT your chain is doing
#   → Automatically logs every chain run when LANGCHAIN_TRACING_V2=true
#   → Shows: inputs, outputs, latency, token usage, errors for every step
#
# WHY USE LANGSMITH?
#   → Debug: see exactly where a chain failed or gave wrong output
#   → Monitor: track latency and cost in production
#   → Evaluate: run test datasets and score your chain automatically
#   → Dataset management: build eval datasets from production traces
#
# SETUP:
#   → Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true
#   → ALL chain calls are automatically logged — no code changes needed!
#   → View at: https://smith.langchain.com

print("\n" + "=" * 55)
print("7. LANGSMITH — OBSERVABILITY")
print("=" * 55)

from langchain_core.tracers.context import collect_runs

with collect_runs() as cb:
    result = chain.invoke({"topic": "reinforcement learning"})
    if cb.traced_runs:
        run_id = cb.traced_runs[0].id
        print(f"Run ID: {run_id}")
    print(f"Result: {result}")

chain_tagged = chain.with_config(
    tags     = ["production", "v2"],
    metadata = {"user_id": "keerthi", "session": "123"}
)
print("Tagged chain ready — metadata visible in LangSmith dashboard")

print("\nAll done! ✓")
print("View traces at: https://smith.langchain.com")
