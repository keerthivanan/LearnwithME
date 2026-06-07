"""
LangChain + LangSmith — Everything
=====================================
pip install langchain langchain-anthropic langchain-community
pip install langsmith chromadb sentence-transformers
"""

# ════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"   # optional
os.environ["LANGCHAIN_TRACING_V2"] = "true"              # enable LangSmith

from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    FewShotPromptTemplate, MessagesPlaceholder
)
from langchain.chains import LLMChain, ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent, create_react_agent
from langchain import hub

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)

# ════════════════════════════════════════════
# 1. BASIC LLM CALL
# ════════════════════════════════════════════
print("1. BASIC LLM CALL")

# Simple invoke
response = llm.invoke("What is machine learning in one sentence?")
print(f"Response: {response.content}")

# With messages
messages = [
    SystemMessage(content="You are a data science tutor. Be concise."),
    HumanMessage(content="What is gradient descent?"),
]
response = llm.invoke(messages)
print(f"Gradient descent: {response.content}")

# ════════════════════════════════════════════
# 2. PROMPT TEMPLATES
# ════════════════════════════════════════════
print("\n2. PROMPT TEMPLATES")

# Basic template
prompt = PromptTemplate(
    input_variables=["topic", "level"],
    template="Explain {topic} to a {level} student in 2 sentences."
)
formatted = prompt.format(topic="neural networks", level="beginner")
print(formatted)
# Explain neural networks to a beginner student in 2 sentences.

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}. Always answer in {language}."),
    ("human",  "Question: {question}"),
])
formatted_chat = chat_prompt.format_messages(
    role="data scientist",
    language="English",
    question="What is overfitting?"
)
print(f"Messages: {[m.content for m in formatted_chat]}")

# Few-shot prompting
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "10 * 5", "output": "50"},
    {"input": "100 / 4", "output": "25"},
]
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)
few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Solve these math problems:",
    suffix="Input: {question}\nOutput:",
    input_variables=["question"]
)
print(few_shot.format(question="6 * 7"))

# ════════════════════════════════════════════
# 3. CHAINS (LCEL — LangChain Expression Language)
# ════════════════════════════════════════════
print("\n3. CHAINS WITH LCEL")

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Simple chain using pipe operator
prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
chain  = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "transformers"})
print(f"Result: {result}")

# Sequential chain
summarize_prompt = ChatPromptTemplate.from_template("Summarize this in 5 words: {text}")
translate_prompt = ChatPromptTemplate.from_template("Translate to Hindi: {text}")

summarize_chain  = summarize_prompt | llm | StrOutputParser()
translate_chain  = translate_prompt | llm | StrOutputParser()

# Chain them: summarize → translate
full_chain = (
    {"text": RunnablePassthrough()}
    | summarize_chain
    | (lambda x: {"text": x})
    | translate_chain
)

result = full_chain.invoke("Machine learning uses data to find patterns and make predictions automatically.")
print(f"Translated summary: {result}")

# JSON output
json_prompt = ChatPromptTemplate.from_template(
    "Return a JSON with keys 'name', 'type', 'use_case' for: {model}"
)
json_chain = json_prompt | llm | JsonOutputParser()
result = json_chain.invoke({"model": "Random Forest"})
print(f"JSON result: {result}")
# {'name': 'Random Forest', 'type': 'Ensemble', 'use_case': 'Classification/Regression'}

# Batch processing
prompts = [{"topic": "CNN"}, {"topic": "LSTM"}, {"topic": "Transformer"}]
results = chain.batch(prompts)
for topic_dict, res in zip(prompts, results):
    print(f"  {topic_dict['topic']}: {res[:60]}...")

# ════════════════════════════════════════════
# 4. MEMORY — CONVERSATION
# ════════════════════════════════════════════
print("\n4. MEMORY")

# Buffer memory — stores all messages
memory = ConversationBufferMemory(return_messages=True)

conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain_with_memory = conv_prompt | llm | StrOutputParser()

def chat(message, memory):
    history = memory.load_memory_variables({})["history"]
    response = chain_with_memory.invoke({"input": message, "history": history})
    memory.save_context({"input": message}, {"output": response})
    return response

r1 = chat("My name is Keerthi", memory)
r2 = chat("What is my name?", memory)
print(f"Bot: {r2}")   # "Your name is Keerthi"

# Summary memory — condenses old messages (saves tokens)
summary_memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# ════════════════════════════════════════════
# 5. RAG — RETRIEVAL AUGMENTED GENERATION
# ════════════════════════════════════════════
print("\n5. RAG PIPELINE")

# Step 1: Load documents
texts = [
    "Python was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "FastAPI is a modern Python web framework for building APIs.",
    "Transformers use self-attention mechanism for NLP tasks.",
    "BERT was introduced by Google in 2018.",
    "GPT stands for Generative Pre-trained Transformer.",
]

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20
)
from langchain.schema import Document
docs = [Document(page_content=t) for t in texts]
chunks = splitter.split_documents(docs)
print(f"Chunks created: {len(chunks)}")

# Step 3: Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: RAG Chain
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If not in context, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Who created Python?",
    "What is BERT?",
    "What is the capital of France?",  # not in context
]
for q in questions:
    ans = rag_chain.invoke(q)
    print(f"  Q: {q}\n  A: {ans}\n")

# ════════════════════════════════════════════
# 6. TOOLS + AGENTS
# ════════════════════════════════════════════
print("\n6. TOOLS + AGENTS")

import math

def calculator(expression: str) -> str:
    """Evaluate a math expression safely"""
    try:
        result = eval(expression, {"__builtins__": {}}, {
            "sqrt": math.sqrt, "pi": math.pi, "abs": abs
        })
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def search_docs(query: str) -> str:
    """Search internal documents"""
    docs = retriever.get_relevant_documents(query)
    return "\n".join(d.page_content for d in docs[:2])

tools = [
    Tool(name="Calculator",   func=calculator,   description="Math calculations"),
    Tool(name="DocumentSearch", func=search_docs, description="Search internal docs about Python, ML, AI"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

result = agent.invoke("What is sqrt(144) + 5?")
print(f"Agent answer: {result['output']}")

result2 = agent.invoke("Who created Python and what is 10 * 7?")
print(f"Agent answer: {result2['output']}")

# ════════════════════════════════════════════
# 7. LANGSMITH — Observability
# ════════════════════════════════════════════
print("\n7. LANGSMITH")

# When LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY is set,
# ALL your chain calls are automatically logged to LangSmith!
# View at: https://smith.langchain.com

# Add metadata to traces
from langchain_core.tracers.context import collect_runs

with collect_runs() as cb:
    result = chain.invoke({"topic": "reinforcement learning"})
    run_id = cb.traced_runs[0].id
    print(f"Run ID: {run_id}")
    print(f"Result: {result}")
    # View trace at: https://smith.langchain.com/o/your-org/runs/{run_id}

# Tag specific chains
chain_tagged = chain.with_config(
    tags=["production", "v2"],
    metadata={"user_id": "keerthi", "session": "123"}
)

print("\nAll done! ✓")
print("View traces at: https://smith.langchain.com")
