# =====================================================================
# PYTHON FOR GenAI ENGINEERS: PANDAS DEEP-DIVE MASTERCLASS
# =====================================================================
# This script covers advanced Pandas data manipulation patterns 
# required for preparing instruction fine-tuning datasets, RLHF data,
# and token optimization in Generative AI production pipelines.
# 
# Run: python practical/13_python_libraries/python_masterclass_pandas.py
# =====================================================================

import pandas as pd
import numpy as np
import io

def heading(title):
    print("\n" + "=" * 65)
    print(f"[{title}]")
    print("=" * 65)

# ---------------------------------------------------------------------
# 1. PARSING CONVERSATION / CHAT TEMPLATES (THE PRODUCTION CORE)
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Instruction and chat datasets often come as nested structures or JSON Lines
#    where each entry has a list of messages:
#    [
#       {"role": "user", "content": "Hello"},
#       {"role": "assistant", "content": "Hi there!"}
#    ]
#    We need to parse, analyze, and format these into single flattened string templates 
#    compatible with base models (e.g. ChatML or LLaMA-3 instruction templates).
#
# THE SOLUTION:
#    Apply custom row-wise lambda operations to format roles and concatenate values.
# ---------------------------------------------------------------------
heading("1. Formatting Dialogue Datasets (SFT Chat Templates)")

# Simulate raw conversation data
raw_chat_data = [
    {
        "id": "conv_001",
        "category": "coding",
        "conversations": [
            {"role": "user", "content": "Write a python function to check if a number is prime."},
            {"role": "assistant", "content": "Here is the code: def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"}
        ]
    },
    {
        "id": "conv_002",
        "category": "general",
        "conversations": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
]

df = pd.DataFrame(raw_chat_data)
print("Original DataFrame:\n", df)

def apply_chatml_template(conv_list):
    """
    Formats the conversation using the standard ChatML template:
    <|im_start|>user
    {content}<|im_end|>
    <|im_start|>assistant
    {content}<|im_end|>
    """
    formatted_str = ""
    for msg in conv_list:
        role = msg["role"]
        content = msg["content"]
        formatted_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted_str

# Apply formatting using a vectorized element-wise lambda function
df["formatted_prompt"] = df["conversations"].apply(apply_chatml_template)

print("\nFormatted ChatML outputs for conv_001:")
print(df["formatted_prompt"].iloc[0])


# ---------------------------------------------------------------------
# 2. TOKEN & CHAR STATISTICS FOR BUCKETING AND PACKING
# ---------------------------------------------------------------------
# THE PROBLEM:
#    To optimize GPU memory and training speed, we must avoid feeding 
#    extremely long sequences that force heavy padding. We must:
#    1. Analyze the token/character distributions.
#    2. Drop outlier examples that exceed max sequence length (e.g. >2048 tokens).
#    3. Analyze dataset statistics (95th percentile) for seq packing boundaries.
# ---------------------------------------------------------------------
heading("2. Dataset Sequence Length Filtering & 95th Percentile")

# Create a mock dataset with variable sentence lengths
sentences = [
    "Short sentence.",
    "Medium length sentence to simulate normal chat conversation context.",
    "This is an extremely long outlier conversation history that goes on and on and contains a massive amount of text that will consume unnecessary GPU RAM and should probably be filtered out during preprocessing to maintain high training throughput.",
    "Another normal user query.",
    "Quick coding task check."
]
df_stats = pd.DataFrame({"text": sentences})

# 1. Compute text characters length
df_stats["char_len"] = df_stats["text"].apply(len)

# 2. Simulate Token Count (a rough rule of thumb is 1 token ~ 4 characters)
df_stats["token_count"] = df_stats["text"].apply(lambda x: len(x.split()) * 2) # Mock tokenizer output
print("Dataset text lengths and simulated token counts:\n", df_stats)

# 3. Calculate 95th Percentile sequence length (crucial for choosing max_seq_length)
p95_len = df_stats["token_count"].quantile(0.95)
print(f"\n95th Percentile Token Length: {p95_len:.2f}")

# 4. Filter out sequences longer than 20 tokens to save GPU VRAM
df_filtered = df_stats[df_stats["token_count"] <= 20]
print("\nFiltered DataFrame (Removed Outliers):\n", df_filtered)


# ---------------------------------------------------------------------
# 3. GROUPING AND AGGREGATING (BALANCING LLM DATASETS)
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Your training dataset might be heavily biased towards one category 
#    (e.g., 90% creative writing, 10% math). If you train directly, the 
#    model will overfit to creative writing and forget how to solve math.
#    We must analyze category distributions and balance the dataset.
# ---------------------------------------------------------------------
heading("3. Aggregation & Balancing Dataset Distribution")

# Mock biased dataset
data_distribution = {
    "prompt": [f"Prompt {i}" for i in range(10)],
    "category": ["general", "general", "coding", "general", "coding", "general", "general", "math", "general", "general"]
}
df_distribution = pd.DataFrame(data_distribution)

# 1. Count distributions
counts = df_distribution["category"].value_counts()
print("Raw Category Counts:\n", counts)

# 2. Get relative percentages
percentages = df_distribution["category"].value_counts(normalize=True) * 100
print("\nRaw Category Percentages:\n", percentages)

# 3. Stratified Downsampling to balance dataset (limit general prompts to 2)
df_balanced = df_distribution.groupby("category").apply(lambda x: x.sample(min(len(x), 2), random_state=42)).reset_index(drop=True)
print("\nBalanced Dataset (Max 2 samples per category):\n", df_balanced)


# ---------------------------------------------------------------------
# 4. MEMORY OPTIMIZATION: LOADING LARGE JSON LINES IN CHUNKS
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Production datasets are often massive JSON Lines (JSONL) files 
#    (e.g. 50GB). Attempting to load this in a single `pd.read_json()` 
#    command will crash the server due to Out-Of-Memory (OOM).
#
# THE SOLUTION:
#    Use chunk-based generator parsing. We process the dataset 
#    incrementally in memory-friendly batch chunks.
# ---------------------------------------------------------------------
heading("4. Chunk-based Memory Optimization for Large Files")

# Simulate a massive file using a StringIO object
fake_jsonl_content = "\n".join([
    '{"id": %d, "text": "Prompt content number %d"}' % (i, i) for i in range(100)
])

# Read in chunks of size 20 (only loads 20 lines at a time into RAM!)
chunk_size = 20
chunk_container = pd.read_json(io.StringIO(fake_jsonl_content), lines=True, chunksize=chunk_size)

print("Processing JSONL file in optimized memory chunks:")
for idx, chunk in enumerate(chunk_container):
    # Process the chunk (e.g. tokenize, clean, filter)
    # In this case, we just show the shape and head of each processed chunk
    print(f"  -> Chunk {idx + 1} loaded. Shape: {chunk.shape}")
    # Perform standard operation: filter out even IDs
    chunk_filtered = chunk[chunk["id"] % 2 == 1]
    # At this point, you can save the processed chunk to disk sequentially


# =====================================================================
# INTERVIEW PRACTICE QUESTIONS (PANDAS FOR LLMs)
# =====================================================================
print("\n" + "=" * 65)
print("[PRO-LEVEL INTERVIEW DRILLS (PANDAS FOR LLMs)]")
print("=" * 65)
print("""
Q1: When preparing a multi-turn chat dataset for training LLaMA models, 
    why shouldn't you just load the entire JSON file directly into RAM?
    
    -> Answer: Fine-tuning datasets can be several gigabytes or terabytes. 
       Loading them directly into memory all at once leads to heavy RAM swapping 
       and Out-of-Memory (OOM) crashes. In production pipelines, we use chunk 
       loading ('chunksize' parameter in Pandas) or stream the dataset directly 
       using generators or Hugging Face's lazy streaming datasets.

Q2: How do you handle missing values or empty strings in prompt columns?
    
    -> Answer: Empty prompts or NaN values will cause model tokenization steps 
       to fail. We drop them using 'df.dropna(subset=["prompt"])' and filter 
       out empty strings by checking string lengths: 'df = df[df["prompt"].str.strip() != ""]'.

Q3: Why is stratified sampling important in ML preprocessing?
    
    -> Answer: Stratified sampling ensures that training, validation, and test 
       splits maintain the exact same ratio of labels or categories as the parent 
       dataset, preventing bias shift in our validation evaluations.
""")
print("=" * 65 + "\n")
