# =====================================================================
# PYTHON FOR GenAI ENGINEERS: PRODUCTION UTILITIES & VRAM MASTERCLASS
# =====================================================================
# This script covers JSONL standard formats, reproducibility seeding, 
# PyTorch/GPU memory leak cleanup models, and progress bar profiling.
# 
# Run: python practical/13_python_libraries/python_masterclass_utils.py
# =====================================================================

import json
import random
import time
import gc
import numpy as np

def heading(title):
    print("\n" + "=" * 65)
    print(f"[{title}]")
    print("=" * 65)

# ---------------------------------------------------------------------
# 1. INDUSTRY STANDARD DATA FORMATTING: JSONL (JSON LINES)
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Instruction fine-tuning models (e.g. OpenAI, Hugging Face SFTTrainer)
#    require datasets in JSON Lines (.jsonl) format. Loading a single giant
#    JSON file parses everything into memory as a list, causing OOM.
#    JSONL lets us stream one line at a time.
#
# THE SOLUTION:
#    Write a clean, memory-efficient generator to stream lines from a JSONL file.
# ---------------------------------------------------------------------
heading("1. Optimized JSON Lines (JSONL) Data Streamer")

# Mock database records to save
dataset_to_write = [
    {"instruction": "Define backpropagation.", "output": "A method to calculate gradients in neural nets."},
    {"instruction": "What is temperature in LLMs?", "output": "A parameter scaling logit probabilities during sampling."},
    {"instruction": "Explain LoRA.", "output": "Low-Rank Adaptation freezes base weights and adds low-rank trainable matrices."}
]

file_path = "practical/13_python_libraries/mock_dataset.jsonl"

# 1. Write data to JSONL format
print("Writing mock dataset to JSONL...")
with open(file_path, "w") as f:
    for entry in dataset_to_write:
        f.write(json.dumps(entry) + "\n")
print(f"Dataset successfully saved to '{file_path}'.")

# 2. Read data as a memory-efficient stream using a generator
def stream_jsonl(path):
    """Memory-efficient generator to stream one sample at a time."""
    with open(path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                yield json.loads(line.strip())

print("\nStreaming parsed JSONL entries:")
for idx, item in enumerate(stream_jsonl(file_path)):
    print(f"  -> Sample #{idx + 1} parsed: '{item['instruction']}'")


# ---------------------------------------------------------------------
# 2. GLOBAL SEEDING FOR ABSOLUTE REPRODUCIBILITY
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Neural networks initialize weights randomly. Data splits use random shuffles.
#    If you do not lock the random seeds, your loss scores and metrics will change
#    every time you run the script, making experiments impossible to compare.
#
# THE SOLUTION:
#    Implement a robust 'seed_everything' function locking all random generators.
# ---------------------------------------------------------------------
heading("2. Seeding Everything for Reproducible ML Experiments")

def seed_everything(seed=42):
    """Locks all core random number generators to ensure identical runs."""
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    print(f"Global experimental seed locked to: {seed}")

# Lock to seed 42
seed_everything(42)
print("Random numbers run 1:", [random.randint(1, 100) for _ in range(5)])
print("NumPy normal run 1  :", np.round(np.random.normal(0, 1, 3), 4))

# Reset seed to verify identical numbers
seed_everything(42)
print("Random numbers run 2:", [random.randint(1, 100) for _ in range(5)])
print("NumPy normal run 2  :", np.round(np.random.normal(0, 1, 3), 4))


# ---------------------------------------------------------------------
# 3. GPU VRAM GARBAGE COLLECTION & MEMORY MANAGEMENT
# ---------------------------------------------------------------------
# THE PROBLEM:
#    When training LLMs, you often hit a CUDA Out of Memory (OOM) error.
#    This happens because PyTorch holds onto historical computation graphs 
#    and deleted variables are not immediately returned to the GPU.
#
# THE SOLUTION:
#    Delete reference variables, trigger Python's manual garbage collector
#    'gc.collect()', and tell PyTorch to release unoccupied cached VRAM.
# ---------------------------------------------------------------------
heading("3. Simulating GPU VRAM Management & Memory Leak Cleanup")

class MockGPUDevice:
    """Mock PyTorch CUDA cache interface for clean demonstration."""
    def __init__(self):
        self.cached_memory_gb = 14.5
        self.allocated_memory_gb = 12.2
        
    def empty_cache(self):
        self.cached_memory_gb = self.allocated_memory_gb
        print("-> PyTorch CUDA cache cleared! Unoccupied memory returned to OS.")

# Create mock model representation
print("Initializing simulated LLM model (consuming 12.2GB GPU VRAM)...")
mock_model = np.zeros((1000, 1000, 1000))  # Simulated memory reservation
device = MockGPUDevice()
print(f"Allocated memory: {device.allocated_memory_gb}GB, Cached memory: {device.cached_memory_gb}GB")

# Perform memory cleanup
print("\nReleasing model and performing garbage collection...")
del mock_model  # Delete the variable reference
gc.collect()    # Manually trigger Python's reference-counting garbage collector
device.empty_cache() # Clear PyTorch CUDA cache
device.allocated_memory_gb = 0.0
device.cached_memory_gb = 0.0

print(f"Allocated memory: {device.allocated_memory_gb}GB, Cached memory: {device.cached_memory_gb}GB")


# ---------------------------------------------------------------------
# 4. PROGRESS BAR & THROUGHPUT PROFILING
# ---------------------------------------------------------------------
# THE PROBLEM:
#    When tokenizing or generating text for millions of records, you need to
#    know how long it will take, what the current processing speed is (seq/sec),
#    and whether the job is stuck.
#
# THE SOLUTION:
#    Implement a robust text-based throughput progress indicator.
# ---------------------------------------------------------------------
heading("4. Custom Iteration & Throughput Profiler (Simulated tqdm)")

def run_profiled_loop(data_list):
    total = len(data_list)
    start_time = time.time()
    
    print(f"Beginning profiling run for {total} elements...")
    for idx, item in enumerate(data_list):
        # Simulate processing step (e.g. LLM generation)
        time.sleep(0.05)
        
        # Calculate stats
        elapsed = time.time() - start_time
        processed = idx + 1
        speed = processed / elapsed  # iterations per second
        percentage = (processed / total) * 100
        
        # Print inline progress status (equivalent to tqdm progress output)
        if processed % 5 == 0 or processed == total:
            print(f"  [Progress] {processed}/{total} ({percentage:.1f}%) | "
                  f"Speed: {speed:.1f} iter/sec | Elapsed: {elapsed:.2f}s")

run_profiled_loop(range(20))


# =====================================================================
# INTERVIEW PRACTICE QUESTIONS (PRODUCTION UTILITIES & MEMORY)
# =====================================================================
print("\n" + "=" * 65)
print("[PRO-LEVEL INTERVIEW DRILLS (UTILITIES & MEMORY)]")
print("=" * 65)
print("""
Q1: Why does deleting a PyTorch model tensor using 'del tensor' not immediately free up VRAM?
    
    -> Answer: Python operates on reference counting and garbage collection. Deleting 
       the variable reference only marks the tensor for collection. Furthermore, PyTorch 
       uses a caching memory allocator to avoid the heavy overhead of frequently calling 
       cudaMalloc. To truly free memory, you must run 'del tensor', trigger Python's 
       garbage collector via 'gc.collect()', and finally call 'torch.cuda.empty_cache()' 
       to release unoccupied memory back to the GPU.

Q2: What is the primary advantage of JSONL over standard JSON for dataset storage?
    
    -> Answer: Standard JSON loads the entire file as a single contiguous object, 
       which requires loading the whole dataset into system RAM. JSONL (JSON Lines) 
       stores each record on a single line. This allows us to parse and stream the 
       dataset incrementally line-by-line, creating highly efficient dataloaders 
       that can process infinite/terabyte-sized datasets on standard RAM.

Q3: Why is seeding essential in machine learning pipelines, and what should be seeded?
    
    -> Answer: Machine learning pipelines rely on random numbers for weight initialization, 
       shuffling datasets, dropout layers, and training updates. Without explicit seeding, 
       experiments are not reproducible, making it impossible to debug metrics or isolate 
       bug causes. A robust pipeline seeds python's 'random', 'numpy.random', and 
       PyTorch's global seed: 'torch.manual_seed()', 'torch.cuda.manual_seed_all()', 
       along with locking CUDA backends: 'torch.backends.cudnn.deterministic = True'.
""")
print("=" * 65 + "\n")
