# 08 — ML Frameworks: PyTorch, TensorFlow and Hugging Face

> The JD specifically mentions PyTorch, TensorFlow, and Hugging Face. Know each well enough to code in them.

---

## 1. PyTorch

### Why PyTorch?

**What it is:** PyTorch is the most widely used deep learning framework for research and increasingly for production. It lets you build and train neural networks in Python with a natural, flexible feel.

Think of it like this: TensorFlow 1.x was like filling out a form before cooking — you declared every step upfront. PyTorch is like cooking live — you write regular Python, run it, and it works immediately. That makes debugging much easier.

- **Dynamic computation graph** (define-by-run): the graph is built as your code runs, not before. You can print tensors mid-forward-pass, use if/else, loops — it's just Python.
- **Pythonic**: feels like writing NumPy, not a special declarative language.
- **Dominant in research**: most AI papers use PyTorch. Hugging Face is PyTorch-native.
- **Used by**: Meta, most AI research labs, OpenAI (historically), Hugging Face.

**WHY you need to know PyTorch:** Every LLM in the Hugging Face ecosystem is PyTorch under the hood. Fine-tuning scripts, LoRA, RAG pipelines — all PyTorch.

---

### Core Concept: Tensors

**What it is:** A tensor is PyTorch's core data structure — a multi-dimensional array, like NumPy but with GPU support and built-in gradient tracking.

Think of it like this: a 1D tensor is a list, a 2D tensor is a table, a 3D tensor is a cube of numbers. A batch of tokenized sentences is a 2D tensor (batch_size × sequence_length).

```python
import torch

# Creating tensors — different ways for different needs:
x = torch.tensor([1.0, 2.0, 3.0])
# Creates a 1D tensor from a Python list
# dtype inferred as float32 from the .0 values

y = torch.zeros(3, 4)
# Creates a 3×4 matrix filled with zeros
# WHY: useful for initializing outputs before filling them in

z = torch.randn(2, 3)
# Creates a 2×3 matrix of random numbers from a normal distribution (mean=0, std=1)
# WHY: random initialization for weights — starting point for learning

# Moving to GPU — critical for fast training:
device = "cuda" if torch.cuda.is_available() else "cpu"
# cuda: NVIDIA GPU — much faster for matrix operations
# cpu: fallback if no GPU available
x = x.to(device)
# WHY: all tensors involved in a computation must be on the same device
# If model is on GPU but data is on CPU, PyTorch will throw an error
```

**Key tensor operations to know:**

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: [2, 2]
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # shape: [2, 2]

# Matrix multiplication — the fundamental LLM operation:
c = torch.matmul(a, b)    # or a @ b  — same thing
# shape: [2, 2]  →  c[i][j] = dot product of row i of a and column j of b
# WHY this matters: every linear layer, every attention head is a matrix multiply

# Shape inspection — you will use this constantly when debugging:
print(a.shape)    # torch.Size([2, 2])
print(a.dtype)    # torch.float32
print(a.device)   # cpu (or cuda:0 if on GPU)

# Reshaping:
flat = a.view(-1)           # shape [4] — flattens to 1D
# -1 means "infer this dimension from total elements"
reshaped = a.reshape(4, 1)  # shape [4, 1]
```

---

### Core Concept: Autograd (Automatic Differentiation)

**What it is:** Autograd is PyTorch's system that automatically computes gradients. You define forward computations, call `.backward()`, and PyTorch fills in `.grad` for every tensor that has `requires_grad=True`.

Think of it like this: you define the recipe (forward pass). Autograd is the food critic who traces back exactly what ingredient made the dish taste wrong and assigns blame scores (gradients) to each one.

```python
x = torch.tensor([2.0], requires_grad=True)
# requires_grad=True tells PyTorch: "track all operations on x"
# WHY: only tensors with this flag get gradients computed for them
# Model parameters automatically have requires_grad=True

y = x ** 2 + 3 * x
# y = x² + 3x
# Mathematically: dy/dx = 2x + 3
# At x=2: dy/dx = 2(2) + 3 = 7

y.backward()
# backward() triggers backpropagation from y back to x
# Computes dy/dx using the chain rule automatically
# The gradient is stored in x.grad

print(x.grad)    # tensor([7.])
# This is the gradient — it tells us: "increase x by 1 → y increases by 7"
# In training: this gradient tells the optimizer which way to adjust each weight
```

**WHY autograd matters:** You never have to manually derive gradients for complex loss functions. PyTorch tracks the computation graph and differentiates it for you — even through loops, conditionals, and custom layers.

---

### Building a Neural Network

**What it is:** All PyTorch models inherit from `nn.Module`. You define layers in `__init__` and the forward computation in `forward()`.

Think of it like a blueprint: `__init__` specifies what rooms (layers) the building has. `forward` specifies how you walk through it.

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # super().__init__() registers this as an nn.Module — required
        # All layers defined here become registered parameters

        self.fc1 = nn.Linear(768, 256)
        # fc1: fully connected layer 1
        # input: 768 features (e.g., BERT embedding size)
        # output: 256 features
        # This layer has 768×256 + 256 = 196,864 trainable parameters (weights + bias)

        self.relu = nn.ReLU()
        # Non-linearity between layers
        # Without this, fc1 and fc2 would collapse to a single linear transformation

        self.fc2 = nn.Linear(256, 2)
        # Final layer: 256 → 2 (e.g., binary classification: positive vs negative)
        # Output is raw logits — apply softmax externally (or CrossEntropyLoss handles it)

    def forward(self, x):
        # forward() defines what happens when you call model(input)
        # x shape: [batch_size, 768]
        x = self.fc1(x)     # [batch_size, 768] → [batch_size, 256]
        x = self.relu(x)    # Apply ReLU — negatives become 0
        x = self.fc2(x)     # [batch_size, 256] → [batch_size, 2]
        return x             # raw logits for 2 classes

model = SimpleModel().to(device)
# .to(device): moves all model parameters (weights) to GPU
# Must do this BEFORE the training loop starts
```

**WHY `nn.Module`:** It automatically tracks all parameters (so `model.parameters()` works), handles saving/loading, switching between train/eval mode, and moving to GPU.

---

### Training Loop

**What it is:** The training loop is the core of all deep learning. It repeatedly feeds batches through the model, computes loss, backpropagates, and updates weights.

Every iteration does exactly 5 things:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# model.parameters(): all trainable weights in the model
# lr=1e-4: learning rate — start here for fine-tuning

criterion = nn.CrossEntropyLoss()
# Loss function: for classification tasks (multi-class)
# Internally applies softmax + negative log likelihood

for epoch in range(num_epochs):
    # epoch: one full pass through all training data
    model.train()   # CRITICAL: sets dropout active, batch norm to training mode

    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)     # move data to same device as model
        labels = labels.to(device)     # MUST be on same device — else error

        # STEP 1: Zero out old gradients
        optimizer.zero_grad()
        # WHY: PyTorch ACCUMULATES gradients by default (adds to existing .grad)
        # If you forget this, gradients from batch 1 corrupt batch 2 updates
        # Rule: always zero_grad() before each forward pass

        # STEP 2: Forward pass — get predictions
        outputs = model(inputs)        # calls model.forward(inputs)
        # outputs shape: [batch_size, num_classes] — raw logits

        # STEP 3: Compute loss
        loss = criterion(outputs, labels)
        # Compares predictions to ground truth
        # Returns a single scalar (the "wrongness score" for this batch)

        # STEP 4: Backward pass — compute gradients
        loss.backward()
        # Backpropagation: computes gradient of loss w.r.t. every parameter
        # Each parameter's .grad is now filled in

        # STEP 5: Update weights
        optimizer.step()
        # Uses the gradients to nudge each weight slightly in the right direction
        # AdamW also applies weight decay here
```

**WHY this exact order matters:** zero_grad → forward → loss → backward → step. Swap any two steps and training breaks silently — gradients accumulate wrong, or weights update before gradients are computed.

---

### Key PyTorch APIs to Know

| Module | Purpose | When you use it |
|--------|---------|-----------------|
| `torch.nn` | Neural network layers and loss functions | Building models |
| `torch.optim` | Optimizers (Adam, AdamW, SGD) | Training loop |
| `torch.utils.data` | Dataset and DataLoader classes | Data pipeline |
| `torch.cuda` | GPU operations and memory management | Moving data/models to GPU |
| `torch.amp` | Mixed precision training (FP16/BF16) | Speed + memory optimization |

---

### Mixed Precision Training (FP16 and BF16)

**What it is:** Training normally uses 32-bit floats (FP32). Mixed precision uses 16-bit floats for most operations — halving memory and speeding up compute on modern GPUs. "Mixed" means: 16-bit for the forward/backward pass, 32-bit for the optimizer step.

Think of it like using shorthand during note-taking (16-bit for speed) but writing out the final summary in full sentences (32-bit for precision in weight updates).

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
# GradScaler: ONLY needed for FP16 (not BF16)
# FP16 has limited range — values below ~6e-5 underflow to zero (gradient vanishes)
# GradScaler multiplies the loss by a large number before backward pass
# so gradients stay in FP16's representable range, then unscales before optimizer step

for batch in dataloader:
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()

    with autocast():
        # autocast: automatically chooses FP16 or FP32 for each operation
        # Matrix multiplications → FP16 (fast, GPU-optimized)
        # Loss computation → FP32 (needs precision)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    # scale: multiplies loss by scaler's current scale factor (e.g., 65536)
    # backward: computes gradients in scaled space (no underflow)

    scaler.step(optimizer)
    # Unscales gradients back to true values, then runs optimizer.step()
    # If gradients contain inf/NaN: skips this step (scale was too large)

    scaler.update()
    # Adjusts scale factor for next step based on whether this step had inf/NaN

# BF16 is simpler — no GradScaler needed:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
# BF16 has the same exponent range as FP32 — no overflow/underflow risk
# Use BF16 on A100 / H100. Use FP16 + GradScaler on V100 / older GPUs.
```

**WHY this matters in interviews:** Every LLM fine-tuning guide mentions BF16. You need to explain the difference between FP16 and BF16 and when GradScaler is needed.

---

### Custom Dataset and DataLoader — Variable Length Sequences

**What it is:** For text data, sequences have different lengths. PyTorch's `Dataset` and `DataLoader` handle batching and padding efficiently.

Think of it like this: you want to ship 32 letters in one box (a batch). The letters are different lengths. You either cut them to the same length or pad shorter ones with blank pages so they all fit in identical-sized slots. Padding to the longest in THAT box (batch) wastes less space than padding everything to the global max.

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        # tokenize all texts upfront and store the result
        self.encodings = tokenizer(
            texts,
            truncation=True,       # cut sequences longer than max_length
            max_length=max_length, # don't exceed this length
            padding=False          # don't pad here — we pad per-batch below
        )
        # WHY no padding here: if we pad to 512 globally, short sentences
        # waste 400+ padding tokens of compute per batch. Pad per-batch instead.

    def __len__(self):
        return len(self.encodings["input_ids"])
        # Required by PyTorch — tells the DataLoader how many samples exist

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
        # Required by PyTorch — returns ONE sample by index
        # Returns a dict: {"input_ids": [...], "attention_mask": [...]}

# DataCollatorWithPadding: handles padding when forming a batch
from transformers import DataCollatorWithPadding
collator = DataCollatorWithPadding(tokenizer=tokenizer)
# This is called once per batch during training
# Takes a list of samples (variable lengths) and pads them to the LONGEST in the batch
# WHY this is efficient: a batch where max length is 64 only pads to 64, not 512

loader = DataLoader(
    dataset,
    batch_size=32,            # 32 samples per batch
    collate_fn=collator,      # handles padding for each batch
    shuffle=True,             # shuffle order each epoch — important for generalization
    num_workers=4             # parallel data loading — 4 CPU workers prefetch batches
    # WHY num_workers: GPU training is fast; you don't want it waiting for data prep
)
```

**WHY dynamic padding matters:** With global padding, a batch of 32 sentences where 31 are short still gets padded to 512. That's wasted GPU compute on 31 × ~400 padding tokens per batch. Dynamic padding pads to the batch's actual maximum — often 5–10x less wasted compute.

---

### torch.compile() — Free 20–40% Speed Boost (PyTorch 2.0+)

**What it is:** A single-line optimization that compiles your model into optimized GPU kernels. No code changes needed beyond adding one line.

```python
model = torch.compile(model)
# That's it. One line.
# PyTorch traces your model's operations with TorchDynamo
# Then generates optimized GPU kernels with TorchInductor
# Result: 20–40% faster training on the same hardware

# Optional: choose a compilation mode:
model = torch.compile(model, mode="default")
# "default": balanced speed vs compilation time — good starting point

model = torch.compile(model, mode="reduce-overhead")
# Optimizes for small models where Python overhead is the bottleneck

model = torch.compile(model, mode="max-autotune")
# Tries many kernel configurations to find the fastest one
# Compilation takes 5–10 minutes but gives best runtime speed
# WHY use it: when you're running long training jobs, the upfront compile cost pays off

# Known limitations:
# - First forward pass is slow (that's when compilation happens)
# - Windows support is partial (as of 2024)
# - Some dynamic shapes may cause recompilation
```

**WHY this matters:** It's a free performance win with zero code changes. Always try it before reaching for more complex optimizations like kernel fusion or custom CUDA code.

---

## 2. TensorFlow and Keras

### Overview

**What it is:** TensorFlow is Google's deep learning framework. Keras is the high-level API built into TensorFlow 2. It's more beginner-friendly than raw PyTorch, especially for standard tasks.

Think of it like this: if PyTorch is cooking from scratch, Keras is using a meal-kit service — all the ingredients are pre-measured and instructions are clear. Less flexibility, faster to set up.

- Originally had a **static computation graph** (TF1) — you defined the full graph before running it, like compiling a program before executing
- TF2 introduced **eager execution** (dynamic, like PyTorch) — runs line by line
- Better production deployment tooling (TF Serving for microservices, TFLite for mobile/edge)
- **Keras** provides a clean, simple API for model building and training

**WHY know TensorFlow in 2025:** Less dominant in research but still widely used in production at large companies (Google, enterprise systems). Some interviewers will ask about it, especially for deployment topics.

---

### Building a Model with Keras

**What it is:** Keras `Sequential` API lets you stack layers one after another in a list. The simplest way to define a model.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(
        256,
        activation='relu',   # applies ReLU after this layer's linear transformation
        input_shape=(768,)   # first layer needs to know input size; subsequent layers infer it
    ),
    # Dense(256, activation='relu'): 768→256 transformation + ReLU non-linearity

    layers.Dropout(0.2),
    # Randomly zeroes 20% of neurons during training — regularization
    # Disabled during model.predict() / model.evaluate() automatically

    layers.Dense(2, activation='softmax')
    # Final layer: 256→2 with softmax
    # softmax: converts raw scores to probabilities that sum to 1
    # e.g., [0.7, 0.3] = 70% class 0, 30% class 1
])

# Compile: configure training before actually training
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    # Adam optimizer with learning rate 1e-4 — same as PyTorch

    loss='sparse_categorical_crossentropy',
    # 'sparse': labels are integers (e.g., 0 or 1), not one-hot vectors
    # WHY sparse vs categorical: if labels are [0, 1, 2], use sparse.
    # If labels are [[1,0,0], [0,1,0]], use categorical.

    metrics=['accuracy']
    # Tracked during training but NOT used for weight updates (only loss is)
)

model.fit(
    train_dataset,
    epochs=3,
    validation_data=val_dataset
    # fit() handles the entire training loop automatically
    # WHY it's convenient: no manual zero_grad/backward/step boilerplate
)
```

---

### Custom Training Loop in TensorFlow

**What it is:** When you need more control than `model.fit()` allows, you write a custom training step using `GradientTape` — TF's equivalent of PyTorch's autograd.

```python
optimizer = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
# @tf.function: compiles this Python function into a TensorFlow graph
# WHY: running as a graph is 2–5x faster than eager execution (line by line)
# Equivalent to torch.compile() in PyTorch — free speed boost

def train_step(x, y):
    with tf.GradientTape() as tape:
        # GradientTape: records all operations inside this block
        # Equivalent to PyTorch's autograd with requires_grad=True
        preds = model(x, training=True)
        # training=True: activates dropout — important!
        # Without this, dropout is disabled even in training
        loss = loss_fn(y, preds)
        # Computes cross-entropy loss between true labels y and predictions preds

    gradients = tape.gradient(loss, model.trainable_variables)
    # tape.gradient: backpropagation — computes ∂loss/∂w for every trainable weight
    # Equivalent to loss.backward() in PyTorch
    # model.trainable_variables: all weights that should be updated (excludes frozen layers)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Pairs each gradient with its corresponding variable and applies the update
    # Equivalent to optimizer.step() in PyTorch
    return loss
```

**WHY GradientTape is designed this way:** TF2 eager execution doesn't track gradients by default (unlike PyTorch's `requires_grad`). You explicitly enter a tape context to say "record operations here for differentiation." It's more explicit — clear where gradient tracking starts and ends.

---

### TensorFlow vs PyTorch — The Real Differences

**What it is:** Both frameworks can do the same things. The differences are ecosystem, defaults, and tooling.

| Feature | PyTorch | TensorFlow / Keras |
|---------|---------|-----------------|
| Computation graph | Dynamic (define by run) | Both — TF2 is dynamic by default |
| Research use | Dominant — most papers and models | Less common in research now |
| Production deployment | TorchServe, ONNX export | TF Serving (mature), TFLite (mobile), TF.js (browser) |
| Ease of use | More Pythonic, more control | Keras is very beginner-friendly |
| LLM ecosystem | Hugging Face native — everything works here | Partial integration — some models only in PyTorch |
| Debugging | Easier — it's just Python | @tf.function can be tricky to debug |

**Interview answer for "Which framework do you prefer and why?"**
> For LLM work, PyTorch is the clear choice — the Hugging Face ecosystem is PyTorch-native, and all major foundation models are released in PyTorch. For production serving at scale, TensorFlow Serving and TFLite have mature tooling. In practice, I use PyTorch for training and experimentation, then export to ONNX for deployment flexibility.

---

## 3. Hugging Face Transformers — The Most Important Section

### What it is

**What it is:** Hugging Face Transformers is the library that gives you ready-to-use pre-trained models, tokenizers, training utilities, and datasets. It is the standard for all LLM work.

Think of it like this: without Hugging Face, loading a pre-trained LLM requires downloading weights, understanding the architecture, writing custom tokenization code, and setting up inference. Hugging Face wraps all of that into three lines.

The Hugging Face ecosystem has several sub-libraries:

```
transformers  → Pre-trained models and tokenizers (the core library)
datasets      → Load and preprocess NLP datasets efficiently
peft          → LoRA, adapters, and other parameter-efficient fine-tuning methods
trl           → SFT (supervised fine-tuning), RLHF, and DPO training utilities
accelerate    → Multi-GPU and distributed training with minimal code changes
evaluate      → Metrics computation (BLEU, ROUGE, accuracy, F1)
```

---

### Loading a Model and Tokenizer

**What it is:** The `AutoTokenizer` and `AutoModelForCausalLM` classes automatically detect the right architecture from the model ID and load the correct implementation.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id: the Hugging Face Hub identifier — "organization/model-name"
# Hugging Face will download the config, tokenizer, and weights on first use
# Subsequent calls load from local cache

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Auto: automatically selects the right tokenizer class (e.g., LlamaTokenizer)
# Downloads vocabulary files, tokenizer config, special token mappings

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # torch_dtype: load weights directly in BF16 — saves memory vs loading FP32 then converting
    # 8B model: ~16GB in FP32, ~8GB in BF16
    # BF16 is safe for inference and fine-tuning (same range as FP32)

    device_map="auto"
    # auto: distribute model layers across all available GPUs automatically
    # If model doesn't fit in GPU: overflow to CPU RAM, then disk
    # WHY device_map: 8B model needs ~16GB GPU memory — often needs 2 GPUs or CPU offload
)
```

**WHY `device_map="auto"` is important:** Without it, you have to manually shard large models across GPUs. `auto` uses the Accelerate library to do this automatically based on available memory.

---

### Auto Classes — The Right Class for the Right Task

**What it is:** Hugging Face has different model classes for different tasks. You need to load the right one. Using the wrong class will give you missing/extra layers.

| Class | Task | Example Use |
|-------|-----|-------------|
| `AutoModel` | Raw transformer output (last hidden states) | Getting token embeddings |
| `AutoModelForCausalLM` | Text generation — predicts next token (GPT-style) | LLaMA, GPT-2, Mistral |
| `AutoModelForMaskedLM` | Predicts masked tokens (BERT-style) | BERT, RoBERTa |
| `AutoModelForSeq2SeqLM` | Encoder-decoder — input sequence → output sequence | T5, BART, mT5 |
| `AutoModelForSequenceClassification` | Classifies whole sequences | Sentiment analysis, intent detection |
| `AutoModelForTokenClassification` | Classifies each token | NER (Named Entity Recognition), POS tagging |
| `AutoModelForQuestionAnswering` | Finds answer span in context | SQuAD-style QA |

```python
# Wrong — AutoModel gives raw embeddings, no LM head:
model = AutoModel.from_pretrained("gpt2")
# model.generate() will NOT work — no language modeling head

# Right — AutoModelForCausalLM has the full generation capability:
model = AutoModelForCausalLM.from_pretrained("gpt2")
# model.generate() works — has the lm_head (linear projection to vocabulary)
```

---

### Tokenizer Usage

**What it is:** A tokenizer converts raw text into token IDs (numbers) that the model can process. It also creates attention masks to handle padding.

```python
# Single sentence — basic usage:
inputs = tokenizer(
    "Hello, how are you?",
    return_tensors="pt"   # "pt" = PyTorch tensors; "tf" = TensorFlow; "np" = NumPy
)
# Returns a dict:
# {
#   "input_ids": tensor([[1, 15043, 29892, 920, 526, 366, 29973]])  — token IDs
#   "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 1]])                — all 1s (no padding)
# }
# input_ids: each number is a token in the vocabulary
# WHY return_tensors: model.forward() needs tensors, not Python lists

# Batch of sentences — different lengths need padding:
inputs = tokenizer(
    ["Hello!", "How are you doing today?"],
    padding=True,        # pad shorter sequences to match the longest in this batch
    truncation=True,     # cut sequences that exceed max_length
    max_length=512,      # maximum allowed sequence length
    return_tensors="pt"  # return as PyTorch tensors
)
# "Hello!" might be 3 tokens, "How are you doing today?" might be 7 tokens
# After padding:
#   input_ids[0]: [1, 15043, 29991, 2, 2, 2, 2]  ← padded with [PAD] token (ID=2)
#   attention_mask[0]: [1, 1, 1, 0, 0, 0, 0]     ← 0s tell model to IGNORE padding
# WHY attention_mask: without it, the model would compute attention over padding tokens
# and treat them as real content — corrupting the output

# Generation + decoding:
output_ids = model.generate(**inputs, max_new_tokens=100)
# **inputs unpacks the dict as keyword args: input_ids=..., attention_mask=...
# generate() runs autoregressive decoding — predicts one token at a time

text = tokenizer.decode(
    output_ids[0],          # decode the first (and often only) generated sequence
    skip_special_tokens=True  # remove [BOS], [EOS], [PAD] tokens from output string
)
print(text)  # "Hello! I'm doing well, thank you for asking..."
```

---

### Text Generation Pipeline

**What it is:** `pipeline()` is the highest-level Hugging Face API. One call sets up the full inference stack: model loading, tokenization, generation, decoding. Best for quick experiments and demos.

```python
from transformers import pipeline
import torch

generator = pipeline(
    "text-generation",                           # task type
    model="meta-llama/Llama-3.1-8B-Instruct",   # model to use
    torch_dtype=torch.bfloat16,                  # load in BF16 to save memory
    device_map="auto"                            # auto GPU distribution
)
# pipeline() handles: loading model + tokenizer, moving to device, etc.

output = generator(
    "Tell me about RAG systems:",
    max_new_tokens=200,    # how many new tokens to generate (not including prompt)
    temperature=0.7,       # controls randomness: 0.0 = deterministic, 1.0 = very random
    # temperature < 1: focuses on high-probability tokens (less creative)
    # temperature > 1: flattens distribution (more creative, more random)
    top_p=0.9,             # nucleus sampling: only consider top 90% of probability mass
    # WHY top_p: prevents sampling from the long tail of unlikely tokens
    # Combined with temperature: gives good quality and diversity
    do_sample=True         # if False: uses greedy decoding (always picks highest probability token)
    # WHY do_sample=True: deterministic decoding produces repetitive, boring text
)

print(output[0]['generated_text'])
# output is a list (one item per input prompt)
# 'generated_text' includes the original prompt + generated continuation
```

---

### Pipeline Tasks — Everything Available

**What it is:** `pipeline()` works for many NLP tasks beyond text generation. One line of code per task.

```python
# Each creates a fully configured, ready-to-use inference pipeline:

pipe = pipeline("text-classification")
# Returns sentiment/classification: {"label": "POSITIVE", "score": 0.99}

pipe = pipeline("ner")
# Named Entity Recognition: finds people, places, organizations in text
# Returns: [{"entity": "PER", "word": "Barack", ...}, ...]

pipe = pipeline("summarization")
# Abstractive summarization using encoder-decoder models (BART, T5)

pipe = pipeline("translation_en_to_fr")
# English to French translation

pipe = pipeline("question-answering")
# Extractive QA: finds answer span in a given context document
# Input: {"question": "...", "context": "..."}

pipe = pipeline("feature-extraction")
# Returns raw embeddings (last hidden states) for each token
# Useful for building semantic search, RAG retrieval, clustering
```

**WHY `pipeline()` matters for interviews:** If asked "how would you quickly prototype a sentiment classifier?" — pipeline() is the answer. If asked about production systems, switch to direct model + tokenizer usage for more control.

---

### Trainer API — Fine-Tuning Made Easy

**What it is:** The `Trainer` class handles the entire fine-tuning training loop for you. You provide the model, data, and configuration — Trainer handles mixed precision, gradient accumulation, checkpointing, logging, and evaluation.

Think of it like this: writing a training loop from scratch is like building a car from parts. Trainer is like buying a car — it works out of the box and has all the safety features already included.

```python
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./results",          # where to save checkpoints and final model
    overwrite_output_dir=True,        # overwrite if directory already exists

    num_train_epochs=3,               # how many full passes through training data
    # For LLM fine-tuning: 1–5 epochs is standard; more risks overfitting

    per_device_train_batch_size=8,
    # batch size PER GPU — if you have 4 GPUs, effective batch = 4×8 = 32
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=4,
    # Run 4 forward passes, accumulate gradients, THEN update weights once
    # Effective batch size = per_device_batch × num_gpus × gradient_accumulation
    # = 8 × 1 × 4 = 32 effective batch size with only 8 samples in GPU memory
    # WHY: simulate large batches on limited GPU memory

    eval_strategy="epoch",            # run evaluation after every epoch
    save_strategy="epoch",            # save checkpoint after every epoch

    learning_rate=2e-5,               # fine-tuning LR — smaller than pretraining (1e-3 to 1e-4)
    # WHY smaller: we're adjusting pre-trained weights, not training from scratch
    # Too large = catastrophic forgetting of original knowledge

    warmup_ratio=0.03,
    # First 3% of training steps: linearly increase LR from 0 to learning_rate
    # WHY warmup: prevents large gradient updates when weights are not yet settled

    lr_scheduler_type="cosine",
    # After warmup: follow cosine curve down to near-zero by end of training
    # Cosine decay: smooth, prevents abrupt stops in learning

    bf16=True,
    # Train in BF16 — halves memory, speeds up training
    # Safe on A100/H100; use fp16=True for older GPUs

    logging_steps=50,                 # log loss/metrics every 50 steps to console + wandb
    report_to="wandb",                # send metrics to Weights & Biases for experiment tracking
)

trainer = Trainer(
    model=model,                   # your pre-trained model (possibly with LoRA adapters)
    args=training_args,            # the TrainingArguments above
    train_dataset=train_dataset,   # your tokenized training dataset
    eval_dataset=eval_dataset,     # your tokenized validation dataset
    tokenizer=tokenizer,           # used for padding and saving
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    # DataCollatorForSeq2Seq: handles encoder-decoder models (T5)
    # Creates labels with -100 for padding tokens (CrossEntropyLoss ignores -100)
)

trainer.train()
# Handles: mixed precision, gradient accumulation, checkpointing, logging, evaluation
# Progress bar with live loss printed
```

**WHY gradient accumulation matters:** With a large model and 24GB GPU, you might only fit batch_size=4. With gradient_accumulation_steps=8, you get effective batch_size=32 without buying a bigger GPU.

---

### Hugging Face Datasets

**What it is:** The `datasets` library provides efficient loading and preprocessing of NLP datasets. It uses Apache Arrow under the hood — data stays memory-mapped on disk, so even 100GB datasets can be used without loading into RAM.

```python
from datasets import load_dataset

# Load well-known public datasets directly:
dataset = load_dataset("squad")       # Stanford QA dataset
dataset = load_dataset("imdb")        # Movie reviews sentiment dataset
dataset = load_dataset("json", data_files="my_data.jsonl")
# Load your own JSONL file — each line is {"text": "...", "label": 0}
# WHY JSONL: one JSON object per line — easy to append, stream, and process

# Inspect structure:
print(dataset)
# DatasetDict({
#   train: Dataset({features: ['text', 'label'], num_rows: 25000})
#   test:  Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# Preprocessing with .map():
def tokenize_function(examples):
    # examples: a batch of rows (dict of lists when batched=True)
    return tokenizer(
        examples["text"],   # the "text" column from the dataset
        truncation=True,
        max_length=512
    )
    # Returns: {"input_ids": [...], "attention_mask": [...]} — added as new columns

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True    # process multiple examples at once — much faster than one-by-one
)
# .map() is lazy-cached: re-running the same map uses the cached result
# WHY this is fast: Apache Arrow memory-mapping + parallel CPU processing
```

---

### Accelerate — Multi-GPU Training

**What it is:** Accelerate is Hugging Face's library for distributed training. It lets you run the same training code on 1 GPU, 8 GPUs, or a cluster with almost no code changes.

Think of it like this: without Accelerate, multi-GPU training requires wrapping your model in `DistributedDataParallel`, setting up process groups, handling device placement manually. Accelerate hides all of that behind `accelerator.prepare()`.

```python
from accelerate import Accelerator

accelerator = Accelerator()
# Accelerator detects: how many GPUs? Which distributed strategy? Mixed precision?
# Reads from a config file or environment — same script works everywhere

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
# .prepare() does:
# 1. Moves model to the right device (GPU 0 on single GPU, sharded on multi-GPU)
# 2. Wraps model in DistributedDataParallel if multi-GPU
# 3. Configures optimizer for distributed setup
# 4. Wraps dataloader to split batches across GPUs (each GPU sees different samples)

for batch in train_dataloader:
    # batch is automatically on the correct device for this GPU
    with accelerator.accumulate(model):
        # accumulate: handles gradient accumulation across steps
        # Only syncs gradients across GPUs when accumulation is complete
        # WHY: syncing after every step is wasteful; sync only before weight update
        outputs = model(**batch)    # forward pass
        loss = outputs.loss          # loss is automatically averaged across GPUs

        accelerator.backward(loss)
        # accelerator.backward instead of loss.backward()
        # Handles mixed precision scaling and gradient sync automatically

        optimizer.step()
        optimizer.zero_grad()
```

**WHY Accelerate instead of raw DDP:** Writing multi-GPU training from scratch requires handling process groups, device IDs, gradient sync barriers, and mixed precision scaler. Accelerate does all of this correctly. Same script runs on 1 or 8 GPUs.

---

### PEFT and LoRA — Loading and Merging Adapters

**What it is:** PEFT (Parameter Efficient Fine-Tuning) manages LoRA adapters on top of base models. You can load a base model + adapter separately, or merge them for faster inference.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model + LoRA adapter (for inference):
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# base_model: the original pre-trained weights — no task-specific fine-tuning

model = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")
# PeftModel: wraps base_model and ADDS the LoRA adapter weights on top
# During forward pass: output = W_base(x) + B(A(x))  ← LoRA adds a low-rank correction
# model is now ready for inference with the adapter's learned behavior

# Merge LoRA into the base model (for production deployment):
merged_model = model.merge_and_unload()
# merge_and_unload() computes: W_merged = W_base + B × A
# (the LoRA correction is baked permanently into the weights)
# WHY merge: removes the PEFT overhead during inference
#   With adapter: two matrix multiplies per layer (base + adapter)
#   After merge: one matrix multiply per layer — faster inference, same output
# After merging: it's just a normal model — deploy like any other model

merged_model.save_pretrained("merged-llama-finetuned")
# Save weights + config — can now be loaded as a regular AutoModelForCausalLM
```

**DataCollator options — important differences:**

```python
# Choose the right collator for your task:

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding
)

# DataCollatorForLanguageModeling:
# For causal LM (GPT-style) training — labels = input_ids shifted by 1
# The model predicts the NEXT token at each position
# labels[i] = input_ids[i+1] — automatically handled by the collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# mlm=False: causal LM (predict next token). mlm=True: masked LM (BERT-style, predict [MASK])

# DataCollatorForSeq2Seq:
# For encoder-decoder models (T5, BART)
# Handles encoder inputs and decoder inputs separately
# Sets label padding to -100 so CrossEntropyLoss ignores padding positions
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# DataCollatorWithPadding:
# For classification tasks — just pads inputs, no label manipulation
# Use when your labels are simple integers (0, 1, 2...) not sequences
collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

---

## 4. Other Important Libraries

### LangChain — Building LLM Applications

**What it is:** LangChain is a framework for chaining LLM calls together to build applications like RAG pipelines, agents, and multi-step reasoning systems.

Think of it like Unix pipes: `cat file | grep keyword | sort | head -10`. LangChain lets you pipe prompts → LLM → parser → tool → LLM → output.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Define a reusable prompt template with a variable:
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    # system message: sets the LLM's persona/behavior
    ("user", "{input}")
    # {input}: a placeholder filled in at call time
])

llm = Ollama(model="llama3")
# Ollama: runs LLMs locally — no API key needed
# WHY Ollama in LangChain: local inference for dev/testing, then swap to OpenAI in prod

chain = prompt | llm
# The | operator creates a pipeline: prompt → llm
# When called, the prompt is formatted first, then passed to the LLM
# LangChain Expression Language (LCEL): composable, lazy, supports streaming

response = chain.invoke({"input": "What is RAG?"})
# invoke fills in {input} = "What is RAG?" and runs the full chain
# Returns the LLM's text response
print(response)
```

---

### vLLM — High-Throughput LLM Serving

**What it is:** vLLM is a serving library optimized for high-throughput LLM inference. Its key innovation is PagedAttention — memory management for the KV cache (the stored key/value tensors from attention).

Think of it like this: normal inference reserves a fixed block of memory per request for the KV cache, even if the sequence is short. PagedAttention manages KV cache memory like virtual memory in an OS — allocates pages on demand, shares pages when possible.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
# LLM: loads model and sets up vLLM's PagedAttention scheduler
# Handles batching of incoming requests automatically

sampling_params = SamplingParams(
    temperature=0.7,  # randomness control
    max_tokens=512    # max response length
)

outputs = llm.generate(
    ["Tell me about transformers"],
    sampling_params
)
# vLLM batches this with other concurrent requests for efficiency
# Much higher throughput than naive HuggingFace pipeline for production APIs
```

**WHY vLLM for production:** Hugging Face's `generate()` is single-request focused. vLLM handles hundreds of concurrent requests, dynamic batching, and memory-efficient KV cache management. Standard in production LLM deployments.

---

### Weights and Biases (wandb) — Experiment Tracking

**What it is:** wandb records training metrics (loss, accuracy, LR) in real time and stores them in the cloud for comparison across experiments.

Think of it like a lab notebook, but automatic and searchable. Every training run is logged. You can compare run A vs run B in a browser dashboard, even months later.

```python
import wandb

wandb.init(
    project="llm-finetuning",   # group name — all runs for this project go here
    name="lora-run-1"           # unique name for this specific run
)
# After this, all training metrics are automatically synced to wandb.ai

# With Hugging Face Trainer: add report_to="wandb" in TrainingArguments
# Trainer automatically logs: loss, eval_loss, learning_rate, epoch, grad_norm
# No extra code needed beyond wandb.init()

# Manual logging (for custom metrics):
wandb.log({"eval_bleu": 0.42, "epoch": 1})
# Logs a custom metric at this point in training
```

---

## 5. Interview Questions — Frameworks

**Q: What is the difference between PyTorch and TensorFlow?**
> Both support dynamic computation graphs (TF2+). PyTorch is more Pythonic and dominates research. TensorFlow has stronger production deployment tooling — TF Serving for microservices, TFLite for mobile, TF.js for the browser. The Hugging Face ecosystem is PyTorch-native, making PyTorch the default choice for all LLM work.

**Q: What does `device_map="auto"` do in Hugging Face?**
> It automatically distributes model layers across all available GPUs — and if the model still doesn't fit, spills to CPU RAM or disk — using Accelerate's big model inference API. Essential for loading large models (7B+) that don't fit in a single GPU's memory.

**Q: What is gradient accumulation and why is it used?**
> Gradient accumulation runs several forward/backward passes without updating weights, then performs one optimizer step with the accumulated gradients. This simulates a larger effective batch size without requiring more GPU memory. Example: `gradient_accumulation_steps=4` with `batch_size=8` = effective batch of 32, but only 8 samples are ever in GPU memory at once.

**Q: What is the `attention_mask` in Hugging Face tokenizers?**
> A binary mask indicating real tokens (1) versus padding tokens (0). When sequences of different lengths are batched together, shorter sequences are padded to match the longest. The attention mask tells the model to completely ignore padding positions when computing self-attention — padding tokens should not influence the meaning of real tokens.

**Q: What is the difference between `model.train()` and `model.eval()`?**
> `model.train()` activates training-specific layers: dropout randomly zeroes neurons, batch norm uses batch statistics. `model.eval()` deactivates them: dropout is bypassed (all neurons active), batch norm uses running averages. Forgetting `model.eval()` during inference means dropout randomly changes your predictions — a common source of confusing bugs.

**Q: Why do you use `optimizer.zero_grad()` before each batch?**
> PyTorch accumulates gradients by default — `.backward()` ADDS new gradients to whatever is already in `.grad`. Without zeroing first, gradients from the previous batch corrupt the current batch's update. Always call `optimizer.zero_grad()` before each forward pass.

---

## Quick Reference Cheat Sheet

```
PyTorch:
  - Dynamic graph — easy to debug
  - model.train() / model.eval() — critical for dropout behavior
  - Always: zero_grad → forward → loss → backward → step
  - torch.compile(model): free 20-40% speedup — try it first
  - BF16 on A100/H100 | FP16 + GradScaler on V100

TensorFlow:
  - Keras Sequential: quick prototyping
  - @tf.function: compiles to graph — 2-5x faster than eager
  - GradientTape: equivalent to PyTorch autograd

Hugging Face:
  - AutoTokenizer + Auto[Task]Model: 3 lines to load any model
  - attention_mask: tells model to ignore padding tokens
  - device_map="auto": auto-distribute large models across GPUs
  - Trainer: handles training loop, mixed precision, checkpointing
  - gradient_accumulation_steps: simulate large batches on small GPUs
  - DataCollatorForLanguageModeling: CLM/MLM training labels
  - DataCollatorWithPadding: classification tasks

Other:
  - PEFT / LoRA: adapter loading + merge_and_unload() for deployment
  - Accelerate: multi-GPU with accelerator.prepare()
  - vLLM: production serving with PagedAttention
  - LangChain: RAG pipelines, agents, chains
  - wandb: experiment tracking — report_to="wandb" in TrainingArguments
```
