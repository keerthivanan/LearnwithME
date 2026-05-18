# 04 — Training & Fine-Tuning LLMs

> One of the most asked topics in GenAI interviews. Know pre-training, fine-tuning, PEFT, LoRA, and RLHF deeply.

---

## 1. The Three-Stage Training Pipeline

**What it is:** Modern LLMs go through three distinct training stages. Each stage has a different goal, uses different data, and costs very different amounts of compute. As a GenAI engineer, you will almost never do Stage 1, rarely do Stage 3, and will frequently do Stage 2.

```
Stage 1: Pre-Training        → General language understanding (huge data, huge compute)
Stage 2: Fine-Tuning (SFT)   → Task-specific behavior (smaller curated data)
Stage 3: Alignment (RLHF)    → Human-preferred behavior (human feedback)
```

---

## 2. Pre-Training

### What it is
Training the model from scratch on massive text data (internet, books, code). The model learns everything it will ever know about language, facts, and reasoning during this stage.

**Analogy:** Pre-training is like sending a student through 12 years of school. They learn math, language, science, history — a broad foundation. Fine-tuning is like a 3-month vocational course after graduation. You build on the broad foundation rather than re-teaching everything.

### Objective
- **GPT-style**: Predict next token (Causal LM) — "given these words, what comes next?"
- **BERT-style**: Predict masked tokens (MLM) — "fill in the [MASK] in this sentence"
- **T5-style**: Reconstruct masked spans (Span Corruption) — "fill in the missing phrase"

### Data Scale
| Model | Training Tokens |
|-------|----------------|
| GPT-3 | 300B tokens |
| LLaMA 2 | 2T tokens |
| LLaMA 3 | 15T tokens |
| Falcon | 1T tokens |

### Compute Scale
- GPT-3: ~3.14 × 10²³ FLOPs, ~$4M+ compute cost
- Training large models requires clusters of 1000s of GPUs/TPUs
- Not something you do at a company unless it is your core product

**WHY:** Pre-training is expensive because every parameter (billions of them) needs gradient updates over trillions of tokens. The Chinchilla scaling law tells us: to train optimally, you need roughly 20 tokens per parameter. LLaMA 3 8B needs ~160B tokens just to be Chinchilla-optimal. Most companies cannot afford this.

---

## 3. Supervised Fine-Tuning (SFT)

### What it is
Take a pre-trained model and continue training it on a smaller, curated dataset for a specific task or behavior. The model already knows language — SFT teaches it *how to use that knowledge* for your specific use case.

**Analogy:** After 12 years of school, you have a job interview. You already know language and math. Now you study a specific textbook about the company's products for a few days. That short study period is SFT.

### Why Fine-Tune?
- Teach the model to follow instructions (base models just continue text, they don't answer questions)
- Specialize for a domain (medical, legal, code)
- Change output format (always return JSON, always be concise)
- Improve performance on a specific task

### SFT Data Format
Typically prompt-completion pairs (input → expected output):

```json
{
  "instruction": "Summarize the following article:",
  "input": "GPT-3 is a large language model...",
  "output": "GPT-3 is a 175B parameter LLM developed by OpenAI..."
}
```

### Full Fine-Tuning
Update **all** model weights.

**WHY:** Full fine-tuning gives the maximum performance because every parameter in the model can be adjusted to fit the new task. But it is the most expensive approach — you need the full model in GPU memory PLUS gradients PLUS optimizer states (Adam stores two moving averages per parameter).

Memory: model params × 16 (for FP32 training with Adam optimizer):
- 7B model in FP32 = 28 GB just for weights
- Gradients = another 28 GB
- Adam optimizer states = another 56 GB
- Total: ~112 GB — requires 4+ A100 80GB GPUs

### When to Use Full Fine-Tuning
- Small models (< 3B) where memory is manageable
- You have a large compute budget
- You need maximum performance on the target task

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)

**What it is:** PEFT is a family of techniques that fine-tune only a small fraction of model parameters (0.01%–5%) while keeping the rest frozen. You get most of the benefit of fine-tuning at a fraction of the memory cost.

**Analogy:** Instead of renovating the entire house, you just change the curtains and add a few picture frames. The house is still fundamentally the same, but it looks and functions differently for your specific needs.

Fine-tune only a **small subset** of parameters, keeping most weights frozen.

Why this matters:
- Full fine-tuning of a 70B model requires 8× A100s at minimum
- PEFT can fine-tune a 70B model on 2× consumer GPUs
- Comparable performance to full fine-tuning on many tasks

### Methods Overview

| Method | Trainable Params | Approach |
|--------|-----------------|---------|
| LoRA | ~0.1-1% | Add low-rank matrices to attention layers |
| QLoRA | ~0.1-1% | LoRA on quantized model (even less memory) |
| Prefix Tuning | Soft prompt tokens | Add trainable tokens to beginning |
| Prompt Tuning | Very few | Trainable embeddings prepended to input |
| Adapter | ~1-5% | Small modules inserted between transformer layers |
| IA3 | < 0.01% | Scale activations with learned vectors |

---

## 5. LoRA — Low-Rank Adaptation (Most Important)

### Core Idea
**What it is:** LoRA is the most widely used PEFT method. Instead of modifying the large weight matrices (d × k dimensions), LoRA adds two small matrices (B and A) whose product approximates the weight update. Since the update is expressed as B×A, and both are much smaller than W, you train a tiny fraction of the parameters.

**Analogy:** Think of a high-resolution image (the full weight matrix W). Instead of storing the whole image, you store a "compressed delta" — a few brush strokes that represent the change you want to apply. LoRA is those brush strokes: a low-rank approximation of the change.

```
W' = W + ΔW = W + B * A
```
Where:
- W: original frozen weight (d × k) — the large matrix you do NOT update
- A: trainable matrix (r × k), r << d,k — small, gets trained
- B: trainable matrix (d × r), initialized to zero — small, gets trained
- r: rank (hyperparameter, typically 4–64) — controls how many parameters LoRA uses

### Why Low-Rank Works
The hypothesis: weight updates during fine-tuning have **low intrinsic rank** — meaning the change the model needs to make can be expressed in a low-dimensional subspace.

**WHY:** Research shows that even though W is a millions-of-parameters matrix, the actual *change* you need to fine-tune it only requires moving it in a few key directions. The update ΔW has low "true dimensionality." LoRA exploits this by factoring the update into two small matrices.

### Computation
```
h = Wx + BAx
  = Wx + ΔWx   # equivalent to adding a residual update to the original output
```
- W is frozen (not updated, does not consume gradient memory)
- Only A and B are trained (tiny fraction of total parameters)
- At inference: W' = W + BA can be pre-computed and merged — ZERO overhead at inference time!

### Savings Example
- LLaMA 7B: 7 billion params, ~56 GB for full fine-tuning (FP32 + Adam)
- LoRA with r=8: ~4 million trainable params (~0.06% of total)
- Memory: from 56 GB (full) down to ~8 GB (LoRA on quantized model)

### LoRA Hyperparameters
| Param | Typical Value | Effect |
|-------|--------------|--------|
| r (rank) | 4–64 | Higher = more capacity to adapt, more memory |
| alpha | 16–64 | Scaling factor — alpha/r scales the magnitude of the LoRA update |
| target_modules | q_proj, v_proj | Which weight matrices to apply LoRA to |
| dropout | 0.05 | Regularization to prevent overfitting |

### LoRA Code (Hugging Face PEFT)
```python
from peft import LoraConfig, get_peft_model, TaskType  # PEFT library from HuggingFace

# Define LoRA configuration — these are the key knobs to tune
config = LoraConfig(
    r=16,                              # rank — 16 is a good default starting point
    lora_alpha=32,                     # scaling: alpha/r = 2.0 is conventional best practice
    target_modules=["q_proj", "v_proj"],  # apply LoRA to query and value projection matrices
    lora_dropout=0.05,                 # 5% dropout for regularization
    bias="none",                       # don't add LoRA to bias terms (saves memory)
    task_type=TaskType.CAUSAL_LM       # tell PEFT this is a causal language model (GPT-style)
)

# Wrap the base model with LoRA — this adds the A and B matrices to specified layers
model = get_peft_model(model, config)

# Print a summary: how many params are trainable vs total
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
# This means only 0.06% of parameters are being trained!
```

**WHY:** `get_peft_model` freezes all original weights and attaches small LoRA matrices. Only A and B participate in the forward pass gradient computation — everything else is frozen and skipped during backprop. This is what makes memory usage so much lower.

---

## 6. QLoRA — Quantized LoRA

### What it is
QLoRA applies LoRA on top of a **4-bit quantized** base model. Quantization compresses the base model weights from 16-bit floats (2 bytes per weight) down to 4-bit integers (0.5 bytes per weight) — 4× compression. This makes very large models fit on consumer GPUs.

**Analogy:** LoRA is like renovating a house without rebuilding it. QLoRA is like doing that renovation while the house is stored in compressed form — like a zip file. You "unzip" small parts as needed, make your changes, and "rezip." The zip file is 4× smaller.

### How it works
1. Quantize the base model to 4-bit (NF4 = Normal Float 4-bit, better than standard int4)
2. Apply LoRA adapters on top of the quantized model
3. LoRA computations happen in BF16 (high precision); base weights are dequantized on the fly

### Result
- 7B model: ~5 GB GPU RAM (vs ~56 GB for full FP32)
- 13B model: ~8 GB GPU RAM
- 70B model: ~40 GB (2 × 24 GB GPUs)

### QLoRA Libraries
```python
from transformers import BitsAndBytesConfig  # config object for quantization

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # use 4-bit instead of 16-bit storage
    bnb_4bit_quant_type="nf4",              # NF4 = Normal Float 4: better than int4 for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute (math) is done in BF16 for accuracy
    bnb_4bit_use_double_quant=True,         # quantize the quantization constants too (saves ~0.4 bits/param)
)

# Load model with quantization — base weights stored as 4-bit, decoded at runtime
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

**WHY:** NF4 (Normal Float 4) is specifically designed for neural network weights, which follow a roughly normal (bell-curve) distribution. By using a quantization grid that matches this distribution, NF4 introduces less error than standard int4 quantization. Double quantization (quantizing the quantization constants) saves an extra 0.37 bits per parameter — small but meaningful at scale.

---

## 7. Instruction Tuning

### What it is
A specific form of SFT where the training data consists of (instruction, response) pairs. The goal is to teach the model to *follow natural language instructions* rather than just continue text. This is what transforms a base model into a chat model.

**Analogy:** A base LLM is like a very knowledgeable person who will only speak if you finish their sentence. Instruction tuning teaches them to listen to questions and give proper answers.

### Example Datasets
- **Alpaca**: 52K GPT-4 generated instructions (self-instruct method)
- **FLAN**: 1000+ NLP tasks with natural language instructions
- **ShareGPT**: Real ChatGPT conversations donated by users
- **OpenAssistant**: Human-generated conversations with quality ratings

### Format (Alpaca template)
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

### Chat Template (LLaMA 2 style)
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST] Paris </s>
```

**WHY:** The chat template tells the model where the system prompt ends, where the user message starts, and where the assistant response starts. The model is trained to recognize these special markers and produce the appropriate response in the right place. Without the correct template, the model generates random text instead of responding helpfully.

---

## 8. RLHF — Reinforcement Learning from Human Feedback

### The Problem
SFT teaches the model to generate text like the training data. But training data quality varies. We want the model to generate responses that **humans prefer** — helpful, harmless, and honest — not just responses that look statistically like the training data.

**Analogy:** SFT is like training a customer service agent by showing them transcripts. RLHF is like actually having customers rate every response the agent gives, then using those ratings to refine the agent's behavior. You get human preference signals, not just pattern matching.

### RLHF Pipeline

**Step 1: SFT (Supervised Fine-Tuning)**
Fine-tune the base model on high-quality demonstration data. This creates the SFT model that will be further improved.

**Step 2: Reward Model Training**
- Show the model a prompt and collect multiple different responses
- Human annotators rank which response is better (pairwise comparison)
- Train a separate reward model to *predict* human preference scores
- This reward model learns "what humans like" from the ranking data

```
Prompt: "Explain quantum entanglement"
Response A: [long technical answer]  → Score: 0.3  (confusing, too dense)
Response B: [clear, simple answer]   → Score: 0.8  (humans prefer this)
```

**Step 3: RL Optimization (PPO)**
Use the reward model as a reward signal to further tune the LLM using Proximal Policy Optimization (PPO).

```
LLM generates response → Reward Model scores it → PPO updates LLM to maximize reward
```

**WHY:** A KL divergence penalty is added to prevent the model from drifting too far from the SFT model. Without this, the model learns to game the reward model (Goodhart's Law) by producing weird text that scores high but is actually terrible.

### DPO — Direct Preference Optimization
**What it is:** A simpler alternative to RLHF that skips the separate reward model entirely. DPO reformulates the RLHF objective so you can directly optimize the language model on preference pairs (chosen vs rejected). Mathematically equivalent to RLHF but with no RL training loop.

**Analogy:** RLHF builds a judge (reward model) and then trains a student to please the judge. DPO skips building the judge and directly trains the student from the human rankings. Same goal, simpler path.

```python
# DPO training data format — each example has the prompt, the preferred response, and the rejected one
{
  "prompt": "Explain quantum entanglement...",
  "chosen": "Quantum entanglement is when two particles become linked...",   # human-preferred response
  "rejected": "Quantum entanglement involves quantum states and wave..."     # less preferred response
}
```

---

## 9. Continual Pre-Training

**What it is:** Fine-tune the base model on domain-specific text data WITHOUT any instruction format — just raw text. This teaches the model new domain facts and vocabulary before you do SFT. It is domain adaptation at the pre-training level.

Different from SFT: no instruction-response format, just raw domain text.
Used for domain adaptation:
- Medical text → BioMedLM (already knows medicine-specific language)
- Code → Code LLaMA
- Legal documents → Legal-specific model

**WHY:** SFT alone may not teach a model enough domain-specific vocabulary and factual knowledge. Continual pre-training first "soaks" the model in domain text so it builds up domain knowledge, then SFT teaches it to apply that knowledge helpfully.

---

## 10. Fine-Tuning Datasets

### Common Datasets
| Dataset | Type | Use |
|---------|------|-----|
| Alpaca 52K | Instruction | General instruction following |
| ShareGPT | Chat | Conversational |
| FLAN Collection | Instruction | Diverse tasks |
| OpenHermes | Instruction | High quality synthetic |
| Code Alpaca | Instruction | Code generation |
| MedAlpaca | Instruction | Medical QA |

### Data Quality > Data Quantity
- 1,000 high-quality examples > 100K noisy ones
- LIMA (2023): 1,000 carefully selected examples can match SFT on 52,000+ example datasets

**WHY:** When you train on noise, the model learns noise. Clean, well-formed, diverse instruction-response pairs teach generalizable instruction-following. Noisy, inconsistent, or poorly written examples teach the model bad habits and introduce conflicting signals that wash out the learning.

---

## 11. Training Setup with Hugging Face

### Full Fine-Tuning
```python
from transformers import Trainer, TrainingArguments  # HuggingFace training infrastructure

# Define all training hyperparameters in one object
training_args = TrainingArguments(
    output_dir="./results",                  # where to save checkpoints and final model
    num_train_epochs=3,                      # how many full passes through the training data
    per_device_train_batch_size=4,           # samples per GPU per forward pass
    gradient_accumulation_steps=8,           # accumulate gradients for 8 steps = effective batch of 32
    learning_rate=2e-4,                      # step size for optimizer — 2e-4 is common for LoRA
    fp16=True,                               # use 16-bit floats for faster computation
    warmup_ratio=0.03,                       # 3% of training steps used for linear LR warmup
    lr_scheduler_type="cosine",              # cosine decay schedule — smoothly reduces LR over training
    logging_steps=10,                        # print loss every 10 steps
    save_steps=100,                          # save a checkpoint every 100 steps
)

# Trainer handles the entire training loop — forward pass, backward pass, optimizer step
trainer = Trainer(
    model=model,                             # the model to train
    args=training_args,                      # the hyperparameter configuration
    train_dataset=train_dataset,             # training data
    eval_dataset=eval_dataset,               # validation data for monitoring
)
trainer.train()                              # start training — runs for num_train_epochs
```

**WHY:** `gradient_accumulation_steps=8` means instead of one large batch of 32 (which won't fit in GPU memory), you do 8 mini-batches of 4 and accumulate the gradients before doing a single optimizer step. You get the same mathematical effect as a batch of 32 without needing 8× the memory.

### Using TRL (Transformer Reinforcement Learning)
```python
from trl import SFTTrainer  # TRL: HuggingFace's library for SFT, DPO, PPO training

# SFTTrainer wraps Trainer with extra features: LoRA integration, tokenization handling
trainer = SFTTrainer(
    model=model,                             # the model to train (can be pre-loaded or lazy-loaded)
    train_dataset=dataset,                   # training data
    peft_config=lora_config,                 # LoRA config — SFTTrainer applies it automatically
    dataset_text_field="text",               # column name in dataset containing the training text
    max_seq_length=2048,                     # truncate training examples to 2048 tokens
    tokenizer=tokenizer,                     # tokenizer for encoding text
    args=training_args,                      # standard TrainingArguments
)
trainer.train()                              # start training
```

**WHY:** TRL's SFTTrainer handles the fiddly details of SFT — correct loss masking (only compute loss on the response, not the prompt), chat template application, and PEFT integration. Using it saves you from writing that boilerplate yourself and from common bugs like computing loss on the prompt tokens.

---

## 12. Merging LoRA Adapters

**What it is:** After LoRA training, the LoRA weights (A and B matrices) are separate from the base model. Merging them bakes the LoRA update directly into the base weights, producing a single clean model with zero inference overhead.

After training, LoRA weights can be merged into the base model for zero-overhead inference:

```python
from peft import PeftModel  # class for loading and working with PEFT-trained models

# Load the base model (without any adapters)
model = AutoModelForCausalLM.from_pretrained(base_model_id)

# Attach the LoRA adapter to the base model
model = PeftModel.from_pretrained(model, adapter_path)

# Merge: compute W' = W + BA for every LoRA layer, producing a clean base model
model = model.merge_and_unload()  # merges LoRA matrices into base weights, removes adapter overhead

# Save the merged model — can now be loaded and used like any regular model
model.save_pretrained("merged_model")
```

**WHY:** During training, keeping LoRA separate is efficient (fewer parameters to compute gradients for). At inference, you want to merge because you don't want the extra matrix multiplication overhead on every forward pass. `merge_and_unload()` pre-computes W' = W + BA once and stores that as the model's weights, eliminating the LoRA overhead permanently.

---

## Catastrophic Forgetting — The Fine-Tuning Silent Killer

**What it is:** When you fine-tune a model on new data, it can forget what it learned during pre-training. The gradient updates that make the model good at the new task overwrite the weight values that encoded the old knowledge.

**Analogy:** Imagine a person who speaks 5 languages fluently. You spend 6 months intensively training them to speak only Mandarin. After 6 months, their Mandarin is excellent, but they have forgotten Spanish and French. Catastrophic forgetting is the neural network equivalent.

Example:
- LLaMA-3 pre-trained: knows math, coding, general knowledge
- Fine-tune ONLY on customer support data
- Result: model becomes great at customer support but loses math ability
- "What is 2+2?" → wrong answer, or "I can only help with customer support topics"

**WHY:** Gradient updates that minimize loss on new task data push weights toward the new task's optimum, overwriting the weight configurations that encoded prior knowledge. The optimizer does not know to "preserve" certain capabilities.

Detection:
- Evaluate on general benchmarks (MMLU, GSM8K) BEFORE and AFTER fine-tuning
- If performance drops >5% on general benchmarks → catastrophic forgetting has occurred

Mitigation strategies:
1. **Mix base data into fine-tuning:** Include 10–20% of general training data so gradients also preserve old knowledge
2. **Lower learning rate:** Smaller gradient steps = less overwriting (use 1e-5 instead of 1e-4)
3. **LoRA:** Only trains the small adapter; base weights stay frozen → forgetting is much less severe
4. **Elastic Weight Consolidation (EWC):** Adds a penalty for changing weights that were important for old tasks
5. **Replay buffer:** Keep a small sample of old task data, train on a mix of old and new

**Best practice:** Always use LoRA for fine-tuning unless you have a very specific reason for full fine-tuning. LoRA preserves base capabilities naturally because base weights are never modified.

**Interview answer:** "How do you prevent catastrophic forgetting?" → "Mix 10–20% of original pre-training data into your fine-tuning dataset. Use LoRA so base weights stay frozen. Evaluate on general benchmarks before and after to detect forgetting. Lower learning rate helps too."

---

## LoRA Rank Selection — How to Choose r

**What it is:** The rank `r` is the most important LoRA hyperparameter. It controls the "expressiveness" of the adapter — how complex a transformation LoRA can represent. Higher rank = more parameters = more expressive adaptation.

r (rank) controls how many new parameters LoRA adds and its expressive power.

| Task Type | Recommended r | Why |
|-----------|--------------|-----|
| Simple style/format change | r=4 | Minimal change needed — the base model just needs gentle steering |
| Domain adaptation | r=8 to r=16 | Moderate adaptation — new vocabulary but same reasoning patterns |
| Complex task learning | r=32 to r=64 | More expressive power — the task requires learning new behaviors |
| Full behavior change | r=128+ | Near full fine-tuning — major reworking of model responses |

**WHY:** Rank r determines the dimensionality of the update subspace. Low rank means you are learning a simpler transformation (like rotating a 2D plane in high-dimensional space). High rank can express more complex weight changes but adds more trainable parameters and can overfit if you have limited data.

Rule of thumb: start with r=16; if validation loss plateaus, try r=32.

**Alpha parameter:** Always set alpha = 2 × r (e.g., r=16 → alpha=32)
- alpha/r is the effective learning rate scaling factor for the LoRA update
- Keeping alpha = 2r is conventional wisdom from the original LoRA paper

**Target modules (which layers to apply LoRA to):**
- Minimal: q_proj, v_proj only (query and value projection in attention)
- Standard: q_proj, k_proj, v_proj, o_proj (all attention projections)
- Aggressive: all linear layers including MLP (gate_proj, up_proj, down_proj)

Memory impact: r=16, targeting q/k/v/o in 7B model ≈ 40M trainable params (0.6% of 7B).

**Interview answer:** "How do you choose LoRA rank?" → "Start with r=16 for most tasks. Complex task learning needs r=32–64. Simple style changes work with r=4–8. Always set alpha=2r. Apply to q,k,v,o projection layers at minimum."

---

## Adapter Merging — Combining Multiple LoRA Adapters

**What it is:** You have trained two separate LoRA adapters — one that makes the model good at math, one that makes it good at coding. You want one single model that does both. This is model merging.

You have LoRA-A (math expert) and LoRA-B (coding expert). Want one model that does both?

**Simple averaging (often fails):**
```
merged = 0.5 × LoRA-A + 0.5 × LoRA-B
```
Problem: conflicting parameters cancel each other out. One adapter says "increase weight X," the other says "decrease weight X," and the average is zero — you get neither.

**TIES-Merging (Trim, Elect Sign, Merge):**
1. **Trim:** Keep only top-k% largest magnitude parameters per adapter (small values are likely noise)
2. **Elect Sign:** For each parameter, take the sign (+ or −) that the majority of adapters agree on
3. **Merge:** Average only parameters that agree on sign (disagreeing parameters cancel out less)
Result: reduces interference between conflicting parameters

**WHY:** Small-magnitude delta weights are likely noise from stochastic training. Trimming them before merging avoids adding noise from two adapters on top of each other. Sign election prevents destructive interference.

**DARE (Drop and Rescale):**
1. Randomly drop p% of delta weights (set them to 0)
2. Rescale remaining by 1/(1−p) to preserve the average magnitude
3. Then merge
Result: sparsifies adapters before merging, reduces conflicts

**WHY:** Sparsification reduces overlap between adapters so their delta weights interfere less when summed. Think of it as thinning out the crowd so there are fewer collisions.

**mergekit library:** The easiest way to do model merging in practice.
- Supports TIES, DARE, SLERP, linear merge
- Used by the community to create "frankenmerge" models on HuggingFace

**Interview answer:** "How do you merge multiple LoRA adapters?" → "Simple averaging often fails due to sign conflicts. TIES-merging trims small parameters and resolves sign disagreements before averaging. DARE randomly drops delta weights before merging. The mergekit library implements all these methods."

---

## 13. Interview Questions — Fine-Tuning

**Q: What is the difference between fine-tuning and pre-training?**
> Pre-training trains a model from scratch on massive general data to learn language representations. Fine-tuning starts from a pre-trained model and further trains on smaller, task-specific data to adapt it for a particular use case. Fine-tuning is vastly cheaper.

**Q: What is LoRA and why is it important?**
> LoRA (Low-Rank Adaptation) adds trainable low-rank matrices (B×A) to frozen pre-trained weights, allowing fine-tuning with ~0.1% of the parameters. It's important because it makes fine-tuning 70B+ models feasible on limited hardware while maintaining most of the quality of full fine-tuning.

**Q: What is the difference between LoRA and QLoRA?**
> QLoRA applies LoRA on top of a 4-bit quantized base model, further reducing memory requirements. A 7B model that needs 14 GB for LoRA fine-tuning can be fine-tuned with QLoRA on a single 8 GB GPU.

**Q: Explain RLHF.**
> RLHF is a three-step process: (1) supervised fine-tuning on demonstrations, (2) training a reward model on human preference rankings of outputs, (3) using RL (PPO) to optimize the LLM to maximize the reward model's score while staying close to the SFT model via a KL penalty. It aligns models to human preferences.

**Q: What is DPO and how does it differ from RLHF?**
> DPO (Direct Preference Optimization) directly optimizes the language model on preference pairs (chosen vs rejected responses) without training a separate reward model. It's mathematically equivalent to RLHF in theory but simpler and more stable in practice.

**Q: When would you use fine-tuning vs RAG?**
> Fine-tuning when: you need to change model behavior/style, teach domain-specific reasoning, or improve performance on a specific task. RAG when: you need access to external or frequently updated knowledge that wasn't in training data. They can be combined.

---

## Quick Reference Cheat Sheet

```
Pre-training:     Scratch training on massive data (expensive, rare)
SFT:              Fine-tune on instruction/response pairs (most common)
LoRA:             Low-rank adapters, ~0.1% params, freezes base model
QLoRA:            LoRA + 4-bit quantization (fits on consumer GPU)
RLHF:             SFT → Reward Model → PPO optimization
DPO:              Simpler RLHF alternative, no reward model needed
Instruction tuning: Teach model to follow natural language instructions
Catastrophic forgetting: Mix old data + use LoRA to prevent
TIES/DARE:        Methods for merging multiple LoRA adapters without conflict
```
