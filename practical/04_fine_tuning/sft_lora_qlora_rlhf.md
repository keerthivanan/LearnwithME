# 04 — Training & Fine-Tuning LLMs

> One of the most asked topics in GenAI interviews. Know pre-training, fine-tuning, PEFT, LoRA, and RLHF deeply.

---

## 1. The Three-Stage Training Pipeline

```
Stage 1: Pre-Training        → General language understanding (huge data, huge compute)
Stage 2: Fine-Tuning (SFT)   → Task-specific behavior (smaller curated data)
Stage 3: Alignment (RLHF)    → Human-preferred behavior (human feedback)
```

---

## 2. Pre-Training

### What it is
Train the model from scratch on massive text data (internet, books, code).

### Objective
- **GPT-style**: Predict next token (Causal LM)
- **BERT-style**: Predict masked tokens (MLM)
- **T5-style**: Reconstruct masked spans (Span Corruption)

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
- Not something you do at a company unless it's your core product

### Pre-training is expensive — most engineers work with fine-tuning.

---

## 3. Supervised Fine-Tuning (SFT)

### What it is
Take a pre-trained model and continue training on a smaller, curated dataset for a specific task or behavior.

### Why Fine-Tune?
- Teach the model to follow instructions
- Specialize for a domain (medical, legal, code)
- Improve performance on specific tasks

### SFT Data Format
Typically prompt-completion pairs:

```json
{
  "instruction": "Summarize the following article:",
  "input": "GPT-3 is a large language model...",
  "output": "GPT-3 is a 175B parameter LLM developed by OpenAI..."
}
```

### Full Fine-Tuning
Update **all** model weights.
- Most effective
- Most expensive (need full model in GPU memory + gradients + optimizer states)
- Memory: model params × 16 (for FP32 training with Adam)

### When to Use Full Fine-Tuning
- Small models (< 3B)
- Have large compute budget
- Need maximum performance on the target task

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)

Fine-tune only a **small subset** of parameters, keeping most weights frozen.

Why?
- Full fine-tuning of a 70B model requires 8× A100s at minimum
- PEFT can fine-tune on a single consumer GPU
- Comparable performance to full fine-tuning on many tasks

### Methods Overview

| Method | Trainable Params | Approach |
|--------|-----------------|---------|
| LoRA | ~0.1-1% | Add low-rank matrices to attention layers |
| QLoRA | ~0.1-1% | LoRA on quantized model |
| Prefix Tuning | Soft prompt tokens | Add trainable tokens to beginning |
| Prompt Tuning | Very few | Trainable embeddings prepended |
| Adapter | ~1-5% | Small modules between transformer layers |
| IA3 | < 0.01% | Scale activations |

---

## 5. LoRA — Low-Rank Adaptation (Most Important)

### Core Idea
Instead of modifying large weight matrix W (d × k), add a low-rank decomposition:

```
W' = W + ΔW = W + B * A
```
Where:
- W: original frozen weight (d × k)
- A: trainable matrix (r × k), r << d,k
- B: trainable matrix (d × r), initialized to zero
- r: rank (hyperparameter, typically 4–64)

### Why Low-Rank Works
The hypothesis: weight updates during fine-tuning have **low intrinsic rank** — meaning the change can be expressed in a low-dimensional subspace.

### Computation
```
h = Wx + BAx
   = Wx + ΔWx        # equivalent to adding a residual
```
- W is frozen (not updated)
- Only A and B are trained
- At inference: W' = W + BA (can be merged — zero overhead!)

### Savings Example
- LLaMA 7B: 7B params
- LoRA with r=8: ~4M trainable params (~0.06% of total)
- Memory: from 56GB (full) to ~8GB (LoRA on quantized)

### LoRA Hyperparameters
| Param | Typical Value | Effect |
|-------|--------------|--------|
| r (rank) | 4–64 | Higher = more capacity, more memory |
| alpha | 16–64 | Scaling factor (α/r scales the update) |
| target_modules | q_proj, v_proj | Which layers to apply LoRA to |
| dropout | 0.05 | Regularization |

### LoRA Code (Hugging Face PEFT)
```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

---

## 6. QLoRA — Quantized LoRA

### What it is
LoRA applied to a **4-bit quantized** model.

### How it works
1. Quantize the base model to 4-bit (NF4 quantization)
2. Apply LoRA adapters on top
3. Compute in BF16 for LoRA, dequantize weights on the fly

### Result
- 7B model: ~5GB GPU RAM
- 13B model: ~8GB GPU RAM
- 70B model: ~40GB (2× 24GB GPUs)

### QLoRA Libraries
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

---

## 7. Instruction Tuning

### What it is
Fine-tune on a dataset of (instruction, response) pairs so the model learns to **follow instructions**.

### Example Datasets
- **Alpaca**: 52K GPT-4 generated instructions
- **FLAN**: 1000+ NLP tasks with natural language instructions
- **ShareGPT**: Real ChatGPT conversations
- **OpenAssistant**: Human-generated conversations

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

---

## 8. RLHF — Reinforcement Learning from Human Feedback

### The Problem
SFT teaches the model to generate text like the training data. But training data quality varies. We want the model to generate responses that **humans prefer**.

### RLHF Pipeline

**Step 1: SFT (Supervised Fine-Tuning)**
Fine-tune the base model on high-quality demonstration data.

**Step 2: Reward Model Training**
- Collect pairs of model outputs
- Human annotators rank which response is better
- Train a reward model to predict human preference score

```
Prompt: "Explain quantum entanglement"
Response A: [long technical answer] → Score: 0.3
Response B: [clear, simple answer]  → Score: 0.8
```

**Step 3: RL Optimization (PPO)**
Use the reward model as the reward signal to further tune the LLM using Proximal Policy Optimization (PPO).

```
LLM generates response → Reward Model scores it → PPO updates LLM to maximize reward
```

KL penalty prevents the model from drifting too far from the SFT model.

### DPO — Direct Preference Optimization
- Simpler alternative to RLHF (no separate reward model)
- Directly optimizes the policy on preference pairs
- More stable, easier to implement
- Used in many modern fine-tuned models

```python
# DPO uses chosen/rejected pairs
{
  "prompt": "...",
  "chosen": "good response",
  "rejected": "bad response"
}
```

---

## 9. Continual Pre-Training

Fine-tune the base model on domain-specific data **without instruction format**.
Used for domain adaptation:
- Medical text → BioMedLM
- Code → Code LLaMA
- Legal documents → Legal-specific model

Different from SFT: no instruction-response format, just raw domain text.

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
- 1000 high-quality examples > 100K noisy ones
- LIMA (2023): 1000 carefully selected examples can match SFT on much larger datasets

---

## 11. Training Setup with Hugging Face

### Full Fine-Tuning
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### Using TRL (Transformer Reinforcement Learning)
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

---

## 12. Merging LoRA Adapters

After training, LoRA weights can be merged into the base model for zero-overhead inference:

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()  # merges adapter into base weights
model.save_pretrained("merged_model")
```

---

## 13. Interview Questions — Fine-Tuning

**Q: What is the difference between fine-tuning and pre-training?**
> Pre-training trains a model from scratch on massive general data to learn language representations. Fine-tuning starts from a pre-trained model and further trains on smaller, task-specific data to adapt it for a particular use case. Fine-tuning is vastly cheaper.

**Q: What is LoRA and why is it important?**
> LoRA (Low-Rank Adaptation) adds trainable low-rank matrices (B×A) to frozen pre-trained weights, allowing fine-tuning with ~0.1% of the parameters. It's important because it makes fine-tuning 70B+ models feasible on limited hardware while maintaining most of the quality of full fine-tuning.

**Q: What is the difference between LoRA and QLoRA?**
> QLoRA applies LoRA on top of a 4-bit quantized base model, further reducing memory requirements. A 7B model that needs 14GB for full fine-tuning can be fine-tuned with QLoRA on a single 8GB GPU.

**Q: Explain RLHF.**
> RLHF is a three-step process: (1) supervised fine-tuning on demonstrations, (2) training a reward model on human preference rankings of outputs, (3) using RL (PPO) to optimize the LLM to maximize the reward model's score while staying close to the SFT model. It aligns models to human preferences.

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
```
