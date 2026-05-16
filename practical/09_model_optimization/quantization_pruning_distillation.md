# 09 — Model Optimization: Quantization, Pruning & Distillation

> The JD asks for scaling, optimizing, and improving efficiency of large models. This file covers all techniques.

---

## 1. Why Optimization Matters

A 70B LLM in FP32:
- 70B × 4 bytes = **280GB** of VRAM just for weights
- Inference: too slow for real-time use without optimization
- Training: requires massive GPU clusters

Goal: make models **smaller**, **faster**, and **cheaper** without losing much quality.

---

## 2. Quantization

### What it is
Reduce the numerical precision of model weights (and/or activations) from 32-bit floats to lower bit widths.

```
FP32 (4 bytes) → FP16 (2 bytes) → INT8 (1 byte) → INT4 (0.5 bytes)
```

### Memory Savings
| Precision | Bits | 7B model size | 70B model size |
|-----------|------|--------------|---------------|
| FP32 | 32 | 28 GB | 280 GB |
| FP16 / BF16 | 16 | 14 GB | 140 GB |
| INT8 | 8 | 7 GB | 70 GB |
| INT4 (NF4) | 4 | 3.5 GB | 35 GB |

### FP16 vs BF16
| Format | Range | Precision | Stability |
|--------|-------|-----------|---------|
| FP16 | Moderate | Higher mantissa | Can overflow (NaN) |
| BF16 | Large (same as FP32) | Lower mantissa | More stable for training |

BF16 is preferred for LLM training. FP16 for inference on older hardware.

### Post-Training Quantization (PTQ)
Quantize a trained model without further training.

**GPTQ** — Row-wise quantization using second-order information
```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

quantized_model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=BaseQuantizeConfig(bits=4, group_size=128)
)
```

**AWQ (Activation-Aware Weight Quantization)**
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_id)
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
```

**bitsandbytes (NF4 for QLoRA)**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # Normal Float 4 (better than INT4)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # quantize quantization constants
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

### Quantization-Aware Training (QAT)
Simulate quantization during training so the model adapts to lower precision.
- Better quality than PTQ
- More expensive (requires training)

### Quantization Formats
| Format | Description |
|--------|------------|
| INT8 | Standard 8-bit integer |
| INT4 | 4-bit integer |
| NF4 | Normal Float 4 (data type for QLoRA, better distribution) |
| GGUF | Format for llama.cpp (quantized CPU inference) |
| GPTQ | Post-training quantization, stored in blocks |
| AWQ | Activation-aware, better for 4-bit |

---

## 3. Pruning

### What it is
Remove weights (or entire neurons/heads) that contribute little to the model's output.

### Unstructured Pruning
Zero out individual weights below a threshold.
- Saves memory in theory but sparse matrices aren't efficient on GPUs
- Limited hardware speedup without sparse support

### Structured Pruning
Remove entire structures: attention heads, layers, neurons.
- Hardware-friendly
- More impact on quality

**Attention Head Pruning**
Not all attention heads are equally important. Some can be removed with minimal quality loss.
- BERT: can prune ~50% of heads with <1% performance drop

**Layer Pruning**
Remove entire transformer layers.
```python
# Example: Keep every other layer
model.encoder.layer = model.encoder.layer[::2]
```

**Width Pruning**
Reduce FFN hidden dimension or embedding dimension.

### Magnitude-Based Pruning
```python
# Prune weights with smallest absolute value
threshold = 0.01
for name, param in model.named_parameters():
    param.data[param.data.abs() < threshold] = 0.0
```

### Iterative Pruning
Prune → Fine-tune → Prune → Fine-tune → ...
Gradually remove weights while retraining to recover performance.

---

## 4. Knowledge Distillation

### What it is
Train a small **student** model to mimic the behavior of a large **teacher** model.

```
Teacher (large): 7B params, high quality
     ↓ (provides soft labels / logits)
Student (small): 1B params, faster inference
```

### How it Works
```
Standard training:   student learns from hard labels (true answers)
Distillation:        student also learns from teacher's soft predictions
```

The teacher's output probabilities carry more information than one-hot labels:
```
One-hot:    [0, 0, 1, 0, 0]     # only says "cat"
Teacher:    [0.02, 0.01, 0.85, 0.08, 0.04]  # says cat, maybe kitten
```

### Distillation Loss
```
L = α * L_CE(student, true_labels) + (1-α) * L_KD(student, teacher)
```
Where L_KD uses softened probabilities (temperature > 1 for softer distribution).

### Famous Distilled Models
| Student | Teacher | Notes |
|---------|---------|-------|
| DistilBERT | BERT | 40% smaller, 60% faster, 97% quality |
| TinyBERT | BERT | Further compressed BERT |
| GPT-4o mini | GPT-4 | OpenAI's distilled model |
| Phi-3 mini | GPT-4 | Very small but capable (Microsoft) |

### Sequence-Level Distillation (for LLMs)
1. Generate responses from teacher on training prompts
2. Fine-tune student on teacher-generated responses
Used to distill capabilities like instruction-following or reasoning.

---

## 5. ONNX & Inference Optimization

### ONNX (Open Neural Network Exchange)
Export model to a framework-agnostic format for optimized inference.

```python
# Export PyTorch model to ONNX
torch.onnx.export(
    model,
    sample_input,
    "model.onnx",
    opset_version=14,
    dynamic_axes={"input": {0: "batch", 1: "seq_len"}}
)
```

### ONNX Runtime
Run ONNX models with hardware-specific optimizations.
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
output = session.run(None, {"input": input_array})
```

### TensorRT (NVIDIA)
NVIDIA's inference optimizer for CUDA GPUs.
- Graph fusion, kernel optimization, FP16/INT8 quantization
- 2-10× speedup over PyTorch on NVIDIA GPUs

---

## 6. Model Compression Comparison

| Technique | Memory | Speed | Quality | Ease |
|-----------|--------|-------|---------|------|
| FP16 quantization | 2× reduction | 2× faster | Near-lossless | Easy |
| INT8 quantization | 4× reduction | 2-3× faster | Minor degradation | Moderate |
| INT4 quantization | 8× reduction | 3-4× faster | Some degradation | Moderate |
| Pruning (unstructured) | Variable | Minimal | Variable | Hard |
| Pruning (structured) | Variable | Real speedup | Variable | Hard |
| Distillation | 3-10× smaller | 3-10× faster | Moderate | Hard |

---

## 7. Inference Optimization Techniques

### Speculative Decoding
Use a small draft model to speculatively generate multiple tokens, then verify with the large model in parallel.
- Same quality as original model (mathematically)
- 2-3× inference speedup
- Used in production at Google, Meta

```
Draft model generates: [" The", " cat", " sat"]
Large model verifies all 3 in parallel
Accept: [" The", " cat"] → reject " sat" → resample
```

### Continuous Batching
Instead of waiting for all requests in a batch to finish, add new requests as slots open up.
- Dramatically improves GPU utilization
- Used in vLLM

### PagedAttention (vLLM)
Inspired by virtual memory in OS. Manages KV cache memory in non-contiguous pages.
- Eliminates memory fragmentation
- Enables more concurrent requests
- Foundation of vLLM's performance

### Flash Attention
(Covered in 02_transformers) — reduces memory I/O, enables longer contexts.

---

## 8. Hardware Acceleration

### GPU Types for LLM Work
| GPU | VRAM | Good For |
|-----|------|---------|
| NVIDIA A100 80GB | 80GB | Large model training |
| NVIDIA A100 40GB | 40GB | Training/inference |
| NVIDIA H100 | 80GB | Fastest, latest |
| NVIDIA RTX 4090 | 24GB | Fine-tuning consumer |
| NVIDIA RTX 3090 | 24GB | Fine-tuning consumer |
| Google TPU v4 | HBM | T5, PaLM training |

### Multi-GPU Strategies
(Covered in detail in 10_distributed_training.md)
- Data Parallelism
- Tensor Parallelism
- Pipeline Parallelism

### CPU Inference (llama.cpp)
For running quantized models on CPU:
```bash
./llama-cli -m llama-3.1-8b.Q4_K_M.gguf -p "Tell me about AI" -n 200
```
GGUF format, Q4 quantization → 7B model on 8GB RAM.

---

## 9. Interview Questions — Optimization

**Q: What is quantization and why is it important for LLMs?**
> Quantization reduces the numerical precision of model weights (e.g., FP32 → INT4). This reduces memory requirements by 4-8×, enabling larger models on limited hardware and faster inference. A 7B model that needs 14GB in FP16 can fit in 3.5GB with INT4 quantization.

**Q: What is the difference between GPTQ and AWQ?**
> GPTQ quantizes layer by layer using second-order (Hessian) information to minimize quantization error. AWQ is activation-aware — it identifies weights that are most important based on activation patterns and protects them during quantization. AWQ generally produces better quality at 4-bit.

**Q: What is knowledge distillation?**
> Training a smaller student model to mimic a larger teacher model. The student learns from the teacher's soft probability distributions (not just hard labels), which contain richer information about the relationships between classes. Result: a smaller, faster model that retains much of the teacher's capabilities.

**Q: What is speculative decoding?**
> A technique where a fast small draft model generates multiple candidate tokens, then the large model verifies them all in a single forward pass. Accepted tokens are kept, rejected ones are resampled. This achieves 2-3× speedup while maintaining identical output quality to the original model.

**Q: What is PagedAttention and why does vLLM use it?**
> PagedAttention manages the KV cache in non-contiguous memory pages (like OS virtual memory), eliminating memory fragmentation and waste. vLLM uses it to serve more concurrent requests efficiently on the same hardware.

---

## Quick Reference Cheat Sheet

```
Quantization:    FP32→FP16→INT8→INT4, reduces memory & speeds inference
GPTQ/AWQ:        Best PTQ methods for 4-bit LLMs
bitsandbytes:    INT8/NF4 quantization for HuggingFace + QLoRA
Distillation:    Large teacher → small student via soft labels
Pruning:         Remove weights/heads/layers (structured = GPU-friendly)
Speculative:     Draft model + large model = 2-3× faster inference
vLLM:            PagedAttention + continuous batching for production
FlashAttention:  Efficient attention computation, memory-efficient
```
