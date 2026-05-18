# 09 — Model Optimization: Quantization, Pruning & Distillation

> The JD asks for scaling, optimizing, and improving efficiency of large models. This file covers all techniques.

---

## 1. Why Optimization Matters

**What it is:** Large language models are computationally expensive. Before optimization they are too slow, too large, and too costly for real production use. Optimization makes them practical.

A 70B LLM in FP32:
- 70B × 4 bytes = **280GB** of VRAM just for weights
- Inference: too slow for real-time use without optimization
- Training: requires massive GPU clusters

**Analogy:** Optimization is like converting a giant textbook into a pocket reference guide. You lose some detail but the guide is 10× smaller and much faster to use — and for most real situations, the pocket guide is good enough.

Goal: make models **smaller**, **faster**, and **cheaper** without losing much quality.

---

## 2. Quantization

### What it is

**What it is:** Quantization reduces the number of bits used to store each number (weight) in the model. Full precision (FP32) uses 32 bits per number. Quantization compresses this down to 16, 8, or even 4 bits per number.

**Analogy:** Quantization is like rounding prices. Instead of storing "£14.73921", you store "£15". You lose a tiny bit of precision but the price list is now much shorter and easier to work with. The key insight: neural network weights are slightly imprecise anyway, so a little more rounding rarely hurts much.

Reduce the numerical precision of model weights (and/or activations) from 32-bit floats to lower bit widths.

```
FP32 (4 bytes) → FP16 (2 bytes) → INT8 (1 byte) → INT4 (0.5 bytes)
```

### Memory Savings

**What it is:** The direct memory reduction from using lower precision. This determines whether you can even load the model onto your available GPUs.

| Precision | Bits | 7B model size | 70B model size |
|-----------|------|--------------|---------------|
| FP32 | 32 | 28 GB | 280 GB |
| FP16 / BF16 | 16 | 14 GB | 140 GB |
| INT8 | 8 | 7 GB | 70 GB |
| INT4 (NF4) | 4 | 3.5 GB | 35 GB |

**WHY this matters:** A single A100 GPU has 80GB VRAM. At FP32, a 70B model needs 280GB — impossible on one GPU. At INT4, it fits in 35GB — possible on one GPU. Quantization is often the difference between "runs" and "doesn't run."

### FP16 vs BF16

**What it is:** Two different 16-bit floating point formats with different trade-offs. The difference is in how they split those 16 bits between range (exponent) and precision (mantissa).

| Format | Range | Precision | Stability |
|--------|-------|-----------|---------|
| FP16 | Moderate | Higher mantissa | Can overflow (NaN) |
| BF16 | Large (same as FP32) | Lower mantissa | More stable for training |

**WHY BF16 is preferred for training:** FP16 can produce NaN (Not a Number) when values get too large — this crashes training. BF16 has the same exponent range as FP32 so it never overflows, making training stable. Use BF16 for training, FP16 for inference on older hardware that does not support BF16.

### Post-Training Quantization (PTQ)

**What it is:** Quantize a model *after* it has already been trained, without any additional training. Fast and easy — you just convert the weights. The trade-off is slightly lower quality than quantization-aware training.

Quantize a trained model without further training.

**GPTQ** — Row-wise quantization using second-order information

**What it is:** GPTQ quantizes one layer at a time, using mathematical information about how important each weight is (second-order Hessian information) to minimize the error introduced by rounding. Smarter than naive quantization.

```python
from transformers import AutoModelForCausalLM   # Hugging Face model loader
from auto_gptq import AutoGPTQForCausalLM       # GPTQ quantization library

quantized_model = AutoGPTQForCausalLM.from_pretrained(
    model_id,                                   # the model to quantize (HF Hub ID or local path)
    quantize_config=BaseQuantizeConfig(bits=4, group_size=128)  # 4-bit quantization, groups of 128 weights
)
```

**AWQ (Activation-Aware Weight Quantization)**

**What it is:** AWQ looks at which weights are most important by analyzing the activation patterns during inference. It protects those critical weights from heavy quantization while aggressively compressing less important ones.

```python
from awq import AutoAWQForCausalLM              # AWQ quantization library

model = AutoAWQForCausalLM.from_pretrained(model_id)  # load the model to quantize
model.quantize(tokenizer, quant_config={
    "zero_point": True,                         # use zero-point quantization (more accurate)
    "q_group_size": 128,                        # quantize in groups of 128 weights
    "w_bit": 4                                  # use 4-bit weights
})
```

**bitsandbytes (NF4 for QLoRA)**

**What it is:** A library that integrates directly with Hugging Face Transformers to quantize models on load. NF4 (Normal Float 4) is a special 4-bit data type designed for neural network weights that follow a normal distribution — it is more accurate than plain INT4.

```python
from transformers import BitsAndBytesConfig     # quantization config class
import torch                                    # PyTorch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # load weights in 4-bit precision
    bnb_4bit_quant_type="nf4",                  # use Normal Float 4 (better than INT4 for neural weights)
    bnb_4bit_compute_dtype=torch.bfloat16,      # do matrix multiplications in BF16 for stability
    bnb_4bit_use_double_quant=True,             # quantize the quantization constants too (saves extra memory)
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)  # load quantized model
```

**WHY double quantization:** The quantization constants (scale and zero-point) for each group also take up memory. Quantizing them too saves another ~0.4 bits per parameter — small but meaningful at scale.

### Quantization-Aware Training (QAT)

**What it is:** During training, simulate the rounding effects of quantization by adding fake quantization operations. The model learns to work well despite the precision loss, resulting in better quality than quantizing after the fact.

Simulate quantization during training so the model adapts to lower precision.
- Better quality than PTQ
- More expensive (requires training resources)

**WHY QAT is better:** When you quantize after training (PTQ), the model was never trained to handle the rounding errors, so quality drops. With QAT, the model sees the rounding errors during training and adjusts its weights to minimize their impact.

### Quantization Formats

**What it is:** The different storage formats you will encounter when working with quantized models.

| Format | Description |
|--------|------------|
| INT8 | Standard 8-bit integer — good balance, widely supported |
| INT4 | 4-bit integer — aggressive compression, some quality loss |
| NF4 | Normal Float 4 — designed for neural weights, better than INT4 |
| GGUF | Format for llama.cpp — quantized models that run on CPU |
| GPTQ | Post-training quantization stored in blocks — GPU inference |
| AWQ | Activation-aware — better for 4-bit inference on GPU |

---

## 3. Pruning

### What it is

**What it is:** Pruning removes parts of the model that contribute little to its output. Just like a tree gets pruned by cutting dead branches, a neural network gets pruned by zeroing out or removing weak weights.

**Analogy:** Pruning is like editing a 300-page report. Most pages have unique content, but some paragraphs are redundant or barely contribute to the message. Cutting those paragraphs makes the report shorter without losing much meaning.

Remove weights (or entire neurons/heads) that contribute little to the model's output.

### Unstructured Pruning

**What it is:** Zero out individual weight values that are below a threshold — regardless of their position in the matrix. Very fine-grained but creates "sparse" matrices that standard hardware handles inefficiently.

Zero out individual weights below a threshold.
- Saves memory in theory but sparse matrices aren't efficient on standard GPUs
- Limited hardware speedup without sparse tensor core support

**WHY this is tricky:** Most GPU matrix multiplication operations are optimized for *dense* matrices. A matrix with 50% zeros does not run 2× faster — the GPU still does the same number of operations. You need special sparse matrix libraries to see real speedup.

### Structured Pruning

**What it is:** Remove entire structural units — whole attention heads, entire layers, complete neurons in the feed-forward network. Because you are removing complete units, the resulting model is still a normal dense model that runs fast on standard hardware.

Remove entire structures: attention heads, layers, neurons.
- Hardware-friendly (results in a smaller dense model, not a sparse one)
- More impact on quality (removing whole units is more drastic than removing individual weights)

**Attention Head Pruning**

**What it is:** Research has shown that not all attention heads are equally important. Some heads can be removed completely with minimal impact on output quality.

Not all attention heads are equally important. Some can be removed with minimal quality loss.
- BERT: can prune ~50% of heads with less than 1% performance drop

**Layer Pruning**

**What it is:** Remove entire transformer layers. The most aggressive form of structured pruning — fewer layers = much faster inference.

Remove entire transformer layers.
```python
# Example: Keep every other layer (removes 50% of layers)
model.encoder.layer = model.encoder.layer[::2]  # [::2] means take every second element from the list
```

**WHY layer pruning works:** Research shows middle layers in large transformers often learn redundant representations. You can skip some without dramatically hurting quality — especially if you fine-tune after pruning.

**Width Pruning**

**What it is:** Reduce the size of weight matrices — narrower feed-forward networks or smaller embedding dimensions.

Reduce FFN hidden dimension or embedding dimension.

### Magnitude-Based Pruning

**What it is:** The simplest pruning approach — assume that weights with small absolute values contribute little to the model's output, and zero them out.

```python
threshold = 0.01                                # weights smaller than this will be zeroed
for name, param in model.named_parameters():    # iterate over every weight matrix in the model
    param.data[param.data.abs() < threshold] = 0.0  # zero out weights with absolute value below threshold
```

**WHY magnitude-based pruning works:** A weight of 0.001 multiplied by an activation barely changes the result. Removing it has negligible effect. Weights of 5.0 are important and should stay.

### Iterative Pruning

**What it is:** Prune a small amount, then fine-tune to recover quality, then prune again, and repeat. Gradual pruning causes less damage than pruning everything at once.

Prune → Fine-tune → Prune → Fine-tune → ...
Gradually remove weights while retraining to recover performance.

**WHY iterative is better:** Pruning 50% at once causes significant quality loss and fine-tuning may not fully recover. Pruning 10% five times with fine-tuning between each step is gentler — the model adapts incrementally.

---

## 4. Knowledge Distillation

### What it is

**What it is:** Train a small "student" model to imitate the behaviour of a large "teacher" model. The student learns not just from correct answers (hard labels) but from the teacher's full probability distribution over answers (soft labels), which contains much richer information.

**Analogy:** Distillation is like mentoring. A senior expert (teacher) does not just tell a junior (student) "the answer is X" — they show their reasoning, their confidence levels, and the alternatives they considered. The junior learns much faster from this richer signal.

Train a small **student** model to mimic the behavior of a large **teacher** model.

```
Teacher (large): 7B params, high quality output
     ↓ (provides soft labels / logits — full probability distributions)
Student (small): 1B params, faster inference, nearly as good
```

### How it Works

**What it is:** The key difference between standard training and distillation is what the student learns from.

```
Standard training:   student learns from hard labels (true answers — one-hot vectors)
Distillation:        student also learns from teacher's soft predictions (full probability vectors)
```

The teacher's output probabilities carry more information than one-hot labels:
```
One-hot:    [0, 0, 1, 0, 0]                  # only says "cat" — no other information
Teacher:    [0.02, 0.01, 0.85, 0.08, 0.04]  # says mostly "cat" but also a bit "kitten" — much richer signal
```

**WHY soft labels are more informative:** The teacher's probability distribution encodes relationships between classes. "Cat" and "kitten" have similar probabilities — the student learns they are related. This knowledge transfer is impossible with one-hot labels where everything else is simply "wrong."

### Distillation Loss

**What it is:** The training objective that combines two signals — learning from real labels (accuracy) and learning from the teacher (knowledge transfer).

```
L = α * L_CE(student, true_labels) + (1-α) * L_KD(student, teacher)
```
Where L_KD uses softened probabilities (temperature > 1 for softer distribution that shows more of the teacher's "dark knowledge").

**WHY temperature in distillation:** The teacher's top token might have probability 0.95 and everything else near 0. This is almost like a hard label. Dividing by temperature > 1 flattens the distribution, making the relative relationships between tokens more visible for the student to learn from.

### Famous Distilled Models

**What it is:** Well-known real-world examples of distillation in production.

| Student | Teacher | Notes |
|---------|---------|-------|
| DistilBERT | BERT | 40% smaller, 60% faster, 97% quality |
| TinyBERT | BERT | Further compressed BERT |
| GPT-4o mini | GPT-4 | OpenAI's distilled model |
| Phi-3 mini | GPT-4 | Very small but capable (Microsoft) |

### Sequence-Level Distillation (for LLMs)

**What it is:** For large LLMs it is impractical to transfer token-level probabilities (vocabulary size × sequence length × data size). Instead, generate complete responses from the teacher and train the student on those.

1. Generate responses from teacher on training prompts (teacher produces training data)
2. Fine-tune student on teacher-generated responses (student learns to produce teacher-quality output)

Used to distill capabilities like instruction-following or step-by-step reasoning.

**WHY this works at scale:** Token-level distillation requires the teacher to output logits for every token, which is enormous. Sequence-level distillation only requires teacher *text* output — much cheaper and easier to store.

---

## 5. ONNX & Inference Optimization

### ONNX (Open Neural Network Exchange)

**What it is:** A standardized format for storing neural network models that is not tied to any specific framework (not PyTorch, not TensorFlow). Once in ONNX format, a model can be run by optimized inference engines on any hardware.

**Analogy:** ONNX is like a PDF file for neural networks. Just as a PDF can be opened by any PDF reader on any operating system, an ONNX model can be run by any ONNX-compatible inference engine.

Export model to a framework-agnostic format for optimized inference.

```python
# Export PyTorch model to ONNX format
torch.onnx.export(
    model,                          # the PyTorch model to export
    sample_input,                   # a sample input — needed to trace the computation graph
    "model.onnx",                   # output file path
    opset_version=14,               # ONNX opset version — use 14+ for modern ops
    dynamic_axes={"input": {0: "batch", 1: "seq_len"}}  # allow variable batch and sequence length
)
```

**WHY dynamic axes:** If you export with fixed sizes, the model can only run on that exact input shape. Dynamic axes let the model handle different batch sizes and sequence lengths, which is essential for a real server.

### ONNX Runtime

**What it is:** Microsoft's high-performance inference engine for ONNX models. It applies hardware-specific optimizations automatically — graph fusions, kernel selection, layout optimization.

Run ONNX models with hardware-specific optimizations.
```python
import onnxruntime as ort                             # ONNX Runtime library
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])  # run on GPU
output = session.run(None, {"input": input_array})   # run inference, return all outputs
```

**WHY ONNX Runtime over plain PyTorch for inference:** ONNX Runtime fuses multiple operations into single GPU kernels, eliminates intermediate buffers, and uses hardware-specific instruction sets. Often 2–3× faster than PyTorch for inference.

### TensorRT (NVIDIA)

**What it is:** NVIDIA's inference optimizer specifically for their GPUs. Takes an ONNX model and compiles it to run as fast as possible on a specific NVIDIA GPU, including INT8 calibration and layer fusion.

NVIDIA's inference optimizer for CUDA GPUs.
- Graph fusion: multiple operations merged into one kernel (reduces memory bandwidth)
- Kernel optimization: picks the fastest GPU kernel for each operation
- FP16/INT8 quantization: can quantize the model at compile time
- 2-10× speedup over plain PyTorch on NVIDIA GPUs

---

## 6. Model Compression Comparison

**What it is:** A practical comparison of all the techniques so you can choose the right one for your situation.

| Technique | Memory | Speed | Quality | Ease |
|-----------|--------|-------|---------|------|
| FP16 quantization | 2× reduction | 2× faster | Near-lossless | Easy |
| INT8 quantization | 4× reduction | 2-3× faster | Minor degradation | Moderate |
| INT4 quantization | 8× reduction | 3-4× faster | Some degradation | Moderate |
| Pruning (unstructured) | Variable | Minimal | Variable | Hard |
| Pruning (structured) | Variable | Real speedup | Variable | Hard |
| Distillation | 3-10× smaller | 3-10× faster | Moderate | Hard |

**WHY the ease column matters:** If you need to deploy tomorrow, INT8 quantization with bitsandbytes takes 10 minutes. Distillation takes weeks of training. Always match technique to your timeline and resources.

---

## 7. Inference Optimization Techniques

### Speculative Decoding

**What it is:** A clever technique using a fast small model to "guess" multiple future tokens ahead, then using the large model to verify all guesses at once. If the large model agrees, you get multiple tokens for the price of one large model forward pass.

**Analogy:** Speculative decoding is like a junior analyst drafting a report and a senior expert reviewing it. The junior is fast but imperfect. The senior reviews the draft in one sitting rather than writing from scratch. Net result: much faster output while maintaining senior quality.

Use a small draft model to speculatively generate multiple tokens, then verify with the large model in parallel.
- Same quality as original model (mathematically proven — rejected tokens are resampled correctly)
- 2-3× inference speedup
- Used in production at Google, Meta

```
Draft model generates: [" The", " cat", " sat"]         ← small model proposes 3 tokens
Large model verifies all 3 in parallel                  ← one forward pass checks all 3
Accept: [" The", " cat"] → reject " sat" → resample    ← take correct ones, resample bad one
```

**WHY same quality:** The rejection sampling step guarantees mathematically identical output distribution to running the large model alone. If draft tokens don't meet the large model's standards, they are rejected and resampled from the large model — quality is preserved.

### Continuous Batching

**What it is:** Standard batching waits for ALL sequences in a batch to finish before starting new ones — this wastes GPU time while short sequences are done and longer ones finish. Continuous batching fills slots immediately when any sequence finishes.

**Analogy:** Continuous batching is like a restaurant that seats new customers the moment a table opens, rather than waiting for all tables in a section to clear before seating anyone new. Much better utilization.

Instead of waiting for all requests in a batch to finish, add new requests as slots open up.
- Dramatically improves GPU utilization (GPU is always busy)
- Used in vLLM
- Essential for serving mixed-length requests (some users ask short questions, some long)

### PagedAttention (vLLM)

**What it is:** A memory management technique for the KV (key-value) cache that the attention mechanism needs. Instead of pre-allocating a large contiguous block of memory for each sequence (wasteful), PagedAttention stores the cache in non-contiguous fixed-size "pages" — like how an operating system manages RAM.

**Analogy:** PagedAttention works like a library's book storage system. Instead of reserving an entire aisle for each patron (wasteful when most aisles are empty), you give each patron shelf space in small blocks wherever there is room — much more efficient use of total storage.

Inspired by virtual memory in OS. Manages KV cache memory in non-contiguous pages.
- Eliminates memory fragmentation (no wasted reserved-but-unused space)
- Enables more concurrent requests on the same GPU
- Foundation of vLLM's performance advantage

### Flash Attention

**What it is:** An algorithmically rewritten attention operation that avoids writing the large attention matrix to GPU memory. Instead it computes attention in small blocks that fit in fast on-chip SRAM, drastically reducing slow high-bandwidth memory reads/writes.

(Covered in depth in the transformers section) — reduces memory I/O, enables longer contexts.

---

## Why LLMs Are Hard to Quantize — Activation Outliers

**What it is:** A critical technical problem unique to LLMs that does not exist for CNNs. Understanding this is key to explaining why naive INT8 fails and why GPTQ/AWQ are needed.

Naive INT8 quantization: scale all values to fit in [-128, 127]

**The problem:** LLMs have "outlier" activations — a small number of values are 100× larger than the average. When you scale to fit the outlier into the INT8 range, all the smaller values collapse to zero — information is destroyed.

```
Example activation tensor: [0.1, 0.2, 0.1, 0.3, 150.0, 0.2, ...]
If you scale to fit 150.0 → 127, then 0.1 → 0 (rounds to zero — information completely lost)
Most values collapse to 0 → model accuracy destroyed
```

**WHY this does not happen with CNNs:** CNN activations follow a relatively smooth distribution without extreme outliers. LLM activations, especially in deeper layers, develop these extreme outliers as an emergent property of the attention mechanism.

This is why naive INT8 quantization fails for LLMs (but works fine for CNNs).

**SmoothQuant solution:**
- Key insight: activations are hard to quantize (have outliers), weights are easy (smooth distribution)
- Mathematically MIGRATE the quantization difficulty from activations to weights
- `Y = (X × diag(s)^-1) × (diag(s) × W)` where s is a per-channel smoothing factor
- After migration: activations are smoother (easier to quantize), weights slightly harder (still fine)
- Result: W8A8 quantization (both weights AND activations in INT8) with minimal accuracy loss

**AWQ (Activation-aware Weight Quantization):**
- Protects the 1% of weights that correspond to salient (high-outlier) activation channels
- Those critical weights kept in higher precision, rest quantized aggressively to 4-bit
- Result: better accuracy than GPTQ at the same compression ratio

**Interview: "Why is quantizing LLMs harder than CNNs?"** → "LLMs have activation outliers — a small number of values are 100x larger than the rest. Naive quantization scales to fit outliers, causing most values to collapse to zero. SmoothQuant migrates this difficulty to weights. AWQ protects weights corresponding to salient activations."

---

## FP8 — H100's Native Precision

**What it is:** A new 8-bit floating point format (not integer — floating point) that is natively accelerated on NVIDIA H100 GPUs. It gives half the memory of BF16 with most of the numerical stability of floating point.

H100 GPUs support FP8 natively (hardware-accelerated). Doubles throughput vs BF16.

FP8 formats:
- E4M3: 4-bit exponent, 3-bit mantissa — better for weights (needs wider range, less precision)
- E5M2: 5-bit exponent, 2-bit mantissa — better for gradients (needs even wider range during backprop)

FP8 training (Transformer Engine):
```python
import transformer_engine.pytorch as te  # NVIDIA's Transformer Engine library
# Replace nn.Linear with TE version — handles FP8 automatically:
self.linear = te.Linear(in_features, out_features, bias=True)  # drop-in replacement for nn.Linear
# Automatically handles FP8 cast + scaling on H100
```

FP8 inference with vLLM:
```python
from vllm import LLM                                    # vLLM inference library
llm = LLM(model="meta-llama/Llama-3.1-70B", quantization="fp8")  # load 70B model in FP8
# 2x throughput vs BF16, minimal accuracy loss
```

Memory: 70B model
```
BF16: 140GB (2 bytes × 70B params)
FP8:   70GB (1 byte × 70B params) — fits on single H100 SXM (80GB)!
```

**WHY FP8 over INT8:** FP8 is floating point (not integer), so it handles the dynamic range of neural network weights more naturally. INT8 can have larger accuracy drops on sensitive layers.

**Interview: "What is FP8 and when do you use it?"** → "FP8 is an 8-bit floating point format natively supported on H100 GPUs. Uses E4M3 for weights and E5M2 for gradients. Provides 2x throughput vs BF16 with minimal accuracy loss. A 70B model in FP8 fits on a single H100 80GB — in BF16 it needs 2 GPUs."

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

## 8. Hardware Acceleration

### GPU Types for LLM Work

**What it is:** Different GPU models have different amounts of memory (VRAM) and different compute capabilities. Choosing the right GPU depends on what size model you are running and whether you are training or just doing inference.

| GPU | VRAM | Good For |
|-----|------|---------|
| NVIDIA A100 80GB | 80GB | Large model training |
| NVIDIA A100 40GB | 40GB | Training/inference |
| NVIDIA H100 | 80GB | Fastest, latest — best for production |
| NVIDIA RTX 4090 | 24GB | Fine-tuning consumer |
| NVIDIA RTX 3090 | 24GB | Fine-tuning consumer |
| Google TPU v4 | HBM | T5, PaLM training |

### Multi-GPU Strategies

(Covered in detail in 11_distributed_training.md)
- Data Parallelism: same model on each GPU, different data
- Tensor Parallelism: split weight matrices across GPUs
- Pipeline Parallelism: split layers across GPUs

### CPU Inference (llama.cpp)

**What it is:** Run quantized models on CPU instead of GPU using the llama.cpp library. Much slower than GPU but accessible to anyone with a laptop — useful for local development, privacy-sensitive applications, or edge deployment.

For running quantized models on CPU:
```bash
./llama-cli -m llama-3.1-8b.Q4_K_M.gguf -p "Tell me about AI" -n 200
# -m: model file in GGUF format   -p: prompt text   -n: max tokens to generate
```
GGUF format, Q4 quantization → 7B model on 8GB RAM.

**WHY CPU inference is still relevant:** Not everyone has a GPU. Privacy-critical applications cannot send data to cloud APIs. Edge devices (phones, embedded systems) have no GPU. llama.cpp makes LLMs accessible everywhere.

---

## Quick Reference Cheat Sheet

```
Quantization:    FP32→FP16→INT8→INT4, reduces memory & speeds inference
GPTQ/AWQ:        Best PTQ methods for 4-bit LLMs (AWQ is usually better quality)
bitsandbytes:    INT8/NF4 quantization for HuggingFace + QLoRA
Distillation:    Large teacher → small student via soft labels
Pruning:         Remove weights/heads/layers (structured = GPU-friendly)
Speculative:     Draft model + large model = 2-3× faster inference
vLLM:            PagedAttention + continuous batching for production
FlashAttention:  Efficient attention computation, memory-efficient
FP8:             H100 native format — 2× throughput vs BF16
Activation outliers: Why LLMs are hard to quantize — SmoothQuant/AWQ solve this
```
