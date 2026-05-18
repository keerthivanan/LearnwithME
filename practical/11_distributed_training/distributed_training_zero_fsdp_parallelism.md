# 10 — Distributed Training & Parallelism

> JD explicitly asks for "distributed computing and parallelism for large-scale training." Know all four strategies.

---

## 1. Why Distributed Training?

**What it is:** Modern large language models are simply too big to train on a single GPU. Distributed training splits the work across many GPUs so training becomes possible.

A 70B model in FP16:
- **140GB VRAM** just for weights
- Training needs: weights + gradients + optimizer states = ~560GB total
- Single A100 = 80GB → **not enough for even 7B model training** (without tricks)

**Analogy:** Distributing training is like building a skyscraper with a construction crew instead of one person. One person cannot lift the steel beams — you need dozens of workers with specialized roles and tools working in coordination.

Solution: split the model and/or data across multiple GPUs.

---

## 2. The Four Parallelism Strategies

**What it is:** Four different ways to split the training work across GPUs. Each solves a different bottleneck.

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  Data Parallel   │ Tensor Parallel  │Pipeline Parallel │ Sequence Parallel│
│  Same model,     │  Split layers    │  Split layers    │  Split sequence  │
│  different data  │  horizontally    │  vertically      │  along length    │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Analogy for each:**
- **Data Parallel** = same assembly line in 4 factories, different raw materials → combine results
- **Tensor Parallel** = one assembly line split across 4 workers side by side at each station
- **Pipeline Parallel** = assembly line with 4 stations in sequence, one after another
- **Sequence Parallel** = one long document split into 4 sections, each processed separately

---

## 3. Data Parallelism (DP / DDP)

### What it is

**What it is:** The simplest approach — every GPU has a complete copy of the entire model, but each GPU processes a different portion of the training data. After each step, the gradients from all GPUs are averaged and the model is updated identically on all GPUs.

Each GPU has a **complete copy** of the model but processes a different subset of data.

```
GPU 0: model_copy + batch_0 → gradient_0   ← GPU 0 computes gradient from its data batch
GPU 1: model_copy + batch_1 → gradient_1   ← GPU 1 computes gradient from its data batch
GPU 2: model_copy + batch_2 → gradient_2   ← GPU 2 computes gradient from its data batch
         ↓ AllReduce (average gradients)    ← all GPUs share and average their gradients
All GPUs update weights with averaged gradient  ← every GPU now has the same updated model
```

### PyTorch DDP (DistributedDataParallel)

**What it is:** PyTorch's built-in distributed training wrapper. Wrap your model with DDP and it automatically handles gradient synchronization across GPUs.

```python
import torch.distributed as dist                              # distributed computing utilities
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP wrapper

dist.init_process_group(backend="nccl")  # initialize distributed training — NCCL is NVIDIA's GPU communication library
model = model.to(local_rank)             # move model to this process's GPU (local_rank = which GPU this process uses)
model = DDP(model, device_ids=[local_rank])  # wrap model with DDP — gradients now sync automatically

# DistributedSampler ensures each GPU gets different non-overlapping data
sampler = DistributedSampler(dataset)         # splits dataset across all GPU processes
dataloader = DataLoader(dataset, sampler=sampler)  # DataLoader uses the sampler to get different data per GPU
```

### Launch Command

```bash
torchrun --nproc_per_node=4 train.py          # launch 4 processes (one per GPU) on this machine
# or
python -m torch.distributed.launch --nproc_per_node=4 train.py  # older launch method
```

**WHY torchrun over the old launch method:** `torchrun` handles elastic training (adding/removing nodes), automatic fault recovery, and is the modern standard.

### Advantages
- Simple to implement (just wrap model with DDP)
- Linear scaling (2× GPUs ≈ 2× throughput)

### Disadvantage
- Each GPU must hold the full model
- **Memory limit**: can only scale batch size, not model size — 70B model still does not fit on one GPU

---

## 4. Tensor Parallelism (TP)

### What it is

**What it is:** Split individual weight matrices across multiple GPUs. Each GPU holds only a "slice" of each weight matrix and computes its slice of the output. Results are combined with an AllGather communication.

**Analogy:** Tensor Parallelism is like four people each computing one column of a spreadsheet formula simultaneously — each person does less work and they share results at the end.

Split individual layers (weight matrices) across multiple GPUs.
Each GPU holds a **slice** of each weight matrix.

### Example: Linear Layer Split

```
W (d × 4d) split across 4 GPUs:              ← weight matrix dimensions: d rows by 4d columns
GPU 0: W[:, 0:d]                              ← GPU 0 holds the first d columns
GPU 1: W[:, d:2d]                             ← GPU 1 holds the next d columns
GPU 2: W[:, 2d:3d]                            ← GPU 2 holds the next d columns
GPU 3: W[:, 3d:4d]                            ← GPU 3 holds the last d columns

Full result = AllGather(GPU_results)           ← combine each GPU's partial result into full output
```

### Megatron-LM Style (Column + Row Parallel)

**What it is:** Megatron-LM (NVIDIA's LLM training library) defines a specific way to split linear layers that minimizes the number of communication operations needed.

```
Column Parallel Linear: Split columns of W across GPUs → each GPU computes part of output independently
Row Parallel Linear:    Split rows of W across GPUs → partial outputs must be AllReduced
→ Arranged to minimize communication (column parallel → row parallel = single AllReduce per transformer block)
```

### When to Use
- When model layers are too large to fit on one GPU
- Usually combined with other strategies (3D Parallelism)

### Communication Overhead

**Why NVLink matters here:** Tensor parallelism requires AllReduce (share all intermediate results) after every single layer. If this communication is slow, GPUs spend most of their time waiting. NVLink (NVIDIA's fast GPU interconnect) runs at 600 GB/s vs PCIe at ~32 GB/s — tensor parallelism essentially requires NVLink to be efficient.

Requires AllReduce after every layer — needs fast interconnect (NVLink). Slow interconnect makes this too expensive.

---

## 5. Pipeline Parallelism (PP)

### What it is

**What it is:** Divide the model's layers into groups ("stages") and assign each stage to a different GPU. Data flows through the GPUs sequentially — GPU 0 processes the first layers and passes its output to GPU 1, and so on.

**Analogy:** Pipeline parallelism is like an assembly line in a factory. Station 1 does the first step on a car, passes it to Station 2 for the next step, etc. Each station specializes in one part of the process.

Split model layers into **stages**, each stage on a different GPU.

```
GPU 0: Layers 1-8      ← this GPU handles the first 8 transformer layers
GPU 1: Layers 9-16     ← this GPU handles the next 8 transformer layers
GPU 2: Layers 17-24    ← this GPU handles the next 8 transformer layers
GPU 3: Layers 25-32    ← this GPU handles the final 8 transformer layers
```

Data flows: GPU0 → GPU1 → GPU2 → GPU3 (forward pass), reverse for backward pass.

### The Pipeline Bubble Problem

**What it is:** The biggest drawback of pipeline parallelism — GPUs at the start and end of the pipeline are idle for significant time while waiting for the pipeline to fill or drain.

**Analogy:** In a car assembly line, Station 1 finishes the first car quickly, but then has to wait idle while that car slowly makes its way through all other stations before Station 1 can start the next car. The idle time is the "bubble."

GPUs idle while waiting for the previous GPU's output:
```
GPU 0: [F1][F2][F3][F4][B4][B3][B2][B1]     ← F = forward micro-batch, B = backward micro-batch
GPU 1:     [F1][F2][F3][F4][B4][B3][B2][B1]  ← GPU 1 starts after GPU 0, creating a gap
               ↑ bubble ↑                     ← this gap is wasted GPU time
```

### GPipe — Micro-batching

**What it is:** Instead of processing one large batch, split each batch into many small "micro-batches." While GPU 0 is processing micro-batch 2, GPU 1 is processing micro-batch 1 — the pipeline fills faster and the bubble is smaller.

Split each batch into micro-batches to reduce the bubble:
```
GPU 0: [F1][F2][F3][F4]         [B4][B3][B2][B1]  ← processes all 4 micro-batches, then backward
GPU 1:     [F1][F2][F3][F4] [B4][B3][B2][B1]       ← starts 1 step behind GPU 0
               pipeline fill ↑ ↑ pipeline drain     ← more micro-batches = smaller bubble fraction
```

**WHY more micro-batches helps:** The bubble is always the same absolute size (P-1 steps where P is pipeline depth). But with more micro-batches (M), the bubble as a fraction of total work becomes (P-1)/(M+P-1) — larger M means the bubble is a smaller fraction.

### 1F1B (One Forward One Backward — PipeDream)

**What it is:** An advanced scheduling strategy that interleaves forward passes (computing outputs) and backward passes (computing gradients) to keep GPUs busy as much as possible.

Interleave forward and backward passes to reduce idle time — keeps all GPUs busier than GPipe.

---

## 6. ZeRO — Zero Redundancy Optimizer (DeepSpeed)

### The Problem

**What it is:** In standard data parallelism (DDP), every single GPU stores complete copies of the model weights, gradients, AND optimizer states. This is massively redundant — the same 18 bytes/parameter are duplicated on every single GPU.

DDP: every GPU stores full model + gradients + optimizer states.

For Adam optimizer:
- FP16 model weights: 2 bytes/param
- FP32 master weights: 4 bytes/param (kept for numerical stability)
- Gradients: 4 bytes/param
- Adam optimizer states (momentum m and variance v): 8 bytes/param
- **Total: ~18 bytes/param** → 7B model = 126GB per GPU!

**Analogy:** Imagine 8 employees all keeping their own complete copy of the entire company's filing system just so they can each update one folder. ZeRO is like saying "each person only keeps their assigned folders — everyone else gets what they need on request."

### ZeRO Solution

**What it is:** Partition the redundant memory across GPUs. Instead of every GPU having everything, each GPU has only its fair share. When a GPU needs something another GPU holds, it communicates to get it.

Partition redundant memory across GPUs.

**ZeRO-1**: Partition optimizer states only
- Each GPU stores optimizer states (Adam m, v) for 1/N of the parameters
- Reduces optimizer state memory by N× (with 8 GPUs, only 1/8 the optimizer memory per GPU)

**ZeRO-2**: Partition optimizer states + gradients
- Each GPU only stores gradients for its partition of parameters
- After backward pass, AllReduce and only keep needed gradients
- Further reduces memory per GPU

**ZeRO-3**: Partition optimizer states + gradients + model parameters
- Model weights themselves are split across GPUs (most aggressive)
- Requires AllGather to reconstruct weights for forward pass
- Requires ReduceScatter to accumulate gradients during backward pass
- Each GPU stores only 1/N of everything — truly minimal memory

```
         | Memory per GPU | Communication |
ZeRO-0   | O(M)          | AllReduce only                          ← standard DDP (no ZeRO)
ZeRO-1   | O(M/3)        | AllReduce + AllGather for optimizer     ← shard optimizer states
ZeRO-2   | O(M/3)        | + ReduceScatter for gradients           ← also shard gradients
ZeRO-3   | O(M/N)        | + AllGather/ReduceScatter for params    ← shard everything
```

**WHY the trade-off:** More aggressive sharding (ZeRO-3) means less memory per GPU but more communication. On fast NVLink interconnects the communication overhead is small; on slow interconnects it can hurt throughput.

### DeepSpeed ZeRO in Practice

**What it is:** DeepSpeed is Microsoft's library that implements ZeRO. You configure it via a JSON file and launch with `deepspeed` instead of `torchrun`.

```python
# deepspeed_config.json — this file configures DeepSpeed behaviour
{
  "zero_optimization": {
    "stage": 3,                           # ZeRO-3: shard everything
    "overlap_comm": true,                 # overlap communication with compute (hides latency)
    "contiguous_gradients": true,         # contiguous memory for gradients (reduces fragmentation)
    "sub_group_size": 1e9,               # process params in sub-groups of this size
    "stage3_max_live_parameters": 1e9,    # max parameters to keep in GPU memory at once
    "stage3_prefetch_bucket_size": 5e8   # prefetch next parameter bucket while computing current
  },
  "bf16": {"enabled": true},            # use BF16 precision for stability
  "gradient_accumulation_steps": 4       # accumulate gradients over 4 steps before updating
}
```

```bash
deepspeed --num_gpus=8 train.py --deepspeed deepspeed_config.json  # launch with 8 GPUs
```

### ZeRO-Infinity

**What it is:** An extension of ZeRO-3 that goes beyond GPU memory and also uses CPU RAM and NVMe SSD storage for parameters. Enables training models far larger than all GPUs combined.

Extend ZeRO-3 to use CPU RAM and NVMe storage for even larger models.
- Train trillion-parameter models on a reasonable number of GPUs
- Much slower due to CPU/NVMe I/O (PCIe bandwidth is much slower than NVLink)
- Used for research on massive models, not latency-sensitive production training

---

## 7. 3D Parallelism

**What it is:** Combine all three main parallelism strategies simultaneously for maximum scale. This is how the largest models (GPT-3, Megatron-Turing NLG 530B) were trained.

**Analogy:** 3D Parallelism is like running a factory with multiple assembly lines (Pipeline Parallel), each station on each line has multiple workers side-by-side (Tensor Parallel), and there are multiple identical factory buildings (Data Parallel). Everything happening at once.

Combine all three for maximum scale:
```
Pipeline Parallel (stages) × Tensor Parallel (within stage) × Data Parallel (replicas)
```

Used for training 100B+ parameter models (GPT-3, Megatron-Turing NLG 530B).

**WHY you need all three:** Pipeline Parallelism handles the model being too tall (too many layers). Tensor Parallelism handles individual layers being too wide (too many hidden dimensions). Data Parallelism handles making good use of all available GPUs for throughput.

---

## 8. FSDP — Fully Sharded Data Parallel (PyTorch)

**What it is:** PyTorch's native built-in equivalent of DeepSpeed ZeRO-3. Shards model parameters, gradients, and optimizer states across GPUs. No external library needed — just PyTorch.

PyTorch's native equivalent of ZeRO-3.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # FSDP wrapper
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch   # configuration options
import torch.distributed as dist                                       # distributed utilities

dist.init_process_group("nccl")  # initialize the process group (NCCL for NVIDIA GPUs)

# Define precision policy — what dtype to use for different parts
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,    # store parameters in BF16 (memory efficient and stable)
    reduce_dtype=torch.float32,    # do gradient reduction in FP32 (more accurate gradient averaging)
    buffer_dtype=torch.bfloat16,   # store buffers in BF16
)

# Wrap model with FSDP
model = FSDP(
    model,
    mixed_precision=mp_policy,                              # apply our precision policy
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,        # prefetch next shard while computing current
    device_id=torch.cuda.current_device()                   # which GPU this process should use
)
```

### FSDP vs DeepSpeed ZeRO

**What it is:** A practical comparison to help you choose which to use.

| Feature | FSDP | DeepSpeed ZeRO |
|---------|------|---------------|
| Integration | Native PyTorch (no extra deps) | External library (install DeepSpeed) |
| Ease | Simpler (fewer configs) | More config options |
| Performance | Similar to ZeRO-3 | ZeRO-3 slightly better in some cases |
| Flexibility | Less features | More features (ZeRO-Infinity, etc.) |

**WHY choose FSDP:** Simpler setup, no extra dependency. **Why choose DeepSpeed:** More advanced features (ZeRO-Infinity, offloading, sparse attention), better support for very large scale.

---

## 9. Gradient Checkpointing

### The Problem

**What it is:** During the forward pass, PyTorch needs to save all intermediate activations (the outputs of each layer) for use in the backward pass. For large models with many layers, this uses enormous GPU memory.

Storing all intermediate activations for backpropagation is memory-intensive.
For N layers: O(N) activation memory — scales linearly with model depth.

**Analogy:** Gradient checkpointing is like doing a road trip where instead of recording video the whole way (expensive memory), you take photos at key checkpoints and only re-trace sections between checkpoints when you need to.

### Solution

**What it is:** During the forward pass, only save activations at certain "checkpoints." When the backward pass needs an intermediate activation, recompute it on the fly from the nearest checkpoint.

During forward pass, only save activations at **checkpoints**.
Recompute non-saved activations during backward pass (recompute from nearest saved checkpoint).

Trade: More compute (~33% extra compute to recompute activations) for less memory (~√N instead of O(N)).

```python
from torch.utils.checkpoint import checkpoint  # gradient checkpointing function

# Without checkpointing — all activations saved in memory
output = model(input)

# With gradient checkpointing — saves memory by recomputing activations when needed
output = checkpoint(model, input)  # saves only at start/end, recomputes intermediate activations during backward
```

```python
# Hugging Face shortcut — enable gradient checkpointing for the whole model
model.gradient_checkpointing_enable()  # one line to enable for HF models
```

**WHY 33% more compute:** During the backward pass, you need to recompute some activations. This is like running part of the forward pass twice. The memory saving (from O(N) to O(√N)) is usually worth this compute cost.

---

## 10. Gradient Accumulation

**What it is:** A way to train with a large effective batch size (needed for stable training of large models) even when your GPU memory cannot fit a large batch. You run multiple small batches, accumulate the gradients, and only update the model after a certain number of small batches.

**Analogy:** Gradient accumulation is like saving up pocket money from multiple days before making one purchase. You can't buy the item in one day, but by accumulating over time you can still make the purchase.

Simulate large batch sizes without using more GPU memory:

```python
accumulation_steps = 8  # simulate 8× larger batch size

for i, batch in enumerate(dataloader):       # iterate over all training batches
    outputs = model(**batch)                 # forward pass — compute predictions
    loss = outputs.loss / accumulation_steps  # normalize loss (so gradient scale stays consistent)
    loss.backward()                          # compute gradients — but DON'T update weights yet

    if (i + 1) % accumulation_steps == 0:   # only update every 8 steps
        optimizer.step()                     # NOW update model weights with accumulated gradient
        optimizer.zero_grad()                # clear gradients for the next accumulation cycle
```

Effective batch size = `per_device_batch_size × num_gpus × accumulation_steps`

**WHY normalized loss:** If you don't divide by accumulation_steps, the gradient you accumulate is 8× larger than a single batch's gradient. The division makes the accumulated gradient equivalent to what you'd get from one large batch.

---

## 11. Communication Primitives

**What it is:** The low-level GPU communication operations that all distributed training strategies are built on. You need to know what each does for interviews.

| Operation | Description | Use In |
|-----------|------------|--------|
| AllReduce | Sum or average a tensor across all GPUs | DDP gradient sync |
| AllGather | Gather different tensor parts from all GPUs into one complete tensor | ZeRO-3 weight gather |
| ReduceScatter | Reduce values then scatter different parts to each GPU | ZeRO gradient accumulation |
| Broadcast | Send from one GPU to all others | Initial weight sync at start of training |
| P2P | Point-to-point communication between two GPUs | Pipeline parallelism (pass activations to next stage) |

**WHY know these:** Interview questions often ask "how do gradients sync in DDP?" (AllReduce) or "what communication does ZeRO-3 use?" (AllGather + ReduceScatter). These are the atomic building blocks.

---

## Sequence Parallelism — Parallelizing the Sequence Dimension

**What it is:** A fourth type of parallelism that addresses a unique challenge of long-context models — at 128K context length, even a single attention layer's activations are so large they need to be split across GPUs.

**Analogy:** If Tensor Parallelism is "split the weight matrix", Sequence Parallelism is "split the document being processed." Imagine four people each reading a different chapter of a long book simultaneously, then comparing notes for the parts where they need to reference each other's chapters.

Tensor Parallelism splits model weights across GPUs.
Sequence Parallelism splits the INPUT SEQUENCE across GPUs.

Why needed: At 128K context length, even a single attention layer's activations are huge.
```
128K tokens × 8192 hidden_dim × 4 bytes = 4GB just for ONE layer's activations
With gradient checkpointing you still need to store some activations — this is unavoidable
```

How it works (Megatron-LM Ring Attention / Sequence Parallelism):
- Split sequence into chunks: GPU0 gets tokens 0-32K, GPU1 gets 32K-64K, etc.
- For attention: each GPU computes attention for its chunk
- To handle cross-chunk attention: GPUs exchange K,V in a ring pattern (Ring Attention)
- Each GPU sends its K,V to next GPU, receives from previous GPU, computes attention

Ring Attention complexity: O(n×d/p) per GPU where p = number of GPUs
- Linear scaling with number of GPUs (double GPUs = half memory per GPU)
- Used in training LLaMA-3 with 128K context

Combined: 3D Parallelism + Sequence Parallelism = 4D Parallelism
```
(Tensor Parallel × Pipeline Parallel × Data Parallel × Sequence Parallel)
```

**Interview: "How do you train models with very long context (128K+)?"** → "Sequence parallelism — split the sequence dimension across GPUs. Ring Attention pattern lets each GPU compute attention for its sequence chunk while exchanging KV with neighbors in a ring. Used by Meta for LLaMA-3's 128K context training."

---

## Fault Tolerance — Handling GPU Failures in Long Training Runs

**What it is:** When you train for weeks on thousands of GPUs, hardware failures are inevitable. You need strategies to lose as little work as possible and resume automatically.

**Analogy:** Fault tolerance in distributed training is like having an autosave feature in a video game. The game saves your progress every few minutes. If the power goes out, you lose at most a few minutes of progress, not the whole run.

LLaMA-3 training ran for months on thousands of GPUs. GPU failures are inevitable.

Checkpointing strategy:
- Save checkpoint every N steps (N=500 is common)
- On failure: restart from last checkpoint, lose at most N×batch_time of work
- ZeRO-3 checkpoint: each GPU saves its own shard, then aggregate

```python
trainer.save_checkpoint("checkpoint-500")  # DeepSpeed handles sharded checkpointing automatically
```

Elastic training (torch.distributed.elastic):
- Handles node failures without stopping training entirely
```bash
torchrun --nnodes=8:10 script.py  # run with 8-10 nodes, tolerate up to 2 failures and continue
# If a node fails: remaining nodes rebalance and continue (re-shard data, adjust batch size)
```

Spot/Preemptible instances:
- AWS Spot instances: up to 90% cheaper than on-demand, but can be interrupted with 2-min warning
- Strategy: checkpoint every 100-200 steps + use elastic training for automatic recovery
- Save checkpoints to S3/GCS (not local disk) so checkpoint survives instance termination

MTIA / hardware failures:
- Nightly health checks on all GPUs before training starts (saves discovering failures mid-run)
- `torch.cuda.is_available()` and memory test before job
- Automatic node replacement in Kubernetes GPU clusters

**Interview: "How do you handle failures in a multi-day training run?"** → "Frequent checkpointing (every 500 steps) to durable storage like S3. Elastic training with torchrun to handle node failures without stopping. Spot instances work well with this pattern since 2-minute preemption warnings give enough time to save a checkpoint."

---

## 12. Interview Questions — Distributed Training

**Q: What is the difference between data parallelism and model parallelism?**
> Data parallelism replicates the full model across GPUs and splits the data batch — each GPU trains on different data. Model parallelism splits the model itself across GPUs — necessary when the model is too large to fit on a single GPU. Pipeline and tensor parallelism are the two main model parallelism strategies.

**Q: What is ZeRO and what problem does it solve?**
> ZeRO (Zero Redundancy Optimizer, DeepSpeed) eliminates the memory redundancy in data parallelism. In standard DDP, every GPU holds full copies of the model, gradients, and optimizer states. ZeRO-3 shards all three across GPUs, reducing memory per GPU from O(M) to O(M/N) where N is the number of GPUs.

**Q: What is gradient checkpointing?**
> A technique that saves memory during training by not storing all intermediate activations. Instead, a subset of activations is saved at checkpoints, and the rest are recomputed during the backward pass when needed. It reduces memory by ~√N layers at the cost of ~33% more compute.

**Q: What is FSDP?**
> Fully Sharded Data Parallel — PyTorch's native implementation of ZeRO-3. It shards model parameters, gradients, and optimizer states across GPUs, enabling training of models much larger than a single GPU's memory.

**Q: What is the pipeline bubble problem?**
> In pipeline parallelism, GPUs at the beginning and end of the pipeline are idle while waiting for micro-batches to fill/drain the pipeline. This "bubble" wastes GPU compute. Techniques like 1F1B scheduling and interleaved stages reduce bubble fraction.

---

## Quick Reference Cheat Sheet

```
Data Parallel (DDP):      Same model × N GPUs, different data, gradient AllReduce
Tensor Parallel:          Split weight matrices across GPUs (Megatron-LM style)
Pipeline Parallel:        Split layers across GPUs, micro-batching reduces bubble
ZeRO (DeepSpeed):         Shard optimizer states/gradients/weights across GPUs
FSDP:                     PyTorch native ZeRO-3 equivalent
Gradient Checkpointing:   Recompute activations during backward to save memory
Gradient Accumulation:    Simulate large batch with small memory via multiple steps
3D Parallelism:           TP × PP × DP for 100B+ models
Sequence Parallelism:     Split sequence across GPUs for 128K+ context training
Fault Tolerance:          Checkpoint every 500 steps to S3 + elastic training
```
