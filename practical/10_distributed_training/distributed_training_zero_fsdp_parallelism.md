# 10 — Distributed Training & Parallelism

> JD explicitly asks for "distributed computing and parallelism for large-scale training." Know all four strategies.

---

## 1. Why Distributed Training?

A 70B model in FP16:
- **140GB VRAM** just for weights
- Training needs: weights + gradients + optimizer states = ~560GB
- Single A100 = 80GB → **not enough for even 7B model training**

Solution: split the model and/or data across multiple GPUs.

---

## 2. The Four Parallelism Strategies

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  Data Parallel   │ Tensor Parallel  │Pipeline Parallel │ Sequence Parallel│
│  Same model,     │  Split layers    │  Split layers    │  Split sequence  │
│  different data  │  horizontally    │  vertically      │  along length    │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## 3. Data Parallelism (DP / DDP)

### What it is
Each GPU has a **complete copy** of the model but processes a different subset of data.

```
GPU 0: model_copy + batch_0 → gradient_0
GPU 1: model_copy + batch_1 → gradient_1
GPU 2: model_copy + batch_2 → gradient_2
         ↓ AllReduce (average gradients)
All GPUs update weights with averaged gradient
```

### PyTorch DDP (DistributedDataParallel)
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")  # NCCL for NVIDIA GPUs
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Sampler ensures each GPU gets different data
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### Launch Command
```bash
torchrun --nproc_per_node=4 train.py
# or
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### Advantages
- Simple to implement
- Linear scaling (2× GPUs ≈ 2× speed)

### Disadvantage
- Each GPU must hold the full model
- **Memory limit**: can only scale batch size, not model size

---

## 4. Tensor Parallelism (TP)

### What it is
Split individual layers (weight matrices) across multiple GPUs.

Each GPU holds a **slice** of each weight matrix.

### Example: Linear Layer Split
```
W (d × 4d) split across 4 GPUs:
GPU 0: W[:, 0:d]
GPU 1: W[:, d:2d]
GPU 2: W[:, 2d:3d]
GPU 3: W[:, 3d:4d]

Full result = AllGather(GPU_results)
```

### Megatron-LM Style (Column + Row Parallel)
```
Column Parallel Linear: Split columns of W across GPUs
Row Parallel Linear:    Split rows of W across GPUs
→ Arranged to minimize communication
```

### When to Use
- When model layers don't fit on one GPU
- Usually combined with other strategies

### Communication Overhead
Requires AllReduce after every layer — needs fast interconnect (NVLink).

---

## 5. Pipeline Parallelism (PP)

### What it is
Split model layers into **stages**, each stage on a different GPU.

```
GPU 0: Layers 1-8
GPU 1: Layers 9-16
GPU 2: Layers 17-24
GPU 3: Layers 25-32
```

Data flows: GPU0 → GPU1 → GPU2 → GPU3 (forward), reverse for backward.

### The Pipeline Bubble Problem
GPUs idle while waiting for the previous GPU's output:
```
GPU 0: [F1][F2][F3][F4][B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4][B4][B3][B2][B1]
               ↑ bubble ↑
```

### GPipe — Micro-batching
Split each batch into micro-batches to reduce the bubble:
```
GPU 0: [F1][F2][F3][F4]         [B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4] [B4][B3][B2][B1]
               pipeline fill ↑ ↑ pipeline drain
```

### 1F1B (One Forward One Backward — PipeDream)
Interleave forward and backward passes to reduce idle time.

---

## 6. ZeRO — Zero Redundancy Optimizer (DeepSpeed)

### The Problem
DDP: every GPU stores full model + gradients + optimizer states.

For Adam: 
- FP16 model weights: 2 bytes/param
- FP32 master weights: 4 bytes/param
- Gradients: 4 bytes/param
- Adam optimizer states (m, v): 8 bytes/param
- **Total: ~18 bytes/param** → 7B model = 126GB per GPU!

### ZeRO Solution
Partition redundant memory across GPUs.

**ZeRO-1**: Partition optimizer states
- Each GPU stores optimizer states for 1/N of the parameters
- Reduces optimizer state memory by N×

**ZeRO-2**: Partition optimizer states + gradients
- Each GPU only stores gradients for its partition
- After backward pass, AllReduce and only keep needed gradients

**ZeRO-3**: Partition optimizer states + gradients + model parameters
- Model weights split across GPUs
- Requires AllGather for forward pass, ReduceScatter for backward

```
         | Memory per GPU | Communication |
ZeRO-0   | O(M)          | AllReduce     |
ZeRO-1   | O(M/3)        | +AllGather    |
ZeRO-2   | O(M/3)        | +             |
ZeRO-3   | O(M/N)        | +             |
```

### DeepSpeed ZeRO in Practice
```python
# deepspeed_config.json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_max_live_parameters": 1e9,
    "stage3_prefetch_bucket_size": 5e8
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": 4
}
```

```bash
deepspeed --num_gpus=8 train.py --deepspeed deepspeed_config.json
```

### ZeRO-Infinity
Extend ZeRO-3 to use CPU RAM and NVMe storage for even larger models.
- Train trillion-parameter models
- Much slower due to CPU/NVMe I/O

---

## 7. 3D Parallelism

Combine all three for maximum scale:
```
Pipeline Parallel (stages) × Tensor Parallel (within stage) × Data Parallel (replicas)
```

Used for training 100B+ parameter models (GPT-3, Megatron-Turing NLG 530B).

---

## 8. FSDP — Fully Sharded Data Parallel (PyTorch)

PyTorch's native equivalent of ZeRO-3.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
import torch.distributed as dist

dist.init_process_group("nccl")

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    mixed_precision=mp_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device()
)
```

### FSDP vs DeepSpeed ZeRO
| Feature | FSDP | DeepSpeed ZeRO |
|---------|------|---------------|
| Integration | Native PyTorch | External library |
| Ease | Simple | More config |
| Performance | Similar | ZeRO-3 slightly better |
| Flexibility | Less | More features |

---

## 9. Gradient Checkpointing

### The Problem
Storing all intermediate activations for backprop is memory-intensive.
For N layers: O(N) activation memory.

### Solution
During forward pass, only save activations at **checkpoints**.
Recompute non-saved activations during backward pass.

Trade: More compute (recompute activations) for less memory (~√N).

```python
from torch.utils.checkpoint import checkpoint

# Without checkpointing
output = model(input)

# With gradient checkpointing (recomputes activations during backward)
output = checkpoint(model, input)
```

```python
# Hugging Face
model.gradient_checkpointing_enable()
```

---

## 10. Gradient Accumulation

Simulate large batch sizes without using more GPU memory:

```python
accumulation_steps = 8

for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # normalize
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = `per_device_batch_size × num_gpus × accumulation_steps`

---

## 11. Communication Primitives

| Operation | Description | Use In |
|-----------|------------|--------|
| AllReduce | Sum/average a tensor across all GPUs | DDP gradient sync |
| AllGather | Gather tensor parts from all GPUs | ZeRO-3 weight gather |
| ReduceScatter | Reduce then scatter different parts | ZeRO gradient accumulation |
| Broadcast | Send from one GPU to all others | Initial weight sync |
| P2P | Point-to-point between two GPUs | Pipeline parallelism |

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
Tensor Parallel:          Split weight matrices across GPUs (Megatron-LM)
Pipeline Parallel:        Split layers across GPUs, micro-batching
ZeRO (DeepSpeed):         Shard optimizer states/gradients/weights across GPUs
FSDP:                     PyTorch native ZeRO-3 equivalent
Gradient Checkpointing:   Recompute activations to save memory
Gradient Accumulation:    Simulate large batch with small memory
3D Parallelism:           TP × PP × DP for 100B+ models
```
