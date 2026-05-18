# 01 — Deep Learning Fundamentals

> Foundation for everything in this JD. Every LLM, Transformer, and fine-tuning technique is built on these concepts.

---

## 1. What is Deep Learning?

**What it is:** Deep Learning is a subset of Machine Learning that uses neural networks with many layers (hence "deep") to automatically learn patterns from raw data — you don't engineer features by hand.

Think of it like this: Traditional ML is like giving a detective a checklist of clues to look for. Deep Learning lets the detective figure out what clues matter entirely on its own, by reading thousands of case files.

- **Machine Learning**: You hand-craft features (e.g., "word count", "has exclamation mark"), then the model learns from those.
- **Deep Learning**: You feed raw data (e.g., raw text), and the model learns which features matter automatically.

```
Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output
# Data flows left to right through stacked layers
# Each layer transforms the data into something more abstract
```

Each layer learns increasingly abstract representations:
- **Layer 1**: basic patterns (edges in images, word co-occurrences in text)
- **Layer N**: high-level concepts (faces, sentence meaning, intent)

**WHY this matters for LLMs:** GPT-4 is just a very deep network. Every concept here — gradients, activations, loss — applies directly to how LLMs are trained.

---

## 2. Neural Network Basics

### Neuron (Perceptron)

**What it is:** A single neuron is the building block. It takes several numbers as input, multiplies each by a learned weight, adds them up, adds a bias, then squishes the result through an activation function.

Think of it like a voting machine: each input casts a weighted vote, the bias is a default lean, and the activation decides how loud the final answer is.

```
output = activation( w1*x1 + w2*x2 + ... + wn*xn + bias )
# x: the input values (e.g., pixel intensities, token embeddings)
# w: weights — learned during training, controls how much each input matters
# bias: a constant offset — lets the neuron fire even when all inputs are 0
# activation: a non-linear function applied at the end (more on this next)
```

**WHY bias matters:** Without bias, every neuron's output is forced through zero. Bias lets the model shift its decision boundary freely.

---

### Layers

**What it is:** Neurons are grouped into layers. Data flows through layers one at a time, left to right.

Here is what each layer type does, in plain English:

| Layer Type | Purpose |
|-----------|---------|
| Input Layer | Receives raw data — no computation, just passes values in |
| Hidden Layers | Do the actual learning — transform data into useful representations |
| Output Layer | Produces the final answer (e.g., probability of each class) |

---

### Fully Connected (Dense) Layer

**What it is:** Every neuron in one layer connects to every neuron in the next. No shortcuts, no skips.

Think of it like a round-table meeting: everyone talks to everyone.

Used in the classification heads of LLMs — the final layer that turns a hidden vector into vocabulary probabilities.

**WHY it's important to know:** When an interviewer says "projection layer" or "classification head", they mean a dense layer. It's everywhere.

---

## 

3. Activation Functions

**What it is:** An activation function is the non-linear "squishing" step applied after each layer's computation. Without it, stacking 100 layers is mathematically identical to having just 1 layer — they would all collapse into one big linear transformation.

Think of it like this: linearity means "double the input, double the output — always." Real problems (understanding language, recognizing faces) are not like that. Activation functions break that linearity, giving the network the power to learn curves and complex patterns.

Here are all the important ones:

| Function | Formula | Use Case |
|----------|---------|---------|
| ReLU | max(0, x) | Hidden layers in most networks — simple, fast |
| GELU | x * Φ(x) | Transformers (GPT, BERT) — smoother than ReLU |
| Sigmoid | 1/(1+e^-x) | Binary output — squashes to 0–1 |
| Softmax | e^xi / Σe^xj | Multi-class output — turns logits into probabilities that sum to 1 |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | RNNs — squashes to -1 to +1 |

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

relu_out  = F.relu(x)        # [-2, -1, 0] become 0; [1, 2] stay
# ReLU: "if negative, kill it; if positive, keep it"
# Simple threshold at zero

gelu_out  = F.gelu(x)        # smooth version of ReLU — slight curve near zero
# GELU: used inside every Transformer feed-forward block (GPT, BERT, LLaMA)
# Smoother gradient near zero = better training

softmax_out = F.softmax(x, dim=0)   # all values sum to 1.0
# Used in the FINAL layer when you need probabilities across multiple classes
# e.g., "70% positive, 30% negative"
```

**WHY GELU specifically:** GELU is the activation inside every Transformer feed-forward network. When you read the GPT-2 paper, they use GELU. When you read LLaMA, they use SwiGLU (a variant). You WILL be asked about this.

**Interview answer for "Why do we need activation functions?"**
> Without them, stacking layers is equivalent to a single linear transformation. Activation functions add non-linearity, allowing the network to learn complex patterns like language understanding.

---

## Weight Initialization — Why Random is Not Good Enough

**What it is:** Before training starts, every weight in the network must be set to some starting value. How you choose those values matters enormously.

Think of it like this: if you start a race facing the wrong direction, you have to run extra far to reach the finish line. Bad weight initialization makes training start in a terrible place.

- **Bad init (all zeros):** Every neuron computes the same thing. Gradients are identical. The network never differentiates. Nothing is learned.
- **Too large:** Outputs explode through layers (exploding gradients from step 1).
- **Too small:** Outputs shrink to zero through layers (vanishing gradients from step 1).

The two smart initialization methods:

```python
import torch.nn as nn

# Xavier / Glorot initialization
# Formula: std = sqrt(2 / (fan_in + fan_out))
# fan_in = number of inputs to the layer
# fan_out = number of outputs from the layer
# Use for: sigmoid and tanh activations
layer = nn.Linear(256, 128)
nn.init.xavier_uniform_(layer.weight)   # in-place: modifies weight directly
# WHY: balances variance going forward AND backward through the layer

# He / Kaiming initialization
# Formula: std = sqrt(2 / fan_in)
# Use for: ReLU activations
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
# WHY: ReLU kills ~half the neurons (zeroes negatives), so He doubles the variance
# to compensate — otherwise outputs shrink layer by layer
```

Key rules to remember:
- **PyTorch default:** Kaiming uniform for Linear layers (safe for ReLU networks)
- **Transformers (GPT-2 style):** small normal distribution with std=0.02
- **Xavier:** for sigmoid/tanh networks

**Interview answer for "Why does initialization matter?"**
> Poor initialization means layer outputs are too large or too small from step one. Xavier accounts for layer width to balance variance in both directions. He initialization accounts for ReLU killing half the neurons by doubling the starting variance.

---

## 4. Forward Pass vs Backward Pass

### Forward Pass

**What it is:** Data flows through the network from input to output, producing a prediction. This is just computation — no learning happens yet.

```
x → Layer1 → Layer2 → ... → ŷ (prediction)
# x: your input (e.g., tokenized sentence as embeddings)
# ŷ: the network's best guess (e.g., "70% probability it's positive sentiment")
# Nothing is updated here — we're just calculating an answer
```

---

### Loss Function

**What it is:** A loss function measures how wrong the prediction is. A single number: 0 means perfect, large means very wrong.

Think of it like a score in golf — lower is better.

```
loss = L(ŷ, y)
# ŷ: what the model predicted
# y: the actual correct answer (ground truth label)
# L: a function that compares them and returns a number (the "wrongness score")
```

Common loss functions in NLP:

| Loss | Used For | Plain English |
|------|---------|---------------|
| Cross-Entropy | Classification, language modeling | "How surprised were you by the correct answer?" |
| MSE | Regression | "How far off was your number guess?" |
| KL Divergence | Probability distributions | "How different are these two probability distributions?" |

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.tensor([[2.0, 0.5, -1.0]])  # model's raw scores for 3 classes
labels = torch.tensor([0])                  # correct class is index 0

loss = criterion(logits, labels)
# CrossEntropyLoss: internally applies softmax, then computes -log(prob of correct class)
# If model was very confident AND correct: loss is near 0
# If model was wrong OR uncertain: loss is large
print(loss)  # e.g., tensor(0.2014) — pretty small, model was right and confident
```

**WHY cross-entropy for LLMs:** LLM training is "predict the next token." That's a classification problem with ~32,000 classes (vocabulary size). Cross-entropy measures how surprised the model is by the actual next token. Perfect model = 0 surprise = 0 loss.

---

## Contrastive Loss — Used in Embedding Models (CLIP, Bi-Encoders)

**What it is:** Instead of predicting a class, contrastive loss trains a model to pull similar things together and push dissimilar things apart in a vector space.

Think of it like this: you're arranging magnets on a table. Similar ones should attract and cluster together. Different ones should repel.

Used in: CLIP (image-text), sentence-transformers, OpenAI's text-embedding models.

The InfoNCE / NT-Xent loss formula (used in CLIP):

```
loss = -log( exp(sim(anchor, positive)/τ) / Σ exp(sim(anchor, negative_i)/τ) )

# anchor: the starting item (e.g., a question)
# positive: the matching item (e.g., the correct answer)
# negative_i: all other items in the batch (treated as wrong answers)
# sim: cosine similarity — measures angle between two vectors (-1 to +1)
# τ (tau): temperature — controls how "sharp" the distribution is
#   small τ (e.g., 0.07): model must be very confident to get low loss
#   large τ (e.g., 1.0): more lenient, softer probabilities
```

```python
# In-batch negatives — the efficient trick:
# If you have a batch of 32 (question, answer) pairs:
# - Each question's CORRECT answer is the positive
# - All OTHER 31 answers in the batch become negatives
# FREE negatives — no extra computation or data labeling needed!

# This is how sentence-transformers trains embedding models
# The bigger the batch, the more negatives, the better the embeddings
```

**Interview answer for "How do you train an embedding model?"**
> Contrastive learning with in-batch negatives. Positive pairs (semantically similar items) are pulled together in vector space. All other items in the batch serve as negatives and are pushed apart. The InfoNCE loss (with temperature τ) controls the sharpness. CLIP and sentence-transformers both use this approach.

---

### Backward Pass (Backpropagation)

**What it is:** After computing the loss, backpropagation figures out how much each weight contributed to the mistake. It then sends "blame" signals backwards through the network using the chain rule from calculus.

Think of it like this: you baked a bad cake. Backprop traces back through every step — was it the oven temperature? The flour quantity? The mixing time? It assigns a blame score to each decision.

```
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w

# ∂L/∂w: "how much did weight w contribute to the loss?" (the gradient)
# ∂L/∂ŷ: "how did the output affect the loss?"
# ∂ŷ/∂w: "how did weight w affect the output?"
# Chain rule: multiply these partial derivatives together — works through any depth
```

Gradients flow backwards from output to input. The gradient for each weight tells you: "increase this weight = loss goes up/down by this much."

**WHY backprop works through 100 layers:** The chain rule lets you decompose complex derivatives into products of simple ones. Each layer just needs to know its own local derivative.

---

## 5. Gradient Descent and Optimizers

### Gradient Descent

**What it is:** After backprop gives us the gradient (the "blame" for each weight), gradient descent uses that gradient to slightly adjust each weight in the direction that reduces loss.

Think of it like this: you're blindfolded on a hill. You feel which direction is downhill with your foot, take a small step that way, then repeat until you reach the bottom.

```python
w = w - lr * gradient
# w: the current value of a weight
# lr: learning rate — how big a step to take
#   too large (e.g., 10.0): you overshoot the bottom and bounce around
#   too small (e.g., 0.000001): you'll get there eventually, but it takes forever
# gradient: the slope — positive means increasing w increases loss
#   so we go the OPPOSITE direction (subtract) to go downhill

# Example:
# w = 5.0, lr = 0.1, gradient = 2.0
# w = 5.0 - 0.1 × 2.0 = 4.8  (nudged slightly toward the minimum)
```

**WHY learning rate matters:** lr=0.01 is safe but slow. lr=10.0 will overshoot the minimum and diverge. LLM training typically uses lr=1e-4 to 3e-4, with warmup and cosine decay.

---

### Types of Gradient Descent

| Type | Description | When to use |
|------|------------|-------------|
| Batch GD | Compute gradient on ALL data, then update once | Very accurate gradient, but impractical for large datasets |
| Stochastic GD (SGD) | One sample at a time, update after each | Fast but noisy — gradient estimate is rough |
| Mini-batch GD | Small batch (32, 64, 128 samples), then update | Best of both worlds — standard in practice |

**WHY mini-batch is standard:** It gives a good gradient estimate without loading all data, and GPUs are designed to process batches efficiently in parallel.

---

### Optimizers

**What it is:** An optimizer is an algorithm that decides how to update weights given the gradients. Gradient descent is the simplest. The others below are smarter.

**SGD (Stochastic Gradient Descent)**

The simplest possible optimizer. Just subtract the gradient times learning rate.

```python
w = w - lr * gradient
# Simple, interpretable, still used for some vision tasks
# Problem: same learning rate for all weights, slow to converge on complex loss landscapes
```

---

**Adam (Adaptive Moment Estimation)**

**What it is:** Adam tracks two running averages of the gradient — a momentum term (smoothed gradient direction) and a velocity term (smoothed gradient magnitude). It uses both to take smarter steps.

Think of it like this: momentum is a ball rolling downhill — it keeps moving in the direction it's been going. Velocity is the ball adjusting its speed based on how steep the slope is.

```python
# At each training step:
m = β1 * m + (1 - β1) * gradient
# m: momentum — exponentially weighted average of PAST gradients
# β1 = 0.9 means: 90% past direction + 10% current gradient
# WHY: smooths out noisy gradients so we don't zig-zag wildly

v = β2 * v + (1 - β2) * gradient**2
# v: velocity — exponentially weighted average of PAST squared gradients
# β2 = 0.999 means: slowly accumulates how large gradients have been
# WHY: weights that get large gradients get smaller learning rates (adaptive)

w = w - lr * m / (torch.sqrt(v) + epsilon)
# epsilon: tiny number (1e-8) to prevent division by zero
# Result: each weight gets its OWN effective learning rate
# Weights with consistently large gradients → smaller steps (don't overshoot)
# Weights with small gradients → larger steps (don't stall)

# Typical values: β1=0.9, β2=0.999, lr=1e-4
```

**WHY Adam dominates:** It handles sparse gradients well, adapts per-parameter, and converges faster than SGD on most deep learning tasks.

---

**AdamW**

**What it is:** Adam with a fix to weight decay. In vanilla Adam, L2 regularization (penalizing large weights) gets accidentally tangled with the adaptive learning rates. AdamW decouples them — weight decay happens separately from the gradient update.

```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,           # learning rate — typical for LLM fine-tuning
    weight_decay=0.01  # L2 penalty on weights — prevents overfitting
)
# AdamW = Adam + proper weight decay
# "Proper" means: weight decay is applied directly to weights, NOT through gradients
# This makes regularization work correctly with adaptive learning rates
```

**WHY AdamW is the default for LLMs:** Every major LLM training recipe (GPT, BERT, LLaMA) uses AdamW. It prevents overfitting via weight decay and trains stably.

---

**AdaFactor**

**What it is:** A memory-efficient optimizer designed for very large models. Instead of storing full second moment matrices (which requires O(n²) memory), AdaFactor stores them in factored form.

```python
# Used for T5, very large transformers where Adam's memory overhead is too high
# Adam requires 2 extra tensors per parameter (m and v) → 3× parameter memory
# AdaFactor compresses second moments → roughly same performance, far less memory
# You will see this in Google's training recipes (T5, PaLM)
```

---

## Gradient Clipping — Prevents Exploding Gradients

**What it is:** During training, gradients can occasionally spike to enormous values (exploding gradient problem). This sends weights flying off to infinity, causing NaN loss and crashing training. Gradient clipping sets a hard cap on how large a gradient can be.

Think of it like a speed limiter on a car: no matter how hard you press the accelerator, the car won't go above a safe top speed.

```python
import torch

# After loss.backward() computes gradients, but BEFORE optimizer.step() updates weights:
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0    # if gradient norm exceeds 1.0, scale ALL gradients down
)
# max_norm=1.0: the total length of the gradient vector is capped at 1
# If gradient norm = 5.0, all gradients are scaled by 0.2 (bringing norm to 1.0)
# This preserves gradient DIRECTION while limiting MAGNITUDE

optimizer.step()  # now safe to update weights — no runaway spikes
```

**WHY it's standard in transformer training:** Transformer gradients can spike suddenly during long training runs (especially on unusual data). The spike is rarely meaningful signal — it's noise. Clipping discards it without affecting normal training steps.

**Rule:** Always use `max_norm=1.0` for transformer training. It appears in every LLM training script.

**Interview answer for "Why do we clip gradients?"**
> Transformer training can have occasional large gradient spikes that destabilize training. Clipping ensures no single step moves weights too far from their current values. Without it, you risk NaN loss and crashed training runs.

---

## Mixed Precision Training — FP16 and BF16

**What it is:** Instead of storing every number as a 32-bit float (FP32), you use 16-bit formats (FP16 or BF16) for most computations. This halves memory usage and speeds up training on modern GPUs.

Think of it like using shorthand notes instead of full sentences — you capture the same information using half the space, but you have to be careful not to abbreviate too aggressively.

```
FP32:  32 bits per number — 4 bytes — full precision — range: ±3.4 × 10^38
FP16:  16 bits per number — 2 bytes — half precision — range: ±65,504 (DANGER: can overflow!)
BF16:  16 bits per number — 2 bytes — same RANGE as FP32 — just less decimal precision
```

```python
# BF16 training — preferred on A100/H100 GPUs (modern hardware):
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(inputs)    # forward pass runs in BF16 — fast + memory efficient
    # WHY bfloat16: it has the same exponent bits as FP32, so no overflow risk
    # The tradeoff is slightly less precision in the mantissa — fine for training

loss.backward()     # gradients computed
optimizer.step()    # weights updated
# No GradScaler needed for BF16 — it handles large values natively

# FP16 training — for older GPUs (V100):
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()   # needed because FP16 can overflow
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()   # scale loss UP before backward to prevent underflow
scaler.step(optimizer)           # unscale gradients before optimizer update
scaler.update()                  # adjust scale factor for next step
```

**Rule:**
- A100 / H100: use BF16, no GradScaler needed
- V100 / older: use FP16 with GradScaler

**WHY this matters for interviews:** Mixed precision training is mentioned in every LLM fine-tuning guide. You need to know the BF16 vs FP16 tradeoff and when to use GradScaler.

---

## 6. Key Hyperparameters

**What it is:** Hyperparameters are the settings you choose before training starts. The model does not learn them — you set them manually. Getting them right is the difference between a model that trains well and one that diverges.

| Hyperparameter | What it controls | Typical value for LLMs |
|---------------|-----------------|------------------------|
| Learning Rate | How large each weight update step is | 1e-4 to 3e-5 for fine-tuning |
| Batch Size | How many samples are processed before each weight update | 8–32 for LLM fine-tuning |
| Epochs | How many full passes through the training data | 1–5 for LLM fine-tuning |
| Dropout | Fraction of neurons randomly zeroed during training (regularization) | 0.1 in Transformers |
| Weight Decay | Penalty on large weights — prevents overfitting | 0.01 in AdamW |

---

### Learning Rate Schedulers

**What it is:** The learning rate is almost never held constant during training. Schedulers change it over time according to a plan.

Think of it like parallel parking: you start slow (careful), speed up once you know where you're going, then slow down again as you pull in.

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,    # ramp learning rate from 0 up to lr over first 100 steps
    num_training_steps=1000  # total training steps
)
# After warmup: cosine decay — LR follows a gentle curve down to near-zero
# WHY warmup: at the start of training, weights are random and gradients are noisy.
#   Starting with a full-sized LR causes wild, destabilizing updates.
#   Warming up lets the optimizer settle into a stable region first.

# Warmup: 0 → lr (first 1-5% of training)
# Cosine decay: lr → near-zero (rest of training)
# Linear decay: also common — LR decreases in a straight line
```

**WHY LLMs use warmup + cosine decay:** Standard for all major LLMs. Warmup prevents instability at the start. Cosine decay helps the model converge smoothly without stopping too early.

---

## 7. Overfitting vs Underfitting

**What it is:** A model can fail in two opposite ways: memorizing the training data without generalizing, or being too simple to capture the patterns at all.

Think of it like studying for an exam:
- **Overfitting:** You memorized the practice exam answers verbatim but can't handle any new question phrasing.
- **Underfitting:** You barely studied and can't answer even the practice questions.
- **Good fit:** You understood the underlying concepts and can answer new questions.

| Problem | Description | Fix |
|---------|------------|-----|
| Overfitting | Model memorizes training data, fails on new examples | Dropout, weight decay, more data, early stopping |
| Underfitting | Model is too simple, can't capture patterns | More layers, more parameters, longer training |
| Good fit | Generalizes well to new data | Right model size + regularization |

---

### Regularization Techniques

**What it is:** Regularization is any technique that prevents a model from memorizing training data and forces it to learn generalizable patterns.

```python
import torch.nn as nn

# Dropout — randomly zeroes out neurons during training
dropout = nn.Dropout(p=0.1)
# p=0.1 means: 10% of neurons are set to 0 each forward pass (randomly chosen)
# WHY: forces the network to not rely on any single neuron
#   If any neuron might be missing, all neurons must learn redundant features → robustness
# KEY: dropout is DISABLED during model.eval() — only active during training

# Weight Decay (L2 regularization)
# Built into AdamW: weight_decay=0.01
# Adds a penalty to the loss: loss_total = loss + λ × Σ(w²)
# WHY: large weights make predictions overly confident about specific patterns
#   Penalizing large weights keeps them small and more general

# Early Stopping — not a code change, a training decision
# Monitor validation loss every N steps
# If validation loss stops improving for K steps → stop training
# WHY: training loss always goes down; validation loss eventually goes back up (overfitting)

# Data Augmentation
# For NLP: synonym replacement, back-translation, random deletion
# WHY: more diverse training examples = harder to memorize, better generalization
```

---

## 8. Embeddings

**What it is:** An embedding is a dense vector (a list of floating-point numbers) that represents a discrete object — like a word, token, or user ID. Similar concepts get similar vectors.

Think of it like GPS coordinates for meaning: "cat" and "kitten" have coordinates close together. "cat" and "Jupiter" are far apart.

```python
# Words become vectors:
# "cat" → [0.2, -0.8, 0.5, 0.1, ...]   # 768 numbers
# "dog" → [0.21, -0.75, 0.48, 0.12, ...] # very similar numbers → similar meaning
# "car" → [-0.9, 0.3, -0.2, 0.8, ...]    # very different numbers → different meaning

import torch.nn as nn

# In an LLM, every token gets an embedding before anything else:
embedding_layer = nn.Embedding(
    num_embeddings=32000,  # vocabulary size — one vector per possible token
    embedding_dim=768      # how many numbers per token (dimension of the vector)
)
# Input: token ID (an integer, e.g., 4231 = the token "cat")
# Output: a 768-dimensional vector

token_id = torch.tensor([4231])           # "cat" token
cat_embedding = embedding_layer(token_id) # shape: [1, 768]
# This 768-number vector is what actually flows into the Transformer layers
```

**WHY embeddings are the foundation of LLMs:**
- You can compute similarity between sentences by comparing their embedding vectors
- Mathematical operations become meaningful: "king" - "man" + "woman" ≈ "queen"
- Every RAG system, every semantic search, every chatbot starts with embeddings

---

## ANN, CNN, RNN — The Three Network Types

### ANN (Artificial Neural Network) — The Basic One

**What it is:** ANN is just another name for a standard feedforward neural network. Every neural network is technically an ANN. When people say "ANN" they usually mean the simplest type — input goes in, passes through hidden layers, output comes out. No memory, no loops.

```
Input → [Hidden Layer 1] → [Hidden Layer 2] → Output
         (fully connected)   (fully connected)
```

- Also called: feedforward network, multilayer perceptron (MLP), dense network
- Used for: classification, regression, the final "head" layer on top of LLMs
- NOT used for: sequences (no memory), images (ignores spatial structure)

---

### CNN (Convolutional Neural Network) — For Spatial Data

**What it is:** CNN uses a sliding window (filter/kernel) that scans across the input, learning local patterns. Instead of connecting every neuron to every other neuron (ANN), it connects neurons only to nearby neighbors.

Think of it like this: you're looking for a cat in a photo. You don't look at the whole photo at once. You scan a small window across the image, checking each region for "cat features" (ears, eyes, fur). That's what a CNN does.

```
Image: 224×224 pixels
↓
Conv Layer 1: 3×3 filter slides across → detects edges
↓
Conv Layer 2: 3×3 filter → detects shapes (eyes, ears)
↓
Conv Layer 3: → detects objects (face, cat)
↓
Flatten → Dense Layer → Output (cat/dog probability)
```

Key concepts:
- **Filter/Kernel:** small window (3×3 or 5×5) that slides across input
- **Feature map:** output of applying one filter to the whole input
- **Pooling:** shrinks feature map (MaxPool takes the max in each region)
- **Stride:** how many pixels the filter moves each step

**Why CNNs matter for GenAI:**
- Vision Transformers (ViT) use CNN-like patch extraction
- CLIP uses CNN (ResNet) or ViT for image encoding
- Stable Diffusion uses CNN-based U-Net for image generation

```python
import torch.nn as nn

# Simple CNN
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 color channels → 32 filters
    nn.ReLU(),
    nn.MaxPool2d(2),                              # shrink by 2x
    nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 → 64 filters
    nn.ReLU(),
    nn.Flatten(),                                 # flatten to 1D
    nn.Linear(64 * 56 * 56, 10)                  # final classification
)
# WHY: CNNs share weights across positions (same filter for whole image)
#      → far fewer parameters than ANN on same image (efficient)
```

---

### Quick Comparison: ANN vs CNN vs RNN

| | ANN | CNN | RNN/LSTM |
|--|-----|-----|----------|
| **Input type** | Tabular/flat data | Images, spatial data | Sequences, text |
| **Key feature** | Fully connected | Sliding filter | Hidden state (memory) |
| **Parameter sharing** | None | Yes (filter reused) | Yes (same weights per step) |
| **Parallelizable** | Yes | Yes | No (sequential) |
| **Used in GenAI** | Classification heads | Vision encoding (ViT, CLIP) | Replaced by Transformers |
| **Still used?** | Yes | Yes (vision) | Rarely (Mamba replacing) |

**Interview answer:** "ANN is the general feedforward network — input to output with no memory. CNN adds spatial awareness through sliding filters — used for images and now in vision transformers. RNN processes sequences with memory but can't parallelize. In 2024, Transformers replaced RNNs for most sequence tasks, but CNNs remain important for vision components in multimodal models like CLIP and LLaVA."

---

## 9. RNN, LSTM, GRU — Why They Existed and Why Transformers Replaced Them

### RNN (Recurrent Neural Network) — The Original Sequence Model

**What it is:** An RNN processes a sequence one step at a time, maintaining a "hidden state" that summarizes everything it has seen so far. It's like reading a book one word at a time and keeping running notes.

```
"The cat sat on the mat" → RNN processes one word at a time:
  h1 = f("The",  h0)   # h0 is a zero vector — no history yet
  h2 = f("cat",  h1)   # h1 carries some memory of "The"
  h3 = f("sat",  h2)   # h2 carries memory of "The cat"
  h4 = f("on",   h3)
  h5 = f("the",  h4)
  h6 = f("mat",  h5)   # h6 must summarize the entire sentence
# f: the RNN cell function — combines previous hidden state + current word

# The problem: h6 must somehow remember "The" from 5 steps ago
# That information has been diluted through 5 multiplications
```

**The Vanishing Gradient Problem:**

```
# To update weights for word 1 based on loss at word 100:
# gradient must travel 99 steps backward through the chain rule
# Each step multiplies by a weight (typically < 1, e.g., 0.9)
# After 100 steps: 0.9^100 = 0.000027
# → gradient is essentially zero → weights for early words NEVER update
# → model forgets what happened 50+ words ago

# This is the vanishing gradient problem — the fundamental flaw of RNNs
```

---

### LSTM (Long Short-Term Memory) — The Fix for Vanishing Gradients

**What it is:** LSTM adds a separate "cell state" — a long-term memory highway that runs alongside the hidden state. Three gates control what gets remembered, forgotten, and output.

Think of it like this: the hidden state is your working memory (what you're thinking right now). The cell state is your long-term memory (facts you've stored). Gates are like switches that decide what to write to memory, what to erase, and what to share.

```
Three gates — all use sigmoid (outputs 0 to 1, where 0 = "forget all" and 1 = "keep all"):

Forget gate:  f_t = sigmoid(W_f × [h_{t-1}, x_t])
# "What fraction of the current cell state should we forget?"
# f_t ≈ 1: keep everything in memory
# f_t ≈ 0: erase it entirely
# The FORGET gate is why LSTM doesn't vanish — it can keep gradients alive

Input gate:   i_t = sigmoid(W_i × [h_{t-1}, x_t])
# "How much of this new information should we write to cell state?"

Output gate:  o_t = sigmoid(W_o × [h_{t-1}, x_t])
# "How much of the cell state should we expose as the hidden state output?"

Cell update:  C_t = f_t × C_{t-1} + i_t × tanh(W_c × [h_{t-1}, x_t])
# C_t: new cell state = (fraction of old state kept) + (fraction of new info added)

Hidden state: h_t = o_t × tanh(C_t)
# h_t: output for this time step = gated version of cell state
```

**WHY it worked:** The cell state can carry information unchanged across hundreds of steps — if the forget gate stays near 1, the gradient can flow back without shrinking.

**WHY it still failed:**
- Still sequential — you must process word 1 before word 2. Cannot parallelize. Slow on GPUs.
- Still struggles with very long sequences (500+ tokens)
- 4x more parameters than a simple RNN — expensive to train

---

### GRU (Gated Recurrent Unit) — Simplified LSTM

**What it is:** GRU collapses the forget and input gates into a single "update gate." Fewer parameters, similar performance, faster training.

```
Update gate: z_t = sigmoid(W_z × [h_{t-1}, x_t])
# How much to update the hidden state vs keep old state
# z_t ≈ 0: keep the old hidden state (like LSTM's forget gate ≈ 1)
# z_t ≈ 1: replace entirely with new information

Reset gate:  r_t = sigmoid(W_r × [h_{t-1}, x_t])
# How much of past hidden state to use when computing new candidate
# r_t ≈ 0: ignore the past — fresh start
# r_t ≈ 1: use the past fully

New hidden:  h_t = (1 - z_t) × h_{t-1} + z_t × tanh(W × [r_t × h_{t-1}, x_t])
# Final hidden = blend of old state and new candidate, controlled by update gate
```

**GRU vs LSTM in practice:**

| Aspect | GRU | LSTM |
|--------|-----|------|
| Parameters | Fewer (2 gates) | More (3 gates) |
| Speed | Faster | Slower |
| Performance | Similar | Slightly better on very long sequences |
| When to use | Smaller datasets, constrained compute | Complex, long sequences |

---

### Why Transformers Replaced Both

**What it is:** Transformers solved the fundamental problem of RNNs — they attend to ALL tokens simultaneously, instead of processing one at a time.

Think of it like this: RNNs read a book page by page and try to remember what they read. Transformers read all pages at once and can directly look at any page when answering a question.

| Problem | RNN / LSTM / GRU | Transformer |
|---------|------------------|-------------|
| Long-range dependencies | Struggles after ~100 tokens — gradient vanishes | Attends to ALL tokens directly — no distance limit |
| Parallelization | Sequential — word 2 waits for word 1 to finish | Fully parallel — process all words at once on GPU |
| Training speed | Slow — cannot use GPU parallelism efficiently | 10–100x faster on GPUs |
| Memory of position 1 at position 512 | Severely degraded | Direct attention — perfect recall |

**Interview answer for "Why did Transformers replace LSTMs?"**
> LSTMs solved vanishing gradients with gating but were still sequential — you couldn't parallelize training. Transformers replaced them by computing attention over all positions simultaneously, enabling massive parallelism on GPUs and direct access to any position in the sequence regardless of distance.

---

### Where LSTM and GRU Still Appear in 2025

**What it is:** Despite losing to Transformers for most NLP, RNN-style models survive in specific niches.

- **Time series forecasting** (stock prices, sensor data) — still competitive, less data hungry
- **Streaming / edge inference** — sequential processing uses constant memory (O(1) vs O(n²) for attention)
- **Mamba / State Space Models (SSMs)** — modern models that are conceptually related to RNNs but with selective memory mechanisms. Growing in 2024–2025.
- **RWKV** — reformulates attention as an RNN for O(1) inference time (no KV cache needed)

**WHY this matters for interviews:** If asked about alternatives to Transformers, Mamba and RWKV are the modern answers. They show you understand the O(n²) attention scaling problem.

---

## 10. Key Concepts for LLM Context

**What it is:** This table maps fundamental deep learning concepts to where they show up inside actual LLMs. Know these connections cold.

| Concept | Relevance to LLMs | Why it matters |
|---------|------------------|----------------|
| Softmax | Used in attention scores and output probabilities | Converts raw scores to probabilities — used everywhere |
| Cross-entropy loss | Used to train language models (predict next token) | The entire LLM training objective |
| Gradient clipping | Prevents exploding gradients in transformer training | Standard in all LLM training scripts |
| Layer normalization | Standard in all modern transformers | Stabilizes training — used before or after every sub-layer |
| Batch normalization | NOT used in transformers | Fails with variable sequence lengths and small batches |
| Residual connections | Critical in transformer architecture | Prevents vanishing gradients in deep networks |

**WHY Layer Norm instead of Batch Norm:** Layer Norm normalizes across the feature dimension (per sample). Batch Norm normalizes across the batch dimension. With variable sequence lengths and small batch sizes (common in LLM training), Batch Norm is unstable. Layer Norm works per-sample and is always stable.

---

## 11. Interview Questions — Deep Learning

**Q: What is backpropagation?**
> The algorithm that computes gradients of the loss with respect to each weight using the chain rule, enabling the model to learn by updating weights in the direction that reduces loss. Gradients flow backwards from the output layer to the input layer.

**Q: Why do we need activation functions?**
> Without them, stacking layers is equivalent to a single linear transformation — all the depth collapses to one matrix multiplication. Activation functions add non-linearity, allowing the network to learn complex, non-linear patterns like language understanding.

**Q: What is the vanishing gradient problem?**
> In deep networks or sequential models (RNNs), gradients become very small as they propagate backwards through many layers, making early layers learn very slowly or not at all. Solved in modern architectures by residual connections (Transformers), gating (LSTM), and better activations like ReLU and GELU.

**Q: Why is AdamW preferred for LLM training?**
> It combines the adaptive learning rates of Adam with proper weight decay decoupled from the gradient update. Vanilla Adam's weight decay interacts incorrectly with adaptive learning rates. AdamW fixes this, giving correct regularization that prevents overfitting and stabilizes training of large models.

**Q: What is the difference between batch size and learning rate?**
> Batch size controls how many samples are seen before a weight update; learning rate controls how large that update is. Larger batches often require a proportionally higher learning rate (the linear scaling rule: double batch size → double learning rate).

**Q: What is Layer Normalization and why do Transformers use it instead of Batch Norm?**
> Layer Norm normalizes across the feature dimension for each individual sample. Batch Norm normalizes across the batch dimension. Transformers use Layer Norm because sequence lengths vary and batch sizes can be very small during LLM training — Layer Norm is more stable in these conditions because it does not depend on other samples in the batch.

**Q: What is contrastive loss and where is it used?**
> Contrastive loss trains a model to pull similar items together and push dissimilar items apart in vector space. Used in CLIP (image-text pairs), sentence-transformers, and OpenAI text-embedding models. The InfoNCE variant uses in-batch negatives — all other items in the batch serve as free negatives, making training efficient.

---

## Quick Reference Cheat Sheet

```
Neural Network:    Input → Hidden Layers → Output
Loss:              Cross-Entropy for LM tasks, MSE for regression
Optimizer:         AdamW (LLMs), Adam (general deep learning)
Activation:        GELU (Transformers), ReLU (general)
Regularization:    Dropout + Weight Decay (in AdamW)
LR Schedule:       Warmup + Cosine Decay (standard for LLMs)
Embeddings:        Dense vectors representing tokens — foundation of LLMs
Gradient clipping: max_norm=1.0 — always use for transformer training
Mixed precision:   BF16 on A100/H100, FP16 + GradScaler on V100
Init:              Kaiming for ReLU nets, Xavier for sigmoid/tanh, std=0.02 for GPT-style
```
