# 01 — Deep Learning Fundamentals

> Foundation for everything in this JD. Every LLM, Transformer, and fine-tuning technique is built on these concepts.

---

## 1. What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses **neural networks with many layers** (hence "deep") to learn representations of data.

- **Machine Learning**: Learns from features you engineer manually
- **Deep Learning**: Learns features automatically from raw data

```
Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Output
```

Each layer learns increasingly abstract representations.
- Layer 1: basic patterns (edges in images, word co-occurrences)
- Layer N: high-level concepts (faces, sentence meaning)

---

## 2. Neural Network Basics

### Neuron (Perceptron)
```
output = activation( w1*x1 + w2*x2 + ... + wn*xn + bias )
```
- **x**: inputs
- **w**: weights (learned during training)
- **bias**: offset term
- **activation**: non-linear function

### Layers
| Layer Type | Purpose |
|-----------|---------|
| Input Layer | Receives raw data |
| Hidden Layers | Learn intermediate representations |
| Output Layer | Produces final prediction |

### Fully Connected (Dense) Layer
Every neuron connects to every neuron in the next layer.
Used in classification heads of LLMs.

---

## 3. Activation Functions

Non-linearity is what makes deep networks powerful. Without it, stacking layers = just one linear transformation.

| Function | Formula | Use Case |
|----------|---------|---------|
| ReLU | max(0, x) | Hidden layers (most common) |
| GELU | x * Φ(x) | Transformers (GPT, BERT) |
| Sigmoid | 1/(1+e^-x) | Binary output |
| Softmax | e^xi / Σe^xj | Multi-class output |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | RNNs, some hidden layers |

**GELU** is especially important — it's the activation used inside Transformer feed-forward networks.

---

## 4. Forward Pass vs Backward Pass

### Forward Pass
Data flows through the network to produce a prediction.
```
x → Layer1 → Layer2 → ... → ŷ (prediction)
```

### Loss Function
Measures how wrong the prediction is.
```
loss = L(ŷ, y)   # predicted vs actual
```

Common loss functions in NLP:
| Loss | Used For |
|------|---------|
| Cross-Entropy | Classification, language modeling |
| MSE | Regression |
| KL Divergence | Probability distributions |

### Backward Pass (Backpropagation)
Computes gradients of the loss with respect to each weight using the **chain rule**.
```
∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂w
```
Gradients flow backwards from output → input.

---

## 5. Gradient Descent & Optimizers

### Gradient Descent
Update weights to minimize loss:
```
w = w - lr * ∂L/∂w
```
- **lr**: learning rate (how big a step to take)

### Types
| Type | Description |
|------|------------|
| Batch GD | Use all data, one update |
| Stochastic GD (SGD) | One sample at a time |
| Mini-batch GD | Small batch (32, 64, 128 samples) |

### Optimizers

**SGD**
```
w = w - lr * gradient
```
Simple, but slow convergence.

**Adam (Adaptive Moment Estimation)**
- Most commonly used optimizer in deep learning and LLMs
- Combines momentum + adaptive learning rates
```
m = β1 * m + (1-β1) * gradient         # momentum
v = β2 * v + (1-β2) * gradient²        # velocity
w = w - lr * m / (√v + ε)
```
Typical values: β1=0.9, β2=0.999, lr=1e-4

**AdamW**
- Adam + weight decay (L2 regularization on weights, not gradients)
- Default optimizer for most LLM training (GPT, BERT, etc.)

**AdaFactor**
- Memory-efficient optimizer used for very large models (T5)
- Stores second moments in factored form to save memory

---

## 6. Key Hyperparameters

| Hyperparameter | What it controls |
|---------------|-----------------|
| Learning Rate | Step size during gradient descent |
| Batch Size | Samples per update |
| Epochs | Full passes through training data |
| Dropout | Regularization — randomly zero neurons |
| Weight Decay | Penalizes large weights (prevents overfitting) |

### Learning Rate Schedulers
LR is usually not constant during training:
- **Warmup**: Start low, ramp up (prevents instability early on)
- **Cosine decay**: Gradually reduce LR in a cosine curve
- **Linear decay**: Linearly reduce LR

LLMs typically use: **warmup + cosine/linear decay**

---

## 7. Overfitting vs Underfitting

| Problem | Description | Fix |
|---------|------------|-----|
| Overfitting | Model memorizes training data, fails on new data | Dropout, weight decay, more data |
| Underfitting | Model too simple, can't capture patterns | More layers, more capacity, longer training |
| Good fit | Generalizes well | Right model size + regularization |

### Regularization Techniques
- **Dropout**: Randomly zero out neurons during training (not inference)
- **Weight Decay (L2)**: Penalize large weights in the loss
- **Early Stopping**: Stop training when validation loss stops improving
- **Data Augmentation**: Create more training samples

---

## 8. Embeddings

An **embedding** is a dense vector representation of discrete data (words, tokens, users).

```
"cat" → [0.2, -0.8, 0.5, 0.1, ...]   # 768-dimensional vector
"dog" → [0.21, -0.75, 0.48, 0.12, ...] # similar vector
```

Why embeddings?
- Capture semantic meaning
- Similar concepts have similar vectors
- Enable mathematical operations on language

In LLMs, every token gets an embedding vector as input to the model.

---

## 9. Key Concepts for LLM Context

| Concept | Relevance to LLMs |
|---------|------------------|
| Softmax | Used in attention scores and output probabilities |
| Cross-entropy loss | Used to train language models (predict next token) |
| Gradient clipping | Prevents exploding gradients in transformer training |
| Batch normalization | Layer norm is used in transformers instead |
| Layer normalization | Standard in all modern transformers |
| Residual connections | Critical in transformer architecture |

---

## 10. Interview Questions — Deep Learning

**Q: What is backpropagation?**
> The algorithm that computes gradients of the loss with respect to each weight using the chain rule, enabling the model to learn by updating weights in the direction that reduces loss.

**Q: Why do we need activation functions?**
> Without them, stacking layers is equivalent to a single linear transformation. Activation functions add non-linearity, allowing the network to learn complex patterns.

**Q: What is the vanishing gradient problem?**
> In deep networks, gradients become very small as they propagate backwards through many layers, making early layers learn very slowly. Solved by residual connections (used in Transformers) and better activations like ReLU/GELU.

**Q: Why is AdamW preferred for LLM training?**
> It combines the adaptive learning rates of Adam with proper weight decay decoupled from the gradient update, preventing overfitting and stabilizing training of large models.

**Q: What is the difference between batch size and learning rate?**
> Batch size controls how many samples are seen before a weight update; learning rate controls how large that update is. Larger batches often require a proportionally higher learning rate (linear scaling rule).

**Q: What is Layer Normalization and why do Transformers use it instead of Batch Norm?**
> Layer Norm normalizes across the feature dimension (per sample), while Batch Norm normalizes across the batch. Transformers use Layer Norm because sequence lengths vary and batch sizes can be small — Layer Norm is more stable in these conditions.

---

## Quick Reference Cheat Sheet

```
Neural Network:    Input → Hidden Layers → Output
Loss:              Cross-Entropy for LM, MSE for regression
Optimizer:         AdamW (LLMs), Adam (general)
Activation:        GELU (Transformers), ReLU (general)
Regularization:    Dropout + Weight Decay
LR Schedule:       Warmup + Cosine Decay
Embeddings:        Dense vectors representing tokens
```
