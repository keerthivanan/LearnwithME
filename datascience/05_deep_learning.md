# Deep Learning

---

## 1. Neural Networks Basics

### What is a Neural Network?
Layers of connected "neurons" that learn by adjusting weights.  
Input → Hidden Layers → Output

### How it Learns
```
Forward pass:  input goes through layers, produces prediction
Loss:          measure how wrong the prediction is
Backprop:      compute gradient of loss w.r.t each weight (chain rule)
Update:        weights -= learning_rate × gradient
Repeat thousands of times → loss decreases → model learns
```

### Activation Functions
```python
# ReLU — most common for hidden layers
# f(x) = max(0, x) → kills negatives, allows positives through
# Pros: Fast, prevents vanishing gradient

# Sigmoid — for binary output
# f(x) = 1 / (1 + e^(-x)) → output ∈ [0,1]

# Softmax — for multiclass output
# converts logits to probabilities that sum to 1

# Tanh — output ∈ [-1, 1], better than sigmoid for hidden layers
```

### Basic Neural Network (PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN(input_dim=20, hidden_dim=128, output_dim=1)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_train).squeeze()
    loss = criterion(output, y_train.float())
    loss.backward()
    optimizer.step()
```

### Basic Neural Network (Keras / TensorFlow)
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')   # binary
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)
```

---

## 2. Optimizers

| Optimizer | Key Idea | Use When |
|-----------|---------|---------|
| SGD | Basic gradient descent | Old standard |
| Momentum | Accumulates gradient direction → faster convergence | With LR scheduling |
| Adam | Adaptive learning rate per parameter | Default choice |
| AdamW | Adam + weight decay (better regularization) | Transformers, modern default |
| RMSprop | Adaptive LR, good for RNNs | RNNs / non-stationary data |

```python
# Adam (default)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW (better for transformers)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

---

## 3. Preventing Overfitting in Deep Learning

```python
# 1. Dropout — randomly zeros some neurons during training
nn.Dropout(0.3)    # 30% of neurons dropped each step

# 2. Batch Normalization — normalizes layer inputs, speeds training
nn.BatchNorm1d(128)

# 3. Early stopping
keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 4. L2 Regularization (weight decay)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)

# 5. Data augmentation (for images)
# 6. More data
```

---

## 4. CNN (Convolutional Neural Networks)

**What**: Designed for images. Learns spatial features using filters.  
**How**: Convolutional layers detect patterns (edges → shapes → objects) at different scales.  
**Use when**: Image classification, object detection, segmentation.

### Key Layers
```
Conv2D:    sliding filter detects local patterns
ReLU:      activation
MaxPool2D: downsampling, keeps dominant features
Flatten:   convert to 1D for dense layers
Dense:     final classification
```

```python
from tensorflow.keras import layers

cnn = keras.Sequential([
    # Feature extraction
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])
```

### Transfer Learning (Best Practice)
```python
# Use pretrained model, fine-tune for your task
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze pretrained weights

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Step 1: Train only top layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10)

# Step 2: Fine-tune (unfreeze some layers)
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=optim.Adam(lr=1e-5), ...)  # very small LR!
model.fit(train_data, epochs=10)
```

**Popular pretrained models**: ResNet50, EfficientNet, VGG16, InceptionV3, MobileNet

---

## 5. RNN / LSTM (Sequence Models)

**What**: Process sequences (text, time series) by maintaining hidden state.  
**Problem with vanilla RNN**: Vanishing gradient → forgets long-term dependencies.  
**LSTM solution**: Uses gates to control what to remember and forget.

### LSTM Gates
```
Forget gate:  "What from the past should I forget?"
Input gate:   "What new info should I store?"
Output gate:  "What should I output?"
Cell state:   long-term memory
Hidden state: short-term memory
```

```python
# Keras LSTM
model = keras.Sequential([
    layers.Embedding(vocab_size, 64),      # word embeddings
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

# Bidirectional LSTM (reads both directions)
layers.Bidirectional(layers.LSTM(128))

# GRU (simpler than LSTM, often similar performance)
layers.GRU(128)
```

---

## 6. Transformers & Attention

**What**: The architecture behind BERT, GPT, and all modern LLMs.  
**Key innovation**: Self-attention — every word can directly attend to every other word.

### Self-Attention (Simply)
```
For each word, compute:
  Query (Q): "What am I looking for?"
  Key   (K): "What do I contain?"
  Value (V): "What will I pass if attended to?"

Attention(Q,K,V) = softmax(QK^T / √d) × V

Score = how much attention word i pays to word j
Output = weighted sum of values based on attention scores
```

### Why Better than LSTM?
```
LSTM: processes words LEFT to RIGHT, sequential (slow, forgets)
Transformer: ALL words attend to ALL words in PARALLEL (fast, no forgetting)
```

### BERT (Bidirectional Encoder)
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize
inputs = tokenizer("I love data science!", return_tensors='pt',
                   max_length=128, truncation=True, padding='max_length')

# Inference
outputs = model(**inputs)
logits = outputs.logits
probs = torch.softmax(logits, dim=1)
```

### GPT (Generative, Decoder-only)
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

inputs = tokenizer("The future of AI is", return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## 7. Key Deep Learning Concepts

### Batch Size Effect
```
Small batch (8-32):  noisier gradients, better generalization, slower
Large batch (256+):  stable gradients, can overfit, faster

Common: 32 or 64 for most tasks
```

### Learning Rate Tips
```
Too high: loss diverges (explodes)
Too low:  training very slow, gets stuck

Good starting points:
  Adam: 1e-3 to 1e-4
  Fine-tuning BERT: 1e-5 to 5e-5

Use LR scheduler:
  ReduceLROnPlateau: reduce LR when loss stops improving
  CosineAnnealing: smooth decay
  Warmup: increase from 0 → peak LR → decay (best for transformers)
```

### GPU Training
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X_batch = X_batch.to(device)
```

---

## 8. Architecture Quick Reference

| Task | Architecture |
|------|-------------|
| Image classification | CNN → ResNet, EfficientNet |
| Object detection | YOLO, Faster-RCNN |
| Image segmentation | U-Net |
| Text classification | BERT |
| Text generation | GPT |
| Translation | Transformer (Encoder-Decoder) |
| Time series | LSTM, Transformer |
| Tabular data | XGBoost > Neural Network |
| Recommendation | Neural collaborative filtering |
| Anomaly (images) | Autoencoder |
| Image generation | GAN, Diffusion models |
