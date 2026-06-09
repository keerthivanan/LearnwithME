"""
PyTorch — Definitions + Code + Outputs
========================================
pip install torch torchvision scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

print(f"PyTorch: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ══════════════════════════════════════════════════════
# 1. TENSORS — The Core Data Structure
# ══════════════════════════════════════════════════════
# WHAT IS A TENSOR?
#   → Multi-dimensional array (like NumPy array) but can run on GPU
#   → PyTorch's fundamental building block — everything is a tensor
#   → 0D = scalar, 1D = vector, 2D = matrix, 3D+ = tensor
#
# PYTORCH vs TENSORFLOW:
#   → PyTorch  : dynamic computation graph — define as you run (Python-like)
#   → TensorFlow: static graph by default (can be dynamic with eager mode)
#   → PyTorch is preferred in research; TF is preferred in production
#
# KEY TENSOR OPS:
#   → .shape / .dtype / .device : inspect tensor properties
#   → .to(device)               : move tensor to CPU or GPU
#   → .reshape() / .view()      : change shape without copying data
#   → @ or torch.matmul()       : matrix multiplication
#   → .T                        : transpose
#
# REQUIRES_GRAD + AUTOGRAD:
#   → requires_grad=True : track all operations for backpropagation
#   → .backward()        : compute gradients (∂loss/∂weight for each weight)
#   → .grad              : the computed gradient value

print("\n" + "=" * 55)
print("1. TENSORS")
print("=" * 55)

t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.zeros(3, 4)
t3 = torch.ones(2, 3)
t4 = torch.randn(3, 3)
t5 = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
t6 = torch.linspace(0, 1, 5)    # [0.00, 0.25, 0.50, 0.75, 1.00]

print(f"t1 shape: {t1.shape}, dtype: {t1.dtype}")     # torch.Size([3]) torch.float32
print(f"t4 shape: {t4.shape}, device: {t4.device}")   # torch.Size([3, 3]) cpu

a = torch.tensor([[1,2],[3,4]], dtype=torch.float)
b = torch.tensor([[5,6],[7,8]], dtype=torch.float)
print(f"a + b:\n{a + b}")
print(f"a @ b:\n{a @ b}")       # [[19,22],[43,50]]
print(f"a.T:\n{a.T}")
print(f"reshape: {a.reshape(4)}")   # [1, 2, 3, 4]
print(f"sum={a.sum()}, mean={a.mean()}, max={a.max()}")

# NumPy ↔ PyTorch (shared memory — changing one changes the other!)
arr      = np.array([1.0, 2.0, 3.0])
t        = torch.from_numpy(arr)
arr_back = t.numpy()
print(f"From numpy: {t}")   # tensor([1., 2., 3.])

# Autograd — automatic differentiation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x          # y = x³ + 2x
y.backward()                  # compute dy/dx
print(f"dy/dx at x=2: {x.grad}")   # 3x² + 2 = 14.0


# ══════════════════════════════════════════════════════
# 2. CUSTOM DATASET & DATALOADER
# ══════════════════════════════════════════════════════
# WHAT IS A DATASET?
#   → A class that wraps your data and makes it accessible by index
#   → Must implement 3 methods:
#     - __init__()   : load/store data
#     - __len__()    : how many samples total
#     - __getitem__(): return one sample (X, y) by index
#
# WHAT IS A DATALOADER?
#   → Wraps a Dataset and handles: batching, shuffling, parallel loading
#   → shuffle=True  : randomly order samples each epoch (training only!)
#   → shuffle=False : keep order (validation/test — for reproducibility)
#   → batch_size    : how many samples per gradient update
#   → num_workers   : parallel data loading processes (0 = main process)

print("\n" + "=" * 55)
print("2. CUSTOM DATASET & DATALOADER")
print("=" * 55)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X_raw, y_raw = make_classification(n_samples=1000, n_features=20, random_state=42)
X_raw = StandardScaler().fit_transform(X_raw)

dataset = TabularDataset(X_raw, y_raw)
n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
# Train batches: 25, Test batches: 4


# ══════════════════════════════════════════════════════
# 3. FEEDFORWARD NEURAL NETWORK (nn.Module)
# ══════════════════════════════════════════════════════
# WHAT IS nn.Module?
#   → Base class for ALL PyTorch models — you MUST inherit from it
#   → You define: __init__() (layers) and forward() (how data flows)
#   → PyTorch automatically computes backward() via autograd
#
# KEY LAYERS:
#   → nn.Linear(in, out)    : fully connected layer (Dense)
#   → nn.BatchNorm1d(dim)   : normalize activations → stable training
#   → nn.ReLU()             : activation function max(0,x)
#   → nn.Dropout(p)         : zero out p% of neurons randomly → prevent overfitting
#
# model.train() vs model.eval():
#   → train(): enables Dropout (randomly zeros neurons) and BatchNorm updates
#   → eval() : disables Dropout (all neurons active) and freezes BatchNorm stats
#   → ALWAYS switch between them correctly — wrong mode = wrong results!

print("\n" + "=" * 55)
print("3. FEEDFORWARD NEURAL NETWORK")
print("=" * 55)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layer_list = []
        prev_dim   = input_dim
        for h in hidden_dims:
            layer_list += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h
        layer_list.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)

model = FeedForward(20, [128, 64, 32], 1, dropout=0.3).to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Total parameters: ~11,000


# ══════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ══════════════════════════════════════════════════════
# WHAT IS THE TRAINING LOOP?
#   → The core of PyTorch — you control every step manually
#   → Each epoch: forward → compute loss → backward → update weights
#
# STANDARD STEPS PER BATCH:
#   1. optimizer.zero_grad()   : clear old gradients (they accumulate by default!)
#   2. output = model(X_batch) : forward pass
#   3. loss = criterion(output, y_batch) : compute loss
#   4. loss.backward()         : compute gradients via autograd
#   5. optimizer.step()        : update weights using gradients
#
# KEY TERMS:
#   → BCEWithLogitsLoss : Binary Cross Entropy + Sigmoid combined (numerically stable)
#   → gradient clipping : cap gradient norm → prevents "exploding gradients" in RNNs
#   → LR scheduler      : reduce learning rate over time for better convergence
#   → state_dict        : ordered dict of model parameters (what to save)

print("\n" + "=" * 55)
print("4. TRAINING LOOP")
print("=" * 55)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.BCEWithLogitsLoss()

best_loss = float("inf")
patience  = 5
counter   = 0

for epoch in range(30):
    # ── Train ─────────────────────────────────
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()                              # 1. clear gradients
        output = model(X_batch)                            # 2. forward pass
        loss   = criterion(output, y_batch)                # 3. compute loss
        loss.backward()                                    # 4. backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients
        optimizer.step()                                   # 5. update weights
        train_loss += loss.item()

    # ── Validate ──────────────────────────────
    model.eval()
    val_loss = 0
    correct  = 0
    total    = 0
    with torch.no_grad():   # no gradients needed for validation
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            output  = model(X_batch)
            loss    = criterion(output, y_batch)
            val_loss += loss.item()
            preds     = (torch.sigmoid(output) > 0.5).float()
            correct  += (preds == y_batch).sum().item()
            total    += y_batch.size(0)

    scheduler.step()
    avg_vloss = val_loss / len(test_loader)
    val_acc   = correct / total

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Loss: {avg_vloss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping
    if avg_vloss < best_loss:
        best_loss = avg_vloss
        counter   = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load("best_model.pt", weights_only=True))
print("Best model loaded!")


# ══════════════════════════════════════════════════════
# 5. CNN IN PYTORCH
# ══════════════════════════════════════════════════════
# WHAT IS nn.Conv2d?
#   → Convolutional layer: apply small filters (kernels) across the image
#   → nn.Conv2d(in_channels, out_channels, kernel_size, padding)
#   → in_channels : RGB image = 3 channels, grayscale = 1
#   → out_channels: how many filters to learn (each learns a different feature)
#
# SHAPE TRACKING:
#   → Input:  (batch, 3, 32, 32)   → 3-channel 32×32 image
#   → Conv2D(3→32) + Pool → (batch, 32, 16, 16)
#   → Conv2D(32→64) + Pool → (batch, 64, 8, 8)
#   → Conv2D(64→128) + Pool → (batch, 128, 4, 4)
#   → Flatten → (batch, 128*4*4=2048)
#   → Linear → (batch, 10)
#
# x.view(x.size(0), -1) = Flatten:
#   → x.size(0) = batch size, -1 = infer remaining dimensions

print("\n" + "=" * 55)
print("5. CNN")
print("=" * 55)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3,32,32)→(32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (32,16,16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (64,8,8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (128,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten spatial dims
        return self.classifier(x)

cnn   = CNN(num_classes=10).to(device)
dummy = torch.randn(4, 3, 32, 32).to(device)   # batch of 4 images (3 channels, 32×32)
out   = cnn(dummy)
print(f"CNN output shape: {out.shape}")   # torch.Size([4, 10])


# ══════════════════════════════════════════════════════
# 6. LSTM IN PYTORCH
# ══════════════════════════════════════════════════════
# WHAT IS nn.LSTM?
#   → Built-in LSTM layer — much faster than implementing from scratch
#   → nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
#   → batch_first=True  : input shape (batch, seq_len, features) — most natural
#   → bidirectional=True: two LSTMs — one forward, one backward
#   → Returns: output (all timestep hidden states), (h_n, c_n) (final states)
#
# TAKING FINAL HIDDEN STATE:
#   → hidden[-2] = final forward direction
#   → hidden[-1] = final backward direction
#   → torch.cat([hidden[-2], hidden[-1]], dim=1) = concatenate both → full context

print("\n" + "=" * 55)
print("6. LSTM")
print("=" * 55)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=2, dropout=0.3)
        self.fc        = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout   = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))         # (batch, seq, embed)
        output, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) # bidirectional final state
        return self.fc(self.dropout(hidden))

lstm      = LSTMClassifier(vocab_size=10000, embed_dim=64, hidden_dim=128, output_dim=1).to(device)
dummy_seq = torch.randint(0, 10000, (8, 200)).to(device)   # batch=8, seq_len=200
out       = lstm(dummy_seq)
print(f"LSTM output shape: {out.shape}")   # torch.Size([8, 1])


# ══════════════════════════════════════════════════════
# 7. SAVE & LOAD MODELS
# ══════════════════════════════════════════════════════
# WHAT TO SAVE:
#   → state_dict (RECOMMENDED): only the weights — no code dependency
#   → Full model             : saves architecture + weights (fragile if code changes)
#   → TorchScript            : compile model for production (no Python needed)
#
# BEST PRACTICE:
#   → Save: torch.save(model.state_dict(), "model.pt")
#   → Load: model.load_state_dict(torch.load("model.pt", map_location=device))
#   → Always pass weights_only=True to avoid security warnings in newer PyTorch

print("\n" + "=" * 55)
print("7. SAVE & LOAD")
print("=" * 55)

torch.save(model.state_dict(), "model_weights.pt")

new_model = FeedForward(20, [128, 64, 32], 1).to(device)
new_model.load_state_dict(torch.load("model_weights.pt",
                                      map_location=device, weights_only=True))
new_model.eval()
print("State dict loaded successfully!")

torch.save(model, "full_model.pt")
loaded = torch.load("full_model.pt", map_location=device, weights_only=False)
print("Full model loaded!")

scripted = torch.jit.script(model)
scripted.save("scripted_model.pt")
print("TorchScript model saved!")


# ══════════════════════════════════════════════════════
# 8. INFERENCE — Making Predictions
# ══════════════════════════════════════════════════════
# WHAT IS INFERENCE?
#   → Using the trained model to predict on new, unseen data
#   → model.eval()    : disable Dropout and BatchNorm training mode
#   → torch.no_grad() : don't track operations for gradient — faster, less memory
#   → torch.sigmoid() : convert raw logit → probability [0,1] for binary classification
#   → argmax()        : pick the class with highest probability (multi-class)

print("\n" + "=" * 55)
print("8. INFERENCE")
print("=" * 55)

model.eval()
sample = torch.randn(1, 20).to(device)

with torch.no_grad():
    logit = model(sample)
    prob  = torch.sigmoid(logit)
    pred  = (prob > 0.5).int()
    print(f"Logit: {logit.item():.4f}")
    print(f"Probability: {prob.item():.4f}")
    print(f"Prediction: {pred.item()}")   # 0 or 1

print("\nAll done! ✓")
