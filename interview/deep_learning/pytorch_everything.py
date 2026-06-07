"""
PyTorch — Everything
=====================
pip install torch torchvision torchaudio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np

print(f"PyTorch: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ════════════════════════════════════════════
# 1. TENSORS — PyTorch Basics
# ════════════════════════════════════════════
print("\n1. TENSORS")

# Create tensors
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.zeros(3, 4)
t3 = torch.ones(2, 3)
t4 = torch.randn(3, 3)           # random normal
t5 = torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
t6 = torch.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1]

print(t1.shape, t1.dtype)        # torch.Size([3]) torch.float32
print(t4.shape, t4.device)       # torch.Size([3, 3]) cpu

# Operations
a = torch.tensor([[1,2],[3,4]], dtype=torch.float)
b = torch.tensor([[5,6],[7,8]], dtype=torch.float)

print(a + b)                     # element-wise add
print(a @ b)                     # matrix multiply → [[19,22],[43,50]]
print(a.T)                       # transpose
print(a.reshape(4))              # reshape → [1,2,3,4]
print(a.sum(), a.mean(), a.max())  # 10, 2.5, 4

# NumPy ↔ PyTorch
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)      # numpy → tensor (shared memory!)
arr_back = t.numpy()             # tensor → numpy

# Move to GPU
x = torch.randn(3, 3).to(device)

# Gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x              # y = x³ + 2x
y.backward()                     # compute gradients
print(f"dy/dx at x=2: {x.grad}")  # 3x² + 2 = 3(4) + 2 = 14.0

# ════════════════════════════════════════════
# 2. CUSTOM DATASET
# ════════════════════════════════════════════
print("\n2. CUSTOM DATASET")

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

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X = StandardScaler().fit_transform(X)

dataset  = TabularDataset(X, y)
n_train  = int(0.8 * len(dataset))
n_test   = len(dataset) - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ════════════════════════════════════════════
# 3. FEEDFORWARD NEURAL NETWORK
# ════════════════════════════════════════════
print("\n3. FEEDFORWARD NEURAL NETWORK")

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = FeedForward(20, [128, 64, 32], 1, dropout=0.3).to(device)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ════════════════════════════════════════════
# 4. TRAINING LOOP
# ════════════════════════════════════════════
print("\n4. TRAINING LOOP")

optimizer  = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion  = nn.BCEWithLogitsLoss()

best_loss = float("inf")
patience  = 5
counter   = 0

for epoch in range(30):
    # ── Train ────────────────────────────────
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        train_loss += loss.item()

    # ── Validate ──────────────────────────────
    model.eval()
    val_loss = 0
    correct  = 0
    total    = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            output  = model(X_batch)
            loss    = criterion(output, y_batch)
            val_loss  += loss.item()
            preds      = (torch.sigmoid(output) > 0.5).float()
            correct   += (preds == y_batch).sum().item()
            total     += y_batch.size(0)

    scheduler.step()
    val_acc  = correct / total
    avg_vloss = val_loss / len(test_loader)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_vloss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping
    if avg_vloss < best_loss:
        best_loss = avg_vloss
        counter   = 0
        torch.save(model.state_dict(), "best_model.pt")  # save best
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
print("Best model loaded!")

# ════════════════════════════════════════════
# 5. CNN IN PYTORCH
# ════════════════════════════════════════════
print("\n5. CNN")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3,32,32)→(32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # → (32,16,16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # → (64,8,8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # → (128,4,4)
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
        x = x.view(x.size(0), -1)   # flatten
        return self.classifier(x)

cnn = CNN(num_classes=10).to(device)
dummy = torch.randn(4, 3, 32, 32).to(device)   # batch of 4 images
out   = cnn(dummy)
print(f"CNN output shape: {out.shape}")   # torch.Size([4, 10])

# ════════════════════════════════════════════
# 6. LSTM
# ════════════════════════════════════════════
print("\n6. LSTM")

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                  bidirectional=True, num_layers=2, dropout=0.3)
        self.fc        = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidir
        self.dropout   = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, _) = self.lstm(embedded)
        # Take final hidden state from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden))

lstm = LSTMClassifier(vocab_size=10000, embed_dim=64,
                       hidden_dim=128, output_dim=1).to(device)
dummy_seq = torch.randint(0, 10000, (8, 200)).to(device)  # batch=8, seq_len=200
out = lstm(dummy_seq)
print(f"LSTM output shape: {out.shape}")  # torch.Size([8, 1])

# ════════════════════════════════════════════
# 7. SAVE & LOAD
# ════════════════════════════════════════════
print("\n7. SAVE & LOAD")

# Save state dict (recommended)
torch.save(model.state_dict(), "model_weights.pt")

# Load
new_model = FeedForward(20, [128, 64, 32], 1).to(device)
new_model.load_state_dict(torch.load("model_weights.pt", map_location=device))
new_model.eval()
print("Model loaded successfully!")

# Save full model
torch.save(model, "full_model.pt")
loaded = torch.load("full_model.pt", map_location=device)

# TorchScript (for production)
scripted = torch.jit.script(model)
scripted.save("scripted_model.pt")
print("TorchScript saved!")

# ════════════════════════════════════════════
# 8. INFERENCE (no gradient, fast)
# ════════════════════════════════════════════
print("\n8. INFERENCE")

model.eval()
sample = torch.randn(1, 20).to(device)

with torch.no_grad():
    logit = model(sample)
    prob  = torch.sigmoid(logit)
    pred  = (prob > 0.5).int()
    print(f"Probability: {prob.item():.4f}, Prediction: {pred.item()}")

print("\nAll done! ✓")
