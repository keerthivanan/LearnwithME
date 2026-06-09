"""
GANs + Autoencoders — Definitions + Code + Outputs
====================================================
pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ══════════════════════════════════════════════════════
# 1. AUTOENCODER — Compress & Reconstruct
# ══════════════════════════════════════════════════════
# WHAT IS AN AUTOENCODER?
#   → A neural network that learns to COMPRESS data and then RECONSTRUCT it
#   → Two parts:
#     - Encoder: Input → Latent Space (small compressed representation)
#     - Decoder: Latent Space → Reconstructed Input
#   → Trained to minimize reconstruction error (MSE Loss)
#
# LATENT SPACE (BOTTLENECK):
#   → The compressed representation (e.g., 784 → 32 numbers)
#   → Forces the model to learn the MOST IMPORTANT features
#   → Similar to PCA but non-linear (can capture complex patterns)
#
# USE CASES:
#   → Dimensionality reduction (like PCA but non-linear)
#   → Anomaly detection: normal data → low reconstruction error
#                        anomalies → high reconstruction error
#   → Denoising: train with noisy input, clean target
#   → Image compression, feature extraction

print("\n" + "=" * 55)
print("1. AUTOENCODER")
print("=" * 55)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)   # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()    # output in [0,1] for pixel values
        )

    def forward(self, x):
        z     = self.encoder(x)      # compress 784 → 32
        recon = self.decoder(z)      # reconstruct 32 → 784
        return recon, z

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))   # flatten 28x28 → 784
])
train_data = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False)

ae       = Autoencoder(input_dim=784, latent_dim=32).to(device)
ae_optim = optim.Adam(ae.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

print("Training Autoencoder...")
for epoch in range(5):
    ae.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        recon, z = ae(imgs)
        loss = mse_loss(recon, imgs)

        ae_optim.zero_grad()
        loss.backward()
        ae_optim.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f}")

# Anomaly detection — high reconstruction error = anomaly
ae.eval()
recon_errors = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs  = imgs.to(device)
        recon, _ = ae(imgs)
        errors = ((recon - imgs) ** 2).mean(dim=1)
        recon_errors.extend(errors.cpu().numpy())

threshold = np.percentile(recon_errors, 95)   # top 5% = anomaly
print(f"\nAnomaly threshold (95th percentile): {threshold:.4f}")
print(f"Anomalies detected: {sum(e > threshold for e in recon_errors)}")
# Anomaly threshold: ~0.05; Anomalies: ~500 (5% of 10000)


# ══════════════════════════════════════════════════════
# 2. VAE — Variational Autoencoder
# ══════════════════════════════════════════════════════
# WHAT IS A VAE?
#   → Upgrade to regular autoencoder — can GENERATE new samples
#   → Regular AE: Input → z (a fixed point)
#   → VAE       : Input → μ (mean) + σ (std) → z ~ N(μ, σ²) (a distribution!)
#
# WHY A DISTRIBUTION?
#   → Regular AE latent space has "gaps" — random points decode to garbage
#   → VAE forces latent space to be smooth and continuous (like N(0,1))
#   → Any point in N(0,1) can be decoded to a meaningful output!
#
# REPARAMETERIZATION TRICK:
#   → Problem: sampling z ~ N(μ, σ²) is not differentiable
#   → Solution: z = μ + ε · σ  where ε ~ N(0,1)
#   → Now gradients can flow through μ and σ!
#
# VAE LOSS = Reconstruction Loss + KL Divergence
#   → Reconstruction: how well we reconstruct the input (MSE or BCE)
#   → KL Divergence : how much learned distribution differs from N(0,1)
#                     KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

print("\n" + "=" * 55)
print("2. VAE — VARIATIONAL AUTOENCODER")
print("=" * 55)

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder    = nn.Sequential(nn.Linear(input_dim, 400), nn.ReLU())
        self.fc_mu      = nn.Linear(400, latent_dim)   # mean vector
        self.fc_log_var = nn.Linear(400, latent_dim)   # log variance (log for stability)
        self.decoder    = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)   # σ = e^(log σ / 2)
        eps = torch.randn_like(std)       # ε ~ N(0,1)
        return mu + eps * std             # z = μ + ε·σ

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def vae_loss(recon, x, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

vae       = VAE(784, latent_dim=20).to(device)
vae_optim = optim.Adam(vae.parameters(), lr=1e-3)

print("Training VAE...")
for epoch in range(5):
    vae.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        recon, mu, log_var = vae(imgs)
        loss = vae_loss(recon, imgs, mu, log_var)

        vae_optim.zero_grad()
        loss.backward()
        vae_optim.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader.dataset):.2f}")

# Generate NEW samples from random latent vectors
vae.eval()
with torch.no_grad():
    z_sample  = torch.randn(16, 20).to(device)   # sample from N(0,1)
    generated = vae.decoder(z_sample)
    print(f"Generated samples shape: {generated.shape}")   # (16, 784)


# ══════════════════════════════════════════════════════
# 3. GAN — Generative Adversarial Network
# ══════════════════════════════════════════════════════
# WHAT IS A GAN?
#   → Two networks competing against each other:
#     - Generator (G)     : takes random noise → produces FAKE images
#     - Discriminator (D) : classifies images as REAL or FAKE
#
# HOW TRAINING WORKS:
#   → Train D: maximize P(real=1) + P(fake=0)
#     - Show D real images → predict 1
#     - Show D G's fakes → predict 0
#   → Train G: fool D — maximize P(fake=1)
#     - Show D fakes → want D to output 1
#   → They improve together: G makes better fakes, D catches better fakes
#   → Equilibrium: G generates perfect fakes, D is 50/50 (can't tell real from fake)
#
# KEY TRICKS:
#   → LeakyReLU (not ReLU) in Discriminator — prevents "dead neurons"
#   → Label smoothing: use 0.9 for real instead of 1.0
#   → .detach() when training D with fake images — don't backprop through G
#   → Adam with betas=(0.5, 0.999) — standard GAN optimizer setting

print("\n" + "=" * 55)
print("3. GAN — GENERATIVE ADVERSARIAL NETWORK")
print("=" * 55)

LATENT_DIM = 100
IMG_DIM    = 784

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()    # output in [-1, 1] — matches normalized images
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()   # output: probability of being real
        )
    def forward(self, x): return self.net(x)

G = Generator(LATENT_DIM, IMG_DIM).to(device)
D = Discriminator(IMG_DIM).to(device)

g_optim      = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optim      = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion_bce = nn.BCELoss()

transform_gan  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),          # normalize to [-1, 1]
    transforms.Lambda(lambda x: x.view(-1))
])
loader_gan = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform_gan),
    batch_size=128, shuffle=True
)

print("Training GAN...")
for epoch in range(5):
    d_losses, g_losses = [], []
    for real_imgs, _ in loader_gan:
        real_imgs  = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ── Step 1: Train Discriminator ──────────────
        D.zero_grad()
        d_real = D(real_imgs)
        loss_real = criterion_bce(d_real, real_labels)         # real → 1

        z         = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = G(z).detach()                               # .detach() = don't train G here
        d_fake    = D(fake_imgs)
        loss_fake = criterion_bce(d_fake, fake_labels)         # fake → 0

        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_optim.step()
        d_losses.append(d_loss.item())

        # ── Step 2: Train Generator ──────────────────
        G.zero_grad()
        z         = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = G(z)
        d_output  = D(fake_imgs)
        g_loss    = criterion_bce(d_output, real_labels)       # fool D: fake → 1

        g_loss.backward()
        g_optim.step()
        g_losses.append(g_loss.item())

    print(f"  Epoch {epoch+1}/5 | D Loss: {np.mean(d_losses):.4f} | G Loss: {np.mean(g_losses):.4f}")

G.eval()
with torch.no_grad():
    z    = torch.randn(16, LATENT_DIM).to(device)
    fake = G(z).reshape(16, 28, 28)
    print(f"Generated fake images shape: {fake.shape}")   # torch.Size([16, 28, 28])


# ══════════════════════════════════════════════════════
# 4. DCGAN — Deep Convolutional GAN
# ══════════════════════════════════════════════════════
# WHAT IS DCGAN?
#   → GAN with convolutional layers instead of fully-connected layers
#   → Much better at generating images (CNN understands spatial structure)
#
# ConvTranspose2d (Transposed Convolution / "Deconvolution"):
#   → The REVERSE of Conv2d — UPSAMPLES the spatial dimensions
#   → Used in Generator to go from small → large (e.g., 7×7 → 28×28)
#   → ConvTranspose2d(in, out, kernel, stride, padding)
#     stride=2 → doubles the spatial size
#
# DCGAN ARCHITECTURE:
#   → Generator   : ConvTranspose2d layers (upsample noise → image)
#   → Discriminator: Conv2d layers (downsample image → real/fake score)

print("\n" + "=" * 55)
print("4. DCGAN — DEEP CONVOLUTIONAL GAN")
print("=" * 55)

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),  # → (256, 7, 7)
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),          # → (128, 14, 14)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),     # → (1, 28, 28)
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.unsqueeze(-1).unsqueeze(-1))   # add spatial dims for conv

dcgan_g = DCGANGenerator(latent_dim=100, channels=1).to(device)
z_test  = torch.randn(4, 100).to(device)
out     = dcgan_g(z_test)
print(f"DCGAN Generator output: {out.shape}")   # torch.Size([4, 1, 28, 28])

print("\nAll done! ✓")
