"""
GANs + Autoencoders — Everything
==================================
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

# ════════════════════════════════════════════
# 1. AUTOENCODER — Compress & Reconstruct
# ════════════════════════════════════════════
print("\n1. AUTOENCODER")
#
#  Input → Encoder → Latent Space (small) → Decoder → Reconstructed Input
#
#  Use for:
#  - Dimensionality reduction (like PCA but non-linear)
#  - Anomaly detection (high reconstruction error = anomaly)
#  - Denoising (train to reconstruct clean from noisy)
#  - Image compression

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()

        # Encoder: compress 784 → 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: reconstruct 32 → 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()            # output in [0,1] for images
        )

    def forward(self, x):
        z = self.encoder(x)         # compress
        x_recon = self.decoder(z)   # reconstruct
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 → 784
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False)

# Train Autoencoder
ae        = Autoencoder(input_dim=784, latent_dim=32).to(device)
ae_optim  = optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training Autoencoder...")
for epoch in range(5):
    ae.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        recon, z = ae(imgs)
        loss = criterion(recon, imgs)

        ae_optim.zero_grad()
        loss.backward()
        ae_optim.step()
        total_loss += loss.item()

    print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f}")

# Anomaly Detection with Autoencoder
ae.eval()
recon_errors = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs  = imgs.to(device)
        recon, _ = ae(imgs)
        errors = ((recon - imgs) ** 2).mean(dim=1)   # per-sample error
        recon_errors.extend(errors.cpu().numpy())

threshold = np.percentile(recon_errors, 95)   # top 5% = anomaly
print(f"\nAnomaly threshold: {threshold:.4f}")
print(f"Anomalies detected: {sum(e > threshold for e in recon_errors)}")

# ════════════════════════════════════════════
# 2. VAE — Variational Autoencoder
# ════════════════════════════════════════════
print("\n2. VAE — VARIATIONAL AUTOENCODER")
#
#  Regular AE:  Input → z (deterministic point)
#  VAE:         Input → μ + σ → z ~ N(μ, σ²)  (latent distribution)
#
#  VAE learns a DISTRIBUTION in latent space — can GENERATE new samples!

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder → outputs mean and log variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )
        self.fc_mu      = nn.Linear(400, latent_dim)  # mean
        self.fc_log_var = nn.Linear(400, latent_dim)  # log variance

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        log_var= self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Trick: z = μ + ε*σ  where ε ~ N(0,1)
        # This makes backprop possible through random sampling!
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z)
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var):
    # Reconstruction loss (BCE or MSE)
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence: how much does learned distribution differ from N(0,1)?
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

vae       = VAE(784, latent_dim=20).to(device)
vae_optim = optim.Adam(vae.parameters(), lr=1e-3)

print("Training VAE...")
for epoch in range(5):
    vae.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs  = imgs.to(device)
        recon, mu, log_var = vae(imgs)
        loss = vae_loss(recon, imgs, mu, log_var)

        vae_optim.zero_grad()
        loss.backward()
        vae_optim.step()
        total_loss += loss.item()

    print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader.dataset):.2f}")

# Generate new samples!
vae.eval()
with torch.no_grad():
    z_sample = torch.randn(16, 20).to(device)   # sample from N(0,1)
    generated = vae.decode(z_sample)             # decode to image space
    print(f"Generated samples shape: {generated.shape}")  # (16, 784)

# ════════════════════════════════════════════
# 3. GAN — Generative Adversarial Network
# ════════════════════════════════════════════
print("\n3. GAN — GENERATIVE ADVERSARIAL NETWORK")
#
#  Generator:     random noise → fake images (tries to fool discriminator)
#  Discriminator: real or fake? (tries to catch the generator)
#
#  Training: Generator and Discriminator play a minimax game
#  Generator improves → better fakes
#  Discriminator improves → catches better fakes
#  Equilibrium: Generator makes perfect fakes

LATENT_DIM = 100
IMG_DIM    = 784   # 28x28 MNIST

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
            nn.Tanh()    # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


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
            nn.Sigmoid()   # real (1) or fake (0)
        )

    def forward(self, x):
        return self.net(x)


G = Generator(LATENT_DIM, IMG_DIM).to(device)
D = Discriminator(IMG_DIM).to(device)

g_optim = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optim = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion_gan = nn.BCELoss()

# Normalize images to [-1, 1] for GAN
transform_gan = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.view(-1))
])
train_data_gan = datasets.MNIST("./data", train=True, download=True, transform=transform_gan)
loader_gan     = DataLoader(train_data_gan, batch_size=128, shuffle=True)

print("Training GAN...")
for epoch in range(5):
    d_losses, g_losses = [], []

    for real_imgs, _ in loader_gan:
        real_imgs  = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ── Train Discriminator ────────────────────
        D.zero_grad()
        d_real = D(real_imgs)
        d_loss_real = criterion_gan(d_real, real_labels)   # classify real as 1

        z         = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = G(z).detach()                          # don't train G here
        d_fake    = D(fake_imgs)
        d_loss_fake = criterion_gan(d_fake, fake_labels)   # classify fake as 0

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optim.step()
        d_losses.append(d_loss.item())

        # ── Train Generator ────────────────────────
        G.zero_grad()
        z         = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = G(z)
        d_output  = D(fake_imgs)
        g_loss    = criterion_gan(d_output, real_labels)   # fool D into thinking fake is real

        g_loss.backward()
        g_optim.step()
        g_losses.append(g_loss.item())

    print(f"  Epoch {epoch+1}/5 | D Loss: {np.mean(d_losses):.4f} | G Loss: {np.mean(g_losses):.4f}")

# Generate fake images
G.eval()
with torch.no_grad():
    z         = torch.randn(16, LATENT_DIM).to(device)
    fake_imgs = G(z).reshape(16, 28, 28)
    print(f"Generated fake images: {fake_imgs.shape}")  # (16, 28, 28)

# ════════════════════════════════════════════
# 4. DCGAN — Deep Convolutional GAN
# ════════════════════════════════════════════
print("\n4. DCGAN — CONVOLUTIONAL GAN")

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # latent → 7x7x256
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            # 7x7 → 14x14
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            # 14x14 → 28x28
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.unsqueeze(-1).unsqueeze(-1))  # reshape z for conv

dcgan_g = DCGANGenerator(latent_dim=100, channels=1).to(device)
z_test  = torch.randn(4, 100).to(device)
out     = dcgan_g(z_test)
print(f"DCGAN output: {out.shape}")   # torch.Size([4, 1, 28, 28])

print("\nAll done! ✓")
