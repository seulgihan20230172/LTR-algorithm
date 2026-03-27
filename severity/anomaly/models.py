import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest


def _to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


class _MLPEncoder(nn.Module):
    def __init__(self, d_in, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class _MLPDecoder(nn.Module):
    def __init__(self, latent_dim, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_out),
        )

    def forward(self, z):
        return self.net(z)


class VanillaAEModel:
    def __init__(self, epochs=25, batch_size=256, lr=1e-3, random_state=42):
        _ = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.enc = None
        self.dec = None

    def fit(self, x_train, x_val=None):
        d = x_train.shape[1]
        self.enc = _MLPEncoder(d, 32)
        self.dec = _MLPDecoder(32, d)
        model = nn.Sequential(self.enc, self.dec)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        x_t = _to_tensor(x_train)
        for _ in range(self.epochs):
            perm = torch.randperm(x_t.shape[0])
            for i in range(0, x_t.shape[0], self.batch_size):
                b = x_t[perm[i : i + self.batch_size]]
                r = model(b)
                loss = crit(r, b)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def score(self, x):
        with torch.no_grad():
            xt = _to_tensor(x)
            z = self.enc(xt)
            r = self.dec(z)
            e = torch.mean((r - xt) ** 2, dim=1)
        return e.numpy().astype(np.float64)


class DenoisingAEModel(VanillaAEModel):
    def __init__(self, epochs=25, batch_size=256, lr=1e-3, noise_std=0.05, random_state=42):
        super().__init__(epochs, batch_size, lr, random_state)
        self.noise_std = noise_std

    def fit(self, x_train, x_val=None):
        d = x_train.shape[1]
        self.enc = _MLPEncoder(d, 32)
        self.dec = _MLPDecoder(32, d)
        model = nn.Sequential(self.enc, self.dec)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        x_t = _to_tensor(x_train)
        for _ in range(self.epochs):
            perm = torch.randperm(x_t.shape[0])
            for i in range(0, x_t.shape[0], self.batch_size):
                b = x_t[perm[i : i + self.batch_size]]
                noisy = b + torch.randn_like(b) * self.noise_std
                r = model(noisy)
                loss = crit(r, b)
                opt.zero_grad()
                loss.backward()
                opt.step()


class _VAE(nn.Module):
    def __init__(self, d, latent=24):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent)
        self.d1 = nn.Linear(latent, 64)
        self.d2 = nn.Linear(64, 128)
        self.out = nn.Linear(128, d)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.d1(z))
        h = torch.relu(self.d2(h))
        return self.out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


class VAEModel:
    def __init__(self, epochs=30, batch_size=256, lr=1e-3, beta=0.01, random_state=42):
        _ = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.model = None

    def fit(self, x_train, x_val=None):
        d = x_train.shape[1]
        self.model = _VAE(d)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        xt = _to_tensor(x_train)
        for _ in range(self.epochs):
            perm = torch.randperm(xt.shape[0])
            for i in range(0, xt.shape[0], self.batch_size):
                b = xt[perm[i : i + self.batch_size]]
                rec, mu, logvar = self.model(b)
                rec_loss = torch.mean((rec - b) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec_loss + self.beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()

    def score(self, x):
        with torch.no_grad():
            xt = _to_tensor(x)
            rec, mu, logvar = self.model(xt)
            rec_e = torch.mean((rec - xt) ** 2, dim=1)
            kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1) / mu.shape[1]
            s = rec_e + self.beta * kl
        return s.numpy().astype(np.float64)


class SequenceAEModel:
    def __init__(self, epochs=20, batch_size=256, lr=1e-3, hidden=32, random_state=42):
        _ = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.enc = None
        self.dec = None

    def fit(self, x_train, x_val=None):
        d = x_train.shape[1]
        self.enc = nn.LSTM(input_size=1, hidden_size=self.hidden, batch_first=True)
        self.dec = nn.LSTM(input_size=self.hidden, hidden_size=1, batch_first=True)
        proj = nn.Linear(self.hidden, self.hidden)
        opt = optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()) + list(proj.parameters()), lr=self.lr)
        xt = _to_tensor(x_train).unsqueeze(-1)
        for _ in range(self.epochs):
            perm = torch.randperm(xt.shape[0])
            for i in range(0, xt.shape[0], self.batch_size):
                b = xt[perm[i : i + self.batch_size]]
                _, (h, _) = self.enc(b)
                z = h[-1]
                z_seq = proj(z).unsqueeze(1).repeat(1, d, 1)
                out, _ = self.dec(z_seq)
                loss = torch.mean((out - b) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.proj = proj

    def score(self, x):
        with torch.no_grad():
            xt = _to_tensor(x).unsqueeze(-1)
            d = xt.shape[1]
            _, (h, _) = self.enc(xt)
            z = h[-1]
            z_seq = self.proj(z).unsqueeze(1).repeat(1, d, 1)
            out, _ = self.dec(z_seq)
            e = torch.mean((out - xt) ** 2, dim=(1, 2))
        return e.numpy().astype(np.float64)


class DeepStackedAEModel(VanillaAEModel):
    def fit(self, x_train, x_val=None):
        d = x_train.shape[1]
        self.enc = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
        )
        self.dec = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, d),
        )
        model = nn.Sequential(self.enc, self.dec)
        opt = optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.MSELoss()
        xt = _to_tensor(x_train)
        for _ in range(self.epochs):
            perm = torch.randperm(xt.shape[0])
            for i in range(0, xt.shape[0], self.batch_size):
                b = xt[perm[i : i + self.batch_size]]
                r = model(b)
                loss = crit(r, b)
                opt.zero_grad()
                loss.backward()
                opt.step()


class IsolationForestModel:
    def __init__(self, random_state=42):
        self.model = IsolationForest(n_estimators=400, contamination="auto", random_state=random_state, n_jobs=-1)

    def fit(self, x_train, x_val=None):
        _ = x_val
        self.model.fit(x_train)

    def score(self, x):
        return (-self.model.score_samples(x)).astype(np.float64)


def build_anomaly_model(model_name: str, epochs: int, random_state: int = 42):
    if model_name == "vanilla_ae":
        return VanillaAEModel(epochs=epochs, random_state=random_state)
    if model_name == "denoising_ae":
        return DenoisingAEModel(epochs=epochs, random_state=random_state)
    if model_name == "vae":
        return VAEModel(epochs=epochs, random_state=random_state)
    if model_name == "sequence_ae":
        return SequenceAEModel(epochs=epochs, random_state=random_state)
    if model_name == "deep_stacked_ae":
        return DeepStackedAEModel(epochs=epochs, random_state=random_state)
    if model_name == "isolation_forest":
        return IsolationForestModel(random_state=random_state)
    raise ValueError(f"지원하지 않는 anomaly 모델: {model_name}")

