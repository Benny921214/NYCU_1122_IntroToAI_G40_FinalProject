import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, vocab_size, hidden_dim, z_dim, pad_idx):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, z_dim)
        self.fc_z = nn.Linear(z_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding(x)
        h, _ = self.encoder_lstm(embedded)
        h = torch.cat((h[:, -1, :hidden_dim], h[:, 0, hidden_dim:]), dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        z = self.fc_z(z).unsqueeze(1).repeat(1, seq_len, 1)
        h, _ = self.decoder_lstm(z)
        return self.fc_out(h)

    def forward(self, x, seq_len):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, seq_len)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=pad_idx)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

vocab_size = len(vocab)
hidden_dim = 256
z_dim = 100
pad_idx = vocab['<pad>']
vae = VAE(vocab_size, hidden_dim, z_dim, pad_idx)

optimizer = optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for x in dataloader:
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x, x.size(1))
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
