from puzzle.dataset import SaeDataSet
from puzzle.model import VAE_gumbel, device, CATEGORICAL_DIM
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np

TEMP_BEGIN = 1
TEMP_MIN = 0.5
ANNEAL_RATE = 0.05

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=(1,2,3)).mean()

    log_qy = torch.log(qy + 1e-20)
    g = torch.log(torch.Tensor([1.0 / CATEGORICAL_DIM])).to(device)
    KLD = torch.sum(qy * (log_qy - g), dim=(-2, -1)).mean() # maximize the kl-divergence

    return BCE - KLD

def train(dataloader, vae,  optimizer, temp):
    vae.train()
    train_loss = 0
    for i, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, qy = vae(data, temp)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader.dataset)

def run(n_epoch):
    train_loader = DataLoader(SaeDataSet(is_train=True), batch_size=64, shuffle=True)
    vae = VAE_gumbel().to(device)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        train_loss = train(train_loader, vae, optimizer, temp)
        print('====> Epoch: {} Average loss: {:.4f}'.format(e, train_loss))
    torch.save(vae.state_dict(), "puzzle/model/0.pth")



if __name__ == "__main__":
    run(10)