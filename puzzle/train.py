from puzzle.dataset import SaeDataSet
from puzzle.model import VAE_gumbel, device, CATEGORICAL_DIM
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np

TEMP_BEGIN = 5
TEMP_MIN = 0.7
ANNEAL_RATE = 0.05
TRAIN_BZ = 100
TEST_BZ = 500

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=(1,2,3)).mean()

    # log_qy = torch.log(qy + 1e-20)
    # g = torch.log(torch.Tensor([1.0 / CATEGORICAL_DIM])).to(device)
    # KLD = torch.sum(qy * (log_qy - g), dim=(-2, -1)).mean() # maximize the kl-divergence

    return BCE

def train(dataloader, vae, temp, optimizer):
    vae.train()
    train_loss = 0
    for i, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, qy, _ = vae(data, temp)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader)

def test(dataloader, vae, temp=TEMP_MIN):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            recon_batch, qy, _ = vae(data, temp)
            loss = loss_function(recon_batch, data, qy)
            test_loss += loss.item()
    return test_loss / len(dataloader)

def run(n_epoch):
    train_set = SaeDataSet(is_train=True)
    test_set = SaeDataSet(is_train=False)
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = VAE_gumbel().to(device)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}".format(e, temp))
        train_loss = train(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = test(test_loader, vae)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "puzzle/model/0.pth")
            best_loss = test_loss



if __name__ == "__main__":
    run(1000)