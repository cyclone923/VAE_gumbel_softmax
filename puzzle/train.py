from puzzle.dataset import SaeDataSet, FoSaeDataSet
from puzzle.sae import Sae
from puzzle.fosae import FoSae
from puzzle.gumble import device
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

TEMP_BEGIN = 5
TEMP_MIN = 0.7
ANNEAL_RATE = 0.03
TRAIN_BZ = 100
TEST_BZ = 720

fo_logic = True

if not fo_logic:
    MODEL_NAME = "Sae"
    DATASET_NAME = "SaeDataSet"
else:
    MODEL_NAME = "FoSae"
    DATASET_NAME = "FoSaeDataSet"

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    sum_dim = [i for i in range(1, x.dim())]
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=sum_dim).mean()
    return BCE

def train(dataloader, vae, temp, optimizer):
    vae.train()
    train_loss = 0
    for i, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        noise = torch.normal(mean=0, std=0.4, size=data.size()).to(device)
        recon_batch, _ , _ = vae(data+noise, temp)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader)

def test(dataloader, vae, temp=0):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            recon_batch, _, _ = vae(data, temp)
            loss = loss_function(recon_batch, data)
            test_loss += loss.item()
    return test_loss / len(dataloader)


def run(n_epoch):
    train_set = eval(DATASET_NAME)(is_train=True)
    test_set = eval(DATASET_NAME)(is_train=False)
    print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1.0 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_lr()))
        train_loss = train(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = test(test_loader, vae)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "puzzle/model/{}.pth".format(MODEL_NAME))
            best_loss = test_loss
        scheculer.step()



if __name__ == "__main__":
    run(1000)