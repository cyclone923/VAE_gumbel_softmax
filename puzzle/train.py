from puzzle.sae import CubeSae
from puzzle.dataset import get_train_and_test_dataset, load_data
from puzzle.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

TEMP_BEGIN_SAE = 5
TEMP_MIN_SAE = 0.7
TEMP_BEGIN_AAE = 5
TEMP_MIN_AAE = 0.1
ANNEAL_RATE = 0.03
TRAIN_BZ = 2000
TEST_BZ = 2000
ALPHA = 0.7

MODEL_NAME = "CubeSae"

torch.manual_seed(0)

# Reconstruction + zero suppressed losses summed over all elements and batch
def rec_loss_function(recon_x, x, criterion):
    BCE = criterion(recon_x, x).sum(dim=[i for i in range(1, x.dim())]).mean()
    # sparsity = latent_z.sum(dim=[i for i in range(1, latent_z.dim())]).mean()
    return BCE

def latent_spasity(z):
    return z.sum(dim=[i for i in range(1, z.dim())]).mean() * ALPHA

def total_loss(output, o1, o2):
    recon_o1, recon_o2, z1, z2, recon_z2 = output
    image_loss = 0
    latent_loss = 0
    spasity = 0
    image_loss += rec_loss_function(recon_o1, o1, nn.BCELoss(reduction='none'))
    image_loss += rec_loss_function(recon_o2, o2, nn.BCELoss(reduction='none'))
    latent_loss += rec_loss_function(recon_z2, z2, nn.MSELoss(reduction='none'))
    spasity += latent_spasity(z1)
    spasity += latent_spasity(z2)
    return image_loss, latent_loss, spasity


def train(dataloader, vae, optimizer, temp, add_spasity):
    vae.train()
    train_loss = 0
    ep_image_loss, ep_latent_loss, ep_spasity = 0, 0, 0
    for i, (data, data_next) in enumerate(dataloader):
        o1 = data.to(device)
        o2 = data_next.to(device)
        optimizer.zero_grad()
        noise1 = torch.normal(mean=0, std=0.4, size=o1.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.4, size=o2.size()).to(device)
        output = vae(o1+noise1, o2+noise2, temp)
        image_loss, latent_loss, spasity = total_loss(output, o1, o2)
        ep_image_loss += image_loss.item()
        ep_latent_loss += latent_loss.item()
        ep_spasity += spasity.item()
        loss = image_loss + latent_loss
        if add_spasity:
            loss += spasity
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print("TRAINING LOSS REC_IMG: {:.3f}, REC_LATENT: {:.3f}, SPASITY_LATENT: {:.3f}".format(
        ep_image_loss/len(dataloader), ep_latent_loss/len(dataloader), ep_spasity/len(dataloader))
    )
    return train_loss / len(dataloader)

def test(dataloader, vae):
    temp = (0,0)
    vae.eval()
    test_loss = 0
    ep_image_loss, ep_latent_loss, ep_spasity = 0, 0, 0
    with torch.no_grad():
        for i, (data, data_next) in enumerate(dataloader):
            o1 = data.to(device)
            o2 = data_next.to(device)
            noise1 = torch.normal(mean=0, std=0.4, size=o1.size()).to(device)
            noise2 = torch.normal(mean=0, std=0.4, size=o2.size()).to(device)
            output = vae(o1 + noise1, o2 + noise2, temp)
            image_loss, latent_loss, spasity = total_loss(output, o1, o2)
            ep_image_loss += image_loss.item()
            ep_latent_loss += latent_loss.item()
            ep_spasity += spasity.item()
            loss = image_loss + latent_loss + spasity
            test_loss += loss.item()
    print("TESTING LOSS REC_IMG: {:.3f}, REC_LATENT: {:.3f}, SPASITY_LATENT: {:.3f}".format(
        ep_image_loss/len(dataloader), ep_latent_loss/len(dataloader), ep_spasity/len(dataloader))
    )
    return test_loss / len(dataloader)

def load_model(vae):
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format(MODEL_NAME)))
    print("puzzle/model/{}.pth loaded".format(MODEL_NAME))


def run(n_epoch):
    train_set, test_set, _, _ = get_train_and_test_dataset(*load_data())
    print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True, num_workers=4)
    vae = CubeSae().to(device)
    # load_model(vae)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1.0 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp1 = np.maximum(TEMP_BEGIN_SAE * np.exp(-ANNEAL_RATE * e), TEMP_MIN_SAE)
        temp2 = np.maximum(TEMP_BEGIN_AAE * np.exp(-ANNEAL_RATE * e), TEMP_MIN_AAE)
        print("Epoch: {}, Temperature: {:.2f} {:.2f}, Lr: {}".format(e, temp1, temp2, scheculer.get_last_lr()))
        train_loss = train(train_loader, vae, optimizer, (temp1, temp2), e >= 20)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = test(test_loader, vae)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "puzzle/model/{}.pth".format(MODEL_NAME))
            best_loss = test_loss
        scheculer.step()



if __name__ == "__main__":
    run(300)