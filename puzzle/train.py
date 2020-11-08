import torch
import torch.nn as nn
from puzzle.sae import CubeSae, LATENT_DIM, N_ACTION
from puzzle.dataset import get_train_and_test_dataset, load_data, TEST_EXAMPLES
from puzzle.gumble import device
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

TEMP_BEGIN_SAE = 5
TEMP_MIN_SAE = 0.7
TEMP_BEGIN_AAE = 5
TEMP_MIN_AAE = 0.5
ANNEAL_RATE = 0.01
TRAIN_BZ = 1000
TEST_BZ = 2000
ALPHA = 1
BETA = 1
LATENT_DIM_SQRT = int(np.sqrt(LATENT_DIM))
N_ACTION_SQTR = int(np.sqrt(N_ACTION))

MODEL_NAME = "CubeSae"

torch.manual_seed(0)

# Reconstruction + zero suppressed losses summed over all elements and batch
def rec_loss_function(recon_x, x, criterion):
    BCE = criterion(recon_x, x).sum(dim=[i for i in range(1, x.dim())]).mean()
    return BCE

def latent_spasity(z):
    return z.sum(dim=[i for i in range(1, z.dim())]).mean() * ALPHA

def total_loss(output, o1, o2):
    recon_o1, recon_o2, recon_o2_tilde, z1, z2, recon_z2, _ = output
    image_loss = 0
    latent_loss = 0
    spasity = 0
    image_loss += rec_loss_function(recon_o1, o1, nn.BCELoss(reduction='none'))
    image_loss += rec_loss_function(recon_o2, o2, nn.BCELoss(reduction='none'))
    image_loss += rec_loss_function(recon_o2_tilde, o2, nn.BCELoss(reduction='none'))
    latent_loss += rec_loss_function(recon_z2, z2, nn.L1Loss(reduction='none')) * BETA
    spasity += latent_spasity(z1)
    spasity += latent_spasity(z2)
    spasity += latent_spasity(recon_z2)
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
        image_loss, latent_loss, sparsity = total_loss(output, o1, o2)
        ep_image_loss += image_loss.item()
        ep_latent_loss += latent_loss.item()
        ep_spasity += sparsity.item()
        loss = image_loss + latent_loss
        if add_spasity:
            loss += sparsity
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print("TRAINING LOSS REC_IMG: {:.3f}, REC_LATENT: {:.3f}, SPASITY_LATENT: {:.3f}".format(
        ep_image_loss/len(dataloader), ep_latent_loss/len(dataloader), ep_spasity/len(dataloader))
    )
    return train_loss / len(dataloader)

def test(dataloader, vae, e, temp=(0, 0)):
    vae.eval()
    test_loss = 0
    ep_image_loss, ep_latent_loss, ep_spasity = 0, 0, 0
    with torch.no_grad():
        all_a = []
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
            if i == 0:
                save_image(output, o1, o2, e)
            all_a.append(output[-1])
            break
        n_action = save_action_histogram(torch.cat(all_a, dim=0), e)

    print("TESTING LOSS REC_IMG: {:.3f}, REC_LATENT: {:.3f}, SPASITY_LATENT: {:.3f}, {} ACTIONS USED".format(
        ep_image_loss/len(dataloader), ep_latent_loss/len(dataloader), ep_spasity/len(dataloader), n_action)
    )
    return test_loss / len(dataloader)

def save_action_histogram(all_a, e):
    all_a = torch.argmax(all_a.squeeze(), dim=-1).detach().cpu()
    fig = plt.figure()
    fig.suptitle('Epoch {}'.format(e), fontsize=12)
    plt.hist(all_a.numpy(), bins=N_ACTION)
    unique_a = torch.unique(all_a).shape[0]
    plt.title('Action used across test dataset of {} example: {}'.format(TEST_EXAMPLES, unique_a))
    plt.savefig("puzzle/image/actions/{}.png".format(e))
    plt.close(fig)
    return unique_a

def save_image(output, b_o1, b_o2, e):
    def show_img(ax, img, t, set_title=False):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if set_title:
            assert t
            ax.title.set_text(t, s=8)
        ax.imshow(img, cmap='gray')

    b_recon_o1, b_recon_o2, b_recon_tilde, b_z1, b_z2, b_recon_z2, b_a = output

    N_SMAPLE= 5
    selected = torch.arange(start=0, end=5)
    pre_process = lambda img: img[selected].squeeze().detach().cpu()

    fig, axs = plt.subplots(N_SMAPLE, 10)
    fig.suptitle('Epoch {}'.format(e), fontsize=12)

    for i, (o1, recon_o1, o2, recon_o2, recon_tilde, z1, z2, recon_z2, a) in enumerate(
        zip(
            pre_process(b_o1), pre_process(b_recon_o1), pre_process(b_o2), pre_process(b_recon_o2),
            pre_process(b_recon_tilde), pre_process(b_z1), pre_process(b_z2), pre_process(b_recon_z2), pre_process(b_a)
        )
    ):
        if i == 0:
            set_title = True
        else:
            set_title = False
        show_img(axs[i,0], o1, r"$o_1$", set_title)
        show_img(axs[i,1], recon_o1, r"$\tilde{o_1}$", set_title)
        show_img(axs[i,2], o2, r"$o_2$", set_title)
        show_img(axs[i,3], recon_o2, r"$\tilde{o_2}$", set_title)
        show_img(axs[i,4], recon_tilde, r"$\tilde{\tilde{o_2}}$", set_title)
        show_img(axs[i,5], z1.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_1$", set_title)
        show_img(axs[i,6], z2.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_2$", set_title)
        show_img(axs[i,7], recon_z2.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$\tilde{z_2}$", set_title)
        show_img(axs[i,8], (z2 - recon_z2).view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_2 - \tilde{z_2}$", set_title)
        show_img(axs[i,9], a.view(N_ACTION_SQTR, N_ACTION_SQTR), "$a$", set_title)

    plt.tight_layout()
    plt.savefig("puzzle/image/samples/{}.png".format(e))
    plt.close(fig)


def load_model(vae):
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format(MODEL_NAME)))
    print("puzzle/model/{}.pth loaded".format(MODEL_NAME))

def run(n_epoch):
    train_set, test_set, _, _ = get_train_and_test_dataset(*load_data())
    print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=False)
    vae = CubeSae().to(device)
    # load_model(vae)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1.0 if e < 1000 else 0.1)
    best_loss = float('inf')
    best_epoch = 0
    for e in range(n_epoch):
        temp1 = np.maximum(TEMP_BEGIN_SAE * np.exp(-ANNEAL_RATE * e), TEMP_MIN_SAE)
        temp2 = np.maximum(TEMP_BEGIN_AAE * np.exp(-ANNEAL_RATE * e), TEMP_MIN_AAE)
        print("Epoch: {}, Temperature: {:.2f} {:.2f}, Lr: {}".format(e, temp1, temp2, scheculer.get_last_lr()))
        train_loss = train(train_loader, vae, optimizer, (temp1, temp2), e >= 100)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = test(test_loader, vae, e, (0, 0))
        print('====> Epoch: {} Average test loss: {:.4f}, best loss {:.4f} in epoch {}'.format(e, test_loss, best_loss, best_epoch))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "puzzle/model/{}.pth".format(MODEL_NAME))
            best_loss = test_loss
            best_epoch = e
        scheculer.step()

if __name__ == "__main__":
    try:
        shutil.rmtree('puzzle/image/')
    except:
        pass

    os.makedirs("puzzle/image/actions", exist_ok=True)
    os.makedirs("puzzle/image/samples", exist_ok=True)
    os.makedirs("puzzle/model", exist_ok=True)
    run(3000)