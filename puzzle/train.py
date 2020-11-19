import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import os
import shutil
from puzzle.sae import CubeSae
from puzzle.dataset import get_train_and_test_dataset, load_data
from puzzle.gumble import device
from puzzle.loss import total_loss
from puzzle.util import save_action_histogram, check_and_clip_grad_norm, save_image, plot_loss, \
    MODEL_DIR, MODEL_PATH, IMG_DIR, BACK_TO_LOGIT, SAMPLE_DIR, SAMPLE_DIR_ARGMAX, ACTION_DIR
from puzzle.make_gif import to_gif
import sys

TEMP_BEGIN_SAE = 5
TEMP_MIN_SAE = 0.3
ANNEAL_RATE_SAE = 0.06

TEMP_BEGIN_AAE = 5
TEMP_MIN_AAE = 0.6
ANNEAL_RATE_AAE = 0.01
TRAIN_BZ = 2000
TEST_BZ = 2000

torch.manual_seed(0)

def train(dataloader, vae, optimizer, temp):
    vae.train()
    train_loss = 0
    ep_image_loss, ep_latent_loss, ep_spasity = 0, 0, 0
    ep_grad_norm = 0
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
        loss = image_loss + sparsity + latent_loss
        loss.backward()
        grad_norm = check_and_clip_grad_norm(vae)
        ep_grad_norm += grad_norm
        train_loss += loss.item()
        optimizer.step()

    print(
        "\nTRAINING Total {:.5f}, Rec: {:.5f}, Latent: {:.5f}, Spasity: {:.5f}, Grad_Norm: {:.5f}".format
        (
        train_loss / len(dataloader),
        ep_image_loss/len(dataloader),
        ep_latent_loss/len(dataloader),
        ep_spasity/len(dataloader),
        ep_grad_norm / len(dataloader),
        )
    )
    return ep_latent_loss / len(dataloader)

def test(dataloader, vae, e, temp):
    vae.eval()
    validation_loss = 0
    ep_image_loss, ep_latent_loss, ep_spasity = 0, 0, 0
    with torch.no_grad():
        all_a = []
        for i, (data, data_next) in enumerate(dataloader):
            o1 = data.to(device)
            o2 = data_next.to(device)
            noise1 = torch.normal(mean=0, std=0.4, size=o1.size()).to(device)
            noise2 = torch.normal(mean=0, std=0.4, size=o2.size()).to(device)
            output = vae(o1 + noise1, o2 + noise2, temp)
            output_argmax = vae(o1 + noise1, o2 + noise2, (0, 0, False))
            image_loss, latent_loss, spasity = total_loss(output, o1, o2)
            ep_image_loss += image_loss.item()
            ep_latent_loss += latent_loss.item()
            ep_spasity += spasity.item()
            loss = image_loss + latent_loss + spasity
            validation_loss += loss.item()
            if i == 0:
                save_image(
                    output, o1+ noise1, o2+ noise2, e, temp,
                    n_latent_z=int(np.sqrt(vae.aae.AAE_LATENT_DIM)),
                    n_latent_a=int(np.sqrt(vae.aae.AAE_N_ACTION)),
                    dir=SAMPLE_DIR
                )

                save_image(
                    output_argmax, o1+ noise1, o2+ noise2, e, temp,
                    n_latent_z=int(np.sqrt(vae.aae.AAE_LATENT_DIM)),
                    n_latent_a=int(np.sqrt(vae.aae.AAE_N_ACTION)),
                    dir=SAMPLE_DIR_ARGMAX
                )
            all_a.append(output[-3])
            break

        save_action_histogram(torch.cat(all_a, dim=0), e, temp, n_bins=vae.aae.AAE_N_ACTION)

    print("\nVALIDATION Total {:.5f}, Rec: {:.5f}, Latent: {:.5f}, Spasity: {:.5f}".format(
        validation_loss / len(dataloader),
        ep_image_loss/len(dataloader),
        ep_latent_loss/len(dataloader),
        ep_spasity/len(dataloader))
    )
    return ep_latent_loss / len(dataloader)

def run(n_epoch):
    train_set, test_set, _, _ = get_train_and_test_dataset(*load_data())
    print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=False)
    vae = CubeSae(BACK_TO_LOGIT).to(device)
    # load_model(vae)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1.0 if e < 100 else 0.1)
    best_loss = float('inf')
    best_epoch = 0
    all_train_loss = []
    all_validation_loss = []
    for e in range(n_epoch):
        sys.stdout.flush()
        temp1 = np.maximum(TEMP_BEGIN_SAE * np.exp(-ANNEAL_RATE_SAE * e), TEMP_MIN_SAE)
        temp2 = np.maximum(TEMP_BEGIN_AAE * np.exp(-ANNEAL_RATE_AAE * e), TEMP_MIN_AAE)
        print("\n" + "-"*50)
        print("Epoch: {}, Temperature: {:.2f} {:.2f}, Lr: {}".format(e, temp1, temp2, scheculer.get_last_lr()))
        train_loss = train(train_loader, vae, optimizer, (temp1, temp2, True))
        validation_loss = test(test_loader, vae, e, (temp1, temp2, False))
        all_train_loss.append(train_loss)
        all_validation_loss.append(validation_loss)
        print("\nBest test loss {:.5f} in epoch {}".format(best_loss, best_epoch))
        if validation_loss < best_loss:
            print("Save model to {}".format(MODEL_PATH))
            torch.save(vae.state_dict(), MODEL_PATH)
            best_loss = validation_loss
            best_epoch = e
        scheculer.step()
    plot_loss(all_train_loss, all_validation_loss, n_epoch)


if __name__ == "__main__":
    try:
        shutil.rmtree(IMG_DIR)
        shutil.rmtree(MODEL_DIR)
    except:
        pass

    os.makedirs(ACTION_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR_ARGMAX, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    run(5)
    to_gif()