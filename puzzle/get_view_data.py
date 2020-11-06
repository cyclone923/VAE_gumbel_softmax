from puzzle.sae import CubeSae
from puzzle.dataset import get_train_and_test_dataset
from puzzle.gumble import device
from puzzle.train import MODEL_NAME
from torch.utils.data import DataLoader
import torch
from puzzle.train import load_data, TEMP_MIN_SAE, TEMP_BEGIN_AAE
import matplotlib.pyplot as plt

N_EXAMPLES = 200

def init():
    train_set, _, view_set, _ = get_train_and_test_dataset(*load_data())

    view_set = train_set

    print("View examples {}".format(len(view_set)))
    view_loader = DataLoader(view_set, batch_size=20, shuffle=True)
    vae = CubeSae().to(device)
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    vae.eval()

    return vae, view_loader

def run(vae, view_loader):

    with torch.no_grad():
        plt.gca()
        for i, (data, data_next) in enumerate(view_loader):
            batch_o1 = data.to(device)
            batch_o2 = data_next.to(device)
            noise1 = torch.normal(mean=0, std=0.4, size=batch_o1.size()).to(device)
            noise2 = torch.normal(mean=0, std=0.4, size=batch_o2.size()).to(device)
            output = vae(batch_o1 + noise1, batch_o2 + noise2, (0, 0))
            for o1, o2, recon_o1, recon_o2, z1, z2, recon_z2, a in zip(*([batch_o1, batch_o2] + [i for i in output])):
                plt.imshow(o1.squeeze(), cmap="gray")
                plt.pause(0.1)
                plt.imshow(recon_o1.squeeze(), cmap="gray")
                plt.pause(0.1)
                plt.imshow(o2.squeeze(), cmap="gray")
                plt.pause(0.1)
                plt.imshow(recon_o2.squeeze(), cmap="gray")
                plt.pause(0.1)
                print(z1.squeeze())
                plt.imshow(z1, cmap="gray")
                plt.pause(0.1)
                print(z2.squeeze(), recon_z2.squeeze())
                plt.imshow(torch.cat([z2, recon_z2 > 0.5], dim=1), cmap="gray")
                plt.pause(0.1)
                print(a.squeeze())
                plt.imshow(a, cmap="gray")
                plt.pause(0.1)















if __name__ == "__main__":
    vae, view_loader = init()
    run(vae, view_loader)