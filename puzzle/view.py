from puzzle.dataset import SaeDataSet
from puzzle.model import VAE_gumbel, device
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch


def run():
    view_set = SaeDataSet(is_train=False)
    view_loader = DataLoader(view_set, batch_size=1, shuffle=True)
    vae = VAE_gumbel().to(device)
    vae.load_state_dict(torch.load("puzzle/model/0.pth", map_location='cpu'))
    vae.eval()

    with torch.no_grad():
        fig, axs = plt.subplots(2, 2)
        for _, ax in np.ndenumerate(axs):
            ax.axis('off')
        plt.gca()
        for _, data in enumerate(view_loader):
            data = data.to(device)
            recon_batch, _, sample = vae(data, 0)

            data = data.squeeze().cpu().numpy()
            recon_batch = recon_batch.squeeze().cpu().numpy()
            axs[0,0].imshow(data, cmap='gray')
            axs[0,1].imshow(recon_batch, cmap='gray')
            diff = data - recon_batch
            axs[1,0].imshow(diff, cmap='gray')
            sample = sample.squeeze().cpu().numpy()
            axs[1,1].imshow(sample, cmap='gray')
            plt.pause(0.2)







if __name__ == "__main__":
    run()