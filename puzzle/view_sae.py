from puzzle.dataset import SaeDataSet
from puzzle.sae import Sae, device
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from puzzle.train import TEST_BZ


def run():
    view_set = SaeDataSet(is_train=False)
    view_loader = DataLoader(view_set, batch_size=TEST_BZ, shuffle=True)
    vae = Sae().to(device)
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format("Sae"), map_location='cpu'))
    vae.eval()

    with torch.no_grad():
        fig, axs = plt.subplots(2, 2)
        for _, ax in np.ndenumerate(axs):
            ax.axis('off')
        plt.gca()

        for _, data in enumerate(view_loader):
            data = data.to(device)
            recon_batch, q_y, sample = vae(data, 0)
            for i in range(data.size()[0]):
                d = data[i].squeeze().cpu().numpy()
                r = recon_batch[i].squeeze().cpu().numpy()
                axs[0,0].imshow(d, cmap='gray')
                axs[0,1].imshow(r, cmap='gray')
                diff = d - r
                axs[1,0].imshow(diff, cmap='gray')
                s = sample[i].squeeze().cpu().numpy()
                q = q_y[i].squeeze().cpu().numpy()

                ones = np.ones(shape=s.shape)
                axs[1,1].imshow(np.concatenate([s, ones, s - q, ones, q], axis=1), cmap='gray')
                plt.pause(0.0001)








if __name__ == "__main__":
    run()