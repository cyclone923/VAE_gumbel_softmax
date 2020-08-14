from puzzle.dataset import SaeDataSet
from puzzle.model import VAE_gumbel, device
from puzzle.train import TEMP_MIN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch





def run():
    view_set = SaeDataSet(is_train=False)
    view_loader = DataLoader(view_set, batch_size=1, shuffle=True)
    vae = VAE_gumbel().to(device)
    vae.eval()
    plt.gca()
    with torch.no_grad():
        for _, data in enumerate(view_loader):
            data = data.to(device)
            recon_batch, qy, sample = vae(data, TEMP_MIN)
            plt.imshow(data.squeeze().cpu().numpy(), cmap='gray')
            plt.pause(0.5)
            plt.imshow(recon_batch.squeeze().cpu().numpy(), cmap='gray')
            plt.pause(0.5)







if __name__ == "__main__":
    run()