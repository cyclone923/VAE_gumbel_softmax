from puzzle.sae import Sae
from puzzle.fosae import FoSae
from puzzle.dataset import get_view_dataset
from puzzle.gumble import device
from puzzle.generate_puzzle import BASE_SIZE
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from puzzle.train import fo_logic, DATA, MODEL_NAME

N_EXAMPLE = 200

def init():
    view_set = get_view_dataset(DATA, N_EXAMPLE)
    view_loader = DataLoader(view_set, batch_size=N_EXAMPLE // 10, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    vae.eval()

    return vae, view_loader

def run(vae, view_loader):

    with torch.no_grad():
        if not fo_logic:
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
                    plt.pause(0.1)
        else:
            fig, axs = plt.subplots(4, 9, figsize=(9, 9))
            for _, ax in np.ndenumerate(axs):
                ax.axis('off')
            plt.gca()

            for _, data in enumerate(view_loader):
                data = data.to(device)
                recon_batch, args, preds = vae(data, 0)
                print(data.size(), recon_batch.size())
                for i in range(data.size()[0]):
                    print(i)
                    ds = data[i].view(9, BASE_SIZE*3, BASE_SIZE*3).squeeze().cpu().numpy()
                    rs = recon_batch[i].view(9, BASE_SIZE*3, BASE_SIZE*3).squeeze().cpu().numpy()
                    for j, (d,r) in enumerate(zip(ds, rs)):
                        axs[0,j].imshow(d, cmap='gray')
                        axs[1,j].imshow(r, cmap='gray')
                        diff = d - r
                        axs[2,j].imshow(diff, cmap='gray')
                    # s = sample[i].squeeze().cpu().numpy()
                    # q = q_y[i].squeeze().cpu().numpy()
                    #
                    # ones = np.ones(shape=s.shape)
                    # axs[1,1].imshow(np.concatenate([s, ones, s - q, ones, q], axis=1), cmap='gray')
                    print("Pause")
                    plt.pause(0.2)












if __name__ == "__main__":
    vae, view_loader = init()
    run(vae, view_loader)