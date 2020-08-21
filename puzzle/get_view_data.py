from puzzle.sae import Sae
from puzzle.fosae import FoSae
from puzzle.dataset import get_view_dataset
from puzzle.gumble import device
from puzzle.generate_puzzle import BASE_SIZE
import numpy as np
from torch.utils.data import DataLoader
import torch
from puzzle.train import fo_logic, load_data, MODEL_NAME

N_EXAMPLE = 200

def init():
    data = load_data(fo_logic)
    view_set = get_view_dataset(data, N_EXAMPLE)
    view_loader = DataLoader(view_set, batch_size=N_EXAMPLE, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    vae.eval()

    return vae, view_loader

def run(vae, view_loader):

    with torch.no_grad():
        if not fo_logic:
            data = view_loader.__iter__().__next__()
            data = data.to(device)
            recon_batch, q_y, sample = vae(data, 0)

            data = data.detach().cpu().numpy()
            recon_batch = recon_batch.detach().cpu().numpy()
            q_y = q_y.detach().cpu().numpy()
            sample = sample.detach().cpu().numpy()

            print(data.shape, recon_batch.shape, q_y.shape, sample.shape)
            np.save("puzzle/puzzle_data/puzzles_data.npy", data)
            np.save("puzzle/puzzle_data/puzzles_rec.npy", recon_batch)
            np.save("puzzle/puzzle_data/puzzles_qy.npy", q_y)
            np.save("puzzle/puzzle_data/puzzles_sample.npy", sample)
        else:
            # fig, axs = plt.subplots(4, 9, figsize=(9, 9))
            # for _, ax in np.ndenumerate(axs):
            #     ax.axis('off')
            #
            # plt.gca()
            data = view_loader.__iter__().__next__()
            data = data.to(device)
            recon_batch, args, preds = vae(data, 0)
            data = data.view(-1, 9 , BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            recon_batch = recon_batch.view(-1, 9, BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            args = args.view(-1, 9 , BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            print(data.shape, recon_batch.shape, args.shape, preds.shape)
            np.save("puzzle/puzzle_data/puzzles_data_fo.npy", data)
            np.save("puzzle/puzzle_data/puzzles_rec_fo.npy", recon_batch)
            np.save("puzzle/puzzle_data/puzzles_args_fo.npy", args)
            np.save("puzzle/puzzle_data/puzzles_preds_fo.npy", preds)














if __name__ == "__main__":
    vae, view_loader = init()
    run(vae, view_loader)