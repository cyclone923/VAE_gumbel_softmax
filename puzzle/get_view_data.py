from puzzle.sae import Sae
from puzzle.fosae import FoSae
from puzzle.dataset import get_view_dataset
from puzzle.gumble import device
from puzzle.generate_puzzle import BASE_SIZE, PUZZLE_FILE_SPC
import numpy as np
from torch.utils.data import DataLoader
import torch
from puzzle.train import fo_logic, load_data, MODEL_NAME, TEMP_MIN, TEMP_BEGIN

N_EXAMPLES = 200

def init(spc=False):
    if not spc:
        data = load_data(fo_logic)
    else:
        data = np.load(PUZZLE_FILE_SPC)

    view_set = get_view_dataset(data, min(N_EXAMPLES, data.shape[0]))
    print("View examples {}".format(len(view_set)))
    view_loader = DataLoader(view_set, batch_size=len(view_set), shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    vae.load_state_dict(torch.load("puzzle/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    vae.eval()

    return vae, view_loader

def run(vae, view_loader, spc=False):

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

            data_np = data.view(-1, 9 , BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()

            # recon_batch, args, preds = vae(data, 0)
            # recon_batch = recon_batch.view(-1, 9, BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            # args = args.view(-1, 9 , BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            # preds = preds.detach().cpu().numpy()
            # print(data_np.shape, recon_batch.shape, args.shape, preds.shape)
            # np.save("puzzle/puzzle_data/puzzles_data_fo{}.npy".format("_spc" if spc else ""), data_np)
            # np.save("puzzle/puzzle_data/puzzles_rec_fo{}.npy".format("_spc" if spc else ""), recon_batch)
            # np.save("puzzle/puzzle_data/puzzles_args_fo{}.npy".format("_spc" if spc else ""), args)
            # np.save("puzzle/puzzle_data/puzzles_preds_fo{}.npy".format("_spc" if spc else ""), preds)

            recon_batch_soft, args_soft, preds_soft = vae(data, TEMP_MIN*30)
            recon_batch_soft = recon_batch_soft.view(-1, 9, BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            args_soft = args_soft.view(-1, 9 , BASE_SIZE*3, BASE_SIZE*3).detach().cpu().numpy()
            preds_soft = preds_soft.detach().cpu().numpy()
            print(data_np.shape, recon_batch_soft.shape, args_soft.shape, preds_soft.shape)
            np.save("puzzle/puzzle_data/puzzles_data_soft_fo{}.npy".format("_spc" if spc else ""), data_np)
            np.save("puzzle/puzzle_data/puzzles_rec_soft_fo{}.npy".format("_spc" if spc else ""), recon_batch_soft)
            np.save("puzzle/puzzle_data/puzzles_args_soft_fo{}.npy".format("_spc" if spc else ""), args_soft)
            np.save("puzzle/puzzle_data/puzzles_preds_soft_fo{}.npy".format("_spc" if spc else ""), preds_soft)














if __name__ == "__main__":
    vae, view_loader = init()
    run(vae, view_loader)
    vae, view_loader = init(spc=True)
    run(vae, view_loader, spc=True)