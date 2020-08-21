import matplotlib.pyplot as plt
import numpy as np
from puzzle.train import fo_logic


if not fo_logic:
    data = np.load("puzzle/puzzle_data/puzzles_data.npy").squeeze()
    rec_batch = np.load("puzzle/puzzle_data/puzzles_rec.npy").squeeze()
    q_y = np.load("puzzle/puzzle_data/puzzles_qy.npy").squeeze()
    sample = np.load("puzzle/puzzle_data/puzzles_sample.npy").squeeze()
    fig, axs = plt.subplots(2, 2)
    for _, ax in np.ndenumerate(axs):
        ax.axis('off')
    plt.gca()


    for d, r, q, s in zip(data, rec_batch, q_y, sample):
        axs[0,0].imshow(d, cmap='gray')
        axs[0,1].imshow(r, cmap='gray')
        axs[1,0].imshow(d-r, cmap='gray')
        all = np.concatenate([q, np.ones(shape=q.shape), s], axis=1)
        axs[1,1].imshow(all, cmap='gray')
        plt.pause(0.1)

else:
    data = np.load("puzzle/puzzle_data/puzzles_data_fo.npy")
    rec_batch = np.load("puzzle/puzzle_data/puzzles_rec_fo.npy")
    args = np.load("puzzle/puzzle_data/puzzles_args_fo.npy")
    preds = np.load("puzzle/puzzle_data/puzzles_preds_fo.npy")
    fig, axs = plt.subplots(5, 9)
    for _, ax in np.ndenumerate(axs):
        ax.axis('off')
    plt.gca()

    for ds, rs, ars, ps in zip(data, rec_batch, args, preds):
        for i, (d, r, a, p) in enumerate(zip(ds, rs, ars, ps)):
            axs[0,i].imshow(d, cmap='gray')
            axs[1,i].imshow(d-r, cmap='gray')
            axs[2,i].imshow(r, cmap='gray')
            axs[3,i].imshow(a, cmap='gray')
            axs[4,i].imshow(p, cmap='gray')
        plt.pause(1)

