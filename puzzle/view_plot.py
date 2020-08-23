import matplotlib.pyplot as plt
import numpy as np

spc = False
fo_logic = True

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
        margin = np.ones(shape=q.shape)
        all = np.concatenate([q, margin, margin, margin, s], axis=1)
        axs[1,1].imshow(all, cmap='gray')
        plt.pause(0.1)

else:
    data = np.load("puzzle/puzzle_data/puzzles_data_fo{}.npy".format("_spc" if spc else ""))
    rec_batch = np.load("puzzle/puzzle_data/puzzles_rec_fo{}.npy".format("_spc" if spc else ""))
    args = np.load("puzzle/puzzle_data/puzzles_args_fo{}.npy".format("_spc" if spc else ""))
    preds = np.load("puzzle/puzzle_data/puzzles_preds_fo{}.npy".format("_spc" if spc else ""))

    rec_batch_soft = np.load("puzzle/puzzle_data/puzzles_rec_soft_fo{}.npy".format("_spc" if spc else ""))
    args_soft = np.load("puzzle/puzzle_data/puzzles_args_soft_fo{}.npy".format("_spc" if spc else ""))
    preds_soft = np.load("puzzle/puzzle_data/puzzles_preds_soft_fo{}.npy".format("_spc" if spc else ""))
    fig, axs = plt.subplots(5, 9, figsize=(6,6))
    for _, ax in np.ndenumerate(axs):
        ax.axis('off')
    plt.gca()

    while True:
        for ds, rs, ars, ps, rts, arts, pts in zip(data, rec_batch, args, preds, rec_batch_soft, args_soft, preds_soft):
            for i, (d, r, ar, p, rt, art, pt) in enumerate(zip(ds, rs, ars, ps, rts, arts, pts)):
                axs[0,i].imshow(d, cmap='gray')
                axs[1,i].imshow(d-r, cmap='gray')
                axs[2,i].imshow(r, cmap='gray')
                axs[3,i].imshow(ar, cmap='gray')
                axs[4,i].imshow(p, cmap='gray')
                # axs[5,i].imshow(d-rt, cmap='gray')
                # axs[6,i].imshow(rt, cmap='gray')
                # axs[7,i].imshow(art, cmap='gray')
                # axs[8,i].imshow(pt, cmap='gray')
            plt.pause(0.1)



