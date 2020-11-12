import matplotlib.pyplot as plt
from puzzle.dataset import VALIDATION_EXAMPLES
from puzzle.sae import LATENT_DIM, N_ACTION
import numpy as np
import torch
import os
import sys

BACK_TO_LOGIT = eval(sys.argv[1])

if BACK_TO_LOGIT:
    print("Back to logit")
    sys.stdout = open("btl.out", "w")
else:
    print("Naive")
    sys.stdout = open("nv.out", "w")

IMG_DIR = "puzzle/image_{}".format("btl" if BACK_TO_LOGIT else "naive")
MODEL_DIR = "puzzle/model_{}".format("btl" if BACK_TO_LOGIT else "naive")
ACTION_DIR = os.path.join(IMG_DIR, "actions")
SAMPLE_DIR = os.path.join(IMG_DIR, "samples")
FIG_SIZE = (12, 9)

MODEL_NAME = "CubeSae"
MODEL_PATH = os.path.join(MODEL_DIR, "{}.pth".format(MODEL_NAME))

LATENT_DIM_SQRT = int(np.sqrt(LATENT_DIM))
N_ACTION_SQTR = int(np.sqrt(N_ACTION))

def save_action_histogram(all_a, e, temp):
    all_a = torch.argmax(all_a.squeeze(), dim=-1).detach().cpu()
    fig = plt.figure(figsize=FIG_SIZE)
    fig.suptitle('Epoch {}'.format(e), fontsize=12)
    plt.hist(all_a.numpy(), bins=N_ACTION)
    unique_a = torch.unique(all_a).shape[0]
    plt.title('Action used: {}/{} in {} test examples, Temp: ({:.2f}, {:.2f})'.format(
        unique_a, N_ACTION, VALIDATION_EXAMPLES, temp[0], temp[1]), fontsize=8
    )
    plt.savefig(os.path.join(ACTION_DIR, "{}.png".format(e)))
    plt.close(fig)
    print("{} action used".format(unique_a))
    return unique_a

def save_image(output, b_o1, b_o2, e, temp):
    def show_img(ax, img, t, set_title=False):
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if set_title:
            assert t
            ax.set_title(t, fontsize=8)
        ax.imshow(img, cmap='gray')

    N_SMAPLE= 10
    selected = torch.arange(start=0, end=N_SMAPLE)
    pre_process = lambda img: img[selected].squeeze().detach().cpu() if img is not None else None

    fig, axs = plt.subplots(N_SMAPLE, 10 + (0 if BACK_TO_LOGIT else 3), figsize=FIG_SIZE)
    fig.suptitle('Epoch {}, Temp: ({:.2f}, {:.2f})'.format(e, temp[0], temp[1]), fontsize=12)

    for i, single in enumerate(
            zip(*([pre_process(b_o1), pre_process(b_o2)] + [pre_process(i) for i in output if i is not None]))
    ):
        if BACK_TO_LOGIT:
            o1, o2, recon_o1, recon_o2, recon_tilde, z1, z2, recon_z2, a = single
        else:
            o1, o2, recon_o1, recon_o2, recon_tilde, z1, z2, recon_z2, a, add, delete = single

        if i == 0:
            set_title = True
        else:
            set_title = False
        show_img(axs[i,0], o1, r"$o_1$", set_title)
        show_img(axs[i,1], recon_o1, r"$\tilde{o_1}$", set_title)
        show_img(axs[i,2], o2, r"$o_2$", set_title)
        show_img(axs[i,3], recon_o2, r"$\tilde{o_2}$", set_title)
        show_img(axs[i,4], recon_tilde, r"$\tilde{\tilde{o_2}}$", set_title)
        show_img(axs[i,5], z1.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_1$", set_title)
        show_img(axs[i,6], z2.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_2$", set_title)
        show_img(axs[i,7], recon_z2.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$\tilde{z_2}$", set_title)
        show_img(axs[i,8], (z2 - recon_z2).view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_2 - \tilde{z_2}$", set_title)
        if not BACK_TO_LOGIT:
            show_img(axs[i,-4], (z2 - z1).view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), r"$z_2 - z1$", set_title)
            show_img(axs[i,-3], add.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), "$add$", set_title)
            show_img(axs[i,-2], delete.view(LATENT_DIM_SQRT, LATENT_DIM_SQRT), "$delete$", set_title)

        show_img(axs[i,-1], a.view(N_ACTION_SQTR, N_ACTION_SQTR), "$a$", set_title)

    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLE_DIR, "{}.png".format(e)))
    plt.close(fig)


def load_model(vae):
    vae.load_state_dict(torch.load(MODEL_PATH))
    print(MODEL_PATH)