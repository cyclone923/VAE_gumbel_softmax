import torch.nn as nn

ALPHA = 0.01
BETA = 1

# Reconstruction + zero suppressed losses summed over all elements and batch
def rec_loss_function(recon_x, x, criterion, weight=1.):
    BCE = criterion(recon_x, x).mean()
    return BCE * weight

def latent_spasity(z, weight=1.):
    return z.mean() * weight

def total_loss(output, o1, o2):
    recon_o1, recon_o2, recon_o2_tilde, z1, z2, recon_z2, _, add, delete = output
    image_loss = 0
    latent_loss = 0
    spasity = 0
    image_loss += rec_loss_function(recon_o1, o1, nn.BCELoss(reduction='none'))
    image_loss += rec_loss_function(recon_o2, o2, nn.BCELoss(reduction='none'))
    # image_loss += rec_loss_function(recon_o2_tilde, o2, nn.BCELoss(reduction='none'))
    latent_loss += rec_loss_function(recon_z2, z2.detach(), nn.MSELoss(reduction='none'), BETA)
    spasity += latent_spasity(z1, ALPHA)
    spasity += latent_spasity(z2, ALPHA)
    # spasity += latent_spasity(recon_z2, ALPHA)
    # if add is not None:
    #     spasity += latent_spasity(add, ALPHA)
    # if delete is not None:
    #     spasity += latent_spasity(delete, ALPHA)
    return image_loss, latent_loss, spasity