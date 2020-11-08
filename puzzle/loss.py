import torch.nn as nn

ALPHA = 0.2

# Reconstruction + zero suppressed losses summed over all elements and batch
def rec_loss_function(recon_x, x, criterion):
    BCE = criterion(recon_x, x).sum(dim=[i for i in range(1, x.dim())]).mean()
    return BCE

def latent_spasity(z):
    return z.sum(dim=[i for i in range(1, z.dim())]).mean() * ALPHA

def total_loss(output, o1, o2):
    recon_o1, recon_o2, recon_o2_tilde, z1, z2, recon_z2, _, _, _ = output
    image_loss = 0
    latent_loss = 0
    spasity = 0
    image_loss += rec_loss_function(recon_o1, o1, nn.BCELoss(reduction='none'))
    image_loss += rec_loss_function(recon_o2, o2, nn.BCELoss(reduction='none'))
    image_loss += rec_loss_function(recon_o2_tilde, o2, nn.BCELoss(reduction='none'))
    latent_loss += rec_loss_function(recon_z2, z2, nn.L1Loss(reduction='none'))
    spasity += latent_spasity(z1)
    spasity += latent_spasity(z2)
    # spasity += latent_spasity(recon_z2)
    return image_loss, latent_loss, spasity