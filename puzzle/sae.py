import torch
from torch import nn
from puzzle.gumble import gumbel_softmax
from puzzle.generate_puzzle import BASE_SIZE

LATENT_DIM = 6 ** 2
N_ACTION = 16 ** 2

def bn_and_dpt(x, bn, dpt):
    return dpt(bn(x))


class Sae(nn.Module):

    SAE_LATENT_DIM = LATENT_DIM
    SAE_N_ACTION = N_ACTION
    SAE_CATEGORICAL_DIM = 1

    def __init__(self):
        super(Sae, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.dpt1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=(BASE_SIZE*3) ** 2 * 16, out_features=self.SAE_LATENT_DIM * self.SAE_CATEGORICAL_DIM)

        self.fc4 = nn.Linear(in_features=self.SAE_LATENT_DIM * self.SAE_CATEGORICAL_DIM, out_features=1000)
        self.bn4 = nn.BatchNorm1d(num_features=1)
        self.dpt4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(in_features=1000, out_features=1000)
        self.bn5 = nn.BatchNorm1d(num_features=1)
        self.dpt5 = nn.Dropout(0.4)
        self.fc6 = nn.Linear(in_features=1000, out_features=(BASE_SIZE*3) ** 2)

    def encode(self, x):
        h1 = bn_and_dpt(torch.tanh(self.conv1(x)), self.bn1, self.dpt1)
        h2 = bn_and_dpt(torch.tanh(self.conv2(h1)), self.bn2, self.dpt2)
        h3 = self.fc3(torch.flatten(h2, start_dim=1, end_dim=-1))
        return h3.view(-1, self.SAE_LATENT_DIM, self.SAE_CATEGORICAL_DIM)

    def decode(self, z_y):
        z = z_y.view(-1, 1, self.SAE_LATENT_DIM * self.SAE_CATEGORICAL_DIM)
        h4 = bn_and_dpt(torch.relu(self.fc4(z)), self.bn4, self.dpt4)
        h5 = bn_and_dpt(torch.relu(self.fc5(h4)), self.bn5, self.dpt5)
        return torch.sigmoid(self.fc6(h5)).view(-1, 1, BASE_SIZE*3, BASE_SIZE*3)

    def forward(self, x, temp, add_noise):
        q_y = self.encode(x)
        z_y = gumbel_softmax(q_y, temp, add_noise)
        return self.decode(z_y), z_y

# Back-to-logits implementation
class Aae(nn.Module):

    AAE_LATENT_DIM = LATENT_DIM
    AAE_N_ACTION = N_ACTION

    def __init__(self, back_to_logit=True):
        super(Aae, self).__init__()
        self.fc1 = nn.Linear(in_features=self.AAE_LATENT_DIM + self.AAE_LATENT_DIM, out_features=1000)
        self.bn1 = nn.BatchNorm1d(num_features=1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=1000 + self.AAE_LATENT_DIM, out_features=1000)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=1000 + self.AAE_LATENT_DIM, out_features=self.AAE_N_ACTION)

        self.fc4 = nn.Linear(in_features=self.AAE_N_ACTION, out_features=1000)
        self.bn4 = nn.BatchNorm1d(num_features=1)
        self.dpt4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(in_features=1000, out_features=1000)
        self.bn5 = nn.BatchNorm1d(num_features=1)
        self.dpt5 = nn.Dropout(0.4)

        self.back_to_logit = back_to_logit
        if self.back_to_logit:
            self.fc6 = nn.Linear(in_features=1000, out_features=self.AAE_LATENT_DIM)
            self.bn_input = nn.BatchNorm1d(num_features=self.AAE_LATENT_DIM)
            self.bn_effect = nn.BatchNorm1d(num_features=self.AAE_LATENT_DIM)
        else:
            self.fc6 = nn.Linear(in_features=1000, out_features=self.AAE_LATENT_DIM * 3)


    def encode(self, s, z, temp, add_noise):
        s = torch.flatten(s, start_dim=1).unsqueeze(1)
        z = torch.flatten(z, start_dim=1).unsqueeze(1)
        h1 = bn_and_dpt(torch.relu(self.fc1(torch.cat([s, z], dim=2))), self.bn1, self.dpt1)
        h2 = bn_and_dpt(torch.relu(self.fc2(torch.cat([s, h1], dim=2))), self.bn2, self.dpt2)
        h3 = self.fc3(torch.cat([s, h2], dim=2))
        return gumbel_softmax(h3.view(-1, 1, self.AAE_N_ACTION), temp, add_noise)

    def decode(self, s, a, temp, add_noise):
        h4 = bn_and_dpt(torch.relu(self.fc4(a)), self.bn4, self.dpt4)
        h5 = bn_and_dpt(torch.relu(self.fc5(h4)), self.bn5, self.dpt5)
        h6 = self.fc6(h5)

        if self.back_to_logit:
            s = self.bn_input(s)
            h6 = h6.view(-1, self.AAE_LATENT_DIM, 1)
            h6 = self.bn_effect(h6)
            s = gumbel_softmax(h6+s, temp, add_noise)
            add = None
            delete = None
        else:
            h6 = h6.view(-1, self.AAE_LATENT_DIM, 3)
            h6 = gumbel_softmax(h6, temp, add_noise)
            add = h6[:,:,[0]]
            delete = h6[:,:,[1]]
            s = torch.min(s, 1-delete)
            s = torch.max(s, add)

        return s, add, delete

    def forward(self, s, z, temp, add_noise):
        a = self.encode(s, z, temp, add_noise)
        recon_z, add, delete = self.decode(s, a, temp, add_noise)
        return recon_z, a, add, delete


class CubeSae(nn.Module):

    def __init__(self, btl):
        super(CubeSae, self).__init__()
        self.sae = Sae()
        self.aae = Aae(back_to_logit=btl)

    def forward(self, o1, o2, temp):
        temp1, temp2, add_noise = temp
        recon_o1, z1 = self.sae(o1, temp1, add_noise)
        recon_o2, z2 = self.sae(o2, temp1, add_noise)
        z_recon, a, add, delete = self.aae(z1.detach(), z2.detach(), temp2, add_noise)
        recon_o2_tilda = self.sae.decode(z_recon)
        return recon_o1, recon_o2, recon_o2_tilda, z1, z2, z_recon, a, add, delete