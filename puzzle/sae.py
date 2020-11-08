import torch
from torch import nn
from puzzle.gumble import gumbel_softmax
from puzzle.generate_puzzle import BASE_SIZE

LATENT_DIM = 6 ** 2
CATEGORICAL_DIM = 1
N_ACTION = 16 ** 2


def bn_and_dpt(x, bn, dpt):
    return dpt(bn(x))

class Sae(nn.Module):

    def __init__(self):
        super(Sae, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.dpt1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=(BASE_SIZE*3) ** 2 * 16, out_features=LATENT_DIM*CATEGORICAL_DIM)

        self.fc4 = nn.Linear(in_features=LATENT_DIM * CATEGORICAL_DIM, out_features=1000)
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
        return h3.view(-1, LATENT_DIM, CATEGORICAL_DIM)

    def decode(self, z_y):
        z = z_y.view(-1, 1, LATENT_DIM * CATEGORICAL_DIM)
        h4 = bn_and_dpt(torch.relu(self.fc4(z)), self.bn4, self.dpt4)
        h5 = bn_and_dpt(torch.relu(self.fc5(h4)), self.bn5, self.dpt5)
        return torch.sigmoid(self.fc6(h5)).view(-1, 1, BASE_SIZE*3, BASE_SIZE*3)

    def forward(self, x, temp):
        q_y = self.encode(x)
        z_y = gumbel_softmax(q_y, temp)
        return self.decode(z_y), z_y

# Back-to-logits implementation
class Aae(nn.Module):
    def __init__(self, back_to_logit=True):
        super(Aae, self).__init__()
        self.fc1 = nn.Linear(in_features=LATENT_DIM + LATENT_DIM, out_features=400)
        self.bn1 = nn.BatchNorm1d(num_features=1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=400 + LATENT_DIM, out_features=400)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=400 + LATENT_DIM, out_features=N_ACTION)

        self.fc4 = nn.Linear(in_features=N_ACTION, out_features=400)
        self.bn4 = nn.BatchNorm1d(num_features=1)
        self.dpt4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(in_features=400, out_features=400)
        self.bn5 = nn.BatchNorm1d(num_features=1)
        self.dpt5 = nn.Dropout(0.4)

        self.back_to_logit = back_to_logit
        if self.back_to_logit:
            self.fc6 = nn.Linear(in_features=400, out_features=LATENT_DIM)
            self.bn_input = nn.BatchNorm1d(num_features=LATENT_DIM)
            self.bn_effect = nn.BatchNorm1d(num_features=LATENT_DIM)
        else:
            self.fc6 = nn.Linear(in_features=400, out_features=LATENT_DIM * 3)

    def encode(self, s, z, temp):
        s = torch.flatten(s, start_dim=1).unsqueeze(1)
        z = torch.flatten(z, start_dim=1).unsqueeze(1)
        h1 = bn_and_dpt(torch.relu(self.fc1(torch.cat([s, z], dim=2))), self.bn1, self.dpt1)
        h2 = bn_and_dpt(torch.relu(self.fc2(torch.cat([s, h1], dim=2))), self.bn2, self.dpt2)
        h3 = self.fc3(torch.cat([s, h2], dim=2))
        return gumbel_softmax(h3.view(-1, 1, N_ACTION), temp)

    def decode(self, s, a, temp):
        h4 = bn_and_dpt(torch.relu(self.fc4(a)), self.bn4, self.dpt4)
        h5 = bn_and_dpt(torch.relu(self.fc5(h4)), self.bn5, self.dpt5)
        h6 = self.fc6(h5)

        if self.back_to_logit:
            h6 = h6.view(-1, LATENT_DIM, 1)
            s = self.bn_input(s)
            h6 = self.bn_effect(h6)
            s = gumbel_softmax(h6+s, temp)
        else:
            h6 = h6.view(-1, LATENT_DIM, 3)
            h6 = gumbel_softmax(h6, temp)
            add = h6[:,:,[0]]
            delete = h6[:,:,[1]]
            s = torch.max(s, add)
            s = torch.min(s, 1-delete)

        return s


    def forward(self, s, z, temp):
        a = self.encode(s, z, temp)
        recon_z = self.decode(s, a, temp)
        return recon_z, a


class CubeSae(nn.Module):
    def __init__(self, btl):
        super(CubeSae, self).__init__()
        self.sae = Sae()
        self.aae = Aae(back_to_logit=btl)

    def forward(self, o1, o2, temp):
        temp1, temp2 = temp
        recon_o1, z1 = self.sae(o1, temp1)
        recon_o2, z2 = self.sae(o2, temp1)
        z_recon, a = self.aae(z1, z2, temp2)
        recon_o2_tilda = self.sae.decode(z_recon)
        return recon_o1, recon_o2, recon_o2_tilda, z1, z2, z_recon, a