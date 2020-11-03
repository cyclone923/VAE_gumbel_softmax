import torch
from torch import nn
from puzzle.gumble import gumbel_softmax
from puzzle.generate_puzzle import BASE_SIZE

LATENT_DIM = 36
CATEGORICAL_DIM = 1
N_ACTION = 128

class Sae(nn.Module):

    def __init__(self):
        super(Sae, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.fc3 = nn.Linear(in_features=(BASE_SIZE*3) ** 2 * 16, out_features=LATENT_DIM*CATEGORICAL_DIM)

        self.fc4 = nn.Linear(in_features=LATENT_DIM * CATEGORICAL_DIM, out_features=1000)
        self.fc5 = nn.Linear(in_features=1000, out_features=1000)
        self.fc6 = nn.Linear(in_features=1000, out_features=(BASE_SIZE*3) ** 2)

    def encode(self, x):
        h1 = torch.tanh(self.conv1(x))
        h2 = torch.tanh(self.conv2(h1))
        h3 = self.fc3(torch.flatten(h2, start_dim=1, end_dim=-1))
        return h3.view(-1, LATENT_DIM, CATEGORICAL_DIM)

    def decode(self, z_y):
        z = z_y.view(-1, 1, LATENT_DIM * CATEGORICAL_DIM)
        h4 = torch.relu(self.fc4(z))
        h5 = torch.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5)).view(-1, 1, BASE_SIZE*3, BASE_SIZE*3)

    def forward(self, x, temp):
        q_y = self.encode(x)
        z_y = gumbel_softmax(q_y, temp)
        return self.decode(z_y), z_y

# Back-to-logits implementation
class Aae(nn.Module):
    def __init__(self):
        super(Aae, self).__init__()
        self.fc1 = nn.Linear(in_features=LATENT_DIM + LATENT_DIM, out_features=400)
        self.fc2 = nn.Linear(in_features=400 + LATENT_DIM, out_features=400)
        self.fc3 = nn.Linear(in_features=400 + LATENT_DIM, out_features=N_ACTION)

        self.effect1_fc = nn.Linear(in_features=N_ACTION, out_features=400)
        self.effect2_fc = nn.Linear(in_features=400, out_features=400)
        self.effect3_fc = nn.Linear(in_features=400, out_features=LATENT_DIM)


    def encode(self, s, z, temp):
        s = torch.flatten(s, start_dim=1).unsqueeze(1)
        z = torch.flatten(z, start_dim=1).unsqueeze(1)
        h1 = torch.relu(self.fc1(torch.cat([s, z], dim=2)))
        h2 = torch.relu(self.fc2(torch.cat([s, h1], dim=2)))
        h3 = self.fc3(torch.cat([s, h2], dim=2))
        return gumbel_softmax(h3.view(-1, 1, N_ACTION), temp)

    def decode(self, s, a, temp):
        h1 = self.effect1_fc(a)
        h2 = self.effect2_fc(h1)
        h3 = self.effect3_fc(h2)
        h3 = h3.view(-1, LATENT_DIM, 1)
        return gumbel_softmax(h3+s, temp)

    def forward(self, s, z, temp):
        a = self.encode(s, z, temp)
        recon_z = self.decode(s, a, temp)
        return recon_z


class CubeSae(nn.Module):

    def __init__(self):
        super(CubeSae, self).__init__()
        self.sae = Sae()
        self.aae = Aae()

    def forward(self, o1, o2, temp):
        recon_o1, z1 = self.sae(o1, temp)
        recon_o2, z2 = self.sae(o2, temp)
        z_recon = self.aae(z1, z2, temp)
        return recon_o1, recon_o2, z1, z2, z_recon



