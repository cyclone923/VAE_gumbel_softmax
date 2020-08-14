import torch
from torch import nn
from torch.nn import functional as F
from puzzle.generate_puzzle import BASE_SIZE


if torch.cuda.is_available():
    device = 'cuda:0'
    print("Using GPU")
else:
    device = 'cpu'
    print("Using CPU")

LATENT_DIM = 36
CATEGORICAL_DIM = 2

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    if temperature == 0:
        _, ind = torch.max(logits, dim=-1)
        ret = torch.zeros(size=logits.size()).to(device).scatter(dim=-1, index=ind.unsqueeze(-1), value=1)
    else:
        noise = sample_gumbel(logits.size())
        y = logits + noise
        ret = F.softmax(y / temperature, dim=-1)
    return ret


def gumbel_softmax(q_y, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """

    y = gumbel_softmax_sample(q_y, temperature)
    if hard:
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros(size=y.size()).to(device).scatter(dim=-1, index=ind.unsqueeze(-1), value=1)
        y_ret = (y_hard - y).detach() + y
    else:
        y_ret = y
    return y_ret


class VAE_gumbel(nn.Module):

    def __init__(self):
        super(VAE_gumbel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.dpt1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=(BASE_SIZE*3) ** 2 * 16, out_features=LATENT_DIM*CATEGORICAL_DIM)

        self.fc4 = nn.Linear(in_features=LATENT_DIM * CATEGORICAL_DIM, out_features=1000)
        self.bn4 = nn.BatchNorm1d(1)
        self.dpt4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(in_features=1000, out_features=1000)
        self.bn5 = nn.BatchNorm1d(1)
        self.dpt5 = nn.Dropout(0.4)
        self.fc6 = nn.Linear(in_features=1000, out_features=(BASE_SIZE*3) ** 2)


    def encode(self, x):
        noise = torch.normal(mean=0, std=0.4, size=x.size()).to(device)
        h1 = self.dpt1(self.bn1(torch.tanh(self.conv1(x + noise))))
        h2 = self.dpt2(self.bn2(torch.tanh(self.conv2(h1))))
        h3 = self.fc3(torch.flatten(h2, start_dim=1, end_dim=-1))
        return h3.view(-1, LATENT_DIM, CATEGORICAL_DIM)

    def decode(self, z_y):
        z = z_y.view(-1, 1, LATENT_DIM * CATEGORICAL_DIM)
        h4 = self.dpt4(self.bn4(torch.relu(self.fc4(z))))
        h5 = self.dpt5(self.bn5(torch.relu(self.fc5(h4))))
        return torch.sigmoid(self.fc6(h5)).view(-1, 1, BASE_SIZE*3, BASE_SIZE*3)

    def forward(self, x, temp=1):
        q_y = self.encode(x)
        z_y = gumbel_softmax(q_y, temp)
        return self.decode(z_y), F.softmax(q_y, dim=-1), z_y