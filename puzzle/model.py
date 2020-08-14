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

LATENT_DIM = 32
CATEGORICAL_DIM = 2

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    noise = sample_gumbel(logits.size())
    y = logits + noise
    return F.softmax(y / temperature, dim=-1)


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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=4)
        self.fc3 = nn.Linear(in_features=16*3*3, out_features=LATENT_DIM*CATEGORICAL_DIM)

        self.fc4 = nn.Linear(in_features=LATENT_DIM * CATEGORICAL_DIM, out_features=16*3*3)
        self.conv_trans5 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=4, stride=4)
        self.conv_trans6 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2(h1))
        h3 = self.relu(self.fc3(h2.view(-1, 16*3*3)))
        return h3.view(-1, LATENT_DIM, CATEGORICAL_DIM)

    def decode(self, z_y):
        z = z_y.view(-1, LATENT_DIM * CATEGORICAL_DIM)
        h4 = self.relu(self.fc4(z).view(-1, 16, 3, 3))
        h5 = self.relu(self.conv_trans5(h4))
        return self.sigmoid(self.conv_trans6(h5))

    def forward(self, x, temp=1):
        q_y = self.encode(x)
        z_y = gumbel_softmax(q_y, temp)
        return self.decode(z_y), F.softmax(q_y, dim=-1), z_y