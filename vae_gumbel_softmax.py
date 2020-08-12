# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca)
# The code has been modified from pytorch example vae code and inspired by the origianl tensorflow implementation of gumble-softmax by Eric Jang.

from __future__ import print_function
import argparse
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                    help='tau(temperature) (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("Using GPU")
else:
    print("Using CPU")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def sample_gumbel(shape, eps=1e-20):
    if args.cuda:
        U = torch.rand(shape).cuda()
    else:
        U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    q_y = logits.view(-1, latent_dim, categorical_dim)
    y = gumbel_softmax_sample(q_y, temperature)
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros(size=y.size())
    if args.cuda:
        y_hard = y_hard.cuda()
    y_hard.scatter(dim=-1, index=ind.unsqueeze(-1), value=1)
    return ((y_hard - y).detach() + y).view(-1, latent_dim*categorical_dim)

class VAE_gumbel(nn.Module):

    def __init__(self, temp):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)


        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp):
        q = self.encode(x.view(-1, 784))
        z = gumbel_softmax(q, temp)
        return self.decode(z), F.softmax(q, dim=1)


latent_dim = 20
categorical_dim = 10  # one-of-K vector

temp_min = 0.2
ANNEAL_RATE = 0.1

model = VAE_gumbel(args.temp)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    log_qy = torch.log(qy + 1e-20)

    if args.cuda:
        g = torch.log(torch.Tensor([1.0 / categorical_dim])).cuda()
    else:
        g = torch.log(torch.Tensor([1.0 / categorical_dim]))

    KLD = torch.sum(qy * (log_qy - g), dim=-1).mean()

    return BCE + KLD


def train(epoch, temp):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format
        #     (
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data))
        #     )

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch, temp=1):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp)
        test_loss += loss_function(recon_batch, data, qy).item()
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run():
    for epoch in range(args.epochs):
        temp = np.maximum(args.temp * np.exp(-ANNEAL_RATE * epoch), temp_min)
        print(temp)
        train(epoch, temp)
        test(epoch)

        ind = torch.randint(low=0, high=10, size=(64, 20, 1))
        z = torch.zeros(size=(64, 20, 10)).scatter(dim=-1, index=ind , value=1).view(64, -1)
        if args.cuda:
            z = z.cuda()
        sample = model.decode(z).cpu()
        save_image(sample.data.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    run()
