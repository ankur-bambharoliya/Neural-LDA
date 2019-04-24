from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from Dataset import NewsGroupDataset
import numpy as np

import os
from utils import load_vocab

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    NewsGroupDataset('./data/20news/train.feat', vocab_size=2000),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    NewsGroupDataset('./data/20news/test.feat', vocab_size=2000),
    batch_size=args.batch_size, shuffle=True, **kwargs)
vocab = load_vocab()


vocab_size = 2000
hidden_size = 500
k_topics = 20

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, k_topics)
        self.fc22 = nn.Linear(hidden_size, k_topics)
        self.fc4 = nn.Linear(k_topics, vocab_size)

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return F.log_softmax(self.fc4(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

print(model)
for param in model.parameters():
    print(type(param.data), param.size())

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_list, x, mu, logvar):
    BCE = -torch.sum(recon_list*x, dim=1)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2*logvar - mu.pow(2) - (2*logvar).exp())
    return KLD + torch.sum(BCE), torch.sum(BCE)

def train(epoch):
    model.train()
    train_loss = 0
    total_count = 0
    for batch_idx, (data,count) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        total_count += np.sum(torch.sum(count).item())
        train_loss += np.sum(loss.item())
        optimizer.step()


    print('====> Epoch: {} Average loss: {:.4f} || perplexity: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), np.exp(train_loss / total_count)))

os.remove("text.txt")
file = open('text.txt', 'a')
def test(epoch):
    file.write("Epoch----->%d\n"%epoch)
    model.eval()
    test_loss = 0.0
    word_count = 0.0
    BCE_SUM = 0.0
    with torch.no_grad():

        for i, (data, count) in enumerate(test_loader):
            data = data.to(device)
            word_count += count.sum().item()
            recon_batch, mu, logvar = model(data)
            loss,  BCE = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            BCE_SUM += BCE.item()

    test_loss /= len(test_loader.dataset)
    perplexity = BCE_SUM/word_count
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print(perplexity)

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    betas = [v.data for v in model.parameters()][6].numpy().T
    print("\n".join([" ".join([vocab[j]
                               for j in np.absolute(v).argsort()[:-20 - 1:-1]]) for v in betas]))

    test(epoch)
