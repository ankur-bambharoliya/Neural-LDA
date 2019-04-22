
import torch
from os import path
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from prod_lda import ProdLDA
from hyper_parameters import HyperParameters, TRAIN_DATA_PATH, TEST_DATA_PATH, USE_CUDA
import time
from Dataset import NewsGroupDataset
import math

kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}

hps = HyperParameters()

train_set = NewsGroupDataset(TRAIN_DATA_PATH, vocab_size=hps.vocab_size)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=hps.batch_size, shuffle=True, **kwargs)

test_set = NewsGroupDataset(TEST_DATA_PATH, vocab_size=2000)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=len(test_set), shuffle=True, **kwargs)

def plot_losses(n_epochs, losses, labels, title,
                save=False, save_prefix=None, y_scale='linear', loss_type=''):
    epochs = list(range(1, n_epochs+1))
    for loss, lab in zip(losses, labels):
        plt.plot(epochs, loss, label=lab)
    plt.xlabel("Epoch")
    plt.ylabel('{} Loss'.format(loss_type))
    plt.yscale(y_scale)
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    f = title+".png"
    if save and save_prefix:
        f = path.join(save_prefix, f)
        plt.savefig(f)

    # show
    plt.show()


def train(model, optimizer):



    total_losses = []
    recon_losses = []
    kl_losses = []


    for epoch in range(1, hps.num_epochs+1):
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        kl_loss_epoch = 0.0

        model.train()
        # Train
        for batch, (xs) in enumerate(train_loader):
            if USE_CUDA:
                xs = xs.cuda()
            reconstructed, kl_params = model(xs)
            recon_loss = model.reconstruction_loss(xs, reconstructed)
            kl_loss = model.kl_loss(*kl_params)
            total_loss = recon_loss + kl_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_loss_epoch += total_loss.data.item()
            recon_loss_epoch += recon_loss.data.item()
            kl_loss_epoch += kl_loss.data.item()

        n_train_batches = len(train_loader)
        total_loss_epoch /= n_train_batches
        recon_loss_epoch /= n_train_batches
        kl_loss_epoch /= n_train_batches
        total_losses.append(total_loss_epoch)
        recon_losses.append(recon_loss_epoch)
        kl_losses.append(kl_loss_epoch)

        print(
            'Epoch {} - Training\n\tLoss : {} '
            '[Reconstruction={}, KL={}]'.format(epoch,total_loss_epoch,
                                                recon_loss_epoch, kl_loss_epoch)
        )

        if epoch % hps.plot_every == 0:
            plot_losses(
                epoch,
                [total_losses],#, val_total_losses],
                ['Training'],#, 'Validation'],
                'Prod-LDA Loss at Epoch {}'.format(epoch),
            )
            plot_losses(
                epoch,
                [recon_losses],#, val_recon_losses],
                ['Training'],#, 'Validation'],
                'Prod-LDA Reconstruction Loss at Epoch {}'.format(epoch),
            )
            plot_losses(
                epoch,
                [kl_losses],#, val_kl_losses],
                ['Training'],#, 'Validation'],
                'Prod-LDA KL Loss at Epoch {}'.format(epoch),
                y_scale='log',
                loss_type='Log'
            )


def print_perp(model):
    model.eval()
    iters = 0
    for batch, (xs) in enumerate(test_loader):
        iters += 1
        if USE_CUDA:
            xs = xs.cuda()
        reconstructed, kl_params = model(xs)
        recon_loss = model.reconstruction_loss(xs, reconstructed, avg=False)
        kl_loss = model.kl_loss(*kl_params, avg=False)
        loss = recon_loss + kl_loss
        counts = xs.sum(1)
        avg = (loss / counts).mean().data.item()
        print('The approximated perplexity is: ', math.exp(avg))
    assert iters == 1

if __name__== '__main__':
    model = ProdLDA(hps)

    if USE_CUDA:
        model.cuda()

    if hps.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), hps.learning_rate,
                                     betas=(hps.momentum, 0.999))
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    hps.learning_rate, momentum=hps.momentum)

    train(model, optimizer)
    print_perp(model)
