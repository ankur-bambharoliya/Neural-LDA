
import numpy as np
import torch
from torch.autograd import Variable
from os import path
import pickle
import matplotlib.pyplot as plt
from prod_lda import ProdLDA
from hyper_parameters import HyperParameters, TRAIN_DATA_PATH, VALID_DATA_PATH, TEST_DATA_PATH, VOCAB_PATH, USE_CUDA
from tqdm import tqdm
import time

def to_one_hot(data, min_length):
    return np.bincount(data, minlength=min_length)

# def make_data():
#     train_np = np.load(TRAIN_DATA_PATH, encoding='bytes')
#     valid_np = np.load(VALID_DATA_PATH, encoding='bytes')
#     test_np = np.load(TEST_DATA_PATH, encoding='bytes')
#     vocab = pickle.load(open(VOCAB_PATH,'rb'))
#     vocab_size = len(vocab)
#     print(valid_np)
#     print('Converting data to one-hot representation')
#     train_np = np.array([
#         to_one_hot(doc.astype('int'), vocab_size) for doc in train_np if np.sum(doc)!=0] )
#     valid_np = np.array([
#         to_one_hot(doc.astype('int'), vocab_size) for doc in valid_np if np.sum(doc)!=0] )
#     test_np = np.array([
#         to_one_hot(doc.astype('int'), vocab_size) for doc in test_np if np.sum(doc)!=0] )
#     print(
#         'Data Loaded\nDim Training Data : {}'
#         '\nDim Validation Data : {}\nDim Test Data : {}'.format(train_np.shape,
#                                                           valid_np.shape,
#                                                           test_np.shape)
#     )
#     train_torch = torch.from_numpy(train_np).float()
#     valid_torch = torch.from_numpy(valid_np).float()
#     test_torch = torch.from_numpy(test_np).float()
#     if USE_CUDA:
#         train_torch = train_torch.cuda()
#         valid_torch = valid_torch.cuda()
#         test_torch = test_torch.cuda()
#     return train_np, valid_np, test_np, train_torch, valid_torch, test_torch, vocab, vocab_size

def make_data():
    train_np = np.load(TRAIN_DATA_PATH, encoding='bytes')
    test_np = np.load(TEST_DATA_PATH, encoding='bytes')
    vocab = pickle.load(open(VOCAB_PATH,'rb'))
    vocab_size = len(vocab)
    print('Converting data to one-hot representation')
    train_np = np.array([
        to_one_hot(doc.astype('int'), vocab_size) for doc in train_np if np.sum(doc)!=0] )
    test_np = np.array([
        to_one_hot(doc.astype('int'), vocab_size) for doc in test_np if np.sum(doc)!=0] )
    print(
        'Data Loaded\nDim Training Data : {}\nDim Test Data : {}'.format(train_np.shape,
                                                                         test_np.shape)
    )
    train_torch = torch.from_numpy(train_np).float()
    test_torch = torch.from_numpy(test_np).float()
    if USE_CUDA:
        train_torch = train_torch.cuda()
        test_torch = test_torch.cuda()
    return train_np, test_np, train_torch, test_torch, vocab, vocab_size

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


def train():
    # train_np, valid_np, test_np, train_torch, valid_torch, test_torch, vocab, vocab_size = make_data()
    train_np, test_np, train_torch, test_torch, vocab, vocab_size = make_data()
    hps = HyperParameters(train_np)
    model = ProdLDA(hps)
    if USE_CUDA:
        model.cuda()

    if hps.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), hps.learning_rate,
                                     betas=(hps.momentum, 0.999))
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    hps.learning_rate, momentum=hps.momentum)

    total_losses = []
    recon_losses = []
    kl_losses = []
    # val_total_losses = []
    # val_recon_losses = []
    # val_kl_losses = []

    for epoch in range(1, hps.num_epochs+1):
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        kl_loss_epoch = 0.0
        val_total_loss_epoch = 0.0
        val_recon_loss_epoch = 0.0
        val_kl_loss_epoch = 0.0
        model.train()
        all_indices = torch.randperm(train_torch.size(0)).split(hps.batch_size)
        # Train
        start = time.time()
        for batch_indices in tqdm(all_indices,
                                  mininterval=2,
                                  desc=' - (Training)',
                                  leave=False):
            if USE_CUDA:
                batch_indices = batch_indices.cuda()
            xs = Variable(train_torch[batch_indices])
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
        n_train_batches = len(all_indices)
        total_loss_epoch /= n_train_batches
        recon_loss_epoch /= n_train_batches
        kl_loss_epoch /= n_train_batches
        total_losses.append(total_loss_epoch)
        recon_losses.append(recon_loss_epoch)
        kl_losses.append(kl_loss_epoch)

        # Validate
        # all_indices = torch.randperm(valid_torch.size(0)).split(hps.batch_size)
        # model.eval()
        # for batch_indices in all_indices:
        #     if USE_CUDA:
        #         batch_indices = batch_indices.cuda()
        #     xs = Variable(train_torch[batch_indices])
        #     reconstructed, kl_params = model(xs)
        #     recon_loss = model.reconstruction_loss(xs, reconstructed).data.item()
        #     kl_loss = model.kl_loss(*kl_params).data.item()
        #     val_total_loss_epoch += recon_loss + kl_loss
        #     val_recon_loss_epoch += recon_loss
        #     val_kl_loss_epoch += kl_loss
        #
        # n_val_batches = len(all_indices)
        # val_total_loss_epoch /= n_val_batches
        # val_recon_loss_epoch /= n_val_batches
        # val_kl_loss_epoch /= n_val_batches
        # val_total_losses.append(val_total_loss_epoch)
        # val_recon_losses.append(val_recon_loss_epoch)
        # val_kl_losses.append(val_kl_loss_epoch)
        print('Epoch {} - (Training)\n\tLoss : {} '
              '[Reconstruction={}, KL={}], Elapse {elapse:3.3f} s'.format(epoch,
                                                                          total_loss_epoch,
                                                                          recon_loss_epoch,
                                                                          kl_loss_epoch,
                                                                          elapse=time.time() - start))

        if epoch % 500 == 0:
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


if __name__== '__main__':
    plt.ion()
    train()
