import torch
from torch import nn
from torch.autograd import Variable
from generation_network import GenerationNetwork
from recognition_nework import RecognitionNetwork
from hyper_parameters import HyperParameters

Tensor = torch.FloatTensor

class ProdLDA(nn.Module):
    
    def __init__(self, hps: HyperParameters):
        super().__init__()
        self.generator = GenerationNetwork(hps)
        self.recognizer = RecognitionNetwork(hps)
        self.hps = hps
        # from
        # https://github.com/hyqneuron/pytorch-avitm/blob/master/pytorch_model.py
        prior_mean   = Tensor(1, hps.num_topics).fill_(0)
        prior_var    = Tensor(1, hps.num_topics).fill_(hps.variance)
        prior_log_var = prior_var.log()
        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_log_var)

    def forward(self, x, get_kl_params=True):
        posterior_mean, posterior_log_var = self.recognizer(x)
        posterior_var = posterior_log_var.exp()
        # sample
        eps = Variable(x.data.new().resize_as_(posterior_mean.data).normal_())  # noise
        z = posterior_mean + posterior_var.sqrt() * eps
        reconstructed = self.generator(z)
        if get_kl_params:
            return reconstructed, (posterior_mean, posterior_log_var, posterior_var)
        else:
            return reconstructed


    def reconstruction_loss(self, input, recon, avg=True):
        loss = -(input * (recon + self.hps.reconstruction_eps).log()).sum(1)
        if avg:
            return loss.mean()
        else:
            return loss


    def kl_loss(self, posterior_mean, posterior_log_var, posterior_var, avg=True):
        prior_mean = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var = Variable(self.prior_var).expand_as(posterior_mean)
        prior_log_var = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division = posterior_var  / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        log_var_division = prior_log_var - posterior_log_var
        loss = 0.5 * (
            (var_division + diff_term + log_var_division).sum(1) -
            self.hps.num_topics
        )
        if avg:
            return loss.mean()
        else:
            return loss