import torch
from generation_network import GenerationNetwork
from reconstruction_network import ReconstructionNetwork
from hyper_parameters import HyperParameters



class ProdLDA(object):
    
    def __init__(self, hps : HyperParameters):
        self.generator = GenerationNetwork(hps)
        self.reconstructor = ReconstructionNetwork(hps)
        self.hps = hps
