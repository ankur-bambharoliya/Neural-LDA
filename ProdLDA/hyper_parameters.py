import numpy as np
from torch import cuda

TRAIN_DATA_PATH = 'data/20news_clean/train.txt.npy'
VALID_DATA_PATH = 'data/20news_clean/valid.txt.npy'
TEST_DATA_PATH = 'data/20news_clean/test.txt.npy'
VOCAB_PATH = 'data/20news_clean/vocab.pkl'
USE_CUDA = cuda.is_available()
class HyperParameters(object):

    def __init__(self, training_data : np.ndarray):
        self.num_topics = 50
        self.learning_rate = 0.001
        self.reconstruction_eps = 1e-10
        self.n_z = self.num_topics
        self.rec_fc1_output_units = 100
        self.rec_fc2_output_units = 100
        self.rec_n_input = training_data.shape[1]
        self.rec_drop_prob = 0.5
        self.gen_drop_prob = 0.5
        self.gen_fc_output_units = self.rec_n_input
        self.variance = 1.
        self.momentum = .99
        self.learning_rate = 0.002
        self.num_epochs = 8000
        self.optimizer = 'Adam'
        self.batch_size = 500

