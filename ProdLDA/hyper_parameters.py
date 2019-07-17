from torch import cuda

TRAIN_DATA_PATH = '../data/20news/train.feat'
TEST_DATA_PATH = '../data/20news/test.feat'

USE_CUDA = cuda.is_available()

class HyperParameters(object):

    def __init__(self):
        self.num_topics = 50
        self.vocab_size = 2000
        self.learning_rate = 0.001
        self.reconstruction_eps = 1e-10
        self.n_z = self.num_topics
        self.rec_fc1_output_units = 100
        self.rec_fc2_output_units = 100
        self.rec_n_input = self.vocab_size
        self.rec_drop_prob = 0.5
        self.gen_drop_prob = 0.5
        self.gen_fc_output_units = self.rec_n_input
        self.variance = 1.
        self.momentum = .99
        self.learning_rate = 0.002
        self.num_epochs = 100
        self.plot_every = 20
        self.optimizer = 'Adam'
        self.batch_size = 100

