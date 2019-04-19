import numpy as np

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
