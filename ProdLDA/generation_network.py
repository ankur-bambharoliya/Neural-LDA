from torch import nn
from hyper_parameters import HyperParameters

class GenerationNetwork(nn.Module):

    def __init__(self, hps: HyperParameters):
        super().__init__()
        self.hps = hps
        self.z_to_x_params = nn.Sequential(
            nn.Softmax(),
            nn.Dropout(hps.gen_drop_prob)
        )

        self.x_reconstruction = nn.Sequential(
            nn.Linear(
                in_features=hps.n_z,
                out_features=hps.gen_fc_output_units,
                bias=False
            ),
            nn.BatchNorm1d(hps.gen_fc_output_units),
            nn.Softmax()
        )

    def forward(self, z):
        return self.x_reconstruction(self.z_to_x_params(z))