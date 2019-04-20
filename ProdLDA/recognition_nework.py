from torch import nn
from hyper_parameters import HyperParameters

class RecognitionNetwork(nn.Module):

    def __init__(self, hps : HyperParameters):
        super().__init__()
        self.hps = hps
        self.x_to_hidden = nn.Sequential(
            nn.Linear(
                in_features=hps.rec_n_input,
                out_features=hps.rec_fc1_output_units,
                bias=True
            ),
            nn.Softplus(),
            nn.Linear(
                in_features=hps.rec_fc1_output_units,
                out_features=hps.rec_fc2_output_units,
                bias=True
            ),
            nn.Softplus(),
            nn.Dropout(hps.rec_drop_prob)
        )
        self.hidden_to_mu = nn.Sequential(
            nn.Linear(
                in_features=hps.rec_fc2_output_units,
                out_features=hps.n_z,
                bias=True,
            ),
            nn.BatchNorm1d(
                num_features=hps.n_z
            )
        )
        self.hidden_to_log_sigma_sq = nn.Sequential(
            nn.Linear(
                in_features=hps.rec_fc2_output_units,
                out_features=hps.n_z,
                bias=True,
            ),
            nn.BatchNorm1d(
                num_features=hps.n_z
            )
        )

    def forward(self, x):
        hidden = self.x_to_hidden(x)
        posterior_mean = self.hidden_to_mu(hidden)
        posterior_log_var = self.hidden_to_log_sigma_sq(hidden)
        return posterior_mean, posterior_log_var