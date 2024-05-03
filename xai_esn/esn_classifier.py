from xai_esn.esn import DeepReservoir
import pytorch_lightning as pl
from torch import nn


class DeepReservoirClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size=1,
        tot_units=100,
        n_layers=1,
        concat=False,
        input_scaling=0.5,
        inter_scaling=1,
        spectral_radius=0.5,
        leaky=1,
        connectivity_recurrent=100,  # = tot_units -> no sparse
        connectivity_input=100,  # = tot_units -> no sparse
        connectivity_inter=10,
    ):
        super().__init__()
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.hidden = DeepReservoir(
            input_size=input_size,
            tot_units=tot_units,
            n_layers=n_layers,
            concat=concat,
            spectral_radius=spectral_radius,
            leaky=leaky,
            input_scaling=input_scaling,
            inter_scaling=inter_scaling,
            connectivity_recurrent=connectivity_recurrent,
            connectivity_input=connectivity_input,
            connectivity_inter=connectivity_inter,
        )
        self.output = None
        self.final_activation = None
        self.save_hyperparameters()

    def build(self, n_classes):
        if n_classes == 2:
            self.output = nn.Linear(self.hidden.layers_units, 1)
            self.final_activation = nn.Sigmoid()
        else:
            self.output = nn.Linear(self.hidden.layers_units, n_classes)
            self.final_activation = nn.Softmax(dim=1)

    def forward_rnn(self, inputs, return_sequences=False):
        h, _ = self.hidden(inputs)
        if not return_sequences:
            return h.mean(dim=1)
        return h

    def forward(self, input_seq, return_sequences=False):
        h = self.forward_rnn(input_seq, return_sequences=return_sequences)
        y = self.final_activation(self.output(h))
        return y
