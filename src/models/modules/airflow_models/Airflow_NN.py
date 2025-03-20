from itertools import chain
from typing import Iterator

import torch
import torch.nn as nn
from .Airflow_MLP import MLP


class AirflowNN(nn.Module):
    def __init__(
        self, nb_hidden_layers, size_hidden_layers, bn_bool, encoder, decoder
    ):
        super(AirflowNN, self).__init__()

        self.nb_hidden_layers = nb_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder = MLP(decoder, batch_norm=False)

        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc]
            + [self.size_hidden_layers] * self.nb_hidden_layers
            + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data):
        z = self.encoder(data)
        z = self.nn(z)
        z = self.decoder(z)
        return z


class AirflowNNMTLDecoderII(nn.Module):
    def __init__(
        self,
        nb_hidden_layers,
        size_hidden_layers,
        bn_bool,
        encoder,
        decoder_surf,
        decoder_vol,
    ):
        super(AirflowNNMTLDecoderII, self).__init__()

        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder_surf = MLP(decoder_surf, batch_norm=False)
        self.decoder_u = MLP(decoder_vol, batch_norm=False)
        self.decoder_v = MLP(decoder_vol, batch_norm=False)
        self.decoder_p = MLP(decoder_vol, batch_norm=False)
        self.decoder_nut = MLP(decoder_vol, batch_norm=False)

        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc]
            + [size_hidden_layers] * nb_hidden_layers
            + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data, surf):

        z = self.encoder(data)
        z = self.nn(z)
        return self.decoder_surf(z[surf]), torch.concatenate(
            (
                self.decoder_u(z[~surf]),
                self.decoder_v(z[~surf]),
                self.decoder_p(z[~surf]),
                self.decoder_nut(z[~surf]),
            ),
            dim=-1,
        )

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.encoder.parameters(),
            self.nn.parameters(),
        )

    def task_specific_parameters(
        self,
    ) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.decoder_surf.parameters(),
            self.decoder_u.parameters(),
            self.decoder_v.parameters(),
            self.decoder_p.parameters(),
            self.decoder_nut.parameters(),
        )

    def last_shared_parameters(self):
        return []

    def reset(self):
        self.encoder.zero_grad(set_to_none=False)
        self.nn.zero_grad(set_to_none=False)


class AirflowNNMTLDecoderI(nn.Module):
    def __init__(
        self,
        nb_hidden_layers,
        size_hidden_layers,
        bn_bool,
        encoder,
        decoder_surf,
        decoder_vol,
    ):
        super(AirflowNNMTLDecoderI, self).__init__()

        self.bn_bool = bn_bool
        self.activation = nn.ReLU()

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder_surf = MLP(decoder_surf, batch_norm=False)
        self.decoder_vol = MLP(decoder_vol, batch_norm=False)

        self.dim_enc = encoder[-1]

        self.nn = MLP(
            [self.dim_enc]
            + [size_hidden_layers] * nb_hidden_layers
            + [self.dim_enc],
            batch_norm=self.bn_bool,
        )

    def forward(self, data, surf):
        z = self.encoder(data)
        z = self.nn(z)
        return self.decoder_surf(z[surf]), self.decoder_vol(z[~surf])

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.encoder.parameters(),
            self.nn.parameters(),
        )

    def reset(self):
        self.encoder.zero_grad(set_to_none=False)
        self.nn.zero_grad(set_to_none=False)

    def task_specific_parameters(
        self,
    ) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.decoder_surf.parameters(),
            self.decoder_vol.parameters(),
        )

    def last_shared_parameters(self):
        return []
