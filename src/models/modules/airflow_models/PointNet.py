from itertools import chain
from typing import Iterator

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from .Airflow_MLP import MLP


class PointNet(nn.Module):
    def __init__(self, base_nb, encoder, decoder):
        super(PointNet, self).__init__()

        self.base_nb = base_nb

        self.in_block = MLP(
            [encoder[-1], self.base_nb, self.base_nb * 2],
            batch_norm=False,
        )
        self.max_block = MLP(
            [
                self.base_nb * 2,
                self.base_nb * 4,
                self.base_nb * 8,
                self.base_nb * 32,
            ],
            batch_norm=False,
        )

        self.out_block = MLP(
            [
                self.base_nb * (32 + 2),
                self.base_nb * 16,
                self.base_nb * 8,
                self.base_nb * 4,
            ],
            batch_norm=False,
        )

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder = MLP(decoder, batch_norm=False)

        self.fcfinal = nn.Linear(self.base_nb * 4, encoder[-1])

    def forward(self, z):

        batch = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        z = self.encoder(z)
        z = self.in_block(z)

        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)
        nb_points = torch.zeros(global_coef.shape[0], device=z.device)

        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        nb_points = nb_points.long()
        global_coef = torch.repeat_interleave(global_coef, nb_points, dim=0)

        z = torch.cat([z, global_coef], dim=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
        z = self.decoder(z)

        return z


class PointNetMTLDecoderI(nn.Module):
    def __init__(self, base_nb, encoder, decoder_surf, decoder_vol):
        super(PointNetMTLDecoderI, self).__init__()

        self.base_nb = base_nb

        self.in_block = MLP(
            [encoder[-1], self.base_nb, self.base_nb * 2],
            batch_norm=False,
        )
        self.max_block = MLP(
            [
                self.base_nb * 2,
                self.base_nb * 4,
                self.base_nb * 8,
                self.base_nb * 32,
            ],
            batch_norm=False,
        )

        self.out_block = MLP(
            [
                self.base_nb * (32 + 2),
                self.base_nb * 16,
                self.base_nb * 8,
                self.base_nb * 4,
            ],
            batch_norm=False,
        )
        self.fcfinal = nn.Linear(self.base_nb * 4, encoder[-1])

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder_surf = MLP(decoder_surf, batch_norm=False)
        self.decoder_vol = MLP(decoder_vol, batch_norm=False)

    def forward(self, z, surf):

        batch = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        z = self.encoder(z)
        z = self.in_block(z)
        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)
        nb_points = torch.zeros(global_coef.shape[0], device=z.device)

        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        nb_points = nb_points.long()
        global_coef = torch.repeat_interleave(global_coef, nb_points, dim=0)

        z = torch.cat([z, global_coef], dim=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
        return self.decoder_surf(z[surf]), self.decoder_vol(z[~surf])

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.encoder.parameters(),
            self.in_block.parameters(),
            self.max_block.parameters(),
            self.out_block.parameters(),
            self.fcfinal.parameters(),
        )

    def reset(self):
        self.encoder.zero_grad(set_to_none=False)
        self.in_block.zero_grad(set_to_none=False),
        self.max_block.zero_grad(set_to_none=False),
        self.out_block.zero_grad(set_to_none=False),
        self.fcfinal.zero_grad(set_to_none=False),

    def task_specific_parameters(
        self,
    ) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(
            self.decoder_surf.parameters(),
            self.decoder_vol.parameters(),
        )

    def last_shared_parameters(self):
        return []


class PointNetMTLDecoderII(nn.Module):
    def __init__(self, base_nb, encoder, decoder_surf, decoder_vol):
        super(PointNetMTLDecoderII, self).__init__()

        self.base_nb = base_nb

        self.in_block = MLP(
            [encoder[-1], self.base_nb, self.base_nb * 2],
            batch_norm=False,
        )
        self.max_block = MLP(
            [
                self.base_nb * 2,
                self.base_nb * 4,
                self.base_nb * 8,
                self.base_nb * 32,
            ],
            batch_norm=False,
        )

        self.out_block = MLP(
            [
                self.base_nb * (32 + 2),
                self.base_nb * 16,
                self.base_nb * 8,
                self.base_nb * 4,
            ],
            batch_norm=False,
        )
        self.fcfinal = nn.Linear(self.base_nb * 4, encoder[-1])

        self.encoder = MLP(encoder, batch_norm=False)
        self.decoder_surf = MLP(decoder_surf, batch_norm=False)
        self.decoder_u = MLP(decoder_vol, batch_norm=False)
        self.decoder_v = MLP(decoder_vol, batch_norm=False)
        self.decoder_p = MLP(decoder_vol, batch_norm=False)
        self.decoder_nut = MLP(decoder_vol, batch_norm=False)

    def forward(self, z, surf):

        batch = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        z = self.encoder(z)
        z = self.in_block(z)
        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)
        nb_points = torch.zeros(global_coef.shape[0], device=z.device)

        for i in range(batch.max() + 1):
            nb_points[i] = (batch == i).sum()
        nb_points = nb_points.long()
        global_coef = torch.repeat_interleave(global_coef, nb_points, dim=0)

        z = torch.cat([z, global_coef], dim=1)
        z = self.out_block(z)
        z = self.fcfinal(z)
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
            self.in_block.parameters(),
            self.max_block.parameters(),
            self.out_block.parameters(),
            self.fcfinal.parameters(),
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
        self.in_block.zero_grad(set_to_none=False),
        self.max_block.zero_grad(set_to_none=False),
        self.out_block.zero_grad(set_to_none=False),
        self.fcfinal.zero_grad(set_to_none=False),
