import torch
import torch.nn as nn
from typing import List


class PatchDiscriminator1D(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
        use_instance_norm: bool = True,
    ) -> None:
        super().__init__()
        in_channels_total = input_channels + output_channels
        norm_layer = nn.InstanceNorm1d if use_instance_norm else nn.BatchNorm1d

        sequence: List[nn.Module] = []
        sequence += [
            nn.Conv1d(in_channels_total, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(base_channels * nf_mult_prev, base_channels * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(base_channels * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(base_channels * nf_mult_prev, base_channels * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(base_channels * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence += [nn.Conv1d(base_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x_cond: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x_cond: (N, C_in, T), y: (N, C_out, T)
        assert x_cond.dim() == 3 and y.dim() == 3
        # Ensure same temporal length by center-cropping the longer one
        if x_cond.shape[-1] != y.shape[-1]:
            if x_cond.shape[-1] > y.shape[-1]:
                diff = x_cond.shape[-1] - y.shape[-1]
                x_cond = x_cond[..., diff // 2 : x_cond.shape[-1] - (diff - diff // 2)]
            else:
                diff = y.shape[-1] - x_cond.shape[-1]
                y = y[..., diff // 2 : y.shape[-1] - (diff - diff // 2)]
        inp = torch.cat([x_cond, y], dim=1)
        return self.model(inp)