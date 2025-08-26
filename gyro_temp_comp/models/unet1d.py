import torch
import torch.nn as nn
from typing import List


class UNet1D(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 1,
        base_channels: int = 64,
        num_downsamples: int = 6,
        use_instance_norm: bool = True,
        use_dropout_in_ups: bool = True,
    ) -> None:
        super().__init__()
        assert num_downsamples >= 1, "num_downsamples must be >= 1"

        norm_layer = nn.InstanceNorm1d if use_instance_norm else nn.BatchNorm1d

        def down_block(in_c: int, out_c: int, normalize: bool) -> nn.Sequential:
            layers: List[nn.Module] = [
                nn.Conv1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(norm_layer(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_c: int, out_c: int, apply_dropout: bool) -> nn.Sequential:
            layers: List[nn.Module] = [
                nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(out_c, affine=True),
                nn.ReLU(inplace=True),
            ]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder
        self.downs = nn.ModuleList()
        in_c = input_channels
        out_c = base_channels
        for i in range(num_downsamples):
            # First down block does not normalize per original pix2pix
            normalize = (i != 0)
            self.downs.append(down_block(in_c, out_c, normalize=normalize))
            in_c = out_c
            out_c = min(out_c * 2, 512)

        # Bottleneck
        bottleneck_c = in_c
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_c, bottleneck_c, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(bottleneck_c, affine=True),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.ups = nn.ModuleList()
        up_in_c = bottleneck_c
        up_out_c = up_in_c

        # mirror the downs; channel sizes reverse order
        channels_list: List[int] = [m[0].out_channels for m in self.downs]  # type: ignore[index]
        skip_channels_reversed = list(reversed(channels_list))

        for i, skip_c in enumerate(skip_channels_reversed):
            # Determine output channels for this up block
            up_out_c = skip_c
            apply_dropout = use_dropout_in_ups and (i < 3)
            self.ups.append(up_block(up_in_c, up_out_c, apply_dropout=apply_dropout))
            # After concatenation with skip, the next in channels grow
            up_in_c = up_out_c + skip_c

        # Final layer to map to output channels without skip connection after last up
        # We do one more deconv to get back to the original resolution
        self.final_deconv = nn.ConvTranspose1d(up_in_c, output_channels, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Tanh()

        self.required_multiple = 2 ** num_downsamples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, T)
        assert x.dim() == 3, "Input must be (N, C, T)"
        length = x.shape[-1]
        if length % self.required_multiple != 0:
            raise ValueError(f"Input length {length} must be divisible by {self.required_multiple} for UNet down/up-sampling")

        skips: List[torch.Tensor] = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h)
            # If due to numerical reasons lengths differ by 1, center-crop the skip
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if abs(diff) <= 2:
                    if diff > 0:
                        skip = skip[..., diff // 2 : skip.shape[-1] - (diff - diff // 2)]
                    else:
                        h = h[..., (-diff) // 2 : h.shape[-1] - ((-diff) - (-diff) // 2)]
                else:
                    raise RuntimeError("Feature map size mismatch beyond tolerance in UNet1D")
            h = torch.cat([h, skip], dim=1)

        h = self.final_deconv(h)
        y = self.final_activation(h)
        return y