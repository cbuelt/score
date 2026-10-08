"""Local Depth Anything DPT decoder components.

Adapted from LiheYoung/Depth-Anything. The decoder is kept local so the
probabilistic wrapper does not require the upstream ``depth_anything`` package.
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualConvUnit(nn.Module):
    """Two residual 3x3 convolutions used by the DPT refinement blocks."""

    def __init__(self, features: int, activation: nn.Module, use_batch_norm: bool) -> None:
        super().__init__()
        self.bn = use_batch_norm
        self.activation = activation
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        output = self.activation(x)
        output = self.conv1(output)
        if self.bn:
            output = self.bn1(output)
        output = self.activation(output)
        output = self.conv2(output)
        if self.bn:
            output = self.bn2(output)
        return self.skip_add.add(output, x)


class FeatureFusionBlock(nn.Module):
    """Fuse and upsample DPT decoder features."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        *,
        use_batch_norm: bool,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.resConfUnit1 = ResidualConvUnit(features, activation, use_batch_norm)
        self.resConfUnit2 = ResidualConvUnit(features, activation, use_batch_norm)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor, residual: Tensor | None = None, *, size: tuple[int, int] | None = None) -> Tensor:
        if residual is not None:
            x = self.skip_add.add(x, self.resConfUnit1(residual))

        x = self.resConfUnit2(x)
        interpolate_kwargs = {"size": size} if size is not None else {"scale_factor": 2}
        x = F.interpolate(
            x,
            **interpolate_kwargs,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return self.out_conv(x)


def _make_scratch(in_channels: Sequence[int], features: int) -> nn.Module:
    """Build the channel adapters used before DPT feature fusion."""
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_channels[0], features, kernel_size=3, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_channels[1], features, kernel_size=3, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_channels[2], features, kernel_size=3, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_channels[3], features, kernel_size=3, padding=1, bias=False)
    return scratch


def _fusion_block(features: int, use_batch_norm: bool) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=False),
        use_batch_norm=use_batch_norm,
        align_corners=True,
    )


class DPTHead(nn.Module):
    """Depth Anything DPT decoder with the original checkpoint module names."""

    def __init__(
        self,
        nclass: int,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        use_clstoken: bool = False,
    ) -> None:
        super().__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList(
            nn.Conv2d(in_channels, channels, kernel_size=1)
            for channels in out_channels
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                for _ in out_channels
            )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _fusion_block(features, use_bn)
        self.scratch.refinenet2 = _fusion_block(features, use_bn)
        self.scratch.refinenet3 = _fusion_block(features, use_bn)
        self.scratch.refinenet4 = _fusion_block(features, use_bn)

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, nclass, kernel_size=1),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                features,
                features // 2,
                kernel_size=3,
                padding=1,
            )
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Identity(),
            )

    def forward(
        self,
        out_features: Sequence[tuple[Tensor, Tensor]],
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        decoder_layers: list[Tensor] = []
        for index, (tokens, cls_token) in enumerate(out_features):
            if self.use_clstoken:
                readout = cls_token.unsqueeze(1).expand_as(tokens)
                tokens = self.readout_projects[index](torch.cat((tokens, readout), dim=-1))

            tokens = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0],
                tokens.shape[-1],
                patch_h,
                patch_w,
            )
            decoder_layers.append(self.resize_layers[index](self.projects[index](tokens)))

        layer1, layer2, layer3, layer4 = decoder_layers
        layer1 = self.scratch.layer1_rn(layer1)
        layer2 = self.scratch.layer2_rn(layer2)
        layer3 = self.scratch.layer3_rn(layer3)
        layer4 = self.scratch.layer4_rn(layer4)
        path4 = self.scratch.refinenet4(layer4, size=layer3.shape[2:])
        path3 = self.scratch.refinenet3(path4, layer3, size=layer2.shape[2:])
        path2 = self.scratch.refinenet2(path3, layer2, size=layer1.shape[2:])
        path1 = self.scratch.refinenet1(path2, layer1)

        output = self.scratch.output_conv1(path1)
        output = F.interpolate(
            output,
            size=(patch_h * 14, patch_w * 14),
            mode="bilinear",
            align_corners=True,
        )
        return self.scratch.output_conv2(output)
