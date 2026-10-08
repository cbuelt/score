"""Probabilistic wrapper around local Depth Anything decoder layers."""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.backbones.depthanything_layers import DPTHead
from src.models.backbones.noise import (
    GaussianNoiseInjector,
    require_n_samples,
    stack_member_groups,
)

GroupWidths = Mapping[str, Optional[int]]
HeadConfig = Mapping[str, int]

_RGB_CHANNELS = 3
_PATCH_SIZE = 14
_UPSTREAM_HEAD_CHANNELS = 32
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


def _validate_dinov2_source(source_path: str) -> Path:
    """Validate the local DINOv2 checkout used by ``torch.hub.load``."""
    source = Path(source_path).expanduser().resolve()
    if not (source / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"DINOv2 source was not found at {source}. Clone "
            "facebookresearch/dinov2 there and set ``dinov2_source_path`` in "
            "configs/backbone/depthanything.yaml."
        )
    return source


# ---------------------------------------------------------------------------
# Probabilistic prediction head
# ---------------------------------------------------------------------------
def _group_norm(channels: int, max_groups: int) -> nn.GroupNorm:
    groups = min(channels, max_groups)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _conv_norm_gelu(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    norm_groups: int,
) -> list[nn.Module]:
    """Build one convolutional refinement unit."""
    padding = kernel_size // 2
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        _group_norm(out_channels, norm_groups),
        nn.GELU(),
    ]


class ConvolutionalHeadBlock(nn.Module):
    """Spatial Conv3x3 refinements followed by a 1x1 channel bottleneck."""

    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        bottleneck_channels: int,
        num_blocks: int,
        norm_groups: int,
        bottleneck_norm_groups: int,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

        layers = _conv_norm_gelu(
            in_channels,
            spatial_channels,
            kernel_size=3,
            norm_groups=norm_groups,
        )
        for _ in range(num_blocks - 1):
            layers.extend(
                _conv_norm_gelu(
                    spatial_channels,
                    spatial_channels,
                    kernel_size=3,
                    norm_groups=norm_groups,
                )
            )
        layers.extend(
            _conv_norm_gelu(
                spatial_channels,
                bottleneck_channels,
                kernel_size=1,
                norm_groups=bottleneck_norm_groups,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ProbabilisticDPTHead(DPTHead):
    """Local DPT decoder with its scalar head replaced by grouped predictions.

    The only upstream forward-path section reproduced below is the decoder
    through ``path_1``.  Upstream does not expose ``path_1`` as a method, and
    its next line immediately applies the scalar ``output_conv1``.  Everything
    before that point remains the original Depth Anything implementation.
    """

    def __init__(
        self,
        in_channels: int,
        decoder_channels: Sequence[int],
        group_widths: GroupWidths,
        features: int,
        use_bn: bool,
        use_clstoken: bool,
        head: Optional[HeadConfig],
        other_zero_init: bool = True,
        noise: Optional[dict] = None,
    ) -> None:
        super().__init__(
            nclass=1,
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=list(decoder_channels),
            use_clstoken=use_clstoken,
        )
        # Retain the official pretrained ``output_conv1`` (features -> 32) as a
        # shared low-resolution adapter. Only ``output_conv2``, which turns those
        # features into one positive scalar depth map, is replaced.
        del self.scratch.output_conv2

        config = head or {}
        head_channels = config.get("head_channels", _UPSTREAM_HEAD_CHANNELS)
        bottleneck_channels = config.get("bottleneck_channels", 16)
        norm_groups = config.get("norm_groups", 8)
        bottleneck_norm_groups = config.get("bottleneck_norm_groups", 4)
        shared_num_blocks = config.get("shared_num_blocks", 2)
        other_num_blocks = config.get("other_num_blocks", 2)

        mu_width = group_widths["mu"]
        if mu_width is None:
            raise ValueError("group_widths['mu'] must be a positive integer.")

        self.num_channels = mu_width
        sigma_width = group_widths["sigma"] or 0
        other_width = group_widths["other"]
        if sigma_width and sigma_width % self.num_channels:
            raise ValueError("group_widths['sigma'] must be a multiple of mu width.")
        if other_width is not None and other_width % self.num_channels:
            raise ValueError("group_widths['other'] must be a multiple of mu width.")

        self.shared_groups = (self.num_channels + sigma_width) // self.num_channels
        self.other_groups = (
            other_width // self.num_channels if other_width is not None else 0
        )
        self.has_other = other_width is not None

        # The retained official output_conv1 maps path_1 from ``features`` (64
        # for vits) to 32 channels before full-resolution interpolation.
        self.shared_block = ConvolutionalHeadBlock(
            _UPSTREAM_HEAD_CHANNELS,
            head_channels,
            bottleneck_channels,
            shared_num_blocks,
            norm_groups,
            bottleneck_norm_groups,
        )
        self.shared_proj = self._zero_init(
            nn.Conv2d(
                bottleneck_channels,
                self.num_channels + sigma_width,
                kernel_size=1,
            )
        )
        if self.has_other:
            self.other_block = ConvolutionalHeadBlock(
                _UPSTREAM_HEAD_CHANNELS,
                head_channels,
                bottleneck_channels,
                other_num_blocks,
                norm_groups,
                bottleneck_norm_groups,
            )
            # Zero-init is only sound when the loss has gradient at zero output;
            # methods whose "other" group enters the covariance quadratically
            # (LorD) set other_zero_init=False so the head starts at a small
            # random symmetry-breaking init instead. See
            # PredictiveMethod.other_zero_init.
            init = self._zero_init if other_zero_init else self._small_init
            self.other_proj = init(
                nn.Conv2d(bottleneck_channels, other_width, kernel_size=1)
            )

        # Optional latent-noise injection for the sample-based (SB) baseline,
        # placed on the retained pretrained ``output_conv1`` output --
        # the last tensor of the official model, and deliberately *before* the
        # bilinear upsample to full resolution. path_1 is at roughly 2/7 of the
        # input resolution, so injecting here rather than after the interpolate
        # makes the mixer ~12x cheaper in area for the same latent structure;
        # the head blocks that follow still run per member at full resolution,
        # which is the irreducible cost of M full-resolution samples.
        self.noise_injector = (
            GaussianNoiseInjector(
                feature_dim=_UPSTREAM_HEAD_CHANNELS, channel_axis=1, **noise
            )
            if noise
            else None
        )

    @staticmethod
    def _zero_init(proj: nn.Conv2d) -> nn.Conv2d:
        nn.init.zeros_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)
        return proj

    @staticmethod
    def _small_init(proj: nn.Conv2d) -> nn.Conv2d:
        # Small enough that the initial prediction stays effectively neutral,
        # nonzero so a quadratic group (U Uᵀ) receives gradient at init.
        nn.init.normal_(proj.weight, std=1e-3)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)
        return proj

    def _path_1(
        self,
        out_features: Sequence[tuple[Tensor, Tensor]],
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        """Run the unchanged official DPT decoder up to its ``path_1`` tensor."""
        decoder_layers = []
        for index, feature in enumerate(out_features):
            tokens = feature[0]
            if self.use_clstoken:
                cls_token = feature[1]
                readout = cls_token.unsqueeze(1).expand_as(tokens)
                tokens = self.readout_projects[index](torch.cat((tokens, readout), -1))

            tokens = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0], tokens.shape[-1], patch_h, patch_w
            )
            decoder_layers.append(self.resize_layers[index](self.projects[index](tokens)))

        layer_1, layer_2, layer_3, layer_4 = decoder_layers
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        return self.scratch.refinenet1(path_2, layer_1_rn)

    def _predict_groups(self, features: Tensor) -> Tensor:
        """Predict and stack groups in the method-compatible group-inner order."""
        shared = self.shared_proj(self.shared_block(features))
        batch, _, height, width = shared.shape
        groups = [
            shared.reshape(batch, self.num_channels, self.shared_groups, height, width)
        ]
        if self.has_other:
            other = self.other_proj(self.other_block(features))
            groups.append(
                other.reshape(
                    batch, self.num_channels, self.other_groups, height, width
                )
            )

        output = torch.cat(groups, dim=2)
        return output.flatten(start_dim=1, end_dim=2)

    def forward(
        self,
        out_features: Sequence[tuple[Tensor, Tensor]],
        patch_h: int,
        patch_w: int,
        output_size: tuple[int, int],
        n_samples: Optional[int] = None,
    ) -> Tensor:
        path_1 = self._path_1(out_features, patch_h=patch_h, patch_w=patch_w)

        # Keep the official pretrained Conv3x3(features -> 32), then replace its
        # old scalar output_conv2 with full-resolution probabilistic branches.
        features = self.scratch.output_conv1(path_1)

        members = 1
        if self.noise_injector is not None:
            # Expand to M members while still at decoder resolution, then let
            # the upsample and the head blocks run per member.
            members = require_n_samples(n_samples, type(self).__name__)
            features = self.noise_injector(features, members)

        features = F.interpolate(
            features, size=output_size, mode="bilinear", align_corners=True
        )
        out = self._predict_groups(features)
        if self.noise_injector is None:
            return out
        # One group per member, so C*G == C here; fold the member axis back into
        # the channel axis group-inner, giving (B, C*M, H, W).
        return stack_member_groups(out, members, self.num_channels, output_size)

    def frozen_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [
            self.projects,
            self.resize_layers,
            self.scratch,
            self.shared_block,
            self.shared_proj,
        ]
        if self.use_clstoken:
            modules.append(self.readout_projects)
        return modules


class DepthAnything(nn.Module):
    """Official Depth Anything encoder/decoder with grouped raw prediction heads."""

    def __init__(
        self,
        dinov2_source_path: str,
        in_channels: int = 3,
        out_channels: int = 1,
        group_widths: Optional[GroupWidths] = None,
        encoder: str = "vits",
        features: int = 64,
        decoder_channels: Sequence[int] = (48, 96, 192, 384),
        use_bn: bool = False,
        use_clstoken: bool = False,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = False,
        head: Optional[HeadConfig] = None,
        other_zero_init: bool = True,
        noise: Optional[dict] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if in_channels != _RGB_CHANNELS:
            raise ValueError(
                "The pretrained Depth Anything v1 encoder accepts RGB input only; "
                f"got in_channels={in_channels}."
            )
        if encoder not in {"vits", "vitb", "vitl"}:
            raise ValueError(f"Unsupported encoder {encoder!r}; use vits, vitb, or vitl.")
        if group_widths is None:
            group_widths = {"mu": out_channels, "sigma": None, "other": None}

        dinov2_source = _validate_dinov2_source(dinov2_source_path)
        self.pretrained = torch.hub.load(
            str(dinov2_source), f"dinov2_{encoder}14", source="local", pretrained=False
        )
        embedding_dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = ProbabilisticDPTHead(
            in_channels=embedding_dim,
            decoder_channels=decoder_channels,
            group_widths=group_widths,
            features=features,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            head=head,
            other_zero_init=other_zero_init,
            noise=noise,
        )
        self.group_widths = group_widths
        # Total emitted channels, like PatchTST/DhariwalUNet expose. Derived from
        # the resolved group_widths so it cannot disagree with what the head
        # actually emits (mirrors PredictiveMethod.out_channels).
        self.out_channels = sum(w for w in group_widths.values() if w is not None)
        self.register_buffer(
            "image_mean", torch.tensor(_IMAGE_MEAN).view(1, _RGB_CHANNELS, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(_IMAGE_STD).view(1, _RGB_CHANNELS, 1, 1)
        )
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)
        self.encoder_frozen = freeze_encoder
        if self.encoder_frozen:
            self._freeze_encoder()

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load official weights while intentionally skipping the old scalar head."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Depth Anything checkpoint not found: {path}. "
                "Download a compatible checkpoint and update ``checkpoint_path``."
            )
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        result = self.load_state_dict(state_dict, strict=False)
        # strict=False exists only to tolerate the *known* mismatches: our new
        # probabilistic head modules (absent from the official checkpoint) and
        # the deleted upstream scalar head (present only there). Anything else
        # missing means encoder/decoder weights silently stayed at random init,
        # so fail loudly instead of training on garbage.
        new_modules = (
            "depth_head.shared_block.",
            "depth_head.shared_proj.",
            "depth_head.other_block.",
            "depth_head.other_proj.",
            # The latent-noise injector of the SB baseline: like the
            # probabilistic heads it has no counterpart upstream, so it is
            # legitimately absent from an official checkpoint.
            "depth_head.noise_injector.",
            "image_mean",
            "image_std",
        )
        missing = [k for k in result.missing_keys if not k.startswith(new_modules)]
        unexpected = [
            k
            for k in result.unexpected_keys
            if not k.startswith("depth_head.scratch.output_conv2.")
        ]
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint {path} does not match the model: "
                f"missing keys {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"unexpected keys {unexpected[:5]}"
                f"{'...' if len(unexpected) > 5 else ''}. Only the new "
                "probabilistic head may be absent from a checkpoint and only "
                "the removed scalar output_conv2 may be extra."
            )

    def forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        height, width = x.shape[-2:]
        patch_h = (height + _PATCH_SIZE - 1) // _PATCH_SIZE
        patch_w = (width + _PATCH_SIZE - 1) // _PATCH_SIZE
        padded_height, padded_width = _PATCH_SIZE * patch_h, _PATCH_SIZE * patch_w

        x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)
        if (padded_height, padded_width) != (height, width):
            x = F.pad(
                x,
                (0, padded_width - width, 0, padded_height - height),
                mode="replicate",
            )
        out_features = self.pretrained.get_intermediate_layers(
            x, n=4, return_class_token=True
        )
        output = self.depth_head(
            out_features,
            patch_h=patch_h,
            patch_w=patch_w,
            output_size=(padded_height, padded_width),
            n_samples=n_samples,
        )
        return output[..., :height, :width]

    def _freeze_encoder(self) -> None:
        """Freeze the pretrained DINOv2 encoder for head-only adaptation."""
        self.pretrained.eval()
        for parameter in self.pretrained.parameters():
            parameter.requires_grad = False

    def _frozen_modules(self) -> list[nn.Module]:
        return [self.pretrained, *self.depth_head.frozen_modules()]

    def freeze_for_stage2(self) -> None:
        """Freeze DINO/DPT/mu-sigma, leaving only the ``other`` head trainable."""
        if not self.depth_head.has_other:
            raise ValueError(
                "freeze_for_stage2() requires group_widths['other'], but this "
                "backbone was built without an other head."
            )
        for module in self._frozen_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in (self.depth_head.other_block, self.depth_head.other_proj):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> "DepthAnything":
        super().train(mode)
        shared_frozen = all(
            not parameter.requires_grad
            for parameter in self.depth_head.shared_proj.parameters()
        )
        if mode:
            if self.encoder_frozen:
                self.pretrained.eval()
            if shared_frozen:
                for module in self._frozen_modules():
                    module.eval()
        return self
