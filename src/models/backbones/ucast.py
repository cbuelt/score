# ----------------------------------------------------------------------------
# Adapted from the U-Cast repository:
#   https://github.com/Rose-STL-Lab/u-cast/tree/main/src/models
# (src/models/networks_edm.py), which itself adapts the ADM / EDM architecture:
#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Changes relative to the source: the `BaseModel` Lightning base class has been
# dropped so that `DhariwalUNet` is a plain `torch.nn.Module`. It now receives
# its input/output channels and spatial resolution directly as constructor
# arguments (like the other backbones in this repo) instead of reading them
# from base-class attributes, and the conditioning logic has been removed. The
# single output layer has been replaced by the group-structured head described
# in `DhariwalUNet`.
# ----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import silu

from src.models.backbones.noise import (
    GaussianNoiseInjector,
    require_n_samples,
    stack_member_groups,
)


# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel**2,
            fan_out=out_channels * kernel**2,
        )
        weight_shape = [out_channels, in_channels, kernel, kernel]
        self.weight = (
            torch.nn.Parameter(weight_init(weight_shape, **init_kwargs) * init_weight)
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0

        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            x = F.avg_pool2d(x, kernel_size=2)
        if w is not None:
            x = F.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(q.dtype) / np.sqrt(
            k.shape[1]
        )
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(k.dtype) / np.sqrt(
            k.shape[1]
        )
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_proj=False,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = (
            0
            if attention is False
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.skip_scale = skip_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, **init
        )

        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                **init,
            )

        if self.num_heads:
            init_attn = init_attn if init_attn is not None else init
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels, out_channels=out_channels * 3, kernel=1, **init_attn
            )
            self.proj = Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero
            )

    def forward(self, x):
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        x = self.conv1(self.dropout(silu(self.norm1(x))))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            b, c, h, w = x.shape
            B2, C2 = b * self.num_heads, c // self.num_heads
            q, k, v = self.qkv(self.norm2(x)).reshape(B2, C2, 3, -1).unbind(2)
            attn_w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", attn_w, v)
            # "(b nh) c (h w) -> b (nh c) h w": merge the head and channel dims
            # back together. Equivalent to einops.rearrange (dropped to avoid the
            # extra dependency).
            a = a.reshape(b, self.num_heads, C2, h, w).reshape(b, c, h, w)
            x = self.proj(a).add_(x) * self.skip_scale

        return x


# ----------------------------------------------------------------------------
# ADM architecture (DhariwalUNet).


def _block_stack(channels: int, n_blocks: int, **block_kwargs) -> torch.nn.Module:
    """A stack of ``n_blocks`` residual ``UNetBlock``s at fixed ``channels`` width.

    Used as the (shared or "other") head trunk. ``UNetBlock.forward`` is a single
    tensor -> tensor map, so a plain ``Sequential`` composes them. ``n_blocks == 1``
    returns a single block (no wrapper). Used to expose head *depth* as a size knob.
    """
    blocks = [
        UNetBlock(in_channels=channels, out_channels=channels, **block_kwargs)
        for _ in range(n_blocks)
    ]
    return blocks[0] if n_blocks == 1 else torch.nn.Sequential(*blocks)


class DhariwalUNet(torch.nn.Module):
    """ADM / EDM U-Net (Dhariwal & Nichol) as a plain ``torch.nn.Module``.

    Follows the common backbone interface of this repo: the input/output channels
    and the spatial resolution are passed directly as constructor arguments.

    The output head is split into two ``UNetBlock``-based sub-heads whose outputs
    are stacked in the channel dim in group-inner order (per physical channel its
    groups are contiguous), matching each predictive method's ``process_output``
    (``reshape(B, C, G, ...)``):

    * A **shared** head produces the ``mu`` and ``sigma`` groups from one shared
      block trunk plus one zero-init projection (they share all parameters and
      differ only in their output-channel slice).
    * An optional **other** head produces the third group (the SCORE core or a
      low-rank factor). It is built only when the method has an ``"other"`` group.

    Which sub-heads exist and how wide each is come from ``group_widths`` (see
    :class:`~src.methods.base.PredictiveMethod`). When ``group_widths is None`` the
    backbone falls back to a single flat head of width ``out_channels`` (e.g. for
    config-only instantiation).

    Args:
        img_resolution: Spatial resolution of the (square) input. Used to name
            the per-level blocks and to size the down/up sampling pyramid; it
            should be the height/width of the input feature map.
        in_channels: Number of input channels.
        out_channels: Number of output channels (fallback head width when
            ``group_widths is None``).
        group_widths: Per-group absolute channel widths
            ``{"mu": C, "sigma": C|None, "other": k*C|None}`` from the predictive
            method, or ``None`` for the single flat-head fallback.
        head: Size-only head config; keys ``shared_num_blocks`` and
            ``other_num_blocks`` set the ``UNetBlock`` depth of each sub-head
            (default 1 each).
        other_zero_init: Whether the ``"other"`` projection starts at exactly
            zero (the default). Methods whose ``"other"`` group has no gradient
            at zero output (LorD) pass ``False``, see
            :attr:`~src.methods.base.PredictiveMethod.other_zero_init`.
        model_channels: Base channel count, scaled by ``channel_mult``.
        channel_mult: Per-resolution channel multipliers.
        num_blocks: Number of residual blocks per resolution.
        attn_levels: Resolution levels (indices into ``channel_mult``) that use
            self-attention. Defaults to none.
        channels_per_head: Channels per attention head.
        dropout: Dropout probability inside the residual blocks.
    """

    def __init__(
        self,
        img_resolution: int,
        in_channels: int = 3,
        out_channels: int = 1,
        group_widths: dict = None,
        head: dict = None,
        model_channels: int = 192,
        channel_mult=(1, 2, 3, 4),
        num_blocks: int = 3,
        attn_levels=None,
        channels_per_head: int = 64,
        dropout: float = 0.10,
        other_zero_init: bool = True,
        noise: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        attn_levels = attn_levels or []

        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            channels_per_head=channels_per_head,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin, cout = cout, model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin, cout = cout, model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(level in attn_levels),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )

            for idx in range(num_blocks + 1):
                cin, cout = cout + skips.pop(), model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(level in attn_levels),
                    **block_kwargs,
                )

        # ---- Output heads -------------------------------------------------
        # Resolve per-group widths. `cout` here is the trunk output width
        # (= model_channels * channel_mult[0]).
        head = head or {}
        shared_num_blocks = head.get("shared_num_blocks", 1)
        other_num_blocks = head.get("other_num_blocks", 1)

        if group_widths is None:
            # Fallback: a single flat head of width `out_channels`.
            group_widths = {"mu": out_channels, "sigma": None, "other": None}
        self._group_widths = group_widths

        mu_w = group_widths["mu"]
        sigma_w = group_widths["sigma"] or 0
        other_w = group_widths["other"]
        # Raw channel count C; mu is always exactly C, so it names the group-inner
        # reshape in forward.
        self._num_channels = mu_w
        shared_w = mu_w + sigma_w  # mu and sigma share a trunk, split at the proj

        # Shared mu/sigma head: shared UNetBlock trunk + single zero-init proj.
        self.shared_block = _block_stack(cout, shared_num_blocks, **block_kwargs)
        self.shared_norm = GroupNorm(num_channels=cout)
        self.shared_proj = Conv2d(
            in_channels=cout, out_channels=shared_w, kernel=3, **init_zero
        )

        # Optional "other" head (SCORE core / low-rank factor), only when the
        # method has an "other" group. Its depth is config-driven.
        self.has_other = other_w is not None
        if self.has_other:
            self.other_block = _block_stack(cout, other_num_blocks, **block_kwargs)
            self.other_norm = GroupNorm(num_channels=cout)
            # Zero-init is only sound when the loss has gradient at zero output;
            # methods whose "other" group enters the covariance quadratically
            # (LorD) set other_zero_init=False so the projection starts at a
            # small random symmetry-breaking init instead (kaiming scaled down to
            # keep the initial prediction effectively neutral). See
            # PredictiveMethod.other_zero_init.
            init_small = dict(init_mode="kaiming_uniform", init_weight=1e-3, init_bias=0)
            self.other_proj = Conv2d(
                in_channels=cout, out_channels=other_w, kernel=3,
                **(init_zero if other_zero_init else init_small),
            )

        # Optional latent-noise injection for the sample-based (SB) baseline.
        # It sits on the decoder output, so the whole encoder/decoder
        # pyramid runs once per case and only the head blocks are replayed per
        # member -- the latent still enters at full spatial resolution, which is
        # what lets a member differ from its siblings pixel by pixel.
        self.noise_injector = (
            GaussianNoiseInjector(feature_dim=cout, channel_axis=1, **noise)
            if noise
            else None
        )

    def _apply_head(self, block, norm, proj, x):
        """Run one sub-head: ``proj(silu(norm(block(x))))`` (as in the original output layer)."""
        return proj(silu(norm(block(x))))

    def forward(self, x, n_samples=None):
        skips = []
        for block in self.enc.values():
            x = block(x)
            skips.append(x)

        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                skip = skips.pop()
                if skip.shape[-2:] != x.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
                x = torch.cat([x, skip], dim=1)
            x = block(x)

        members = 1
        if self.noise_injector is not None:
            # (B, cout, H, W) -> (B*M, cout, H, W): the head below is replayed
            # once per member, and the members are folded back into the channel
            # axis at the end.
            members = require_n_samples(n_samples, type(self).__name__)
            x = self.noise_injector(x, members)

        # Group-inner stacking: (B, group_w, H, W) -> (B, C, groups_per_channel, H, W)
        # per head, concat on the group axis, then flatten back to (B, C*G, H, W).
        # This is the exact inverse of each method's process_output reshape.
        b, _, h, w = x.shape
        shared = self._apply_head(
            self.shared_block, self.shared_norm, self.shared_proj, x
        )
        groups = [shared.reshape(b, self._num_channels, -1, h, w)]
        if self.has_other:
            other = self._apply_head(
                self.other_block, self.other_norm, self.other_proj, x
            )
            groups.append(other.reshape(b, self._num_channels, -1, h, w))
        out = torch.cat(groups, dim=2)  # (B, C, G, H, W)
        out = out.reshape(b, self._num_channels * out.shape[2], h, w)
        if self.noise_injector is None:
            return out
        # One group per member, so C*G == C here; fold the member axis back into
        # the channel axis group-inner, giving (B, C*M, H, W).
        return stack_member_groups(out, members, self._num_channels, (h, w))

    def freeze_for_stage2(self):
        """Freeze the trunk + shared mu/sigma head, leaving only ``other`` trainable.

        Used by stage 2 of the two-stage fit: disables grads on the encoder/decoder
        and the shared (mu/sigma) head and puts them in eval mode so their dropout
        is deterministic, so mu and D = diag(sqrt(tau)) stay bit-exactly fixed
        while the correlation core (the ``other`` head) is trained.
        """
        if not self.has_other:
            raise ValueError(
                "freeze_for_stage2() requires an 'other' head, but this backbone "
                "was built without one (group_widths['other'] is None)."
            )
        frozen = [
            self.enc,
            self.dec,
            self.shared_block,
            self.shared_norm,
            self.shared_proj,
        ]
        for module in frozen:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        for module in (self.other_block, self.other_norm, self.other_proj):
            for p in module.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen trunk + shared head in eval mode once stage 2 froze them.

        Lightning calls ``model.train()`` at the start of every train epoch, which
        would otherwise flip the frozen ``UNetBlock`` dropout back into training
        mode. We detect the stage-2 freeze via ``shared_proj`` having no trainable
        params and re-assert eval mode so mu/D stay fixed.
        """
        super().train(mode)
        shared_frozen = all(
            not p.requires_grad for p in self.shared_proj.parameters()
        )
        if mode and shared_frozen:
            for module in (
                self.enc,
                self.dec,
                self.shared_block,
                self.shared_norm,
                self.shared_proj,
            ):
                module.eval()
        return self
