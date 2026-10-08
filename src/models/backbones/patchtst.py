# ----------------------------------------------------------------------------
# Adapted from the PatchTST repository:
#   https://github.com/yuqinie98/PatchTST (PatchTST_supervised/models/PatchTST.py)
#
# Changes relative to the source: the `configs`-namespace constructor and the
# `series_decomp` two-backbone branch have been dropped so that `PatchTST` is a
# plain `torch.nn.Module` matching this repo's backbone contract (it receives
# `in_channels` / `out_channels` / `group_widths` directly, like the other
# backbones, e.g. `DhariwalUNet`). The single `Flatten_Head` has been replaced by
# the group-structured head described below.
# ----------------------------------------------------------------------------

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.backbones.noise import (
    GaussianNoiseInjector,
    require_n_samples,
    stack_member_groups,
)
from src.models.backbones.patchtst_layers import (
    RevIN,
    Transpose,
    TSTEncoder,
    TSTiEncoder,
    get_activation_fn,
)


def _norm_layer(norm: str, d_model: int) -> nn.Module:
    """The head's pre-projection norm, matching `TSTEncoderLayer`'s convention.

    ``BatchNorm`` normalizes over the token axis and so has to be sandwiched
    between transposes (the tokens are ``(bs, patch_num, d_model)``); LayerNorm
    applies to the feature axis directly.
    """
    if "batch" in norm.lower():
        return nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
        )
    return nn.LayerNorm(d_model)


def _softplus_inv(y: Tensor, eps: float) -> Tensor:
    """Inverse of ``softplus``: the ``x`` with ``softplus(x) = y`` (for ``y > 0``).

    Used to push a multiplicative rescaling of a *variance* back through the
    ``softplus`` that the predictive method applies to the head output (see
    :meth:`PatchTST._denorm_groups`). ``log(expm1(y))`` overflows for large ``y``,
    so use the equivalent ``y + log(-expm1(-y))``, which is stable: the correction
    term vanishes as ``y`` grows.
    """
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y))


class PatchTST(nn.Module):
    """Channel-independent patch transformer with a group-structured output head.

    The trunk is stock PatchTST: RevIN -> patching -> a channel-independent
    transformer encoder (:class:`TSTiEncoder`) producing per-variable patch
    embeddings ``(B, C, d_model, patch_num)``.

    The head mirrors :class:`~src.models.backbones.ucast.DhariwalUNet`: instead of
    a single ``Flatten_Head`` linear it is split into two sub-heads whose outputs
    are stacked in the channel dim in group-inner order (per variable its groups
    are contiguous), matching each predictive method's ``process_output``
    (``reshape(B, C, G, ...)``):

    * A **shared** head produces the ``mu`` and ``sigma`` groups from one shared
      block trunk plus one zero-init projection (they share all parameters and
      differ only in their output slice).
    * An optional **other** head produces the third group (the SCORE core or a
      low-rank factor). It is built only when the method has an ``"other"`` group.

    Each sub-head is a stack of ``TSTEncoderLayer``s -- the same block the trunk
    uses, so head depth and trunk depth are the same currency -- followed by a
    norm, an activation, dropout, and a zero-init ``Linear`` flattening the patch
    embeddings to the forecast horizon. With the default ``n_layers=2`` trunk and
    depth-1 sub-heads every path through the network is 3 encoder layers deep, the
    depth upstream PatchTST was tuned at.

    Which sub-heads exist and how wide each is come from ``group_widths`` (see
    :class:`~src.methods.base.PredictiveMethod`). When ``group_widths is None`` the
    backbone falls back to a single flat head of width ``out_channels`` (the
    deterministic / config-only case).

    Args:
        in_channels: Number of input variables.
        out_channels: Number of target variables (fallback head width when
            ``group_widths is None``). Must equal ``in_channels``: the encoder is
            channel-independent, so it emits one stream per input variable.
        seq_len: Length of the input (context) window.
        pred_len: Length of the forecast horizon; the target event axis.
        group_widths: Per-group absolute channel widths
            ``{"mu": C, "sigma": C|None, "other": k*C|None}`` from the predictive
            method, or ``None`` for the single flat-head fallback.
        head: Size-only head config; keys ``shared_num_blocks`` /
            ``other_num_blocks`` set the ``TSTEncoderLayer`` depth of each sub-head
            (default 1 each) and ``head_dropout`` the dropout before each
            projection.
        group_denorm: Per-group RevIN denormalization rule, see
            :meth:`_denorm_groups`.
        other_zero_init: Whether the ``"other"`` projection starts at exactly
            zero (the default). Methods whose ``"other"`` group has no gradient
            at zero output (LorD) pass ``False``, see
            :attr:`~src.methods.base.PredictiveMethod.other_zero_init`.
        patch_len: Patch (token) length.
        stride: Patch stride.
        padding_patch: ``"end"`` replication-pads the context so the final partial
            patch is kept; ``None`` drops it.
        n_layers: Number of trunk encoder layers.
        d_model: Token embedding width.
        n_heads: Attention heads per encoder layer.
        d_ff: Feed-forward width inside an encoder layer.
        norm: ``"BatchNorm"`` or ``"LayerNorm"``, used by trunk and head alike.
        dropout: Residual dropout inside the encoder layers.
        attn_dropout: Dropout on the attention weights.
        act: Activation inside the encoder layers and before each head projection.
        revin: Enable reversible per-window instance normalization.
        affine: Give RevIN learnable per-variable affine parameters.
        subtract_last: Center on the last context value instead of the window mean.
        eps: Numerical floor, shared with the ``softplus`` inversion in the denorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        group_widths: dict = None,
        head: dict = None,
        group_denorm: dict = None,
        other_zero_init: bool = True,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = "end",
        n_layers: int = 2,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        eps: float = 1e-5,
        noise: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pred_len = pred_len
        self.eps = eps

        # ---- RevIN ---------------------------------------------------------
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                in_channels, eps=eps, affine=affine, subtract_last=subtract_last
            )

        # ---- Patching ------------------------------------------------------
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num

        # ---- Trunk ---------------------------------------------------------
        self.backbone = TSTiEncoder(
            in_channels,
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            res_attention=res_attention,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
            **kwargs,
        )

        # ---- Output heads --------------------------------------------------
        head = head or {}
        shared_num_blocks = head.get("shared_num_blocks", 1)
        other_num_blocks = head.get("other_num_blocks", 1)
        head_dropout = head.get("head_dropout", 0.0)

        if group_widths is None:
            # Fallback: a single flat head of width `out_channels`.
            group_widths = {"mu": out_channels, "sigma": None, "other": None}
        self._group_widths = group_widths

        mu_w = group_widths["mu"]
        sigma_w = group_widths["sigma"] or 0
        other_w = group_widths["other"]
        # Raw variable count C; mu is always exactly C, so it names the group-inner
        # reshape in forward. (`out_channels` is the *total* head width C*G, not C.)
        self._num_channels = mu_w
        if mu_w != in_channels:
            # The encoder is channel-independent: it emits exactly one stream per
            # input variable and the head is shared across them. A target with
            # fewer channels than the input (features=MS) would need the head to
            # select or mix streams, which is not implemented.
            raise ValueError(
                "PatchTST is channel-independent, so the target must have one "
                f"channel per input variable; got {mu_w} target channels for "
                f"{in_channels} input variables. Use datamodule.features=M (or S), "
                "not MS."
            )
        # Groups *per variable*: the head is shared across variables (the encoder
        # is channel-independent), so it projects to `g * pred_len` per variable
        # rather than to the absolute width.
        self._g_shared = (mu_w + sigma_w) // mu_w  # 1 (mu only) or 2 (mu + sigma)
        self.has_other = other_w is not None
        self._g_other = (other_w // mu_w) if self.has_other else 0

        # RevIN denorm rule per group, in stacking order. Defaults suit every
        # method whose "other" group is scale-free (SCORE's core is
        # unit-mean by construction); LorD's U factor scales linearly with the
        # data and so needs `other: "scale"`.
        self.group_denorm = {"mu": "loc", "sigma": "var", "other": "none"}
        self.group_denorm.update(group_denorm or {})

        block_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=act,
            res_attention=res_attention,
            pre_norm=pre_norm,
        )
        nf = d_model * patch_num

        self.head_act = get_activation_fn(act)
        self.head_dropout = nn.Dropout(head_dropout)

        # Shared mu/sigma head: shared encoder-block trunk + single zero-init proj.
        self.shared_block = TSTEncoder(
            patch_num, n_layers=shared_num_blocks, **block_kwargs
        )
        self.shared_norm = _norm_layer(norm, d_model)
        self.shared_proj = self._zero_init(nn.Linear(nf, self._g_shared * pred_len))

        # Optional "other" head (SCORE core / low-rank factor), only when the
        # method has an "other" group. Its depth is config-driven.
        if self.has_other:
            self.other_block = TSTEncoder(
                patch_num, n_layers=other_num_blocks, **block_kwargs
            )
            self.other_norm = _norm_layer(norm, d_model)
            # Zero-init is only sound when the loss has gradient at zero output;
            # methods whose "other" group enters the covariance quadratically
            # (LorD) set other_zero_init=False so the projection starts at a
            # small random symmetry-breaking init instead. See
            # PredictiveMethod.other_zero_init.
            init = self._zero_init if other_zero_init else self._small_init
            self.other_proj = init(nn.Linear(nf, self._g_other * pred_len))

        # Optional latent-noise injection for the sample-based (SB) baseline.
        # It sits on the trunk's token stream, i.e. after the whole
        # channel-independent transformer encoder and immediately before the
        # head: the encoder is the expensive, transferable part and runs once
        # per case however many members are drawn, while the head -- two of the
        # same encoder blocks plus the flattening projection -- is what gets
        # replayed. Note the latent enters *inside* RevIN, so each member is
        # denormalized by that case's own window statistics.
        self.noise_injector = (
            GaussianNoiseInjector(feature_dim=d_model, channel_axis=-1, **noise)
            if noise
            else None
        )

    @staticmethod
    def _zero_init(linear: nn.Linear) -> nn.Linear:
        """Zero the projection so the head starts at a neutral prediction.

        Mirrors ucast's ``init_zero`` output convs. At init the head emits 0, i.e.
        ``mu`` = the RevIN window mean, ``diag = softplus(0) ~ 0.69``, and (for
        SCORE) a flat spectrum, i.e. an identity core.
        """
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        return linear

    @staticmethod
    def _small_init(linear: nn.Linear) -> nn.Linear:
        """Small random projection init for quadratic ``"other"`` groups.

        Small enough that the initial prediction stays effectively neutral,
        nonzero so a quadratic group (LorD's ``U Uᵀ``) receives gradient at
        init -- at exactly zero it would sit on a stationary point forever.
        """
        nn.init.normal_(linear.weight, std=1e-3)
        nn.init.zeros_(linear.bias)
        return linear

    def _apply_head(self, block, norm, proj, u: Tensor, n_vars: int) -> Tensor:
        """Run one sub-head: ``proj(flatten(dropout(act(norm(block(u))))))``.

        ``u`` is the trunk's token stream ``(B*C, patch_num, d_model)``; the result
        is ``(B, C, g, pred_len)``.
        """
        z = self.head_dropout(self.head_act(norm(block(u))))
        z = z.reshape(-1, n_vars, z.shape[-2] * z.shape[-1])  # (B, C, nf)
        z = proj(z)  # (B, C, g * pred_len)
        return z.reshape(z.shape[0], n_vars, -1, self.pred_len)

    def _denorm_groups(self, out: Tensor, n_samples: int = 1) -> Tensor:
        """Undo the RevIN normalization group-wise on ``(B, C, G, pred_len)``.

        RevIN maps ``x -> ((x - m) / s) * gamma + beta`` per (batch, variable), so
        the inverse map on the target is ``y = (y_n - beta) / gamma * s + m``: a
        location shift ``m`` and a scale ``a = s / gamma``. A Gaussian predictive
        therefore denormalizes as ``mu = a * mu_n + m`` and ``Sigma = a^2 Sigma_n``
        (``a`` is constant over the horizon, so it commutes with the transform B
        and the SCORE core is unaffected).

        The head emits *pre-activation* values and the predictive method applies
        ``softplus`` afterwards, so a variance cannot simply be multiplied here.
        Each group is denormalized by its own rule:

        * ``loc`` (``mu``): the plain affine inverse.
        * ``var`` (``sigma``): ``softplus_inv(a^2 * softplus(x))``, so that the
          method's ``softplus(.) + eps`` yields exactly ``a^2`` times the
          normalized variance.
        * ``scale``: multiply by ``a`` (a quantity that scales linearly, e.g.
          LorD's low-rank factor ``U``).
        * ``none``: identity -- for a scale-free group such as
          :class:`~src.methods.score.SCORE`'s unit-mean core (scale gauge),
          whose scale is carried by ``sigma`` instead.

        With a noise injector the head output is ``(B*M, C, G, pred_len)`` while
        the RevIN statistics are still one per *case*, so ``n_samples`` repeats
        them member-wise. It must be ``repeat_interleave``, matching how the
        injector expanded the batch: every member of a case has to be
        denormalized by that case's own window mean and scale.
        """
        revin = self.revin_layer
        # RevIN's stats are (B, 1, C); the head output is (B, C, G, pred_len).
        center = revin.last if revin.subtract_last else revin.mean
        m = center.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        s = revin.stdev.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        if n_samples > 1:
            m = m.repeat_interleave(n_samples, dim=0)
            s = s.repeat_interleave(n_samples, dim=0)
        if revin.affine:
            gamma = revin.affine_weight.view(1, -1, 1, 1) + revin.eps * revin.eps
            beta = revin.affine_bias.view(1, -1, 1, 1)
        else:
            gamma, beta = 1.0, 0.0
        a = s / gamma  # per-(batch, variable) scale of the inverse map

        # Group widths per variable, in stacking order: mu is always 1 group wide,
        # sigma is present iff the shared head is 2 wide, other iff has_other.
        groups, start = [], 0
        for name, width in (
            ("mu", 1),
            ("sigma", self._g_shared - 1),
            ("other", self._g_other),
        ):
            if width == 0:
                continue
            x = out[:, :, start : start + width]
            start += width
            rule = self.group_denorm[name]
            if rule == "loc":
                x = (x - beta) / gamma * s + m
            elif rule == "var":
                x = _softplus_inv(a.pow(2) * F.softplus(x), self.eps)
            elif rule == "scale":
                x = x * a
            elif rule != "none":
                raise ValueError(
                    f"Unknown group_denorm rule {rule!r} for group {name!r}; "
                    "expected one of 'loc', 'var', 'scale', 'none'."
                )
            groups.append(x)
        return torch.cat(groups, dim=2)

    def forward(self, x: Tensor, n_samples: Optional[int] = None) -> Tensor:
        # x: (B, C, seq_len) -> (B, C*G, pred_len), or (B, C*M, pred_len) when a
        # noise injector expands the batch into M members.
        n_vars = x.shape[1]

        if self.revin:
            x = self.revin_layer(x.permute(0, 2, 1), "norm").permute(0, 2, 1)

        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        # (B, C, seq_len) -> (B, C, patch_num, patch_len) -> (B, C, patch_len, patch_num)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)

        z = self.backbone(x)  # (B, C, d_model, patch_num)
        # Back to the token layout the encoder blocks consume.
        u = z.permute(0, 1, 3, 2)  # (B, C, patch_num, d_model)

        members = 1
        if self.noise_injector is not None:
            # (B, C, patch_num, d_model) -> (B*M, C, patch_num, d_model). The
            # global latent is shared across variables *and* patches, so it can
            # correlate the whole forecast field; the local one varies per
            # (variable, patch).
            members = require_n_samples(n_samples, type(self).__name__)
            u = self.noise_injector(u, members)
        u = u.reshape(-1, self.patch_num, u.shape[-1])

        # Group-inner stacking: each sub-head gives (B, C, g, pred_len); concat on
        # the group axis, then flatten back to (B, C*G, pred_len). This is the exact
        # inverse of each method's process_output reshape.
        groups = [
            self._apply_head(
                self.shared_block, self.shared_norm, self.shared_proj, u, n_vars
            )
        ]
        if self.has_other:
            groups.append(
                self._apply_head(
                    self.other_block, self.other_norm, self.other_proj, u, n_vars
                )
            )
        out = torch.cat(groups, dim=2)  # (B, C, G, pred_len)

        if self.revin:
            out = self._denorm_groups(out, members)

        out = out.reshape(
            out.shape[0], self._num_channels * out.shape[2], self.pred_len
        )
        if self.noise_injector is None:
            return out
        # One group per member, so C*G == C here; fold the member axis back into
        # the channel axis group-inner, giving (B, C*M, pred_len).
        return stack_member_groups(
            out, members, self._num_channels, (self.pred_len,)
        )

    def _frozen_modules(self) -> list:
        """Trunk + shared mu/sigma head: everything stage 2 holds fixed."""
        modules = [self.backbone, self.shared_block, self.shared_norm, self.shared_proj]
        if self.revin:
            # RevIN's affine params rescale mu and sigma, so they are part of the
            # mu/D pathway and must be frozen with it.
            modules.append(self.revin_layer)
        return modules

    def freeze_for_stage2(self):
        """Freeze the trunk + shared mu/sigma head, leaving only ``other`` trainable.

        Mirrors :meth:`src.models.backbones.ucast.DhariwalUNet.freeze_for_stage2`
        for the two-stage fit: disables grads on the encoder and the shared
        (mu/sigma) head and puts them in eval mode, so mu and D = diag(sqrt(tau))
        stay bit-exactly fixed while the correlation core (the ``other`` head) is
        trained. Eval mode matters twice over here: it freezes the encoder dropout
        *and* stops ``TSTEncoderLayer``'s BatchNorm running stats from drifting,
        which would otherwise move mu even with every parameter frozen.
        """
        if not self.has_other:
            raise ValueError(
                "freeze_for_stage2() requires an 'other' head, but this backbone "
                "was built without one (group_widths['other'] is None)."
            )
        for module in self._frozen_modules():
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        for module in (self.other_block, self.other_norm, self.other_proj):
            for p in module.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen trunk + shared head in eval mode once stage 2 froze them.

        Lightning calls ``model.train()`` at the start of every train epoch, which
        would otherwise flip the frozen encoder's dropout and BatchNorm back into
        training mode. We detect the stage-2 freeze via ``shared_proj`` having no
        trainable params and re-assert eval mode so mu/D stay fixed.
        """
        super().train(mode)
        shared_frozen = all(not p.requires_grad for p in self.shared_proj.parameters())
        if mode and shared_frozen:
            for module in self._frozen_modules():
                module.eval()
        return self
