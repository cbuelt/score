"""Latent-noise injection for the sample-based generative baseline (SB).

Every other method in :mod:`src.methods` is a *parametric* predictive: the
backbone emits distribution parameters and the randomness is added afterwards,
outside the network. A generative network works the other way round -- it maps
``(input, latent noise) -> one sample`` -- so the randomness has to enter
*inside* the model, and the same input run with ``M`` latents produces the ``M``
ensemble members the fair scoring rule (by default the Gaussian kernel score)
is computed from.

This module holds the one block that does that, shared by all four backbones.
Each of them has the same shape at the end: a trunk producing a feature tensor,
then a head projecting it to the output channels. The injector is spliced
between the two, so the head, its group-inner stacking, and every
``process_output`` reshape stay exactly as they are -- the only thing that
changes is that the batch axis is now ``B*M`` long.

Injecting *late* is deliberate. The trunk is the expensive part (and, for
DepthAnything and PatchTST, the pretrained part); running it once and only
replaying the head per member keeps the cost of ``M`` samples close to the cost
of one, which is what makes ``M = 10`` affordable per training step.

**Two latents, not one.** The block draws a *global* latent, shared across every
spatial position of a member, and a *local* one drawn independently per
position. Both are needed, and for the same reason ``LorD`` needs
``Sigma = diag(d) + U Uᵀ`` rather than either term alone:

* with only a global latent of width ``k``, all ``M`` members of a case differ
  along a ``k``-dimensional subspace of the ``V``-dimensional event space, so the
  predictive is degenerate -- rank ``k`` out of ``V`` -- and its marginals are far
  too narrow wherever the global directions do not reach;
* with only a local latent, members differ independently at every position, so
  the predictive has no spatial dependence at all, which is exactly the
  structure this repo exists to model.

Together they give a full-rank predictive that can still carry long-range
dependence, and the mixer's nonlinearity keeps the map from latent to sample
from collapsing to the affine (i.e. exactly Gaussian) case.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class GaussianNoiseInjector(nn.Module):
    """Expand a trunk feature tensor to ``M`` members by mixing in Gaussian latents.

    Consumes ``(B, ...)`` features and returns ``(B*M, ...)`` with the member
    index *fastest-varying*, i.e. row ``b*M + m``. Downstream code therefore
    recovers the members with ``reshape(B, M, ...)``, and
    :func:`stack_member_groups` turns that into the group-inner layout the
    predictive methods expect.

    The mixer is a residual two-layer perceptron over the feature axis (a 1x1
    convolution, for the conv-shaped backbones), so its output has the trunk's
    own width and the head that follows it is untouched.

    Args:
        feature_dim: Width of the trunk's feature axis.
        channel_axis: Which axis carries the features -- ``1`` for the
            conv-shaped backbones (``(B, F, *spatial)``), ``-1`` for the
            token-shaped ones (``(B, *spatial, F)``).
        global_dim: Width of the per-member latent shared across all positions.
            This is what carries spatial dependence between the samples.
        local_dim: Width of the latent drawn independently per position. This is
            what keeps the predictive full-rank.
        hidden_dim: Width of the mixer's hidden layer; defaults to
            ``feature_dim``.
    """

    def __init__(
        self,
        feature_dim: int,
        channel_axis: int = -1,
        global_dim: int = 32,
        local_dim: int = 8,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if global_dim < 1 or local_dim < 1:
            raise ValueError(
                "GaussianNoiseInjector needs both a global and a local latent "
                f"(see the module docstring); got global_dim={global_dim}, "
                f"local_dim={local_dim}."
            )
        if channel_axis not in (1, -1):
            raise ValueError(
                f"channel_axis must be 1 or -1, got {channel_axis}."
            )
        self.feature_dim = feature_dim
        self.channel_axis = channel_axis
        self.global_dim = global_dim
        self.local_dim = local_dim
        hidden_dim = hidden_dim or feature_dim
        self.mix = nn.Sequential(
            nn.Linear(feature_dim + global_dim + local_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Small random, *not* zero, for the same reason LorD's factor head
        # sets `other_zero_init=False`: the spread term of a fair ensemble score
        # depends on the member differences x_i - x_j and has zero gradient when
        # they all vanish (the Gaussian kernel is stationary at zero distance,
        # and torch defines the gradient of a norm at zero as 0). With a
        # zero-init residual every member of a case is bit-identical, so the
        # spread term -- the only part of the score that asks for any spread --
        # would start with no gradient. A small init breaks that symmetry while
        # leaving the initial prediction effectively the trunk's own features.
        nn.init.normal_(self.mix[-1].weight, std=1e-3)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, features: Tensor, n_samples: int) -> Tensor:
        """``(B, ...) -> (B*M, ...)``, member index fastest-varying."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        x = features.movedim(1, -1) if self.channel_axis == 1 else features
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} features on axis "
                f"{self.channel_axis}, got {x.shape[-1]} (shape "
                f"{tuple(features.shape)})."
            )

        # repeat_interleave, not repeat: it puts the member index in the fast
        # position, which is what makes the reshape(B, M, ...) downstream --
        # and `stack_member_groups` -- correct.
        x = x.repeat_interleave(n_samples, dim=0)  # (B*M, *spatial, F)
        rows, spatial = x.shape[0], x.shape[1:-1]
        opts = dict(device=x.device, dtype=x.dtype)

        # One latent per (case, member), broadcast over every position ...
        z_global = torch.randn(rows, *([1] * len(spatial)), self.global_dim, **opts)
        z_global = z_global.expand(rows, *spatial, self.global_dim)
        # ... and one drawn independently at each position.
        z_local = torch.randn(rows, *spatial, self.local_dim, **opts)

        out = x + self.mix(torch.cat([x, z_global, z_local], dim=-1))
        return out.movedim(-1, 1) if self.channel_axis == 1 else out


def require_n_samples(n_samples: Optional[int], backbone: str) -> int:
    """Validate the member count a noise-injecting backbone was called with.

    Defaulting a missing ``n_samples`` to 1 would be the worst option available:
    the fair ensemble scores divide by ``M(M-1)``, so a silent single-member
    ensemble turns the training loss into a division by zero several layers away
    from the cause. Fail here instead, naming the caller that should have
    supplied it.
    """
    if n_samples is None:
        raise ValueError(
            f"{backbone} was built with a noise injector, so it needs an "
            "explicit n_samples. LightningPredictiveModel.forward supplies it "
            "from the predictive method; a direct call must pass it too."
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    return int(n_samples)


def stack_member_groups(
    out: Tensor, n_samples: int, num_channels: int, spatial: Sequence[int]
) -> Tensor:
    """Fold a member axis into the channel axis, group-inner.

    ``out`` is the head's output for the expanded batch, ``(B*M, C, *spatial)``
    with the member index fastest-varying (as :class:`GaussianNoiseInjector`
    leaves it). The backbones' contract is a single ``(B, C*G, *spatial)``
    tensor whose channel axis is *group-inner* -- for each channel ``c`` its
    ``G`` groups are contiguous -- so with the members playing the part of the
    groups this returns ``(B, C*M, *spatial)`` laid out as
    ``[c0_m0, c0_m1, ..., c1_m0, ...]``.

    That is exactly the layout ``SampleBased.process_output`` un-stacks, and the
    same one every Gaussian method's ``process_output`` assumes, so the sample
    axis costs no special case anywhere downstream.
    """
    rows = out.shape[0]
    batch = rows // n_samples
    members = out.reshape(batch, n_samples, num_channels, *spatial)
    return members.transpose(1, 2).reshape(batch, num_channels * n_samples, *spatial)
