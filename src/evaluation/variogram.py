"""Closed-form variogram score of order p=1 for the Gaussian predictives.

The variogram score of order ``p`` (Scheuerer & Hamill, 2015) is

    VS_p(P, y) = sum_ij w_ij ( |y_i - y_j|^p - E_{X~P} |X_i - X_j|^p )^2

It scores the *pairwise* structure of the predictive rather than its marginals,
which is exactly where the covariance parameterizations differ: two models with
identical marginals are indistinguishable under ``crps`` but not under this.

For a Gaussian ``P`` the expectation is available in closed form. The difference
of two jointly Gaussian coordinates is itself Gaussian,

    X_i - X_j ~ N(m_ij, v_ij),  m_ij = mu_i - mu_j,
                                v_ij = Sigma_ii + Sigma_jj - 2 Sigma_ij,

so for ``p = 1`` the expectation is a folded-normal mean (:func:`_folded_normal_mean`)
and no sampling is needed. This module owns everything about the score that is
*independent* of the parameterization; the single quantity it asks of the method
is :meth:`~src.methods.base.PredictiveMethod.pair_diff_var`, which every family
computes without materializing ``Sigma``.

**Blocking.** The pair set is all ``V^2`` ordered pairs. At ERA5's
``V = 35 200`` that is 1.24e9 pairs, and a single ``(V, V)`` float32 tensor is
4.6 GiB -- with the ~12 live intermediates of the closed-form expression, a
materialized score would need ~55 GiB for one sample (and ~440 GiB at the
configured eval batch size of 8). So the score is accumulated over row blocks of
``block_size`` coordinates, bounding the working set to ``(B, block_size, V)``.

Note that peak memory scales with ``B * block_size``, not ``block_size`` alone --
a block size tuned at ``B = 1`` will OOM at a realistic eval batch size.
Throughput, meanwhile, is *flat*: the kernel is compute-bound by ~128 rows, so
larger blocks cost memory linearly and buy nothing. Measured on ERA5
(``V = 35 200``, ``B = 8``, one RTX 4090) for the full exact ``V^2``:

    block_size    128     256     512    1024
    peak VRAM    1.25    2.48    4.92    9.82  GiB
    per sample   0.36    0.37    0.37    0.38  s

Hence the default of 256: ~2.5 GiB at the configured ERA5 eval batch size, full
speed, and ~0.37 s per sample -- about 7 min over the 1095-sample test split.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.methods import PredictiveMethod

# Selectable pair-weight functions (see `pair_weights`).
VARIOGRAM_WEIGHTS = ("none", "inv_dist")

# Standardized-normal tail cutoff for the folded-normal mean. `erf(12/sqrt(2))`
# is 1.0 and `exp(-72)` is ~5e-32 in every dtype used here, so clamping at 12
# leaves the value unchanged while making the degenerate `v = 0` limit exact
# rather than a 0/0 (see `_folded_normal_mean`).
_Z_MAX = 12.0

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _folded_normal_mean(m: Tensor, v: Tensor) -> Tensor:
    """``E|Z|`` for ``Z ~ N(m, v)`` -- the mean of a folded normal.

        E|Z| = sqrt(2 v / pi) exp(-m^2 / (2 v)) + m erf( m / sqrt(2 v) )

    Degenerate ``v = 0`` (the deterministic method, whose predictive is a Dirac)
    is handled without a branch: the first term carries the *unclamped* ``sqrt(v)``
    so it vanishes exactly, while the ratio ``m / sqrt(v)`` is clamped to
    :data:`_Z_MAX` before the ``erf``, which then returns ``sign(m)``. The result
    is exactly ``|m|``, the correct limit.
    """
    sd = v.clamp_min(0.0).sqrt()
    # Divide by a strictly positive std, then clamp: for any genuinely tiny `v`
    # the true value is already `|m|` to far beyond machine precision.
    z = (m / sd.clamp_min(torch.finfo(sd.dtype).tiny)).clamp(-_Z_MAX, _Z_MAX)
    return sd * _SQRT_2_OVER_PI * torch.exp(-0.5 * z * z) + m * torch.erf(z * _INV_SQRT_2)


def _grid_coords(spatial: Sequence[int], device, dtype) -> Tensor:
    """Per-axis grid coordinate of every flat spatial position, ``(n, len(spatial))``."""
    n = math.prod(spatial)
    pos = torch.arange(n, device=device)
    coords = []
    stride = 1
    for size in reversed(spatial):
        coords.append((pos // stride) % size)
        stride *= size
    return torch.stack(list(reversed(coords)), dim=-1).to(dtype)


def pair_weights(
    spatial: Sequence[int],
    channels: int,
    rows: Tensor,
    kind: str,
    device,
    dtype,
    dist: Optional[Tensor] = None,
) -> Tensor:
    """Pair weights ``w_ij`` for a block of rows, shape ``(len(rows), V)``.

    ``rows`` are flat event indices into ``V = channels * prod(spatial)``.
    Self-pairs always get weight zero -- they contribute nothing to the score
    (both the observed and the expected increment vanish) and would otherwise be
    a division by zero under ``inv_dist``.

    * ``"none"``     -- ``w_ij = 1``: the unweighted variogram score. The default,
      and the sensible choice for the 1-D time-series tasks.
    * ``"inv_dist"`` -- ``w_ij = 1 / d_ij``. This is the standard weighting for
      spatial fields: it emphasizes short-range pairs, where the covariance
      parameterizations actually differ, instead of letting the O(V^2) long-range
      pairs dominate.

    The distance ``d_ij`` comes from one of two sources:

    * an explicit ``dist`` matrix (``(V, V)``, any consistent unit -- the score
      normalizes by ``sum w_ij``, so a global scale cancels). This is what
      irregular geometries need: for a station network the event axis is an
      arbitrary ordering (here, by country), so index distance is meaningless
      and only the real geodesic separation is.
    * otherwise the implied *planar, non-periodic* grid coordinates. Sound only
      because a regular grid (ERA5 is 0.25 deg throughout) has grid units
      proportional to physical units, so the constant cancels and no lat/lon
      plumbing is needed. Note this is deliberately *not* the periodic distance:
      the circulant models assume a torus, but the ERA5 domain is a regional box
      and the score's geometry should be the data's, not the model's.
    """
    if kind not in VARIOGRAM_WEIGHTS:
        raise ValueError(
            f"Unknown variogram weight {kind!r}; expected one of {VARIOGRAM_WEIGHTS}"
        )

    n = math.prod(spatial)
    V = channels * n
    r_idx = torch.arange(rows.shape[0], device=device)

    if kind == "none":
        w = torch.ones(rows.shape[0], V, device=device, dtype=dtype)
        w[r_idx, rows] = 0.0
        return w

    # inv_dist: a purely spatial weight, so a multi-channel event has no
    # well-defined pair distance (two variables at the same pixel would be at
    # distance zero). Fail loudly rather than invent a convention.
    if channels != 1:
        raise ValueError(
            "variogram weight 'inv_dist' is a spatial weight and is only defined "
            f"for single-channel targets, but the event has {channels} channels. "
            "Use weight='none' for multi-channel tasks."
        )

    if dist is not None:
        if dist.shape != (V, V):
            raise ValueError(
                f"Explicit distance matrix has shape {tuple(dist.shape)}, "
                f"expected ({V}, {V}) to match the event."
            )
        block = dist.to(device=device, dtype=dtype)[rows]  # (len(rows), V)
    else:
        coords = _grid_coords(spatial, device, dtype)  # (n, ndim)
        block = torch.cdist(coords[rows], coords)  # (len(rows), n)

    w = torch.where(block > 0, 1.0 / block, torch.zeros_like(block))
    # A distance matrix can carry a tiny nonzero on the diagonal; pin self-pairs.
    w[r_idx, rows] = 0.0
    return w


def variogram_score_p1(
    method: PredictiveMethod,
    pred: Tensor,
    y: Tensor,
    weight: str = "none",
    block_size: int = 256,
    dist: Optional[Tensor] = None,
) -> Tensor:
    """Closed-form p=1 variogram score per batch element, shape ``(B,)``.

    Returns the **weighted mean** over pairs,

        sum_ij w_ij ( |y_i - y_j| - E|X_i - X_j| )^2  /  sum_ij w_ij

    rather than the raw sum. Dividing by the (constant) total weight is a
    positive rescaling, so the score stays proper, but it keeps the number
    readable and comparable across tasks -- a raw sum over ERA5's 1.24e9 pairs
    would be ~1e9 and would not be comparable to the time-series tasks at all.
    Lower is better.

    :param weight: pair weight function, see :func:`pair_weights`.
    :param block_size: rows of the pair matrix per block; bounds the working set
        to ``(B, block_size, V)``. Peak memory scales with ``B * block_size``,
        while throughput does not -- see the module docstring for measurements.
    :param dist: optional ``(V, V)`` physical distance matrix for the
        ``inv_dist`` weighting, for geometries where the event ordering carries
        no spatial meaning (see :func:`pair_weights`).

    Missing targets (``NaN``) are handled by masking whole *pairs*: a pair
    contributes only when both endpoints are observed, i.e. ``w_ij`` is
    multiplied by ``m_i m_j``. The normalization then varies per sample, so the
    weight total is accumulated per batch element rather than once.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    mu = method.get_mean(pred)  # (B, C, *spatial)
    channels, spatial = mu.shape[1], tuple(mu.shape[2:])
    mu_flat = mu.flatten(start_dim=1)  # (B, V)
    y, mask = method._mask_of(y)
    y_flat = y.flatten(start_dim=1)  # (B, V)
    V = mu_flat.shape[1]

    total = torch.zeros(mu_flat.shape[0], device=mu.device, dtype=mu.dtype)
    weight_sum = torch.zeros(mu_flat.shape[0], device=mu.device, dtype=mu.dtype)

    for start in range(0, V, block_size):
        rows = torch.arange(start, min(start + block_size, V), device=mu.device)

        # Difference distribution X_i - X_j ~ N(m, v) for i in `rows`, all j.
        v = method.pair_diff_var(pred, rows)  # (B, R, V)
        m = mu_flat[:, rows].unsqueeze(-1) - mu_flat.unsqueeze(-2)  # (B, R, V)
        expected = _folded_normal_mean(m, v)

        observed = (y_flat[:, rows].unsqueeze(-1) - y_flat.unsqueeze(-2)).abs()
        w = pair_weights(spatial, channels, rows, weight, mu.device, mu.dtype, dist)
        if mask is not None:
            w = w.unsqueeze(0) * mask[:, rows].unsqueeze(-1) * mask.unsqueeze(-2)

        total = total + (w * (observed - expected).pow(2)).sum(dim=(-2, -1))
        weight_sum = weight_sum + w.sum(dim=(-2, -1)) if mask is not None else weight_sum + w.sum()

    return total / weight_sum.clamp_min(torch.finfo(mu.dtype).tiny)
