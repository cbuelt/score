"""Registry of evaluation metrics for the decoupled evaluation protocol.

Every metric is a callable ``(method, pred, y) -> scalar Tensor`` (already
reduced with ``reduction="mean"``), registered by name in :data:`EVAL_METRICS`.
A novel metric is added with a single ``@register("name")`` decorator -- no edits
to the predictive methods or the :class:`~src.evaluation.evaluator.Evaluator`.

Two kinds of metric live here:

* **Closed-form** (``mse``, ``mae``, ``nll``): thin forwarders to the predictive
  method's own implementation, so the closed-form maths stays a single source of
  truth on the method classes and is shared with training.
* **Sampling-based** (``crps``, ``kernel_score``, ``energy_score``): drawn from
  ``method.get_samples`` and scored with an ensemble estimator. This is what lets
  methods *without* a closed-form covariance be compared fairly against the
  Gaussian ones -- every method is scored the same way. ``crps`` is the
  per-coordinate univariate score; the other two are O(M²) multivariate scores.

``variogram_score`` is a third kind: closed-form like the first group, but its
maths lives in :mod:`src.evaluation.variogram` rather than on the methods,
because only one small per-parameterization piece (``pair_diff_var``) differs
between families.

Metrics may take extra keyword arguments; the :class:`~src.evaluation.evaluator.Evaluator`
supplies them per metric name from ``metric_kwargs`` (config-driven).
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.evaluation.variogram import variogram_score_p1
from src.methods import PredictiveMethod

# name -> metric callable
EVAL_METRICS: Dict[str, Callable[[PredictiveMethod, Tensor, Tensor], Tensor]] = {}

# Station distance matrices, keyed by the metadata path they came from. Loaded
# once per path: the file is small but the metric runs on every eval batch.
_DIST_CACHE: Dict[str, Tensor] = {}


def _load_dist(path: str) -> Tensor:
    """The ``(V, V)`` geodesic distance matrix from a dataset's ``meta.npz``."""
    if path not in _DIST_CACHE:
        with np.load(path, allow_pickle=False) as f:
            _DIST_CACHE[path] = torch.from_numpy(np.asarray(f["dist"]))
    return _DIST_CACHE[path]


def _masked_ensemble(
    method: PredictiveMethod, pred: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """``(fct, obs, mask)`` for the sampling-based scores, with missing targets neutralized.

    Masked coordinates are overwritten by the observation *in every member*, so
    the predictive there is a point mass at the truth. A point mass contributes
    nothing to any norm- or kernel-based score -- ``x_m - y`` and ``x_m - x_j``
    both vanish on those coordinates -- so what the estimator sees is exactly
    the predictive marginal on the observed coordinates. Unlike the closed-form
    path this needs no structure from the parameterization, so it is exact for
    every family, including the spectral ones and SCORE.

    :return: ``fct`` of shape ``(B, V, M)``, ``obs`` of shape ``(B, V)``, and the
        ``(B, V)`` observation mask (``None`` when nothing is missing).
    """
    y, mask = method._mask_of(y)
    obs = y.flatten(start_dim=1)  # (B, V)
    fct = method.get_samples(pred, flatten=True)  # (B, V, M)
    if mask is not None:
        fct = fct * mask.unsqueeze(-1) + (obs * (1.0 - mask)).unsqueeze(-1)
    return fct, obs, mask


def register(name: str):
    """Register a metric callable under ``name`` in :data:`EVAL_METRICS`."""

    def decorator(fn):
        if name in EVAL_METRICS:
            raise ValueError(f"Metric {name!r} is already registered")
        EVAL_METRICS[name] = fn
        return fn

    return decorator


def _forward_closed_form(name: str):
    """Register a metric that simply delegates to ``method.<name>(pred, y)``."""

    @register(name)
    def _metric(method: PredictiveMethod, pred: Tensor, y: Tensor) -> Tensor:
        return getattr(method, name)(pred, y)

    _metric.__name__ = f"eval_{name}"
    return _metric


# Closed-form metrics: keep their single source of truth on the methods.
# `crps` is deliberately *not* here: the methods' closed-form `crps` scores the
# stage-1 marginals (std sqrt(tau), independent of the core) and doubles as a
# training loss, so at eval time the full predictive is scored via samples
# instead (below), which also keeps CRPS comparable for sample-based methods.
for _name in ("mse", "mae", "nll"):
    _forward_closed_form(_name)


@register("crps")
def eval_crps(method: PredictiveMethod, pred: Tensor, y: Tensor) -> Tensor:
    """Per-coordinate CRPS of the predictive via the ensemble estimator.

    Draws from ``method.get_samples`` and scores each flattened coordinate with
    the univariate ensemble CRPS, then averages over the event and batch --
    mirroring the closed-form reduction while capturing the marginals of the
    full predictive covariance (e.g. SCORE's marginal variance ``tau * mean(s)``
    when the scale gauge ``mean(s) = 1`` is not enforced, which the
    core-independent closed form omits).
    Sampling-based so every method, including sample-based ones, is scored the
    same way.
    """
    fct, obs, mask = _masked_ensemble(method, pred, y)
    # The fair (unbiased M(M-1)) estimator, computed in coordinate blocks so the
    # pair tensor never exceeds ~1 GiB (see _crps_ensemble_fair); matches
    # scoringrules' "fair" estimator exactly.
    crps = _crps_ensemble_fair(obs, fct)  # (B, V)
    # Averaged over the *observed* coordinates, so the number stays on a
    # per-station scale as the number of reporting stations varies.
    return method._masked_mean(crps, mask).mean()


#: Element budget per intermediate block in the chunked estimators. The
#: scoringrules "fair" estimators materialize the full (M, M, V) pair tensor,
#: which at depth resolution (V = 307,200, M = 100) is 11.4 GiB in float32 and
#: does not fit next to the model on a 24-48 GB card. 2^28 elements cap an
#: intermediate at ~1 GiB.
_PAIR_BLOCK_ELEMENTS = 2 ** 28


def _crps_ensemble_fair(obs: Tensor, fct: Tensor) -> Tensor:
    """Fair-estimator ensemble CRPS, chunked over the coordinate axis.

    Identical to ``scoringrules.crps_ensemble(..., estimator="fair")`` -- the
    per-coordinate decomposition ``E|X - y| - 1/(2M(M-1)) sum_{i!=j}|x_i - x_j|``
    -- but the (M, M) pair tensor is built for a block of coordinates at a
    time, so memory stays bounded for any V.

    :param obs: observations, shape (B, V).
    :param fct: ensemble members, shape (B, V, M).
    :return: per-coordinate CRPS, shape (B, V).
    """
    M = fct.shape[-1]
    block = max(1, _PAIR_BLOCK_ELEMENTS // (M * M))
    out = torch.empty_like(obs)
    for lo in range(0, obs.shape[1], block):
        f = fct[:, lo:lo + block, :]                       # (B, v, M)
        o = obs[:, lo:lo + block]                          # (B, v)
        cross = (f - o.unsqueeze(-1)).abs().mean(dim=-1)   # (B, v)
        pair = (f.unsqueeze(-1) - f.unsqueeze(-2)).abs()   # (B, v, M, M)
        spread = pair.sum(dim=(-2, -1)) / (2 * M * (M - 1))
        out[:, lo:lo + block] = cross - spread
    return out


@register("energy_score")
def eval_energy_score(
    method: PredictiveMethod, pred: Tensor, y: Tensor
) -> Tensor:
    """Energy score via the fair O(M²) ensemble estimator.

    Same estimator as ``scoringrules.es_ensemble(..., estimator="fair")``,
    ``E||X - y|| - 1/(2M(M-1)) sum_{i!=j} ||x_i - x_j||``, but the pairwise
    member distances go through ``torch.cdist`` (matmul path, O(M² + MV)
    memory) instead of materializing the (M, M, V) difference tensor that the
    scoringrules torch backend builds -- 11.4 GiB at depth resolution.

    Masked coordinates drop out of every norm (see :func:`_masked_ensemble`), so
    this is the energy score of the marginal on the observed coordinates. Note
    the score's magnitude grows with the event dimension, so samples with more
    missing stations contribute slightly smaller values -- inherent to a
    multivariate score on a varying coordinate set, and not a normalization that
    can be divided out.
    """
    fct, obs, _ = _masked_ensemble(method, pred, y)
    M = fct.shape[-1]
    members = fct.transpose(1, 2)  # (B, M, V)
    cross = (members - obs.unsqueeze(1)).norm(dim=-1).mean(dim=-1)  # (B,)
    pair = torch.cdist(members, members)  # (B, M, M), no M²V intermediate
    spread = pair.sum(dim=(-2, -1)) / (2 * M * (M - 1))  # diagonal is zero
    return (cross - spread).mean()


@register("kernel_score")
def eval_kernel_score(
    method: PredictiveMethod, pred: Tensor, y: Tensor
) -> Tensor:
    """Gaussian kernel score via an O(M²) ensemble estimator.

    Uses the repo's bandwidth ``k(a, b) = exp(-||a - b||² / gamma²)`` (not the
    fixed bandwidth of ``scoringrules.gksmv_ensemble``) so the sampled score is
    on the same scale as the closed-form kernel score used in training. The
    self-term uses the unbiased ``M(M-1)`` estimator (the diagonal m=j is
    excluded), matching

        score = 0.5 (1 + E[k(X, X')]) - E[k(X, y)].

    When the method carries multiple eval bandwidths, the per-bandwidth score is
    averaged over them (``method.gamma2_list``), mirroring the multi-bandwidth
    combine rule of the closed-form training score. A single bandwidth reduces to
    the plain kernel score.
    """
    fct, obs, _ = _masked_ensemble(method, pred, y)
    B, V, M = fct.shape

    # Distances are bandwidth-independent, so compute them once and reuse across
    # bandwidths. cross_sq: (B, M); sq_dist: (B, M, M) pairwise over the V axis.
    cross_sq = (fct - obs.unsqueeze(-1)).pow(2).sum(dim=1)  # (B, M)
    members = fct.transpose(1, 2)  # (B, M, V)
    sq_dist = torch.cdist(members, members).pow(2)  # (B, M, M)

    gamma2_list = method.gamma2_list
    score = None  # (B,), averaged over bandwidths
    for g2 in gamma2_list:
        # cross term  E[k(X, y)] ≈ mean_m exp(-||x_m - y||² / γ²)
        cross = torch.exp(-cross_sq / g2).mean(dim=-1)  # (B,)

        # self term  E[k(X, X')] ≈ 1/(M(M-1)) Σ_{m≠j} exp(-||x_m - x_j||² / γ²)
        self_kernel = torch.exp(-sq_dist / g2)  # (B, M, M)
        # drop the diagonal (m == j) before averaging over the M(M-1) off-diagonal
        diag = torch.diagonal(self_kernel, dim1=-2, dim2=-1)  # (B, M)
        self_term = (self_kernel.sum(dim=(-2, -1)) - diag.sum(dim=-1)) / (M * (M - 1))

        s = 0.5 * (1.0 + self_term) - cross  # (B,)
        score = s if score is None else score + s

    score = score / len(gamma2_list)
    return score.mean()


@register("variogram_score")
def eval_variogram_score(
    method: PredictiveMethod,
    pred: Tensor,
    y: Tensor,
    weight: str = "none",
    block_size: int = 256,
    dist_path: Optional[str] = None,
) -> Tensor:
    """Closed-form p=1 variogram score over all coordinate pairs.

    Unlike the O(M²) ensemble scores above this needs no samples: for a Gaussian
    predictive the pairwise increment ``X_i - X_j`` is itself Gaussian, so its
    expected absolute value is a folded-normal mean. See
    :mod:`src.evaluation.variogram` for the maths, the pair weights (``weight``)
    and why the pair matrix is accumulated in row blocks (``block_size``).

    This is the one metric here that probes pairwise *spatial* structure, so it
    is what separates parameterizations that agree on the marginals.

    :param dist_path: optional path to a dataset ``meta.npz`` carrying a
        ``dist`` matrix of true pairwise distances, used by the ``inv_dist``
        weighting. Required for irregular geometries: on a station network the
        event ordering is arbitrary, so the grid-index fallback would weight by
        a meaningless distance.
    """
    dist = _load_dist(dist_path) if dist_path is not None else None
    return variogram_score_p1(
        method, pred, y, weight=weight, block_size=block_size, dist=dist
    ).mean()
