"""Tests for the closed-form p=1 variogram score.

The score is

    VS_1(P, y) = sum_ij w_ij ( |y_i - y_j| - E|X_i - X_j| )^2  /  sum_ij w_ij

with the expectation a folded-normal mean, exact for a Gaussian ``P``. The
load-bearing check is against the ensemble estimator (``scoringrules.vs_ensemble``
at ``p=1``) over a large sample from each method: closed form and estimator must
agree to within Monte-Carlo error.

That error decays only as ``1/sqrt(M)`` and -- because every pair's plug-in mean
is built from the *same* draw -- does not average down over the ~400 pairs, so
these tests use a fixed seed and a ~1.5% relative tolerance. Averaged over
independent replicates the two agree to well within one standard error.
"""

import math

import pytest
import scoringrules as sr
import torch

from src.evaluation.metrics import EVAL_METRICS
from src.evaluation.variogram import (
    VARIOGRAM_WEIGHTS,
    _folded_normal_mean,
    pair_weights,
    variogram_score_p1,
)
from src.methods import (
    Deterministic,
    Diag,
    LorD,
    SpectralDiag,
    SCORE,
)

EPS = 1e-9
N_TRAIN_SAMPLES = 7
N_EVAL_SAMPLES = 11
GAMMA = 1.0
RANK = 3

B, C, H, W = 2, 1, 4, 5
V = C * H * W

# Ensemble size for the reference estimator, and the resulting tolerance. The
# estimator's error is dominated by the shared draw across pairs, so it shrinks
# like 1/sqrt(M) rather than 1/sqrt(M * n_pairs).
M_REF = 200_000
REL_TOL = 0.015

FAMILIES = ("deterministic", "diagonal", "lora", "spectral", "whitened")


def _symmetric_s(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(B, C, H, W, generator=g, dtype=torch.float64) + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def _build(name: str):
    """Return ``(method, pred, y)`` for one family."""
    torch.manual_seed(0)
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    var = torch.rand(B, C, H, W, dtype=torch.float64) + 0.5
    y = torch.randn(B, C, H, W, dtype=torch.float64)

    if name == "deterministic":
        m = Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return m, mu, y
    if name == "diagonal":
        m = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return m, torch.stack([mu, var], dim=-1), y
    if name == "lora":
        lora = 0.4 * torch.randn(B, C, H, W, RANK, dtype=torch.float64)
        m = LorD("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=RANK)
        pred = torch.cat([mu.unsqueeze(-1), var.unsqueeze(-1), lora], dim=-1)
        return m, pred, y
    if name == "spectral":
        m = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return m, torch.stack([mu, _symmetric_s(1)], dim=-1), y
    if name == "whitened":
        m = SCORE("crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return m, torch.stack([mu, var, _symmetric_s(1)], dim=-1), y
    raise AssertionError(name)


# ---------------------------------------------------------------------------
# The folded-normal mean
# ---------------------------------------------------------------------------


def test_folded_normal_mean_zero_mean_case():
    """For m = 0 the folded normal is a half-normal with mean sqrt(2v/pi)."""
    v = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64)
    got = _folded_normal_mean(torch.zeros_like(v), v)
    assert torch.allclose(got, (2.0 * v / math.pi).sqrt())


def test_folded_normal_mean_matches_monte_carlo():
    torch.manual_seed(0)
    m = torch.tensor([-2.0, -0.3, 0.0, 0.7, 3.0], dtype=torch.float64)
    v = torch.tensor([0.5, 2.0, 1.0, 0.1, 3.0], dtype=torch.float64)
    draws = m + v.sqrt() * torch.randn(2_000_000, 5, dtype=torch.float64)
    assert torch.allclose(_folded_normal_mean(m, v), draws.abs().mean(0), atol=5e-3)


def test_folded_normal_mean_degenerate_variance_is_abs_mean():
    """v = 0 (a Dirac) must give exactly |m|, with no NaN from the 0/0."""
    m = torch.tensor([-3.0, -1e-8, 0.0, 1e-8, 5.0], dtype=torch.float64)
    got = _folded_normal_mean(m, torch.zeros_like(m))
    assert torch.all(torch.isfinite(got))
    assert torch.allclose(got, m.abs())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_folded_normal_mean_no_overflow_in_either_dtype(dtype):
    """Large |m| with vanishing v must not overflow the exp/erf arguments."""
    m = torch.tensor([1e4, -1e4], dtype=dtype)
    got = _folded_normal_mean(m, torch.zeros_like(m))
    assert torch.all(torch.isfinite(got))
    assert torch.allclose(got, m.abs())


# ---------------------------------------------------------------------------
# Pair weights
# ---------------------------------------------------------------------------


def test_weight_none_is_ones_off_the_diagonal():
    rows = torch.arange(V)
    w = pair_weights((H, W), C, rows, "none", torch.device("cpu"), torch.float64)
    assert w.shape == (V, V)
    assert torch.all(w.diagonal() == 0.0)
    assert torch.all(w[~torch.eye(V, dtype=torch.bool)] == 1.0)


def test_weight_inv_dist_is_reciprocal_planar_distance():
    rows = torch.arange(V)
    w = pair_weights((H, W), C, rows, "inv_dist", torch.device("cpu"), torch.float64)

    assert torch.all(w.diagonal() == 0.0)
    assert torch.all(torch.isfinite(w))
    # Nearest neighbours (distance 1) get weight 1; the opposite corners of the
    # 4x5 grid are at sqrt(3^2 + 4^2) = 5.
    assert w[0, 1] == pytest.approx(1.0)  # (0,0) -> (0,1)
    assert w[0, W] == pytest.approx(1.0)  # (0,0) -> (1,0)
    assert w[0, V - 1] == pytest.approx(1.0 / 5.0)
    assert torch.allclose(w, w.T)


def test_weight_inv_dist_is_non_periodic():
    """Deliberately not the torus distance the circulant models assume: the two
    ends of a grid row are far apart, not adjacent."""
    rows = torch.arange(V)
    w = pair_weights((H, W), C, rows, "inv_dist", torch.device("cpu"), torch.float64)
    assert w[0, W - 1] == pytest.approx(1.0 / (W - 1))


def test_weight_inv_dist_rejects_multichannel():
    """Two variables at the same pixel have no meaningful spatial separation."""
    with pytest.raises(ValueError, match="single-channel"):
        pair_weights((H, W), 2, torch.arange(2 * V), "inv_dist",
                     torch.device("cpu"), torch.float64)


def test_unknown_weight_raises():
    with pytest.raises(ValueError, match="Unknown variogram weight"):
        pair_weights((H, W), C, torch.arange(V), "gaussian",
                     torch.device("cpu"), torch.float64)


def test_weight_row_blocks_match_the_full_matrix():
    full = pair_weights((H, W), C, torch.arange(V), "inv_dist",
                        torch.device("cpu"), torch.float64)
    blocks = torch.cat(
        [
            pair_weights((H, W), C, torch.arange(i, min(i + 6, V)), "inv_dist",
                         torch.device("cpu"), torch.float64)
            for i in range(0, V, 6)
        ]
    )
    assert torch.allclose(blocks, full)


# ---------------------------------------------------------------------------
# The score itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("weight", VARIOGRAM_WEIGHTS)
def test_closed_form_matches_ensemble_estimator(name, weight):
    """The closed form agrees with `scoringrules.vs_ensemble` at p=1."""
    method, pred, y = _build(name)
    torch.manual_seed(11)

    closed = variogram_score_p1(method, pred, y, weight=weight, block_size=7)

    w = pair_weights((H, W), C, torch.arange(V), weight, y.device, y.dtype)
    fct = method.get_samples(pred, n_samples=M_REF, flatten=True)  # (B, V, M)
    ensemble = (
        sr.vs_ensemble(
            y.flatten(start_dim=1), fct, w=w,
            m_axis=-1, v_axis=-2, p=1.0, estimator="nrg", backend="torch",
        )
        / w.sum()
    )
    assert torch.allclose(closed, ensemble, rtol=REL_TOL)


def test_deterministic_reduces_to_the_plain_increment_error():
    """With a Dirac predictive E|X_i - X_j| = |mu_i - mu_j| exactly, so the score
    collapses to a weighted mean of squared increment errors -- no sampling
    tolerance needed."""
    method, pred, y = _build("deterministic")
    w = pair_weights((H, W), C, torch.arange(V), "inv_dist", y.device, y.dtype)

    mu_f, y_f = pred.flatten(start_dim=1), y.flatten(start_dim=1)
    obs = (y_f.unsqueeze(-1) - y_f.unsqueeze(-2)).abs()
    exp = (mu_f.unsqueeze(-1) - mu_f.unsqueeze(-2)).abs()
    expected = (w * (obs - exp).pow(2)).sum(dim=(-2, -1)) / w.sum()

    got = variogram_score_p1(method, pred, y, weight="inv_dist", block_size=6)
    assert torch.allclose(got, expected)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("weight", VARIOGRAM_WEIGHTS)
def test_score_is_block_size_invariant(name, weight):
    """Blocking is an implementation detail of the memory budget, not of the
    score: every block size must give the same number."""
    method, pred, y = _build(name)
    scores = [
        variogram_score_p1(method, pred, y, weight=weight, block_size=bs)
        for bs in (1, 3, 7, V, 4 * V)
    ]
    for score in scores[1:]:
        assert torch.allclose(score, scores[0], atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
def test_perfect_forecast_scores_lower_than_a_wrong_one(name):
    """Sanity of orientation: lower is better."""
    method, pred, y = _build(name)
    good = variogram_score_p1(method, pred, method.get_mean(pred), weight="inv_dist")
    bad = variogram_score_p1(method, pred, y + 5.0 * torch.randn_like(y),
                             weight="inv_dist")
    assert torch.all(good < bad)


def test_score_is_nonnegative_and_shaped_per_batch_element():
    method, pred, y = _build("whitened")
    score = variogram_score_p1(method, pred, y, weight="inv_dist")
    assert score.shape == (B,)
    assert torch.all(score >= 0.0)


def test_invalid_block_size_raises():
    method, pred, y = _build("diagonal")
    with pytest.raises(ValueError, match="block_size"):
        variogram_score_p1(method, pred, y, block_size=0)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_registered_and_reduces_to_a_scalar():
    method, pred, y = _build("whitened")
    metric = EVAL_METRICS["variogram_score"]
    got = metric(method, pred, y, weight="inv_dist", block_size=6)
    assert got.ndim == 0
    expected = variogram_score_p1(method, pred, y, weight="inv_dist", block_size=6)
    assert torch.allclose(got, expected.mean())


def test_metric_defaults_to_unweighted():
    method, pred, y = _build("diagonal")
    assert torch.allclose(
        EVAL_METRICS["variogram_score"](method, pred, y),
        variogram_score_p1(method, pred, y, weight="none").mean(),
    )


# ---------------------------------------------------------------------------
# Irregular geometry: explicit distances and missing observations
# ---------------------------------------------------------------------------


def _diag_method(v, gamma=None):
    return Diag(
        loss="kernel_score",
        eps=EPS,
        n_train_samples=N_TRAIN_SAMPLES,
        n_eval_samples=N_TRAIN_SAMPLES,
        out_channels=1,
        gamma=[gamma if gamma is not None else math.sqrt(v)],
    )


def test_explicit_distances_override_the_grid_fallback():
    """`inv_dist` must use a supplied distance matrix, not the event index.

    On a station network the event axis is an arbitrary ordering (the real
    dataset concatenates by country), so index distance is meaningless. Feeding
    a permuted geometry must change the score; if it did not, the weighting
    would be silently ignoring the argument.
    """
    torch.manual_seed(0)
    v = 6
    rows = torch.arange(v)
    coords = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    inline = (coords[:, None] - coords[None, :]).abs()

    # A distance matrix matching the implied grid reproduces the fallback exactly
    # (both are 1-D unit spacing).
    from_grid = pair_weights((v,), 1, rows, "inv_dist", torch.device("cpu"), torch.float32)
    from_dist = pair_weights(
        (v,), 1, rows, "inv_dist", torch.device("cpu"), torch.float32, dist=inline
    )
    assert torch.allclose(from_grid, from_dist)

    # A genuinely different geometry must give different weights.
    scrambled = inline[torch.tensor([3, 0, 5, 1, 4, 2])][:, torch.tensor([3, 0, 5, 1, 4, 2])]
    from_scrambled = pair_weights(
        (v,), 1, rows, "inv_dist", torch.device("cpu"), torch.float32, dist=scrambled
    )
    assert not torch.allclose(from_grid, from_scrambled)


def test_distance_matrix_shape_is_validated():
    """A distance matrix that does not match the event must fail loudly."""
    rows = torch.arange(4)
    with pytest.raises(ValueError, match="expected"):
        pair_weights(
            (4,), 1, rows, "inv_dist", torch.device("cpu"), torch.float32,
            dist=torch.ones(3, 3),
        )


@pytest.mark.parametrize("weight", list(VARIOGRAM_WEIGHTS))
def test_missing_observations_mask_whole_pairs(weight):
    """A pair contributes only when both endpoints are observed.

    Checked against the score recomputed on the explicitly sub-selected
    coordinates, which is the definition the pair mask is meant to implement.
    """
    torch.manual_seed(0)
    v, b = 6, 1
    method = _diag_method(v)
    pred = method.process_output(torch.randn(b, 2, v))
    y = torch.randn(b, 1, v)

    keep = [0, 2, 3, 5]
    idx = torch.tensor(keep)
    y_nan = y.clone()
    y_nan[0, 0, [1, 4]] = float("nan")

    # An explicit geometry, so that dropping coordinates does not silently
    # renumber them: under the grid fallback the sub-selected event would be
    # read as four *adjacent* points, a different geometry from the four that
    # survive masking.
    coords = torch.tensor([0.0, 1.7, 2.9, 4.1, 5.6, 7.3])
    dist = (coords[:, None] - coords[None, :]).abs()
    kwargs = {"dist": dist} if weight == "inv_dist" else {}
    sub_kwargs = {"dist": dist[idx][:, idx]} if weight == "inv_dist" else {}

    masked = variogram_score_p1(method, pred, y_nan, weight=weight, **kwargs)

    # The same score over just the observed coordinates: a diagonal Gaussian is
    # closed under sub-selection, so this is an exact reference.
    sub_method = _diag_method(len(keep), gamma=method.gamma)
    sub = variogram_score_p1(
        sub_method, pred[:, :, idx], y[:, :, idx], weight=weight, **sub_kwargs
    )
    assert torch.allclose(masked, sub, atol=1e-6)


def test_missing_observations_do_not_produce_nan():
    """The station data always has gaps, so a NaN target must not poison the score."""
    torch.manual_seed(0)
    v = 8
    method = _diag_method(v)
    pred = method.process_output(torch.randn(3, 2, v))
    y = torch.randn(3, 1, v)
    y[:, 0, 0] = float("nan")  # a station that never reports
    y[1, 0, 3] = float("nan")

    score = variogram_score_p1(method, pred, y, weight="inv_dist")
    assert torch.isfinite(score).all()
