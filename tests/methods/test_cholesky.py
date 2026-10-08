"""Closed-form tests for :class:`~src.methods.cholesky.Cholesky`.

This is the one family with no structure to exploit: ``Sigma = L Lᵀ`` ranges over
the whole SPD cone. That makes it the easiest family to check -- every quantity
has an independent reference in ``torch.distributions`` or in plain dense linear
algebra -- and the most useful one to check, because it is the reference arm the
structured parameterizations are compared against.

Two references are used throughout:

* ``MultivariateNormal(mu, scale_tril=L)`` for the likelihood and the samples,
  which knows nothing about this repo's conventions; and
* :class:`~src.methods.diagonal.Diag`, which the dense family must reproduce
  *exactly* once the strictly-lower head is zero. That equivalence is what pins
  the ``L_ii = sqrt(d)`` convention: with ``L_ii = softplus(d)`` instead, every
  score would be off by a squaring and only this test would notice.
"""

import math

import pytest
import torch
from torch.distributions import MultivariateNormal

from src.evaluation.metrics import EVAL_METRICS
from src.methods import Cholesky, Diag, PredictiveMethod
from src.methods.base import masked_gaussian_kernel_score

EPS = 1e-9
N_TRAIN_SAMPLES = 7
N_EVAL_SAMPLES = 11
B, C, D = 4, 1, 6
V = C * D
GAMMA = math.sqrt(V)


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _method(out_channels=C, event_dim=V, loss="nll", **kwargs):
    return Cholesky(
        loss,
        EPS,
        N_TRAIN_SAMPLES,
        N_EVAL_SAMPLES,
        out_channels,
        GAMMA,
        event_dim=event_dim,
        **kwargs,
    )


def _raw(method, spatial=(D,), seed=0):
    """A random head output of the width this method's ``group_widths`` asks for."""
    torch.manual_seed(seed)
    return torch.randn(B, method.out_channels, *spatial)


def _reference_chol(pred, event_dim=V):
    """``L`` rebuilt from ``pred`` without going through the method under test."""
    diag = pred[..., 1].flatten(start_dim=1)
    L = torch.diag_embed(diag.sqrt())
    if event_dim < 2:
        return L
    strict = pred[..., 2:].flatten(start_dim=1, end_dim=-2)
    return L + torch.nn.functional.pad(strict, (0, 1)).tril(-1)


# ---------------------------------------------------------------------------
# The factor itself
# ---------------------------------------------------------------------------


def test_factor_is_lower_triangular_with_positive_diagonal():
    """``L`` is a valid Cholesky factor for *any* head output, by construction."""
    method = _method()
    pred = method.process_output(_raw(method) * 5.0)  # large values: no saturation
    L = method._chol(pred)

    assert L.shape == (B, V, V)
    assert torch.equal(L, L.tril())
    assert torch.all(torch.diagonal(L, dim1=-2, dim2=-1) > 0.0)
    # Positive-definite by construction: no head output can make this fail.
    torch.linalg.cholesky(L @ L.transpose(-1, -2))


def test_strictly_lower_head_is_read_row_wise():
    """Coordinate ``i``'s slots become row ``i``; entries from the diagonal on are
    dropped, so slot ``j`` of row ``i`` is used only when ``j < i``."""
    method = _method()
    pred = method.process_output(_raw(method))
    strict = pred[..., 2:].flatten(start_dim=1, end_dim=-2)  # (B, V, V-1)
    L = method._chol(pred)

    for i in range(V):
        for j in range(i):
            assert torch.allclose(L[:, i, j], strict[:, i, j])


def test_process_output_rejects_a_mismatched_event_shape():
    """A head whose flattened event size is not ``event_dim`` is a config error."""
    method = _method(event_dim=V)
    with pytest.raises(ValueError, match="event_dim"):
        method.process_output(torch.randn(B, method.out_channels, D + 1))


# ---------------------------------------------------------------------------
# Exact reduction to the diagonal family
# ---------------------------------------------------------------------------


def _paired_preds(seed=0):
    """``(chol_method, chol_pred, diag_method, diag_pred)`` on the same mu/diag.

    The dense head's strictly-lower slots are zeroed *before* ``process_output``,
    so both families see identical ``mu`` and identical pre-softplus variances.
    """
    torch.manual_seed(seed)
    chol = _method()
    raw = torch.randn(B, C, V + 1, D)
    raw[:, :, 2:] = 0.0
    chol_pred = chol.process_output(raw.reshape(B, C * (V + 1), D))

    diag = Diag("nll", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    diag_pred = diag.process_output(raw[:, :, :2].reshape(B, C * 2, D))
    return chol, chol_pred, diag, diag_pred


@pytest.mark.parametrize(
    "score", ["mse", "mae", "nll", "crps", "kernel_score", "marginal_kernel_score"]
)
def test_matches_diag_when_off_diagonal_is_zero(score):
    """With ``L`` diagonal the dense family *is* the diagonal family, exactly.

    This is what pins ``L_ii = sqrt(d)``: the ``sigma`` group carries a variance
    in both families, so every score must agree bit-for-bit in float64.
    """
    chol, chol_pred, diag, diag_pred = _paired_preds()
    y = torch.randn(B, C, D)

    got = getattr(chol, score)(chol_pred, y, reduction="none")
    ref = getattr(diag, score)(diag_pred, y, reduction="none")
    assert torch.allclose(got, ref, atol=1e-12)


def test_matches_diag_on_the_covariance_accessors():
    """Mean, marginal std and pairwise difference variance also coincide."""
    chol, chol_pred, diag, diag_pred = _paired_preds()
    rows = torch.arange(V)

    assert torch.allclose(chol.get_mean(chol_pred), diag.get_mean(diag_pred))
    assert torch.allclose(chol.get_marg_std(chol_pred), diag.get_marg_std(diag_pred))
    assert torch.allclose(
        chol.pair_diff_var(chol_pred, rows), diag.pair_diff_var(diag_pred, rows)
    )


def test_zero_head_is_the_neutral_prediction():
    """A zero-init head gives ``Sigma = diag(softplus(0))`` -- the same neutral
    start as Diag, which is why ``other_zero_init`` may stay ``True``."""
    method = _method()
    pred = method.process_output(torch.zeros(B, method.out_channels, D))
    sigma = method._cov(pred)

    neutral = math.log(2.0) + EPS  # softplus(0)
    assert torch.allclose(sigma, torch.eye(V).expand(B, V, V) * neutral)


def test_off_diagonal_head_has_a_nonzero_gradient_at_zero():
    """The strictly-lower head is not a stationary point at zero.

    LorD's ``U`` is (``Sigma`` is quadratic in it, so ``dL/dU = 0`` at
    ``U = 0`` and a zero-init head never trains); here the diagonal anchors it,
    so ``other_zero_init`` is ``True``. Pin the difference numerically.
    """
    method = _method()
    head = torch.zeros(B, method.out_channels, D, requires_grad=True)
    y = torch.randn(B, C, D)

    method.nll(method.process_output(head), y).backward()

    grad = head.grad.reshape(B, C, V + 1, D)[:, :, 2:]
    assert method.other_zero_init is True
    assert grad.abs().max() > 1e-6


# ---------------------------------------------------------------------------
# Closed forms vs. independent dense references
# ---------------------------------------------------------------------------


def test_nll_matches_multivariate_normal():
    """The structured NLL equals ``-MultivariateNormal(mu, scale_tril=L).log_prob``."""
    method = _method()
    pred = method.process_output(_raw(method))
    y = torch.randn(B, C, D)

    L = _reference_chol(pred)
    ref = -MultivariateNormal(pred[..., 0].flatten(start_dim=1), scale_tril=L).log_prob(
        y.flatten(start_dim=1)
    )
    assert torch.allclose(method.nll(pred, y, reduction="none"), ref)


def test_dense_score_terms_return_the_deployed_covariance():
    """``_dense_score_terms`` short-circuits the base reconstruction; both must
    agree, since the masked scores go through it."""
    method = _method()
    pred = method.process_output(_raw(method))
    y = torch.randn(B, C, D)

    cov, resid = method._dense_score_terms(pred, y)
    L = _reference_chol(pred)

    assert torch.allclose(cov, L @ L.transpose(-1, -2))
    assert torch.allclose(resid, (pred[..., 0] - y).flatten(start_dim=1))
    # ... and it agrees with the base class's own reconstruction from
    # `get_marg_std` + `pair_diff_var`, which is the identity the override skips.
    base_cov, _ = PredictiveMethod._dense_score_terms(method, pred, y)
    assert torch.allclose(cov, base_cov, atol=1e-12)
    # Nothing about the offset: the scored object is the predictive covariance.
    assert torch.all(method._dense_logdet_offset(pred, None) == 0.0)


def test_kernel_score_matches_the_dense_closed_form():
    """The closed form is the shared dense one evaluated at Sigma = L Lᵀ."""
    method = _method(loss="kernel_score")
    pred = method.process_output(_raw(method))
    y = torch.randn(B, C, D)

    L = _reference_chol(pred)
    ref = masked_gaussian_kernel_score(
        L @ L.transpose(-1, -2), (pred[..., 0] - y).flatten(start_dim=1), GAMMA**2, None
    )
    assert torch.allclose(method.kernel_score(pred, y, reduction="none"), ref)


def test_quadratic_form_matches_dense_sigma():
    """``aᵀ Sigma a`` for random weight vectors, including cross-channel ones."""
    torch.manual_seed(3)
    method = _method(out_channels=2, event_dim=2 * D)
    pred = method.process_output(_raw(method))
    L = _reference_chol(pred, event_dim=2 * D)
    sigma = L @ L.transpose(-1, -2)

    for _ in range(3):
        a = torch.randn(B, 2, D)
        ref = torch.einsum("bi,bij,bj->b", a.flatten(1), sigma, a.flatten(1))
        assert torch.allclose(method.quadratic_form(pred, a), ref)


def test_samples_reproduce_the_covariance():
    """``mu + L z`` has mean ``mu`` and covariance ``L Lᵀ``."""
    method = _method()
    pred = method.process_output(_raw(method))
    L = _reference_chol(pred)

    torch.manual_seed(0)
    samples = method.get_samples(pred, n_samples=200_000, flatten=True)
    centered = samples - samples.mean(dim=-1, keepdim=True)
    emp = centered @ centered.transpose(-1, -2) / (samples.shape[-1] - 1)

    assert torch.allclose(
        samples.mean(dim=-1), pred[..., 0].flatten(start_dim=1), atol=0.03
    )
    assert torch.allclose(emp, L @ L.transpose(-1, -2), atol=0.06)


def test_sliced_energy_score_matches_the_ensemble_estimator():
    """The O(M) sliced estimator agrees with the O(M²) ensemble estimator."""
    method = _method(loss="energy_score")
    method.n_eval_samples = 20_000
    pred = method.process_output(_raw(method))
    y = torch.randn(B, C, D)

    torch.manual_seed(1)
    sliced = method.energy_score(pred, y, n_samples=200_000)
    torch.manual_seed(1)
    ensemble = EVAL_METRICS["energy_score"](method, pred, y)

    assert sliced.item() == pytest.approx(ensemble.item(), rel=0.01)


# ---------------------------------------------------------------------------
# Degenerate and guarded cases
# ---------------------------------------------------------------------------


def test_scalar_target_has_no_other_head():
    """At ``V = 1`` there is no off-diagonal, and the family is exactly Diag."""
    method = _method(event_dim=1)
    assert method.group_widths["other"] is None

    torch.manual_seed(0)
    raw = torch.randn(B, 2 * C, 1)
    pred = method.process_output(raw)
    y = torch.randn(B, C, 1)

    diag = Diag("nll", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    diag_pred = diag.process_output(raw)
    assert torch.allclose(
        method.nll(pred, y, reduction="none"), diag.nll(diag_pred, y, reduction="none")
    )


def test_refuses_a_gridded_event_dimension():
    """A dense factor at gridded scale is refused up front, before running out of memory."""
    with pytest.raises(NotImplementedError, match="max_event_dim"):
        _method(event_dim=Cholesky.max_event_dim + 1)


def test_requires_event_dim():
    """``event_dim`` sizes the head, so it cannot be inferred later."""
    with pytest.raises(ValueError, match="event_dim"):
        Cholesky("nll", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
