"""Tests for ``quadratic_form`` on each predictive method.

``quadratic_form(pred, a)`` returns ``aᵀ Sigma a = Var(aᵀX)`` -- the one thing
the linear pre-rank functions of :mod:`src.evaluation.calibration` need from the
covariance. Each family computes it from its own parameterization without ever
forming ``Sigma``, so the load-bearing check is against a densely constructed
reference, exactly as in ``test_pair_diff_var.py``.

Beyond the families that file covers, the SCORE cases here also exercise the
non-Fourier transform path (a two-axis separable DCT) and the low-rank
correction of the core, because ``quadratic_form`` is the only covariance
accessor that handles *every* SCORE path -- ``_dense_score_terms`` raises on the
native Fourier path and on multi-axis transforms.

The ``C = 2`` cases pin the channel structure: the spectral covariances are
block-diagonal across channels (their transform runs over the spatial axes only)
while LorD's factor ``U`` and Cholesky's factor ``L`` span the whole
flattened event and couple them.
"""

import math

import pytest
import torch

from src.methods import (
    Deterministic,
    Cholesky,
    Diag,
    LorD,
    SpectralDiag,
    SCORE,
)
from src.methods.transforms import SeparableOrthogonalTransform, dct_matrix

EPS = 1e-9
N_TRAIN_SAMPLES = 7
N_EVAL_SAMPLES = 11
GAMMA = 1.0
RANK = 3

B, H, W = 2, 4, 5


# ---------------------------------------------------------------------------
# Dense reference covariance helpers (mirror test_pair_diff_var.py)
# ---------------------------------------------------------------------------


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _blockdiag(per_channel: torch.Tensor) -> torch.Tensor:
    """``(B, C, n, n)`` per-channel blocks -> a ``(B, C*n, C*n)`` block diagonal."""
    B_, C, n, _ = per_channel.shape
    out = per_channel.new_zeros(B_, C * n, C * n)
    for c in range(C):
        out[:, c * n : (c + 1) * n, c * n : (c + 1) * n] = per_channel[:, c]
    return out


def _circulant_core(s: torch.Tensor) -> torch.Tensor:
    """Dense Σ = Fᴴ diag(s) F, blocked over channels -> ``(B, V, V)``."""
    B_, C, H_, W_ = s.shape
    n = H_ * W_
    fkron = torch.kron(_fft_matrix(H_), _fft_matrix(W_))
    sb = s.reshape(B_, C, n).to(torch.complex128)
    return _blockdiag((fkron.conj().T @ (sb.unsqueeze(-1) * fkron)).real)


def _transform_core(s: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Dense core ``Bᵀ diag(s) B`` for a real orthogonal basis, per channel."""
    B_, C, H_, W_ = s.shape
    n = H_ * W_
    sb = s.reshape(B_, C, n)
    return _blockdiag(basis.T @ (sb.unsqueeze(-1) * basis))


def _symmetric_s(C: int, seed: int) -> torch.Tensor:
    """Positive conjugate-symmetric spectral diagonal -> real circulant Σ."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(B, C, H, W, generator=g, dtype=torch.float64) + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def _dct_transform() -> tuple:
    """A two-axis separable DCT transform and its dense Kronecker basis."""
    mats = [dct_matrix(H), dct_matrix(W)]
    return SeparableOrthogonalTransform(mats), torch.kron(mats[0], mats[1])


def _whitened_sigma(diag, core, lora=None, renorm=True):
    """``Sigma_P = D̃ (core + UUᵀ) D̃`` with ``D̃² = diag / diag(core + UUᵀ)``.

    The renormalization by the implied marginal variances is derived here from
    the dense core rather than from the method's ``_scale2``, so the reference
    stays independent of the code under test. ``renorm=False`` is the
    native-Fourier rank-0 path, where the predictive scale is ``diag`` itself.
    """
    total = core
    if lora is not None:
        U = lora.reshape(lora.shape[0], -1, lora.shape[-1])  # (B, V, r)
        total = total + U @ U.transpose(-1, -2)
    d_flat = diag.flatten(start_dim=1)
    if renorm:
        d_flat = d_flat / total.diagonal(dim1=-2, dim2=-1)
    scale = d_flat.sqrt()
    return scale.unsqueeze(-1) * total * scale.unsqueeze(-2)


def _build(name: str, C: int):
    """Return ``(method, pred, dense_sigma)`` for one family at ``C`` channels."""
    torch.manual_seed(0)
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    var = torch.rand(B, C, H, W, dtype=torch.float64) + 0.5
    V = C * H * W

    def _lora_fields():
        return 0.4 * torch.randn(B, C, H, W, RANK, dtype=torch.float64)

    if name == "deterministic":
        method = Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return method, mu, torch.zeros(B, V, V, dtype=torch.float64)

    if name == "diagonal":
        method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        pred = torch.stack([mu, var], dim=-1)
        return method, pred, torch.diag_embed(var.flatten(start_dim=1))

    if name == "lora":
        lora = _lora_fields()
        method = LorD(
            "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=RANK
        )
        pred = torch.cat([mu.unsqueeze(-1), var.unsqueeze(-1), lora], dim=-1)
        U = lora.reshape(B, -1, RANK)
        sigma = torch.diag_embed(var.flatten(start_dim=1)) + U @ U.transpose(-1, -2)
        return method, pred, sigma

    if name == "cholesky":
        # Unstructured Sigma = L Lᵀ: `var` is L's squared diagonal, a random
        # field of V-1 channels its strictly-lower part, read row-wise.
        strict = 0.3 * torch.randn(B, C, H, W, V - 1, dtype=torch.float64)
        method = Cholesky(
            "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, event_dim=V
        )
        pred = torch.cat([mu.unsqueeze(-1), var.unsqueeze(-1), strict], dim=-1)
        L = torch.diag_embed(var.flatten(start_dim=1).sqrt())
        L = L + torch.nn.functional.pad(
            strict.flatten(start_dim=1, end_dim=-2), (0, 1)
        ).tril(-1)
        return method, pred, L @ L.transpose(-1, -2)

    if name == "spectral":
        s = _symmetric_s(C, seed=1)
        method = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return method, torch.stack([mu, s], dim=-1), _circulant_core(s)

    if name == "whitened":
        # Native Fourier, rank 0: no renormalization, Sigma_P = D Circ(s) D.
        s = _symmetric_s(C, seed=1)
        method = SCORE(
            "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA
        )
        sigma = _whitened_sigma(var, _circulant_core(s), renorm=False)
        return method, torch.stack([mu, var, s], dim=-1), sigma

    if name == "whitened_lora":
        # Native Fourier + rank: the renormalization is always on when rank > 0.
        s = _symmetric_s(C, seed=1)
        lora = _lora_fields()
        method = SCORE(
            "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=RANK
        )
        pred = torch.cat(
            [mu.unsqueeze(-1), var.unsqueeze(-1), s.unsqueeze(-1), lora], dim=-1
        )
        return method, pred, _whitened_sigma(var, _circulant_core(s), lora)

    if name == "whitened_dct":
        # Two-axis separable DCT: a localized (non-flat) basis, so renormalized.
        transform, basis = _dct_transform()
        g = torch.Generator().manual_seed(2)
        s = torch.rand(B, C, H, W, generator=g, dtype=torch.float64) + 0.5
        method = SCORE(
            "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, transform=transform
        )
        sigma = _whitened_sigma(var, _transform_core(s, basis))
        return method, torch.stack([mu, var, s], dim=-1), sigma

    if name == "whitened_dct_lora":
        transform, basis = _dct_transform()
        g = torch.Generator().manual_seed(2)
        s = torch.rand(B, C, H, W, generator=g, dtype=torch.float64) + 0.5
        lora = _lora_fields()
        method = SCORE(
            "crps",
            EPS,
            N_TRAIN_SAMPLES,
            N_EVAL_SAMPLES,
            C,
            GAMMA,
            transform=transform,
            rank=RANK,
        )
        pred = torch.cat(
            [mu.unsqueeze(-1), var.unsqueeze(-1), s.unsqueeze(-1), lora], dim=-1
        )
        return method, pred, _whitened_sigma(var, _transform_core(s, basis), lora)

    raise AssertionError(name)


FAMILIES = (
    "deterministic",
    "diagonal",
    "lora",
    "cholesky",
    "spectral",
    "whitened",
    "whitened_lora",
    "whitened_dct",
    "whitened_dct_lora",
)


def _reference(sigma: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``aᵀ Sigma a`` from a dense Σ."""
    a_flat = a.flatten(start_dim=1)
    return torch.einsum("bi,bij,bj->b", a_flat, sigma, a_flat)


# ---------------------------------------------------------------------------
# Correctness vs. the dense reference Σ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("C", [1, 2])
def test_matches_dense_reference_for_random_weights(name, C):
    """The closed form equals aᵀΣa built from a dense Σ, for arbitrary a."""
    method, pred, sigma = _build(name, C)
    g = torch.Generator().manual_seed(7)
    a = torch.randn(B, C, H, W, generator=g, dtype=torch.float64)

    got = method.quadratic_form(pred, a)

    assert got.shape == (B,)
    assert torch.allclose(got, _reference(sigma, a), atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("C", [1, 2])
def test_uniform_weights_match_dense_reference(name, C):
    """The mean pre-rank's weight vector a = 1/d is the case that must be right."""
    method, pred, sigma = _build(name, C)
    V = C * H * W
    a = torch.full((B, C, H, W), 1.0 / V, dtype=torch.float64)

    got = method.quadratic_form(pred, a)

    assert torch.allclose(got, _reference(sigma, a), atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("C", [1, 2])
def test_indicator_weights_recover_the_marginal_variance(name, C):
    """``a = e_i`` picks out Σ_ii, which is `get_marg_std`² -- a cross-check
    against a second, independently implemented closed form."""
    method, pred, _ = _build(name, C)
    V = C * H * W
    marg_var = method.get_marg_std(pred).flatten(start_dim=1).pow(2)

    for i in (0, 3, V - 1):
        a = torch.zeros(B, C * H * W, dtype=torch.float64)
        a[:, i] = 1.0
        got = method.quadratic_form(pred, a.reshape(B, C, H, W))
        assert torch.allclose(got, marg_var[:, i], atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
def test_masked_weights_match_the_subselected_gaussian(name):
    """A weight vector supported on a coordinate subset sees exactly the
    corresponding sub-block -- which is what makes the per-case restriction to
    the reporting stations exact rather than an approximation."""
    method, pred, sigma = _build(name, 1)
    V = H * W
    obs = torch.tensor([0, 1, 4, 7, 9, 11, V - 1])

    a = torch.zeros(B, V, dtype=torch.float64)
    a[:, obs] = 1.0 / obs.numel()
    got = method.quadratic_form(pred, a.reshape(B, 1, H, W))

    sub = sigma[:, obs][:, :, obs]
    a_sub = torch.full((B, obs.numel()), 1.0 / obs.numel(), dtype=torch.float64)
    expected = torch.einsum("bi,bij,bj->b", a_sub, sub, a_sub)
    assert torch.allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Agreement with the samplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", [f for f in FAMILIES if f != "deterministic"]
)
def test_agrees_with_the_empirical_variance_of_the_samples(name):
    """Var(aᵀX) estimated from `get_samples` matches the closed form.

    The two paths share no algebra -- one contracts the parameterization, the
    other draws from it -- so this catches a factor that is consistent within
    the closed forms but wrong against the actual predictive.
    """
    method, pred, _ = _build(name, 1)
    torch.manual_seed(3)
    a = torch.randn(B, 1, H, W, dtype=torch.float64)

    samples = method.get_samples(pred, n_samples=200_000, flatten=True)  # (B, V, M)
    proj = torch.einsum("bv,bvm->bm", a.flatten(start_dim=1), samples)
    empirical = proj.var(dim=-1, unbiased=True)
    closed = method.quadratic_form(pred, a)

    assert torch.allclose(empirical, closed, rtol=0.05)


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_weights_broadcast_over_the_batch():
    """A single ``(1, C, ...)`` weight field applies to every case."""
    method, pred, sigma = _build("lora", 1)
    a = torch.randn(1, 1, H, W, dtype=torch.float64)

    got = method.quadratic_form(pred, a)

    assert got.shape == (B,)
    assert torch.allclose(got, _reference(sigma, a.expand(B, -1, -1, -1)), atol=1e-12)


def test_wrong_event_shape_raises():
    """A flattened ``a`` is the likely mistake; it must not silently broadcast."""
    method, pred, _ = _build("spectral", 1)
    with pytest.raises(ValueError, match="event shape"):
        method.quadratic_form(pred, torch.randn(B, H * W, dtype=torch.float64))


def test_result_is_never_negative():
    """A variance, including where the algebra sums terms of mixed sign."""
    for name in FAMILIES:
        method, pred, _ = _build(name, 1)
        a = torch.randn(B, 1, H, W, dtype=torch.float64)
        assert bool((method.quadratic_form(pred, a) >= 0).all()), name
