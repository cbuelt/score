"""Tests for the ``get_samples`` method on each predictive method.

The sampling-based evaluation protocol draws ensembles from each method's
predictive multivariate Gaussian. These tests check the output layout
(``flatten`` toggle, device/dtype), that the member count follows the
train/eval sample-count switch, and -- the load-bearing check -- that the
empirical covariance of a large ensemble matches the dense reference Σ each
method represents.
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

EPS = 1e-9
N_TRAIN_SAMPLES = 7
N_EVAL_SAMPLES = 11
GAMMA = 1.0


# ---------------------------------------------------------------------------
# Layout / sample-count behaviour (shared across methods)
# ---------------------------------------------------------------------------


def _diag_pred(B=4, C=2, D=3):
    mu = torch.randn(B, C, D)
    var = torch.rand(B, C, D) + 0.5
    return torch.stack([mu, var], dim=-1), (B, C, D)


def test_shape_default_and_flatten():
    """Default keeps the event shape (B, C, *spatial, M); flatten gives (B, V, M)."""
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    pred, (B, C, D) = _diag_pred()
    M = 5

    event = method.get_samples(pred, n_samples=M)
    assert event.shape == (B, C, D, M)

    flat = method.get_samples(pred, n_samples=M, flatten=True)
    assert flat.shape == (B, C * D, M)

    # The two layouts must hold the same numbers (up to the reshape).
    torch.manual_seed(0)
    a = method.get_samples(pred, n_samples=M)
    torch.manual_seed(0)
    b = method.get_samples(pred, n_samples=M, flatten=True)
    assert torch.allclose(a.flatten(start_dim=1, end_dim=-2), b)


def test_n_samples_follows_training_flag():
    """With no explicit count, eval mode draws n_eval_samples, train mode n_train."""
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    pred, _ = _diag_pred()

    method.training = False
    assert method.get_samples(pred).shape[-1] == N_EVAL_SAMPLES
    method.training = True
    assert method.get_samples(pred).shape[-1] == N_TRAIN_SAMPLES


def test_dtype_preserved():
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    pred, _ = _diag_pred()
    pred = pred.to(torch.float64)
    samples = method.get_samples(pred, n_samples=3)
    assert samples.dtype == torch.float64


# ---------------------------------------------------------------------------
# Covariance correctness vs. the dense reference Σ each method represents
# ---------------------------------------------------------------------------

# A large ensemble so the empirical covariance is tight. atol on the covariance
# scales like sigma^2 / sqrt(M), so we keep variances ~1 and M large.
M_COV = 60000
COV_ATOL = 0.06


def _empirical_cov(samples_flat: torch.Tensor) -> torch.Tensor:
    """Empirical covariance per batch element from (B, V, M) samples."""
    mean = samples_flat.mean(dim=-1, keepdim=True)
    centered = samples_flat - mean
    M = samples_flat.shape[-1]
    return centered @ centered.transpose(-1, -2) / (M - 1)


def test_deterministic_samples_are_degenerate():
    """A deterministic head is a point mass: all members equal the prediction."""
    method = Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    pred = torch.randn(4, 2, 3)
    samples = method.get_samples(pred, n_samples=8)
    assert samples.shape == (4, 2, 3, 2)
    # Every member equals the mean -> zero spread.
    assert torch.allclose(samples, pred.unsqueeze(-1).expand_as(samples))
    assert samples.var(dim=-1).max().item() == pytest.approx(0.0, abs=1e-12)


def test_diagonal_empirical_covariance():
    """Empirical covariance ≈ diag(var) for the diagonal Gaussian."""
    torch.manual_seed(0)
    B, C, D = 3, 1, 4
    mu = torch.randn(B, C, D, dtype=torch.float64)
    var = (torch.rand(B, C, D, dtype=torch.float64) + 0.5)
    pred = torch.stack([mu, var], dim=-1)

    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    emp = _empirical_cov(samples)
    target = var.flatten(start_dim=1).diag_embed()
    assert torch.allclose(emp, target, atol=COV_ATOL)


def test_lora_empirical_covariance():
    """Empirical covariance ≈ diag(d) + U Uᵀ for the low-rank Gaussian."""
    torch.manual_seed(0)
    B, C, D, R = 3, 1, 4, 2
    mu = torch.randn(B, C, D, dtype=torch.float64)
    diag = (torch.rand(B, C, D, dtype=torch.float64) + 0.5)
    lora = 0.3 * torch.randn(B, C, D, R, dtype=torch.float64)
    pred = torch.cat([mu.unsqueeze(-1), diag.unsqueeze(-1), lora], dim=-1)

    method = LorD("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=R)
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    emp = _empirical_cov(samples)
    d = diag.flatten(start_dim=1)
    U = lora.reshape(B, -1, R)
    target = d.diag_embed() + U @ U.transpose(-1, -2)
    assert torch.allclose(emp, target, atol=COV_ATOL)


def test_cholesky_empirical_covariance():
    """Empirical covariance ≈ L Lᵀ for the dense Gaussian.

    Sampling is ``mu + L z``, so this is the check that the factor is assembled
    the way the closed forms assume: row-indexed strictly-lower part, ``sqrt(d)``
    on the diagonal, nothing above it.
    """
    torch.manual_seed(0)
    B, C, D = 3, 1, 4
    V = C * D
    mu = torch.randn(B, C, D, dtype=torch.float64)
    diag = torch.rand(B, C, D, dtype=torch.float64) + 0.5
    strict = 0.3 * torch.randn(B, C, D, V - 1, dtype=torch.float64)
    pred = torch.cat([mu.unsqueeze(-1), diag.unsqueeze(-1), strict], dim=-1)

    method = Cholesky(
        "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, event_dim=V
    )
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    L = torch.diag_embed(diag.flatten(start_dim=1).sqrt())
    L = L + torch.nn.functional.pad(
        strict.flatten(start_dim=1, end_dim=-2), (0, 1)
    ).tril(-1)
    emp = _empirical_cov(samples)
    assert torch.allclose(emp, L @ L.transpose(-1, -2), atol=COV_ATOL)
    assert torch.allclose(samples.mean(dim=-1), mu.flatten(start_dim=1), atol=COV_ATOL)


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _spectral_covariance(s: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Dense Σ = Fᴴ diag(s) F for the 2D unitary DFT (mirrors test_spectral.py)."""
    fkron = torch.kron(_fft_matrix(H), _fft_matrix(W))
    sb = s.reshape(s.shape[0], -1).to(torch.complex128)
    sigma = fkron.conj().T @ (sb.unsqueeze(-1) * fkron)
    return sigma.real


def _symmetric_s(B, H, W, seed):
    """Positive conjugate-symmetric spectral diagonal -> real circulant Σ."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(B, 1, H, W, generator=g, dtype=torch.float64) + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def test_spectral_empirical_covariance():
    """Empirical covariance ≈ Fᴴ diag(s) F for the circulant Gaussian.

    This is the check that catches an FFT factor mistake in spectral sampling.
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 1, 4, 4
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    s = _symmetric_s(B, H, W, seed=1)
    pred = torch.stack([mu, s], dim=-1)

    method = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    emp = _empirical_cov(samples)
    target = _spectral_covariance(s, H, W)
    assert torch.allclose(emp, target, atol=COV_ATOL)


def test_whitened_empirical_covariance():
    """Empirical covariance ≈ D Fᴴ diag(s) F D for SCORE.

    Catches errors in the per-pixel D scaling on top of the core sampler.
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 1, 4, 4
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    diag = torch.rand(B, C, H, W, dtype=torch.float64) + 0.5
    s = _symmetric_s(B, H, W, seed=1)
    pred = torch.stack([mu, diag, s], dim=-1)

    method = SCORE("crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    emp = _empirical_cov(samples)
    core = _spectral_covariance(s, H, W)
    sqrt_d = diag.flatten(start_dim=1).sqrt()
    target = sqrt_d.unsqueeze(-1) * core * sqrt_d.unsqueeze(-2)
    assert torch.allclose(emp, target, atol=COV_ATOL)


def test_whitened_marginal_variance_is_diag_times_mean_s():
    """Marginal variance ≈ diag * mean(s): the per-pixel scale times the core's
    constant marginal variance (Parseval). With a unit-mean core it equals diag."""
    torch.manual_seed(0)
    B, C, H, W = 3, 1, 4, 4
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    diag = torch.rand(B, C, H, W, dtype=torch.float64) + 0.5
    s = _symmetric_s(B, H, W, seed=2)
    pred = torch.stack([mu, diag, s], dim=-1)

    method = SCORE("crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    emp_var = samples.var(dim=-1)  # (B, V)
    mean_s = s.flatten(start_dim=1).mean(dim=1, keepdim=True)
    target = diag.flatten(start_dim=1) * mean_s
    assert torch.allclose(emp_var, target, atol=COV_ATOL)


def test_whitened_normalized_core_marginal_variance_is_diag():
    """End-to-end: with the normalized core (`process_output`, the default), the
    predictive marginal variance is the stage-1 `diag` itself -- mean(s) = 1 leaves
    no residual scale factor. This is the point of the unit-mean constraint."""
    torch.manual_seed(0)
    B, C, H, W = 3, 2, 4, 4
    method = SCORE("crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)

    head = torch.randn(B, C * 3, H, W, dtype=torch.float64)
    pred = method.process_output(head)
    samples = method.get_samples(pred, n_samples=M_COV, flatten=True)

    emp_var = samples.var(dim=-1)  # (B, V)
    target = pred[..., 1].flatten(start_dim=1)
    assert torch.allclose(emp_var, target, atol=COV_ATOL)
