"""Tests for the ``get_marg_std`` method on each predictive method.

``get_marg_std`` returns the closed-form per-grid-point marginal standard
deviation ``sqrt(diag(Sigma))`` of each method's predictive Gaussian. These
tests check the output layout (event shape, device/dtype) and -- the
load-bearing check -- that the analytic marginal std matches both the diagonal
of the dense reference Σ the method represents and the empirical spread of a
large ensemble drawn by ``get_samples``.
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

# A large ensemble so the empirical std is tight enough to cross-check the
# analytic value. atol on a std ~1 from M samples scales like 1/sqrt(2M).
M_COV = 60000
STD_ATOL = 0.03


# ---------------------------------------------------------------------------
# Dense reference covariance helpers (mirror test_samples.py)
# ---------------------------------------------------------------------------


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _spectral_covariance(s: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Dense Σ = Fᴴ diag(s) F for the 2D unitary DFT."""
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


def _empirical_std(method, pred) -> torch.Tensor:
    """Per-coordinate empirical std of a large ensemble, event shape (B, C, ...)."""
    samples = method.get_samples(pred, n_samples=M_COV)
    return samples.std(dim=-1)


# ---------------------------------------------------------------------------
# Layout / dtype (shared)
# ---------------------------------------------------------------------------


def test_shape_matches_mean_event_shape():
    """The marginal std has the mean's event shape (B, C, *spatial)."""
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    B, C, D = 4, 2, 3
    mu = torch.randn(B, C, D)
    var = torch.rand(B, C, D) + 0.5
    pred = torch.stack([mu, var], dim=-1)

    std = method.get_marg_std(pred)
    assert std.shape == (B, C, D)


def test_dtype_and_device_preserved():
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    mu = torch.randn(4, 2, 3, dtype=torch.float64)
    var = torch.rand(4, 2, 3, dtype=torch.float64) + 0.5
    pred = torch.stack([mu, var], dim=-1)
    std = method.get_marg_std(pred)
    assert std.dtype == torch.float64
    assert std.device == pred.device


# ---------------------------------------------------------------------------
# Correctness vs. the dense reference diagonal (and the ensemble)
# ---------------------------------------------------------------------------


def test_deterministic_marg_std_is_zero():
    """A deterministic head is a point mass: zero marginal std everywhere."""
    method = Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, 2, GAMMA)
    pred = torch.randn(4, 2, 3)
    std = method.get_marg_std(pred)
    assert std.shape == pred.shape
    assert torch.all(std == 0.0)


def test_diagonal_marg_std_is_sqrt_diag():
    """Diagonal Gaussian: marginal std = sqrt(diag), matching the ensemble."""
    torch.manual_seed(0)
    B, C, D = 3, 2, 4
    mu = torch.randn(B, C, D, dtype=torch.float64)
    var = torch.rand(B, C, D, dtype=torch.float64) + 0.5
    pred = torch.stack([mu, var], dim=-1)

    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    std = method.get_marg_std(pred)

    assert torch.allclose(std, var.sqrt())
    assert torch.allclose(std, _empirical_std(method, pred), atol=STD_ATOL)


def test_lora_marg_std_is_diag_plus_row_norm():
    """Low-rank Gaussian: marginal var = d_i + ||U_i||², matching the dense Σ."""
    torch.manual_seed(0)
    B, C, D, R = 3, 1, 4, 2
    mu = torch.randn(B, C, D, dtype=torch.float64)
    diag = torch.rand(B, C, D, dtype=torch.float64) + 0.5
    lora = 0.3 * torch.randn(B, C, D, R, dtype=torch.float64)
    pred = torch.cat([mu.unsqueeze(-1), diag.unsqueeze(-1), lora], dim=-1)

    method = LorD("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=R)
    std = method.get_marg_std(pred)

    # Reference: diagonal of Σ = diag(d) + U Uᵀ.
    d = diag.flatten(start_dim=1)
    U = lora.reshape(B, -1, R)
    ref_diag = d + (U * U).sum(dim=-1)
    assert torch.allclose(std.flatten(start_dim=1), ref_diag.sqrt())
    assert torch.allclose(std, _empirical_std(method, pred), atol=STD_ATOL)


def test_cholesky_marg_std_is_row_norm():
    """Dense Gaussian: marginal var = ||l_i||², the squared row norm of ``L``.

    Not ``sqrt(d)``: the ``sigma`` group holds only ``L_ii²``, the *conditional*
    variance of coordinate ``i`` given ``1..i-1``. Mixing the two up is the
    obvious way to get this family wrong, so pin that the two really differ.
    """
    torch.manual_seed(0)
    B, C, D = 3, 1, 5
    V = C * D
    mu = torch.randn(B, C, D, dtype=torch.float64)
    diag = torch.rand(B, C, D, dtype=torch.float64) + 0.5
    strict = 0.4 * torch.randn(B, C, D, V - 1, dtype=torch.float64)
    pred = torch.cat([mu.unsqueeze(-1), diag.unsqueeze(-1), strict], dim=-1)

    method = Cholesky(
        "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, event_dim=V
    )
    std = method.get_marg_std(pred)

    # Reference: diagonal of Σ = L Lᵀ, built independently of the method.
    L = torch.diag_embed(diag.flatten(start_dim=1).sqrt())
    L = L + torch.nn.functional.pad(
        strict.flatten(start_dim=1, end_dim=-2), (0, 1)
    ).tril(-1)
    ref = (L @ L.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(std.flatten(start_dim=1), ref.sqrt())
    assert torch.allclose(std, _empirical_std(method, pred), atol=STD_ATOL)
    # The first coordinate is the only one whose row is bare, so it alone has
    # marginal std sqrt(d); every other coordinate is strictly wider.
    assert torch.allclose(std[:, 0, 0], diag[:, 0, 0].sqrt())
    assert torch.all(std[:, 0, 1:] > diag[:, 0, 1:].sqrt())


def test_spectral_marg_std_matches_dense_diag():
    """Circulant Gaussian: constant marginal std = sqrt(mean(s)) = sqrt(diag Σ)."""
    torch.manual_seed(0)
    B, C, H, W = 2, 1, 4, 4
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    s = _symmetric_s(B, H, W, seed=1)
    pred = torch.stack([mu, s], dim=-1)

    method = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    std = method.get_marg_std(pred)

    ref_diag = _spectral_covariance(s, H, W).diagonal(dim1=-2, dim2=-1)  # (B, V)
    assert torch.allclose(std.flatten(start_dim=1), ref_diag.sqrt())
    assert torch.allclose(std, _empirical_std(method, pred), atol=STD_ATOL)


def test_whitened_marg_std_matches_dense_diag():
    """SCORE: marginal var = diag * mean(s) = diag(D Fᴴ diag(s) F D)."""
    torch.manual_seed(0)
    B, C, H, W = 2, 1, 4, 4
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    diag = torch.rand(B, C, H, W, dtype=torch.float64) + 0.5
    s = _symmetric_s(B, H, W, seed=1)
    pred = torch.stack([mu, diag, s], dim=-1)

    method = SCORE("crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    std = method.get_marg_std(pred)

    core = _spectral_covariance(s, H, W)
    sqrt_d = diag.flatten(start_dim=1).sqrt()
    ref = (sqrt_d.unsqueeze(-1) * core * sqrt_d.unsqueeze(-2)).diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(std.flatten(start_dim=1), ref.sqrt())
    assert torch.allclose(std, _empirical_std(method, pred), atol=STD_ATOL)


def test_whitened_normalized_core_marg_std_is_sqrt_diag():
    """End-to-end: with the default normalized core (mean(s) = 1) the marginal std
    is exactly sqrt of the stage-1 `diag`."""
    torch.manual_seed(0)
    B, C, H, W = 3, 2, 4, 4
    method = SCORE("crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)

    head = torch.randn(B, C * 3, H, W, dtype=torch.float64)
    pred = method.process_output(head)
    std = method.get_marg_std(pred)

    assert torch.allclose(std, pred[..., 1].sqrt(), atol=1e-6)
    assert torch.allclose(std, _empirical_std(method, pred), atol=STD_ATOL)
