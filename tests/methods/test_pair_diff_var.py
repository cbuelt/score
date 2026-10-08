"""Tests for ``pair_diff_var`` on each predictive method.

``pair_diff_var(pred, rows)`` returns ``Var(X_i - X_j) = Sigma_ii + Sigma_jj -
2 Sigma_ij`` for a block of rows against all columns -- the one thing the p=1
variogram score needs from the covariance. Each family computes it from its own
parameterization without ever forming ``Sigma``, so the load-bearing check is
against a densely constructed reference ``Sigma``.

The ``C=2`` cases pin the channel structure, which differs by family: the
spectral covariances are block-diagonal across channels (their FFT runs over the
spatial axes only) while LorD's factor ``U`` and Cholesky's factor ``L``
span the whole flattened event and therefore couple channels.
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
RANK = 3

B, H, W = 2, 4, 5


# ---------------------------------------------------------------------------
# Dense reference covariance helpers (mirror test_marg_std.py)
# ---------------------------------------------------------------------------


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _spectral_covariance(s: torch.Tensor) -> torch.Tensor:
    """Per-channel dense Σ = Fᴴ diag(s) F, blocked over channels -> (B, V, V)."""
    B_, C, H_, W_ = s.shape
    n = H_ * W_
    fkron = torch.kron(_fft_matrix(H_), _fft_matrix(W_))
    sb = s.reshape(B_, C, n).to(torch.complex128)
    per_channel = (fkron.conj().T @ (sb.unsqueeze(-1) * fkron)).real  # (B, C, n, n)
    out = per_channel.new_zeros(B_, C * n, C * n)
    for c in range(C):
        out[:, c * n : (c + 1) * n, c * n : (c + 1) * n] = per_channel[:, c]
    return out


def _symmetric_s(C: int, seed: int) -> torch.Tensor:
    """Positive conjugate-symmetric spectral diagonal -> real circulant Σ."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(B, C, H, W, generator=g, dtype=torch.float64) + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def _cholesky_pred(mu: torch.Tensor, var: torch.Tensor, V: int):
    """``(pred, dense_sigma)`` for the dense family: Σ = L Lᵀ.

    ``var`` becomes ``L``'s squared diagonal and a random field of ``V-1``
    channels its strictly-lower part, row-indexed exactly as
    ``Cholesky._chol`` reads it.
    """
    strict = 0.3 * torch.randn(*var.shape, V - 1, dtype=torch.float64)
    pred = torch.cat([mu.unsqueeze(-1), var.unsqueeze(-1), strict], dim=-1)
    L = torch.diag_embed(var.flatten(start_dim=1).sqrt())
    L = L + torch.nn.functional.pad(
        strict.flatten(start_dim=1, end_dim=-2), (0, 1)
    ).tril(-1)
    return pred, L @ L.transpose(-1, -2)


def _reference(sigma: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``Sigma_ii + Sigma_jj - 2 Sigma_ij`` from a dense Σ, self-pairs zeroed."""
    diag = sigma.diagonal(dim1=-2, dim2=-1)  # (B, V)
    ref = diag[:, rows].unsqueeze(-1) + diag.unsqueeze(-2) - 2.0 * sigma[:, rows, :]
    ref[:, torch.arange(rows.shape[0]), rows] = 0.0
    return ref


def _build(name: str, C: int):
    """Return ``(method, pred, dense_sigma)`` for one family at ``C`` channels."""
    torch.manual_seed(0)
    mu = torch.randn(B, C, H, W, dtype=torch.float64)
    var = torch.rand(B, C, H, W, dtype=torch.float64) + 0.5
    V = C * H * W

    if name == "deterministic":
        method = Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return method, mu, torch.zeros(B, V, V, dtype=torch.float64)

    if name == "diagonal":
        method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        pred = torch.stack([mu, var], dim=-1)
        return method, pred, torch.diag_embed(var.flatten(start_dim=1))

    if name == "lora":
        lora = 0.4 * torch.randn(B, C, H, W, RANK, dtype=torch.float64)
        method = LorD(
            "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=RANK
        )
        pred = torch.cat([mu.unsqueeze(-1), var.unsqueeze(-1), lora], dim=-1)
        U = lora.reshape(B, -1, RANK)
        sigma = torch.diag_embed(var.flatten(start_dim=1)) + U @ U.transpose(-1, -2)
        return method, pred, sigma

    if name == "cholesky":
        method = Cholesky(
            "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, event_dim=V
        )
        pred, sigma = _cholesky_pred(mu, var, V)
        return method, pred, sigma

    if name == "spectral":
        s = _symmetric_s(C, seed=1)
        method = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
        return method, torch.stack([mu, s], dim=-1), _spectral_covariance(s)

    if name == "whitened":
        s = _symmetric_s(C, seed=1)
        method = SCORE(
            "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA
        )
        core = _spectral_covariance(s)
        sqrt_d = var.flatten(start_dim=1).sqrt()
        sigma = sqrt_d.unsqueeze(-1) * core * sqrt_d.unsqueeze(-2)
        return method, torch.stack([mu, var, s], dim=-1), sigma

    raise AssertionError(name)


FAMILIES = ("deterministic", "diagonal", "lora", "cholesky", "spectral", "whitened")


# ---------------------------------------------------------------------------
# Correctness vs. the dense reference Σ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("C", [1, 2])
def test_matches_dense_reference(name, C):
    """The closed form equals Σ_ii + Σ_jj - 2Σ_ij built from a dense Σ."""
    method, pred, sigma = _build(name, C)
    V = C * H * W
    rows = torch.tensor([0, 3, 5, V - 1])

    got = method.pair_diff_var(pred, rows)

    assert got.shape == (B, rows.shape[0], V)
    assert torch.allclose(got, _reference(sigma, rows), atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("C", [1, 2])
def test_self_pairs_are_zero_and_result_nonnegative(name, C):
    """Var(X_i - X_i) = 0 exactly, and a variance is never negative."""
    method, pred, _ = _build(name, C)
    V = C * H * W
    rows = torch.arange(V)

    got = method.pair_diff_var(pred, rows)

    assert torch.all(got.diagonal(dim1=-2, dim2=-1) == 0.0)
    assert torch.all(got >= 0.0)


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("C", [1, 2])
def test_symmetric_in_the_pair(name, C):
    """Var(X_i - X_j) = Var(X_j - X_i): the full block is a symmetric matrix."""
    method, pred, _ = _build(name, C)
    got = method.pair_diff_var(pred, torch.arange(C * H * W))
    assert torch.allclose(got, got.transpose(-1, -2), atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
def test_consistent_with_marg_std(name):
    """Independent coordinates aside, the block's own diagonal terms must agree
    with ``get_marg_std``: for the diagonal method Var(X_i - X_j) is exactly
    ``sigma_i^2 + sigma_j^2``, and in general it is bounded by that."""
    method, pred, _ = _build(name, C=1)
    V = H * W
    rows = torch.arange(V)

    var = method.get_marg_std(pred).flatten(start_dim=1).pow(2)  # (B, V)
    sum_var = var.unsqueeze(-1) + var.unsqueeze(-2)
    got = method.pair_diff_var(pred, rows)

    off = ~torch.eye(V, dtype=torch.bool)
    # |2 Sigma_ij| <= Sigma_ii + Sigma_jj by Cauchy-Schwarz + AM-GM.
    assert torch.all(got[:, off] <= 2.0 * sum_var[:, off] + 1e-12)
    if name in ("diagonal", "deterministic"):
        assert torch.allclose(got[:, off], sum_var[:, off])


@pytest.mark.parametrize("name", ["spectral", "whitened"])
def test_spectral_covariance_is_channel_block_diagonal(name):
    """The spectral FFT runs over spatial axes only, so different channels are
    uncorrelated: cross-channel pairs reduce to the sum of marginal variances."""
    C = 2
    method, pred, _ = _build(name, C)
    n, V = H * W, C * H * W
    rows = torch.arange(V)

    var = method.get_marg_std(pred).flatten(start_dim=1).pow(2)
    got = method.pair_diff_var(pred, rows)

    cross = (torch.arange(V) // n).unsqueeze(-1) != (torch.arange(V) // n).unsqueeze(0)
    expected = (var.unsqueeze(-1) + var.unsqueeze(-2))[:, cross]
    assert torch.allclose(got[:, cross], expected, atol=1e-12)


def test_lora_couples_channels():
    """LorD reshapes U over the whole flattened event, so -- unlike the
    spectral families -- its cross-channel covariance is *not* zero."""
    C = 2
    method, pred, _ = _build("lora", C)
    n, V = H * W, C * H * W
    rows = torch.arange(V)

    var = method.get_marg_std(pred).flatten(start_dim=1).pow(2)
    got = method.pair_diff_var(pred, rows)

    cross = (torch.arange(V) // n).unsqueeze(-1) != (torch.arange(V) // n).unsqueeze(0)
    sum_var = (var.unsqueeze(-1) + var.unsqueeze(-2))[:, cross]
    assert not torch.allclose(got[:, cross], sum_var, atol=1e-6)


# ---------------------------------------------------------------------------
# Layout / blocking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_row_blocks_agree_with_the_full_matrix(name):
    """Slicing `rows` selects the corresponding rows of the full block -- this is
    what makes the blockwise accumulation in the variogram score exact."""
    method, pred, _ = _build(name, C=1)
    V = H * W
    full = method.pair_diff_var(pred, torch.arange(V))

    blocks = torch.cat(
        [
            method.pair_diff_var(pred, torch.arange(start, min(start + 7, V)))
            for start in range(0, V, 7)
        ],
        dim=1,
    )
    assert torch.allclose(blocks, full, atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
def test_dtype_and_device_preserved(name):
    method, pred, _ = _build(name, C=1)
    got = method.pair_diff_var(pred, torch.tensor([0, 2]))
    assert got.dtype == pred.dtype
    assert got.device == pred.device


def test_non_square_and_1d_spatial_grids():
    """The circulant lag arithmetic is written for an arbitrary spatial rank; a
    1-D grid is the time-series layout (C_out, pred_len)."""
    torch.manual_seed(0)
    C, L = 1, 9
    mu = torch.randn(B, C, L, dtype=torch.float64)
    g = torch.Generator().manual_seed(2)
    base = torch.rand(B, C, L, generator=g, dtype=torch.float64) + 0.5
    s = 0.5 * (base + base[:, :, (-torch.arange(L)) % L])

    method = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    pred = torch.stack([mu, s], dim=-1)
    rows = torch.arange(L)
    got = method.pair_diff_var(pred, rows)

    fkron = _fft_matrix(L)
    sb = s.to(torch.complex128)
    sigma = (fkron.conj().T @ (sb.unsqueeze(-1) * fkron)).real[:, 0]  # (B, L, L)
    assert torch.allclose(got, _reference(sigma, rows), atol=1e-12)
