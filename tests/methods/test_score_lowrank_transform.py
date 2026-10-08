"""Tests for the low-rank core correction on the general (non-Fourier) transform path.

The core is ``Sigma_hat = diag(s) + Uhat Uhat^T`` in the B-domain, with the
rank-r factors parameterized as real physical-space fields (``Uhat = B u_j``).
The physical core is ``C = B^T diag(s) B + sum_j u_j u_j^T`` with diagonal
``Delta_i = (A^T s)_i + sum_j u_j(i)^2``, ``A = |B|^2`` (the implied marginal
variances), and the predictive distribution is ``N(mu, D_t C D_t)`` with
``D_t^2 = diag / Delta`` -- the renormalization is always active for rank > 0, so
``diag(Sigma_P) = diag`` exactly for every spectrum and every factor set.

All references are dense computations on the Kronecker-product matrix,
mirroring ``test_score_lowrank.py`` (Fourier) and
``test_score_transform.py`` (rank 0). The DCT is the primary basis here
(real-orthogonal, non-flat, defined for any n); one test uses the flat
Hadamard basis, where the rank term alone makes Delta position-dependent.
"""

import math

import pytest
import torch

from src.methods import SCORE
from src.methods.transforms import (
    SeparableOrthogonalTransform,
    dct_matrix,
    hadamard_matrix,
)

GAMMA = 10.0
OUT_CHANNELS = 1
BATCH_SIZE = 4
H, W = 4, 6
N = OUT_CHANNELS * H * W
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 100
RANK = 2

torch.manual_seed(0)


@pytest.fixture
def transform():
    return SeparableOrthogonalTransform([dct_matrix(H), dct_matrix(W)])


def _method(transform, rank=RANK, **kwargs):
    return SCORE(
        "kernel_score",
        EPS,
        N_TRAIN_SAMPLES,
        N_EVAL_SAMPLES,
        OUT_CHANNELS,
        GAMMA,
        transform=transform,
        rank=rank,
        **kwargs,
    )


def _dense_B() -> torch.Tensor:
    return torch.kron(dct_matrix(H), dct_matrix(W))


def _positive_field(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64) + 0.5


def _random_field(seed: int, extra=()) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(
        BATCH_SIZE, OUT_CHANNELS, H, W, *extra, generator=g, dtype=torch.float64
    )


def _pred(mu, diag, s, u) -> torch.Tensor:
    return torch.cat(
        [mu.unsqueeze(-1), diag.unsqueeze(-1), s.unsqueeze(-1), u], dim=-1
    )


def _dense_core(s: torch.Tensor, u: torch.Tensor, Bmat=None) -> torch.Tensor:
    """Dense physical core B^T diag(s) B + U U^T, (B, N, N)."""
    Bmat = _dense_B() if Bmat is None else Bmat
    sb = s.reshape(s.shape[0], -1)
    core = Bmat.T @ (sb.unsqueeze(-1) * Bmat)
    U = u.reshape(u.shape[0], -1, u.shape[-1])  # (B, N, r)
    return core + U @ U.transpose(-1, -2)


def _dense_deployed(diag, s, u, Bmat=None) -> torch.Tensor:
    core = _dense_core(s, u, Bmat)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    scale2 = diag.flatten(start_dim=1) / delta
    rt = scale2.sqrt()
    return rt.unsqueeze(-1) * core * rt.unsqueeze(-2)


def test_delta_renorm_always_on(transform):
    assert _method(transform)._use_delta
    # Even for a flat basis the rank term forces the renormalization.
    hada = SeparableOrthogonalTransform([hadamard_matrix(4), hadamard_matrix(8)])
    assert _method(hada)._use_delta
    assert not _method(hada, rank=0)._use_delta


def test_marg_std_exact(transform):
    m = _method(transform)
    diag, s = _positive_field(4), _positive_field(5)
    u = _random_field(6, (RANK,)) * 0.5
    pred = _pred(_random_field(7), diag, s, u)
    assert torch.allclose(m.get_marg_std(pred), diag.sqrt(), atol=1e-12)
    Sigma = _dense_deployed(diag, s, u)
    assert torch.allclose(
        torch.diagonal(Sigma, dim1=-2, dim2=-1),
        diag.reshape(BATCH_SIZE, -1),
        atol=1e-12,
    )


def test_nll_matches_dense(transform):
    m = _method(transform)
    mu, diag, s, y = (
        _random_field(7),
        _positive_field(8),
        _positive_field(9),
        _random_field(10),
    )
    u = _random_field(11, (RANK,)) * 0.5
    pred = _pred(mu, diag, s, u)

    Sigma = _dense_deployed(diag, s, u)
    diff = (y - mu).reshape(BATCH_SIZE, -1, 1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    nll_ref = 0.5 * (N * math.log(2 * math.pi) + torch.logdet(Sigma) + maha)
    result = m.nll(pred, y)
    assert torch.isclose(result, nll_ref.mean(), atol=1e-8)


def test_kernel_score_matches_dense(transform):
    m = _method(transform)
    mu, diag, s, y = (
        _random_field(12),
        _positive_field(13),
        _positive_field(14),
        _random_field(15),
    )
    u = _random_field(16, (RANK,)) * 0.5
    pred = _pred(mu, diag, s, u)

    core = _dense_core(s, u)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    scale2 = diag.flatten(start_dim=1) / delta
    mu_w = mu.flatten(start_dim=1) / scale2.sqrt()
    y_w = y.flatten(start_dim=1) / scale2.sqrt()

    eye = torch.eye(N, dtype=core.dtype)
    g2 = GAMMA**2
    diff = (mu_w - y_w).unsqueeze(-1)
    log_det1 = torch.logdet(eye + 2.0 * core / g2)
    log_det2 = torch.logdet(eye + 4.0 * core / g2)
    inner = g2 * eye + 2.0 * core
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(inner, diff)).reshape(-1)
    ref = (
        0.5 * (1.0 + torch.exp(-0.5 * log_det2))
        - torch.exp(-0.5 * log_det1 - maha)
    ).mean()

    result = m.kernel_score(pred, y)
    assert torch.isclose(result, ref, atol=1e-10)


def test_zero_factors_match_rank_zero_transform_path(transform):
    """rank > 0 with zero factors equals the rank-0 transform path exactly.

    Both paths renormalize on a non-flat basis, so unlike the Fourier
    case no unit-mean spectrum is needed for the predictive models to coincide.
    """
    m_lora = _method(transform)
    m_diag = _method(transform, rank=0)
    mu, diag, s, y = (
        _random_field(17),
        _positive_field(18),
        _positive_field(19),
        _random_field(20),
    )
    u0 = torch.zeros(BATCH_SIZE, OUT_CHANNELS, H, W, RANK, dtype=torch.float64)
    pred_lora = _pred(mu, diag, s, u0)
    pred_diag = torch.stack([mu, diag, s], dim=-1)
    assert torch.isclose(m_lora.nll(pred_lora, y), m_diag.nll(pred_diag, y), atol=1e-9)
    assert torch.isclose(
        m_lora.kernel_score(pred_lora, y), m_diag.kernel_score(pred_diag, y), atol=1e-11
    )


def test_samples_match_deployed_covariance(transform):
    m = _method(transform)
    torch.manual_seed(123)
    mu = _random_field(21)[:1]
    diag = _positive_field(22)[:1]
    s = _positive_field(23)[:1]
    u = (_random_field(24, (RANK,)) * 0.5)[:1]
    pred = _pred(mu, diag, s, u)

    samples = m.get_samples(pred, n_samples=200_000, flatten=True)[0]
    emp_mean = samples.mean(dim=-1)
    centered = samples - emp_mean.unsqueeze(-1)
    emp_cov = centered @ centered.T / samples.shape[-1]

    Sigma = _dense_deployed(diag, s, u)[0]
    assert torch.allclose(emp_mean, mu.reshape(-1), atol=2e-2)
    assert (emp_cov - Sigma).abs().max() < 3e-2


def test_flat_basis_with_rank_matches_dense():
    """Hadamard (flat) + rank: Delta = mean(s) + sum_j u_j^2 varies over the
    grid through the factors alone; the dense NLL must still match."""
    hada_H, hada_W = 4, 8
    n = OUT_CHANNELS * hada_H * hada_W
    transform = SeparableOrthogonalTransform(
        [hadamard_matrix(hada_H), hadamard_matrix(hada_W)]
    )
    m = _method(transform)
    g = torch.Generator().manual_seed(30)
    shape = (BATCH_SIZE, OUT_CHANNELS, hada_H, hada_W)
    mu = torch.randn(*shape, generator=g, dtype=torch.float64)
    y = torch.randn(*shape, generator=g, dtype=torch.float64)
    diag = torch.rand(*shape, generator=g, dtype=torch.float64) + 0.5
    s = torch.rand(*shape, generator=g, dtype=torch.float64) + 0.5
    u = torch.randn(*shape, RANK, generator=g, dtype=torch.float64) * 0.5
    pred = _pred(mu, diag, s, u)

    Bmat = torch.kron(hadamard_matrix(hada_H), hadamard_matrix(hada_W))
    Sigma = _dense_deployed(diag, s, u, Bmat)
    assert torch.allclose(
        torch.diagonal(Sigma, dim1=-2, dim2=-1),
        diag.reshape(BATCH_SIZE, -1),
        atol=1e-12,
    )
    diff = (y - mu).reshape(BATCH_SIZE, -1, 1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    nll_ref = 0.5 * (n * math.log(2 * math.pi) + torch.logdet(Sigma) + maha)
    assert torch.isclose(m.nll(pred, y), nll_ref.mean(), atol=1e-8)


def test_gradients_flow(transform):
    m = _method(transform)
    head = torch.randn(
        BATCH_SIZE, OUT_CHANNELS * (3 + RANK), H, W, dtype=torch.float64,
        requires_grad=True,
    )
    y = _random_field(25)
    for loss_name in ("kernel_score", "nll"):
        pred = m.process_output(head)
        loss = getattr(m, loss_name)(pred, y)
        (grad,) = torch.autograd.grad(loss, head)
        g = grad.reshape(BATCH_SIZE, OUT_CHANNELS, 3 + RANK, H, W)
        assert torch.isfinite(grad).all(), loss_name
        assert g[:, :, 2].abs().max() > 0, loss_name      # spectrum
        assert g[:, :, 3:].abs().max() > 0, loss_name     # factors


def test_pair_diff_var_matches_dense(transform):
    """Var(y_i - y_j) from the closed-form hook equals the dense computation
    on the renormalized DCT + low-rank covariance."""
    m = _method(transform)
    diag, s = _positive_field(30), _positive_field(31)
    u = _random_field(32, (RANK,)) * 0.5
    pred = _pred(_random_field(33), diag, s, u)
    Sigma = _dense_deployed(diag, s, u)
    rows = torch.tensor([0, 7, N - 1])
    v = m.pair_diff_var(pred, rows)
    dSig = torch.diagonal(Sigma, dim1=-2, dim2=-1)
    ref = dSig[:, rows].unsqueeze(-1) + dSig.unsqueeze(-2) - 2.0 * Sigma[:, rows, :]
    assert torch.allclose(v, ref.clamp_min(0.0), atol=1e-9)


def test_pair_diff_var_rank0_transform_matches_dense(transform):
    m = _method(transform, rank=0)
    diag, s = _positive_field(34), _positive_field(35)
    pred = torch.stack([_random_field(36), diag, s], dim=-1)
    Bmat = _dense_B()
    sb = s.reshape(s.shape[0], -1)
    core = Bmat.T @ (sb.unsqueeze(-1) * Bmat)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    rt = (diag.flatten(start_dim=1) / delta).sqrt()
    Sigma = rt.unsqueeze(-1) * core * rt.unsqueeze(-2)
    rows = torch.tensor([1, N // 2])
    v = m.pair_diff_var(pred, rows)
    dSig = torch.diagonal(Sigma, dim1=-2, dim2=-1)
    ref = dSig[:, rows].unsqueeze(-1) + dSig.unsqueeze(-2) - 2.0 * Sigma[:, rows, :]
    assert torch.allclose(v, ref.clamp_min(0.0), atol=1e-9)
