"""Tests for the low-rank correction of the SCORE core (rank > 0).

The core is ``Sigma_hat = Circ(s) + U U^T`` with real physical-space factor
fields (paper Eqs. (11)-(12)); the predictive distribution is
``N(mu, D_t C D_t)`` with ``D_t^2 = diag / Delta`` and the implied marginal
variances ``Delta_i = mean(s) + sum_j u_j(i)^2``. All references are dense
computations on the Kronecker DFT matrix, mirroring ``test_score.py`` /
``test_score_transform.py``.
"""

import math

import torch

from src.methods import SCORE

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


def _method(rank=RANK, **kwargs):
    return SCORE(
        "kernel_score",
        EPS,
        N_TRAIN_SAMPLES,
        N_EVAL_SAMPLES,
        OUT_CHANNELS,
        GAMMA,
        rank=rank,
        **kwargs,
    )


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _symmetric_s(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)
    base = base + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


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


def _dense_core(s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Dense core Circ(s) + U U^T, (B, N, N) real."""
    fkron = torch.kron(_fft_matrix(H), _fft_matrix(W))
    sb = s.reshape(s.shape[0], -1).to(torch.complex128)
    circ = (fkron.conj().T @ (sb.unsqueeze(-1) * fkron)).real
    U = u.reshape(u.shape[0], -1, u.shape[-1])  # (B, N, r)
    return circ + U @ U.transpose(-1, -2)


def _dense_deployed(diag, s, u) -> torch.Tensor:
    core = _dense_core(s, u)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    scale2 = diag.flatten(start_dim=1) / delta
    rt = scale2.sqrt()
    return rt.unsqueeze(-1) * core * rt.unsqueeze(-2)


def test_group_widths_and_zero_init():
    m = _method()
    assert m.group_widths == {
        "mu": OUT_CHANNELS,
        "sigma": OUT_CHANNELS,
        "other": OUT_CHANNELS * (1 + RANK),
    }
    assert m.out_channels == OUT_CHANNELS * (3 + RANK)
    assert not m.other_zero_init
    assert _method(rank=0).other_zero_init


def test_transform_plus_rank_supported():
    """rank > 0 combines with a general transform; the renormalization is forced on.

    The dense ground-truth comparisons for this combination live in
    ``test_score_lowrank_transform.py``.
    """
    from src.methods.transforms import SeparableOrthogonalTransform, haar_matrix

    m = _method(transform=SeparableOrthogonalTransform([haar_matrix(4), haar_matrix(8)]))
    assert m._use_delta


def test_process_output_layout():
    m = _method()
    head = torch.randn(
        BATCH_SIZE, OUT_CHANNELS * (3 + RANK), H, W, dtype=torch.float64
    )
    pred = m.process_output(head)
    assert pred.shape == (BATCH_SIZE, OUT_CHANNELS, H, W, 3 + RANK)
    diag, s = pred[..., 1], pred[..., 2]
    assert (diag > 0).all()
    assert (s > 0).all()
    assert torch.allclose(
        s.mean(dim=(2, 3)),
        torch.ones(BATCH_SIZE, OUT_CHANNELS, dtype=torch.float64),
        atol=1e-12,
    )
    # The factor channels pass through the identity map.
    raw = head.reshape(BATCH_SIZE, OUT_CHANNELS, 3 + RANK, H, W).movedim(2, -1)
    assert torch.allclose(pred[..., 3:], raw[..., 3:])


def test_marg_std_exact():
    m = _method()
    diag, s = _positive_field(4), _symmetric_s(5)
    u = _random_field(6, (RANK,)) * 0.5
    pred = _pred(_random_field(7), diag, s, u)
    assert torch.allclose(m.get_marg_std(pred), diag.sqrt(), atol=1e-12)
    Sigma = _dense_deployed(diag, s, u)
    assert torch.allclose(
        torch.diagonal(Sigma, dim1=-2, dim2=-1),
        diag.reshape(BATCH_SIZE, -1),
        atol=1e-12,
    )


def test_nll_matches_dense():
    m = _method()
    mu, diag, s, y = (
        _random_field(7),
        _positive_field(8),
        _symmetric_s(9),
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


def test_kernel_score_matches_dense():
    m = _method()
    mu, diag, s, y = (
        _random_field(12),
        _positive_field(13),
        _symmetric_s(14),
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


def test_rank_zero_padding_matches_diag_core():
    """rank > 0 with zero factors reproduces the pure diagonal-core scores.

    The rank path always renormalizes by the implied marginal variances while
    the diagonal Fourier path does not, so the two predictive models coincide
    exactly for unit-mean
    spectra (the softmax regime), which is what we compare here.
    """
    m_lora = _method()
    m_diag = _method(rank=0)
    mu, diag, s, y = (
        _random_field(17),
        _positive_field(18),
        _symmetric_s(19),
        _random_field(20),
    )
    s = s / s.mean(dim=(2, 3), keepdim=True)
    u0 = torch.zeros(BATCH_SIZE, OUT_CHANNELS, H, W, RANK, dtype=torch.float64)
    pred_lora = _pred(mu, diag, s, u0)
    pred_diag = torch.stack([mu, diag, s], dim=-1)
    assert torch.isclose(m_lora.nll(pred_lora, y), m_diag.nll(pred_diag, y), atol=1e-9)
    assert torch.isclose(
        m_lora.kernel_score(pred_lora, y), m_diag.kernel_score(pred_diag, y), atol=1e-11
    )


def test_samples_match_deployed_covariance():
    m = _method()
    torch.manual_seed(123)
    mu = _random_field(21)[:1]
    diag = _positive_field(22)[:1]
    s = _symmetric_s(23)[:1]
    u = (_random_field(24, (RANK,)) * 0.5)[:1]
    pred = _pred(mu, diag, s, u)

    samples = m.get_samples(pred, n_samples=200_000, flatten=True)[0]
    emp_mean = samples.mean(dim=-1)
    centered = samples - emp_mean.unsqueeze(-1)
    emp_cov = centered @ centered.T / samples.shape[-1]

    Sigma = _dense_deployed(diag, s, u)[0]
    assert torch.allclose(emp_mean, mu.reshape(-1), atol=2e-2)
    assert (emp_cov - Sigma).abs().max() < 3e-2


def test_gradients_flow():
    m = _method()
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


def test_pair_diff_var_matches_dense():
    """Fourier + low-rank core: closed-form pair-difference variances vs dense Sigma."""
    m = _method()
    diag, s = _positive_field(30), _symmetric_s(31)
    u = _random_field(32, (RANK,)) * 0.5
    pred = _pred(_random_field(33), diag, s, u)
    Sigma = _dense_deployed(diag, s, u)
    rows = torch.tensor([0, 3, N - 2])
    v = m.pair_diff_var(pred, rows)
    dSig = torch.diagonal(Sigma, dim1=-2, dim2=-1)
    ref = dSig[:, rows].unsqueeze(-1) + dSig.unsqueeze(-2) - 2.0 * Sigma[:, rows, :]
    assert torch.allclose(v, ref.clamp_min(0.0), atol=1e-8)
