"""Tests for the general (non-Fourier) transform path of SCORE.

With a :class:`SeparableOrthogonalTransform` the predictive distribution is
``N(mu, Sigma_P)`` with ``Sigma_P = D_t C D_t``, ``C = B^T diag(s) B`` the raw
core and ``D_t^2 = diag / Delta(s)`` when the automatic renormalization by the
implied marginal variances is active (non-flat basis), ``D_t^2 = diag``
otherwise. All references are independent dense computations on the
Kronecker-product matrix, mirroring ``test_score.py``.
"""

import math

import pytest
import torch

from src.methods import SCORE
from src.methods.transforms import (
    SeparableOrthogonalTransform,
    haar_matrix,
    hadamard_matrix,
)

GAMMA = 10.0
OUT_CHANNELS = 1
BATCH_SIZE = 4
H, W = 4, 8
N = OUT_CHANNELS * H * W
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 100

torch.manual_seed(0)


@pytest.fixture
def transform():
    return SeparableOrthogonalTransform([haar_matrix(H), haar_matrix(W)])


def _method(transform, **kwargs):
    return SCORE(
        "kernel_score",
        EPS,
        N_TRAIN_SAMPLES,
        N_EVAL_SAMPLES,
        OUT_CHANNELS,
        GAMMA,
        transform=transform,
        **kwargs,
    )


def _dense_B() -> torch.Tensor:
    """The full transform matrix, rows = basis functions of the 2-D grid."""
    return torch.kron(haar_matrix(H), haar_matrix(W))


def _positive_field(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64) + 0.5


def _random_field(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)


def _core_cov(s: torch.Tensor) -> torch.Tensor:
    """Dense raw core C = B^T diag(s) B, (B, N, N)."""
    Bmat = _dense_B()
    sb = s.reshape(s.shape[0], -1)
    return Bmat.T @ (sb.unsqueeze(-1) * Bmat)

def _deployed_cov(diag, s, use_delta: bool) -> torch.Tensor:
    core = _core_cov(s)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    scale2 = diag.flatten(start_dim=1)
    if use_delta:
        scale2 = scale2 / delta
    rt = scale2.sqrt()
    return rt.unsqueeze(-1) * core * rt.unsqueeze(-2)


def test_flatness_detection():
    haar = SeparableOrthogonalTransform([haar_matrix(H), haar_matrix(W)])
    hada = SeparableOrthogonalTransform([hadamard_matrix(H), hadamard_matrix(W)])
    assert not haar.is_flat
    assert hada.is_flat
    # Automatic normalization choice follows the flatness.
    assert _method(haar)._use_delta
    assert not _method(hada)._use_delta


def test_forward_matches_dense_and_roundtrips(transform):
    x = _random_field(1)
    z = transform.forward(x)
    z_dense = (_dense_B() @ x.reshape(BATCH_SIZE, -1).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(z.reshape(BATCH_SIZE, -1), z_dense, atol=1e-12)
    assert torch.allclose(transform.inverse(z), x, atol=1e-12)


def test_delta_matches_dense_core_diagonal(transform):
    s = _positive_field(2)
    delta = transform.delta(s)
    dense_diag = torch.diagonal(_core_cov(s), dim1=-2, dim2=-1)
    assert torch.allclose(delta.reshape(BATCH_SIZE, -1), dense_diag, atol=1e-12)


def test_flat_delta_is_mean_spectrum():
    hada = SeparableOrthogonalTransform([hadamard_matrix(H), hadamard_matrix(W)])
    s = _positive_field(3)
    expected = s.mean(dim=(2, 3), keepdim=True).expand_as(s)
    assert torch.allclose(hada.delta(s), expected, atol=1e-12)


def test_process_output_no_symmetrization(transform):
    """The general path keeps the softmax (mean(s) = 1) but imposes no
    conjugate-symmetry constraint on the spectrum."""
    method = _method(transform)
    head = torch.randn(BATCH_SIZE, OUT_CHANNELS * 3, H, W, dtype=torch.float64)
    pred = method.process_output(head)
    diag, s = pred[..., 1], pred[..., 2]
    assert (diag > 0).all()
    assert (s > 0).all()
    assert torch.allclose(
        s.mean(dim=(2, 3)),
        torch.ones(BATCH_SIZE, OUT_CHANNELS, dtype=torch.float64),
        atol=1e-12,
    )


def test_marg_std_exact_with_delta(transform):
    """With the renormalization the predictive marginal variance is diag exactly,
    for an arbitrary (asymmetric, non-unit-mean) spectrum."""
    method = _method(transform)
    diag, s = _positive_field(4), _positive_field(5)
    pred = torch.stack([_random_field(6), diag, s], dim=-1)
    assert torch.allclose(method.get_marg_std(pred), diag.sqrt(), atol=1e-12)
    # Dense cross-check on the predictive covariance.
    Sigma = _deployed_cov(diag, s, use_delta=True)
    assert torch.allclose(
        torch.diagonal(Sigma, dim1=-2, dim2=-1),
        diag.reshape(BATCH_SIZE, -1),
        atol=1e-12,
    )


def test_marg_std_free_without_delta(transform):
    method = _method(transform, delta_renorm=False)
    diag, s = _positive_field(4), _positive_field(5)
    pred = torch.stack([_random_field(6), diag, s], dim=-1)
    expected = (diag * transform.delta(s)).sqrt()
    assert torch.allclose(method.get_marg_std(pred), expected, atol=1e-12)


@pytest.mark.parametrize("use_delta", [True, False], ids=["delta", "free"])
def test_nll_matches_dense(transform, use_delta):
    method = _method(transform, delta_renorm=use_delta)
    mu, diag, s, y = (
        _random_field(7),
        _positive_field(8),
        _positive_field(9),
        _random_field(10),
    )
    pred = torch.stack([mu, diag, s], dim=-1)
    Sigma = _deployed_cov(diag, s, use_delta=use_delta)
    diff = (y - mu).reshape(BATCH_SIZE, -1, 1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    nll_ref = 0.5 * (N * math.log(2 * math.pi) + torch.logdet(Sigma) + maha)
    result = method.nll(pred, y)
    assert torch.isclose(result, nll_ref.mean(), atol=1e-8)


@pytest.mark.parametrize("use_delta", [True, False], ids=["delta", "free"])
def test_kernel_score_matches_dense(transform, use_delta):
    """Stage-2 kernel score = dense closed form of the raw core C scored on the
    standardized residual D_t^{-1}(mu - y)."""
    method = _method(transform, delta_renorm=use_delta)
    mu, diag, s, y = (
        _random_field(11),
        _positive_field(12),
        _positive_field(13),
        _random_field(14),
    )
    pred = torch.stack([mu, diag, s], dim=-1)

    core = _core_cov(s)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    scale2 = diag.flatten(start_dim=1)
    if use_delta:
        scale2 = scale2 / delta
    mu_w = mu.flatten(start_dim=1) / scale2.sqrt()
    y_w = y.flatten(start_dim=1) / scale2.sqrt()

    eye = torch.eye(N, dtype=core.dtype)
    g2 = GAMMA**2
    diff = (mu_w - y_w).unsqueeze(-1)
    log_det1 = torch.logdet(eye + 2.0 * core / g2)
    log_det2 = torch.logdet(eye + 4.0 * core / g2)
    inner = g2 * eye + 2.0 * core
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(inner, diff)).reshape(-1)
    ref = (0.5 * (1.0 + torch.exp(-0.5 * log_det2)) - torch.exp(-0.5 * log_det1 - maha)).mean()

    result = method.kernel_score(pred, y)
    assert torch.isclose(result, ref, atol=1e-10)


def test_samples_match_deployed_covariance(transform):
    """Empirical mean/covariance of the sampler match Sigma_P = D_t C D_t."""
    method = _method(transform)
    torch.manual_seed(123)
    mu = _random_field(15)[:1]
    diag = _positive_field(16)[:1]
    s = _positive_field(17)[:1]
    pred = torch.stack([mu, diag, s], dim=-1)

    samples = method.get_samples(pred, n_samples=200_000, flatten=True)[0]  # (N, M)
    emp_mean = samples.mean(dim=-1)
    centered = samples - emp_mean.unsqueeze(-1)
    emp_cov = centered @ centered.T / samples.shape[-1]

    Sigma = _deployed_cov(diag, s, use_delta=True)[0]
    assert torch.allclose(emp_mean, mu.reshape(-1), atol=2e-2)
    assert (emp_cov - Sigma).abs().max() < 3e-2


def test_gradient_flows_through_delta(transform):
    """The stage-2 losses are differentiable through the renormalized
    standardization: gradients w.r.t. the raw spectral head are finite and nonzero."""
    method = _method(transform)
    head = torch.randn(
        BATCH_SIZE, OUT_CHANNELS * 3, H, W, dtype=torch.float64, requires_grad=True
    )
    y = _random_field(18)
    for loss_name in ("kernel_score", "nll"):
        pred = method.process_output(head)
        loss = getattr(method, loss_name)(pred, y)
        (grad,) = torch.autograd.grad(loss, head)
        s_grad = grad.reshape(BATCH_SIZE, OUT_CHANNELS, 3, H, W)[:, :, 2]
        assert torch.isfinite(grad).all(), loss_name
        assert s_grad.abs().max() > 0, loss_name
