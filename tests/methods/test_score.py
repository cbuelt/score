"""Tests for :class:`SCORE`.

The predictive distribution is ``N(mu, Sigma_P)`` with
``Sigma_P = D Fᴴ diag(s) F D`` and ``D = diag(sqrt(diag))`` the per-pixel scale.
Stage 2 of the two-stage fit scores the *correlation* core on the standardized
target, so ``kernel_score`` here is the core score (validated against the dense
core covariance and the standardized residual), and ``nll`` scores the full
predictive covariance. The closed-form ``crps`` is the stage-1 marginal loss
(std sqrt(diag), core-independent); the predictive-marginal CRPS is a
sample-based eval metric (see ``test_evaluator.py``). References are independent dense /
per-coord closed forms, mirroring ``test_spectral.py``.
"""

import math

import pytest
import torch

from src.methods import SCORE

GAMMA = 10.0
OUT_CHANNELS = 1
BATCH_SIZE = 8
H, W = 4, 6  # spatial grid; the event dimension is C * H * W
N = OUT_CHANNELS * H * W
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 4000
SQRT_2PI = math.sqrt(2 * math.pi)


def test_out_channels():
    """The SCORE head predicts mu, diag and a spectrum per channel, so it
    exposes three times the deterministic head's channels."""
    method = SCORE(
        "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA
    )
    assert method.out_channels == OUT_CHANNELS * 3


@pytest.fixture
def method():
    return SCORE(
        "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA
    )


def _full(value: float) -> torch.Tensor:
    return torch.full((BATCH_SIZE, OUT_CHANNELS, H, W), value, dtype=torch.float64)


def _symmetric_s(seed: int) -> torch.Tensor:
    """A positive, conjugate-symmetric spectral diagonal s (real circulant core)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)
    base = base + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def _positive_field(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64) + 0.5


def _random_field(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)


def _pred(mu: torch.Tensor, diag: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Pack mu, diag and the spectrum the way ``process_output`` returns it."""
    return torch.stack([mu, diag, s], dim=-1)


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _core_covariance(s: torch.Tensor) -> torch.Tensor:
    """Dense core covariance Σ_hat = Fᴴ diag(s) F, (B, N, N) real."""
    fkron = torch.kron(_fft_matrix(H), _fft_matrix(W))
    sb = s.reshape(s.shape[0], -1).to(torch.complex128)
    sigma = fkron.conj().T @ (sb.unsqueeze(-1) * fkron)
    return sigma.real


def _deployed_covariance(diag: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Dense predictive Σ_P = D Σ_hat D with D = diag(sqrt(diag)), (B, N, N)."""
    core = _core_covariance(s)
    sqrt_d = diag.flatten(start_dim=1).sqrt()  # (B, N)
    return sqrt_d.unsqueeze(-1) * core * sqrt_d.unsqueeze(-2)


def _deployed_nll(mu, diag, s, y) -> float:
    """Dense Gaussian NLL for the predictive Σ_P, averaged over the batch."""
    mu_f = mu.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    Sigma = _deployed_covariance(diag, s)
    diff = (y_f - mu_f).unsqueeze(-1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    log_det = torch.logdet(Sigma)
    nll = 0.5 * (N * math.log(2 * math.pi) + log_det + maha)
    return nll.mean().item()


def _whitened_kernel_score(mu, diag, s, y) -> float:
    """Closed-form kernel score of the core Σ_hat against the standardized residual.

    Stage 2 scores ``N(D⁻¹ mu, Σ_hat)`` against ``D⁻¹ y``: the dense kernel score
    on the core covariance with standardized mean/target.
    """
    sqrt_d = diag.sqrt()
    mu_w = (mu / sqrt_d).flatten(start_dim=1)
    y_w = (y / sqrt_d).flatten(start_dim=1)
    Sigma = _core_covariance(s)
    eye = torch.eye(N, dtype=Sigma.dtype)
    g2 = GAMMA**2

    diff = (mu_w - y_w).unsqueeze(-1)
    log_det1 = torch.logdet(eye + 2.0 * Sigma / g2)
    log_det2 = torch.logdet(eye + 4.0 * Sigma / g2)
    inner = g2 * eye + 2.0 * Sigma
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(inner, diff)).reshape(-1)

    fac1 = torch.exp(-0.5 * log_det1 - maha)
    fac2 = torch.exp(-0.5 * log_det2)
    score = 0.5 * (1.0 + fac2) - fac1
    return score.mean().item()


def _marginal_kernel_score(mu, diag, y) -> float:
    """Per-coordinate (d=1) Gaussian kernel score with the 1-D bandwidth.

    gamma2 = factor * d (joint), so the per-marginal bandwidth is gamma2 / d.
    """
    v = diag.flatten(start_dim=1)
    diff = (mu - y).flatten(start_dim=1)
    d = v.shape[-1]
    g2 = (GAMMA**2) / d

    diag1 = 1.0 + 2.0 * v / g2
    diag2 = 1.0 + 4.0 * v / g2
    fac1 = torch.exp(-0.5 * torch.log(diag1) - diff.pow(2) / (diag1 * g2))
    fac2 = torch.exp(-0.5 * torch.log(diag2))
    score = 0.5 * (1.0 + fac2) - fac1
    return score.mean(dim=-1).mean().item()


def _crps(mu, diag, s, y) -> float:
    """Stage-1 marginal CRPS with std sqrt(diag), independent of the core s.

    The closed-form ``crps`` is the stage-1 loss that fits mu/diag as the physical
    marginals; it deliberately ignores the core (the core is expected to
    learn mean(s) ~ 1 on its own). The predictive marginal std sqrt(diag * mean(s))
    is instead scored by the sample-based eval metric in src/evaluation/metrics.py.
    """
    sigma = torch.sqrt(diag)
    w = (y - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(w / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * w.pow(2)) / SQRT_2PI
    per_elem = sigma * (w * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return per_elem.flatten(start_dim=1).mean(dim=1).mean().item()


def _case(mu, diag, s, y) -> dict:
    return dict(
        mu=mu,
        diag=diag,
        s=s,
        y=y,
        pred=_pred(mu, diag, s),
        mse=torch.nn.functional.mse_loss(mu, y).item(),
        mae=torch.nn.functional.l1_loss(mu, y).item(),
        nll=_deployed_nll(mu, diag, s, y),
        kernel_score=_whitened_kernel_score(mu, diag, s, y),
        marginal_kernel_score=_marginal_kernel_score(mu, diag, y),
        crps=_crps(mu, diag, s, y),
    )


CASES = {
    # Flat unit core, unit diag: Σ_P = I.
    "white_zero": _case(_full(0.0), _full(1.0), _full(1.0), _full(0.0)),
    # Mean shift, non-unit diag, flat core.
    "white_shift": _case(_full(1.0), _full(2.0), _full(1.0), _full(0.0)),
    # Coloured core with a non-trivial per-pixel diag scale.
    "coloured": _case(
        _random_field(2), _positive_field(5), _symmetric_s(3), _random_field(4)
    ),
}

CASE_IDS = list(CASES)
CASE_VALUES = list(CASES.values())


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mse(method, case):
    result = method.mse(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mse"], dtype=torch.float64))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mae(method, case):
    result = method.mae(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mae"], dtype=torch.float64))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_deployed_nll_matches_dense(method, case):
    # Validates the log|det D| Jacobian correction on top of the core NLL.
    result = method.nll(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["nll"], dtype=torch.float64))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_whitened_kernel_score_matches_dense(method, case):
    # The training-time kernel_score scores the correlation core on the
    # standardized residual, matching the dense closed form on Σ_hat.
    result = method.kernel_score(case["pred"], case["y"])
    assert torch.isclose(
        result, torch.tensor(case["kernel_score"], dtype=torch.float64)
    )


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_marginal_kernel_score_matches_per_coord(method, case):
    result = method.marginal_kernel_score(case["pred"], case["y"])
    assert torch.isclose(
        result, torch.tensor(case["marginal_kernel_score"], dtype=torch.float64)
    )


def test_marginal_kernel_score_uses_1d_bandwidth(method):
    """The marginal score must use the 1-D bandwidth gamma2/d, not the joint gamma2.

    Equivalently, a method built with the 1-D bandwidth gamma_1d = sqrt(factor)
    evaluated with its *own* (then unscaled) gamma2 reproduces the marginal score,
    pinning the gamma2/d recovery. Here we just assert the joint-bandwidth score
    differs, so a regression that forgets the /d is caught.
    """
    mu, diag, s, y = (
        _random_field(2),
        _positive_field(5),
        _symmetric_s(3),
        _random_field(4),
    )
    pred = _pred(mu, diag, s)
    marginal = method.marginal_kernel_score(pred, y)

    # Score with the joint bandwidth (no /d): clearly over-smoothed -> ~0.
    v = diag.flatten(start_dim=1)
    diff = (mu - y).flatten(start_dim=1)
    g2_joint = GAMMA**2
    diag1 = 1.0 + 2.0 * v / g2_joint
    diag2 = 1.0 + 4.0 * v / g2_joint
    fac1 = torch.exp(-0.5 * torch.log(diag1) - diff.pow(2) / (diag1 * g2_joint))
    fac2 = torch.exp(-0.5 * torch.log(diag2))
    joint = (0.5 * (1.0 + fac2) - fac1).mean()
    assert not torch.isclose(marginal, joint)
    # The 1-D bandwidth keeps the kernel discriminating; the joint bandwidth
    # over-smooths each marginal, so its score is much smaller (a regression that
    # forgets the /d would collapse the marginal score toward this value).
    assert marginal.item() > 5.0 * joint.item()


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_crps_matches_marginal(method, case):
    result = method.crps(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["crps"], dtype=torch.float64))


def _mirror_spectrum(s: torch.Tensor) -> torch.Tensor:
    """s_{-k} on both spatial axes (the conjugate mirror)."""
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return s[:, :, idx_h][:, :, :, idx_w]


def test_process_output(method):
    """process_output yields positive diag/spectral, a conjugate-symmetric core,
    and (by default) a unit-mean core: mean(s) == 1 per (batch, channel), so the
    circulant is a correlation matrix and diag(Sigma_P) == diag."""
    head = torch.randn(BATCH_SIZE, OUT_CHANNELS * 3, H, W, dtype=torch.float64)
    pred = method.process_output(head)
    assert pred.shape == (BATCH_SIZE, OUT_CHANNELS, H, W, 3)
    diag, s = pred[..., 1], pred[..., 2]
    assert (diag > 0).all()
    assert (s > 0).all()
    assert torch.allclose(s, _mirror_spectrum(s), atol=1e-10)
    # Unit mean over the frequency axes, per (batch, channel).
    mean_s = s.mean(dim=(2, 3))
    assert torch.allclose(
        mean_s, torch.ones(BATCH_SIZE, OUT_CHANNELS, dtype=torch.float64), atol=1e-12
    )


def test_process_output_unnormalized_core():
    """With normalize_core=False the core keeps the free-scale softplus map: still
    positive and conjugate-symmetric, but its mean is not pinned to 1."""
    method = SCORE(
        "crps",
        EPS,
        N_TRAIN_SAMPLES,
        N_EVAL_SAMPLES,
        OUT_CHANNELS,
        GAMMA,
        normalize_core=False,
    )
    head = torch.randn(BATCH_SIZE, OUT_CHANNELS * 3, H, W, dtype=torch.float64)
    s = method.process_output(head)[..., 2]
    assert (s > 0).all()
    assert torch.allclose(s, _mirror_spectrum(s), atol=1e-10)
    assert not torch.allclose(
        s.mean(dim=(2, 3)),
        torch.ones(BATCH_SIZE, OUT_CHANNELS, dtype=torch.float64),
    )


def test_process_output_softplus_normalized_core():
    """core_activation='softplus' keeps the softmax's gauge (mean(s) = 1,
    positive, conjugate-symmetric) but uses the elementwise map."""
    method = SCORE(
        "crps",
        EPS,
        N_TRAIN_SAMPLES,
        N_EVAL_SAMPLES,
        OUT_CHANNELS,
        GAMMA,
        core_activation="softplus",
    )
    head = torch.randn(BATCH_SIZE, OUT_CHANNELS * 3, H, W, dtype=torch.float64)
    s = method.process_output(head)[..., 2]
    assert (s > 0).all()
    assert torch.allclose(s, _mirror_spectrum(s), atol=1e-10)
    assert torch.allclose(
        s.mean(dim=(2, 3)),
        torch.ones(BATCH_SIZE, OUT_CHANNELS, dtype=torch.float64),
        atol=1e-12,
    )
    # It is a genuinely different map than the softmax on the same head.
    softmax_method = SCORE(
        "crps", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA
    )
    s_softmax = softmax_method.process_output(head)[..., 2]
    assert not torch.allclose(s, s_softmax)


def test_invalid_core_activation_raises():
    with pytest.raises(ValueError, match="core_activation"):
        SCORE(
            "crps",
            EPS,
            N_TRAIN_SAMPLES,
            N_EVAL_SAMPLES,
            OUT_CHANNELS,
            GAMMA,
            core_activation="relu",
        )
