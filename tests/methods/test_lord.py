import math

import pytest
import scoringrules as sr
import torch
from torch.distributions import LowRankMultivariateNormal

from src.methods import Diag, LorD

GAMMA = 0.5
OUT_CHANNELS = 1
BATCH_SIZE = 8
D = 2
RANK = 1
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 50
SQRT_2PI = math.sqrt(2 * math.pi)


def test_out_channels():
    """
    Test correct dimensionality.

    LorD predicts a mean, a diagonal variance and ``rank`` low-rank
    factors per channel, so it exposes ``2 + rank`` times as many output
    channels as the deterministic head.
    """
    approximation = LorD(
        "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA, rank=RANK
    )
    out_channels = approximation.out_channels
    assert out_channels == OUT_CHANNELS * (RANK + 2)


@pytest.fixture
def method():
    """A LorD instance shared across the metric tests."""
    return LorD("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA, rank=RANK)


def _full(value: float) -> torch.Tensor:
    """A constant tensor with the (mu/diag/y) shape every example uses."""
    return torch.full((BATCH_SIZE, OUT_CHANNELS, D), value)


def _diag(values: tuple[float, ...]) -> torch.Tensor:
    """A per-feature tensor, one variance value per feature (len == D)."""
    assert len(values) == D
    row = torch.tensor(values).reshape(1, 1, D)
    return row.expand(BATCH_SIZE, OUT_CHANNELS, D).contiguous()


def _lora(columns: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    """A constant low-rank factor U of shape (B, C, D, RANK).

    ``columns`` holds one tuple of length D per rank column.
    """
    assert len(columns) == RANK
    for col in columns:
        assert len(col) == D
    # (D, RANK) then broadcast over batch and channel.
    factor = torch.tensor(columns).transpose(0, 1).reshape(1, 1, D, RANK)
    return factor.expand(BATCH_SIZE, OUT_CHANNELS, D, RANK).contiguous()


def _zero_lora() -> torch.Tensor:
    """An all-zero low-rank factor (covariance is purely diagonal)."""
    return torch.zeros(BATCH_SIZE, OUT_CHANNELS, D, RANK)


def _pred(mu: torch.Tensor, diag: torch.Tensor, lora: torch.Tensor) -> torch.Tensor:
    """Pack mu, diag and lora the way ``LorD.process_output`` does."""
    return torch.cat([mu.unsqueeze(-1), diag.unsqueeze(-1), lora], dim=-1)


def _covariance(diag: torch.Tensor, lora: torch.Tensor) -> torch.Tensor:
    """Dense covariance Σ = diag(diag) + U Uᵀ, flattened over (C, D)."""
    d = diag.flatten(start_dim=1)
    U = lora.reshape(diag.shape[0], -1, RANK)
    return d.diag_embed() + U @ U.transpose(-1, -2)


def _nll(mu: torch.Tensor, diag: torch.Tensor, lora: torch.Tensor, y: torch.Tensor):
    """Closed-form Gaussian NLL for Σ = diag(diag) + U Uᵀ, averaged over batch.

    ``diag`` is the variance (not the standard deviation), matching the
    convention LorD feeds to ``cov_diag``. The channel axis is flattened
    into the event dimension, mirroring the implementation, so Σ is the dense
    (C*D)-dimensional covariance.
    """
    mu_f = mu.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    Sigma = _covariance(diag, lora)
    dim = mu_f.shape[-1]
    diff = (y_f - mu_f).unsqueeze(-1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    log_det = torch.logdet(Sigma)
    nll = 0.5 * (dim * math.log(2 * math.pi) + log_det + maha)
    return nll.mean().item()


def _kernel_score(
    mu: torch.Tensor, diag: torch.Tensor, lora: torch.Tensor, y: torch.Tensor
):
    """Closed-form Gaussian kernel score for a dense covariance Σ.

    With the Gaussian kernel k(x, x') = exp(-||x - x'||² / γ²),

        E[k(X, y)]  = |I + 2Σ/γ²|^{-1/2} exp(-(μ-y)ᵀ(γ²I + 2Σ)⁻¹(μ-y))
        E[k(X, X')] = |I + 4Σ/γ²|^{-1/2}
        score       = 0.5 (1 + E[k(X, X')]) - E[k(X, y)].

    Computed from the dense Σ here so the test checks the Woodbury
    implementation against independent linear algebra.
    """
    mu_f = mu.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    Sigma = _covariance(diag, lora)
    dim = mu_f.shape[-1]
    eye = torch.eye(dim, dtype=Sigma.dtype)
    g2 = GAMMA**2

    diff = (mu_f - y_f).unsqueeze(-1)
    log_det1 = torch.logdet(eye + 2.0 * Sigma / g2)
    log_det2 = torch.logdet(eye + 4.0 * Sigma / g2)
    inner = g2 * eye + 2.0 * Sigma
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(inner, diff)).reshape(-1)

    fac1 = torch.exp(-0.5 * log_det1 - maha)
    fac2 = torch.exp(-0.5 * log_det2)
    score = 0.5 * (1.0 + fac2) - fac1
    return score.mean().item()


def _crps(mu: torch.Tensor, diag: torch.Tensor, y: torch.Tensor):
    """Closed-form CRPS, averaged over features then over the batch.

    The CRPS head only uses the marginal variances (the diagonal), so the
    standard deviation per feature is sqrt(diag).
    """
    sigma = torch.sqrt(diag)
    w = (y - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(w / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * w.pow(2)) / SQRT_2PI
    per_elem = sigma * (w * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return per_elem.flatten(start_dim=1).mean(dim=1).mean().item()


def _energy_score_reference(
    mu: torch.Tensor, diag: torch.Tensor, lora: torch.Tensor, y: torch.Tensor
):
    """A high-sample energy-score estimate from the multivariate Gaussian.

    Samples are drawn from N(mu, diag(diag) + U Uᵀ) and scored with the
    classical ensemble estimator (``sr.es_ensemble``). The implementation
    under test uses a low-sample slicing estimator, so we build a tight
    reference here and assert closeness with a generous tolerance.
    """
    torch.manual_seed(0)
    mu_f = mu.flatten(start_dim=1)
    diag_f = diag.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    U = lora.reshape(mu.shape[0], -1, RANK)

    dist = LowRankMultivariateNormal(mu_f, U, diag_f)
    n = 2000
    samples = dist.sample((n,))  # (n, B, C*D)
    es = sr.es_ensemble(
        obs=y_f,
        fct=samples.permute(1, 2, 0),  # (B, C*D, n)
        m_axis=-1,
        v_axis=-2,
        backend="torch",
    )
    return es.mean().item()


# Each case carries mu, diag, lora and y; the expected metric values come from
# the closed-form/reference helpers above, so the tests check the
# implementation against independent maths rather than against itself.
def _case(
    mu: torch.Tensor, diag: torch.Tensor, lora: torch.Tensor, y: torch.Tensor
) -> dict:
    return dict(
        mu=mu,
        diag=diag,
        lora=lora,
        y=y,
        pred=_pred(mu, diag, lora),
        mse=torch.nn.functional.mse_loss(mu, y).item(),
        mae=torch.nn.functional.l1_loss(mu, y).item(),
        nll=_nll(mu, diag, lora, y),
        kernel_score=_kernel_score(mu, diag, lora, y),
        crps=_crps(mu, diag, y),
        energy_score=_energy_score_reference(mu, diag, lora, y),
    )


CASES = {
    # Purely diagonal cases (lora = 0): these mirror the diagonal tests with a
    # unit variance, so LorD and Diag must agree (see test_matches_diag).
    "both_zero": _case(_full(0.0), _full(1.0), _zero_lora(), _full(0.0)),
    "pred_one": _case(_full(1.0), _full(1.0), _zero_lora(), _full(0.0)),
    "pred_minus_two": _case(_full(-2.0), _full(1.0), _zero_lora(), _full(0.0)),
    # mu = y = 0 with a non-uniform diagonal variance diag(1, 2), lora = 0.
    "zero_diag": _case(_full(0.0), _diag((1.0, 2.0)), _zero_lora(), _full(0.0)),
    # A simple genuine low-rank structure: variance diag(1, 1) plus a single
    # rank-one factor u = (1, 0.5), so Σ = [[2, 0.5], [0.5, 1.25]].
    "simple_lora": _case(
        _full(0.5), _diag((1.0, 1.0)), _lora(((1.0, 0.5),)), _full(0.0)
    ),
}

CASE_IDS = list(CASES)
CASE_VALUES = list(CASES.values())

# The cases whose covariance is purely diagonal with unit variance, used to
# check LorD reduces to Diag.
DIAG_EQUIV_IDS = ["both_zero", "pred_one", "pred_minus_two"]


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mse(method, case):
    # mse only depends on mu, so it is independent of the low-rank factor.
    result = method.mse(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mse"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mae(method, case):
    # mae only depends on mu, so it is independent of the low-rank factor.
    result = method.mae(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mae"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_nll(method, case):
    result = method.nll(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["nll"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_kernel_score(method, case):
    # Checks the Woodbury kernel score against a dense closed form, covering
    # both the diagonal cases and the genuine low-rank case.
    result = method.kernel_score(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["kernel_score"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_crps(method, case):
    result = method.crps(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["crps"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_energy_score(method, case):
    # The energy score is estimated from N_EVAL_SAMPLES slices, so it is a
    # noisy Monte-Carlo estimate. Seed the RNG for reproducibility and check
    # closeness to a high-sample reference with a generous tolerance.
    torch.manual_seed(0)
    result = method.energy_score(case["pred"], case["y"])
    assert torch.isclose(
        result, torch.tensor(case["energy_score"]), atol=0.05, rtol=0.01
    )


@pytest.mark.parametrize("case_id", DIAG_EQUIV_IDS)
def test_matches_diag(method, case_id):
    """With lora = 0 and unit variance, LorD must equal Diag.

    Diag packs sigma (a standard deviation) while LorD packs the
    variance, but for unit variance the two coincide, so every probabilistic
    score should match between the heads.
    """
    case = CASES[case_id]
    diag_method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)
    diag_pred = torch.stack([case["mu"], _full(1.0)], dim=-1)
    lora_pred = case["pred"]
    y = case["y"]

    for name in ("nll", "kernel_score", "crps"):
        lhs = getattr(diag_method, name)(diag_pred, y)
        rhs = getattr(method, name)(lora_pred, y)
        assert torch.isclose(lhs, rhs), name

    # The energy score shares the slicing estimator and RNG draw between the
    # two heads, so seeding identically makes the comparison exact.
    torch.manual_seed(0)
    diag_es = diag_method.energy_score(diag_pred, y)
    torch.manual_seed(0)
    lora_es = method.energy_score(lora_pred, y)
    assert torch.isclose(diag_es, lora_es)
