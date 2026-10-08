import math

import pytest
import scoringrules as sr
import torch

from src.methods import Diag

GAMMA = 0.5
OUT_CHANNELS = 1
BATCH_SIZE = 8
D = 2
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 50
SQRT_2PI = math.sqrt(2 * math.pi)


def test_out_channels():
    """
    Test correct dimensionality.

    Diag predicts a mean and a standard deviation per channel, so it
    exposes twice as many output channels as the deterministic head.
    """
    approximation = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)
    out_channels = approximation.out_channels
    assert out_channels == OUT_CHANNELS * 2


def test_n_samples_follows_training_flag():
    """``n_samples`` returns the eval count by default and the train count in
    training mode, so sampling-based scores use the right number of draws."""
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)
    assert method.n_samples == N_EVAL_SAMPLES
    method.training = True
    assert method.n_samples == N_TRAIN_SAMPLES


@pytest.fixture
def method():
    """A Diag instance shared across the metric tests."""
    return Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)


def _full(value: float) -> torch.Tensor:
    """A constant tensor with the (mu/var/y) shape every example uses."""
    return torch.full((BATCH_SIZE, OUT_CHANNELS, D), value)


def _diag(values: tuple[float, ...]) -> torch.Tensor:
    """A per-feature tensor, one value per feature (len == D)."""
    assert len(values) == D
    row = torch.tensor(values).reshape(1, 1, D)
    return row.expand(BATCH_SIZE, OUT_CHANNELS, D).contiguous()


def _pred(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Pack mu and the variance the way ``Diag.process_output`` does.

    The second channel holds the variance (sigma**2), not the standard
    deviation, so the closed-form helpers below derive sigma as needed.
    """
    return torch.stack([mu, var], dim=-1)


def _nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form Gaussian NLL, summed over features, averaged over batch."""
    full = torch.log(sigma) + math.log(SQRT_2PI) + (y - mu).pow(2) / (2 * sigma.pow(2))
    return full.flatten(start_dim=1).sum(dim=1).mean().item()


def _kernel_score(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form Gaussian kernel score for a diagonal covariance."""
    gamma2 = GAMMA**2
    var = sigma.pow(2).flatten(start_dim=1)
    diff = (mu - y).flatten(start_dim=1)

    diag1 = 1.0 + 2.0 * var / gamma2
    diag2 = 1.0 + 4.0 * var / gamma2
    log_det1 = torch.log(diag1).sum(dim=-1)
    log_det2 = torch.log(diag2).sum(dim=-1)
    maha = (diff.pow(2) / diag1).sum(dim=-1)

    fac1 = torch.exp(-0.5 * log_det1 - maha / gamma2)
    fac2 = torch.exp(-0.5 * log_det2)
    score = 0.5 * (1.0 + fac2) - fac1
    return score.mean().item()


def _crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form CRPS of a normal predictive distribution.

    CRPS(N(mu, sigma), y) = sigma * [w (2 Phi(w) - 1) + 2 phi(w) - 1/sqrt(pi)]
    with w = (y - mu) / sigma. Averaged over features, then over the batch.
    """
    w = (y - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(w / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * w.pow(2)) / SQRT_2PI
    per_elem = sigma * (w * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return per_elem.flatten(start_dim=1).mean(dim=1).mean().item()


def _energy_score_reference(
    mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor
) -> float:
    """A high-sample Monte-Carlo estimate of the energy score.

    The implementation under test draws only ``N_EVAL_SAMPLES`` samples, so
    individual evaluations are noisy. We compute a tight reference here with
    many samples and assert closeness with a generous tolerance.
    """

    torch.manual_seed(0)
    n = 1000
    mu_e = mu.unsqueeze(-1).repeat_interleave(n, dim=-1)
    sigma_e = sigma.unsqueeze(-1).repeat_interleave(n, dim=-1)
    samples = mu_e + sigma_e * torch.randn_like(mu_e)
    es = sr.es_ensemble(
        obs=y.flatten(start_dim=1),
        fct=samples.flatten(start_dim=1, end_dim=-2),
        m_axis=-1,
        v_axis=-2,
        backend="torch",
    )
    return es.mean().item()


# Each case carries mu, var and y; the expected metric values are derived
# from the closed-form helpers above so the tests check the implementation
# against independent maths rather than against itself. ``var`` is the
# variance packed into the prediction; sigma = sqrt(var) feeds the helpers.
def _case(mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> dict:
    sigma = torch.sqrt(var)
    return dict(
        mu=mu,
        var=var,
        sigma=sigma,
        y=y,
        pred=_pred(mu, var),
        mse=torch.nn.functional.mse_loss(mu, y).item(),
        mae=torch.nn.functional.l1_loss(mu, y).item(),
        nll=_nll(mu, sigma, y),
        kernel_score=_kernel_score(mu, sigma, y),
        crps=_crps(mu, sigma, y),
        energy_score=_energy_score_reference(mu, sigma, y),
    )


CASES = {
    # The three cases mirror test_deterministic, here with unit variance.
    "both_zero": _case(_full(0.0), _full(1.0), _full(0.0)),
    "pred_one": _case(_full(1.0), _full(1.0), _full(0.0)),
    "pred_minus_two": _case(_full(-2.0), _full(1.0), _full(0.0)),
    # mu = y = 0 with a non-uniform diagonal variance = diag(1, 4).
    "zero_diag_var": _case(_full(0.0), _diag((1.0, 4.0)), _full(0.0)),
    # pred = 1 with a strongly anisotropic variance = diag(100, 0.01).
    "pred_one_skewed_var": _case(_full(1.0), _diag((100.0, 0.01)), _full(0.0)),
}

CASE_IDS = list(CASES)
CASE_VALUES = list(CASES.values())


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mse(method, case):
    result = method.mse(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mse"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mae(method, case):
    result = method.mae(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mae"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_nll(method, case):
    # Unlike the deterministic head, Diag has a proper density, so the
    # NLL is finite and matches the closed-form Gaussian NLL.
    result = method.nll(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["nll"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_kernel_score(method, case):
    result = method.kernel_score(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["kernel_score"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_crps(method, case):
    result = method.crps(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["crps"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_energy_score(method, case):
    # The energy score is estimated from N_EVAL_SAMPLES draws, so it is a
    # noisy Monte-Carlo estimate. Seed the RNG for reproducibility and check
    # closeness to a high-sample reference with a generous tolerance.
    torch.manual_seed(0)
    result = method.energy_score(case["pred"], case["y"])
    assert torch.isclose(
        result, torch.tensor(case["energy_score"]), atol=0.05, rtol=0.01
    )


def test_energy_score_isotropic_closed_form(method):
    """Energy score has a closed form when y = mu and sigma is isotropic.

    The slicing estimator computes a univariate CRPS per random direction v
    (a unit vector). For an isotropic Gaussian the pushed-forward standard
    deviation is s = sqrt(v^T diag(sigma^2) v) = sigma for every direction,
    and y = mu makes the standardised residual w = v^T(y - mu) / s = 0. The
    per-slice CRPS therefore collapses to the same constant for every slice,

        CRPS = sigma * [2 phi(0) - 1/sqrt(pi)]
             = sigma * (sqrt(2) - 1) / sqrt(pi),

    so the Monte-Carlo average over directions is exact (no sampling noise).
    The energy score is that CRPS scaled by the slicing constant C_d, which
    lets us check the two metrics against each other and against the maths.
    """
    sigma_val = 1.3
    mu = _full(0.7)
    var = _full(sigma_val**2)
    y = mu.clone()
    pred = _pred(mu, var)

    closed_form = sigma_val * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)

    # CRPS matches the closed form directly.
    crps = method.crps(pred, y)
    assert torch.isclose(crps, torch.tensor(closed_form))

    # The energy score is C_d * CRPS; the per-slice value is constant, so the
    # estimate is exact regardless of the seed or the number of eval samples.
    log_cd = (
        0.5 * math.log(math.pi)
        + math.lgamma((D + 1) / 2)
        - math.lgamma(D / 2)
    )
    expected_es = math.exp(log_cd) * closed_form

    torch.manual_seed(0)
    energy_score = method.energy_score(pred, y)
    assert torch.isclose(energy_score, torch.tensor(expected_es))
