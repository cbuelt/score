import math

import pytest
import scoringrules as sr
import torch

from src.methods import SpectralDiag

# A larger gamma than the diagonal tests keeps the Gaussian kernel score in a
# discriminating (non-saturated) regime for these variances.
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
    """SpectralDiag predicts a mean and a spectral variance per channel, so it
    exposes twice as many output channels as the deterministic head."""
    approximation = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)
    assert approximation.out_channels == OUT_CHANNELS * 2


@pytest.fixture
def method():
    """A SpectralDiag instance shared across the metric tests."""
    return SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)


def _full(value: float) -> torch.Tensor:
    """A constant field with the (mu / s / y) shape every example uses."""
    return torch.full((BATCH_SIZE, OUT_CHANNELS, H, W), value, dtype=torch.float64)


def _symmetric_s(seed: int) -> torch.Tensor:
    """A positive, conjugate-symmetric spectral diagonal s.

    Hermitian symmetry s_k == s_{-k} (over both spatial axes) makes the implied
    covariance Σ = Fᴴ diag(s) F a real symmetric PSD matrix, so the dense
    references below are exact real-arithmetic computations.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)
    base = base + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def _random_field(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)


def _pred(mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Pack mu and the spectral variance the way ``process_output`` does."""
    return torch.stack([mu, s], dim=-1)


def _fft_matrix(n: int) -> torch.Tensor:
    j = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * math.pi * torch.outer(j, j) / n) / math.sqrt(n)


def _covariance(s: torch.Tensor) -> torch.Tensor:
    """Dense physical covariance Σ = Fᴴ diag(s) F, flattened over (C, H, W).

    With the 2D unitary DFT F = F_H ⊗ F_W, this is the real circulant covariance
    the spectral-diagonal model represents. Returned as (B, N, N) real.
    """
    fkron = torch.kron(_fft_matrix(H), _fft_matrix(W))  # (N, N) complex
    sb = s.reshape(s.shape[0], -1).to(torch.complex128)  # (B, N)
    sigma = fkron.conj().T @ (sb.unsqueeze(-1) * fkron)  # (B, N, N)
    return sigma.real


def _nll(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form Gaussian NLL for the dense Σ, averaged over the batch."""
    mu_f = mu.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    Sigma = _covariance(s)
    diff = (y_f - mu_f).unsqueeze(-1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    log_det = torch.logdet(Sigma)
    nll = 0.5 * (N * math.log(2 * math.pi) + log_det + maha)
    return nll.mean().item()


def _kernel_score(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form Gaussian kernel score for the dense Σ.

    With k(x, x') = exp(-||x - x'||² / γ²),
        E[k(X, y)]  = |I + 2Σ/γ²|^{-1/2} exp(-(μ-y)ᵀ(γ²I + 2Σ)⁻¹(μ-y))
        E[k(X, X')] = |I + 4Σ/γ²|^{-1/2}
        score       = 0.5 (1 + E[k(X, X')]) - E[k(X, y)].
    """
    mu_f = mu.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    Sigma = _covariance(s)
    eye = torch.eye(N, dtype=Sigma.dtype)
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


def _crps(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form marginal CRPS.

    The spectral-diagonal covariance is stationary, so every location shares the
    marginal variance σ̄² = mean(s) (Parseval). CRPS is then the univariate
    Gaussian CRPS with that constant σ, averaged over locations then the batch.
    """
    sigma = torch.sqrt(s.flatten(start_dim=1).mean(dim=1)).reshape(-1, 1, 1, 1)
    sigma = sigma.expand_as(mu)
    w = (y - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(w / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * w.pow(2)) / SQRT_2PI
    per_elem = sigma * (w * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return per_elem.flatten(start_dim=1).mean(dim=1).mean().item()


def _energy_score_reference(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> float:
    """High-sample energy-score estimate from the dense Gaussian N(mu, Σ).

    The implementation under test uses a sliced estimator, so we build a tight
    reference with the classical ensemble estimator and assert closeness with a
    generous tolerance.
    """
    torch.manual_seed(0)
    mu_f = mu.flatten(start_dim=1)
    y_f = y.flatten(start_dim=1)
    Sigma = _covariance(s)
    eye = torch.eye(N, dtype=Sigma.dtype)
    chol = torch.linalg.cholesky(Sigma + 1e-9 * eye)  # (B, N, N)

    n = 2000
    noise = torch.randn(mu.shape[0], n, N, dtype=Sigma.dtype)
    samples = mu_f.unsqueeze(1) + noise @ chol.transpose(-1, -2)  # (B, n, N)
    es = sr.es_ensemble(
        obs=y_f,
        fct=samples.permute(0, 2, 1),  # (B, N, n)
        m_axis=-1,
        v_axis=-2,
        backend="torch",
    )
    return es.mean().item()


# Each case carries mu, s and y; the expected metric values come from the
# closed-form / dense references above, so the tests check the implementation
# against independent maths rather than against itself.
def _case(mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> dict:
    return dict(
        mu=mu,
        s=s,
        y=y,
        pred=_pred(mu, s),
        mse=torch.nn.functional.mse_loss(mu, y).item(),
        mae=torch.nn.functional.l1_loss(mu, y).item(),
        nll=_nll(mu, s, y),
        kernel_score=_kernel_score(mu, s, y),
        crps=_crps(mu, s, y),
        energy_score=_energy_score_reference(mu, s, y),
    )


CASES = {
    # mu = y = 0 with a flat unit spectrum (white noise, Σ = I).
    "white_zero": _case(_full(0.0), _full(1.0), _full(0.0)),
    # A constant mean shift over a flat spectrum.
    "white_shift": _case(_full(1.0), _full(1.0), _full(0.0)),
    # mu = y = 0 with a non-flat (coloured) conjugate-symmetric spectrum.
    "coloured_zero": _case(_full(0.0), _symmetric_s(1), _full(0.0)),
    # Random mean / target with a coloured spectrum: a genuinely non-trivial FFT.
    "coloured_random": _case(_random_field(2), _symmetric_s(3), _random_field(4)),
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
def test_nll(method, case):
    # The spectral NLL equals the physical NLL of N(mu, Fᴴ diag(s) F); the rfft
    # half-spectrum with conjugacy weights makes this exact.
    result = method.nll(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["nll"], dtype=torch.float64))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_kernel_score(method, case):
    # Checks the spectral kernel score against the dense closed form on Σ.
    result = method.kernel_score(case["pred"], case["y"])
    assert torch.isclose(
        result, torch.tensor(case["kernel_score"], dtype=torch.float64)
    )


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_crps(method, case):
    result = method.crps(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["crps"], dtype=torch.float64))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_energy_score(method, case):
    # The energy score is a sliced Monte-Carlo estimate, so it is noisy. Seed the
    # RNG for reproducibility and check closeness to the dense reference.
    torch.manual_seed(0)
    result = method.energy_score(case["pred"], case["y"])
    assert torch.isclose(
        result, torch.tensor(case["energy_score"], dtype=torch.float64),
        atol=0.05, rtol=0.01,
    )


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_parseval(method, case):
    """The residual is unitary-invariant: the conjugacy-weighted half-spectrum
    energy of (rfft(mu) - rfft(y)) equals ||mu - y||² in physical space.

    This is the property the spectral losses rely on, and it is the part of the
    physical link that is exact (not the covariance weighting alone)."""
    re_diff, im_diff, _, w = method._spectral_terms(case["pred"], case["y"])
    weighted_energy = (w * (re_diff.pow(2) + im_diff.pow(2))).sum(dim=1)
    phys_energy = (case["mu"] - case["y"]).flatten(start_dim=1).pow(2).sum(dim=1)
    assert torch.allclose(weighted_energy, phys_energy, atol=1e-10)


def test_energy_score_isotropic_closed_form(method):
    """Energy score has a closed form when y = mu and the spectrum is flat.

    A flat spectrum s_k = σ² gives an isotropic Σ = σ² I, so every slice has the
    same pushed-forward std σ, and y = mu makes the standardised residual zero.
    The per-slice CRPS collapses to the constant σ (√2 - 1) / √π, so the
    Monte-Carlo average over directions is exact. The energy score is that CRPS
    scaled by the slicing constant C_d with d = N, which pins down that the
    estimator runs over the physical dimension."""
    var = 1.7
    mu = _full(0.3)
    s = _full(var)
    y = mu.clone()
    pred = _pred(mu, s)

    sigma = math.sqrt(var)
    closed_form = sigma * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)

    # CRPS uses the constant marginal std and matches the closed form directly.
    crps = method.crps(pred, y)
    assert torch.isclose(crps, torch.tensor(closed_form, dtype=torch.float64))

    log_cd = (
        0.5 * math.log(math.pi)
        + math.lgamma((N + 1) / 2)
        - math.lgamma(N / 2)
    )
    expected_es = math.exp(log_cd) * closed_form

    torch.manual_seed(0)
    energy_score = method.energy_score(pred, y)
    assert torch.isclose(energy_score, torch.tensor(expected_es, dtype=torch.float64))


def _mirror(s: torch.Tensor) -> torch.Tensor:
    """s_{-k} over both spatial axes (the conjugate mirror k -> -k mod N)."""
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return s[:, :, idx_h][:, :, :, idx_w]


def test_process_output_symmetrizes_spectrum(method):
    """``process_output`` must emit a conjugate-symmetric spectral diagonal.

    The closed-form spectral losses (``_spectral_terms``) keep only the rfft half
    of ``s`` and weight it as if the dropped half mirrored it, while the sampler
    realises ``Re(Fᴴ diag(s) F)``. These coincide only when ``s_k == s_{-k}``; an
    unconstrained head emits an asymmetric ``s`` and the two paths then score
    different distributions. ``process_output`` enforces the symmetry, so an
    arbitrary head output must come back Hermitian-symmetric.
    """
    g = torch.Generator().manual_seed(7)
    head = torch.randn(
        BATCH_SIZE, OUT_CHANNELS * 2, H, W, generator=g, dtype=torch.float64
    )
    pred = method.process_output(head)
    s = pred[..., 1]
    assert torch.allclose(s, _mirror(s), atol=1e-12)
    # and the variance stays strictly positive after the averaging
    assert (s > 0).all()


def test_closed_form_matches_sampled_kernel_score(method):
    """Closed-form and ensemble kernel scores agree on a real ``process_output``.

    This is the end-to-end guard for the validation-vs-test discrepancy: feeding
    an arbitrary head output through ``process_output`` and scoring the *same*
    prediction with the analytic ``kernel_score`` and the O(M²) ensemble
    estimator must match to Monte-Carlo precision, because the symmetrised ``s``
    makes both score the one covariance ``Fᴴ diag(s) F``.
    """
    from src.evaluation.metrics import EVAL_METRICS

    g = torch.Generator().manual_seed(11)
    head = torch.randn(
        BATCH_SIZE, OUT_CHANNELS * 2, H, W, generator=g, dtype=torch.float64
    )
    y = _random_field(12)
    pred = method.process_output(head)

    method.training = False  # use the large eval sample count
    closed = method.kernel_score(pred, y)
    torch.manual_seed(0)
    sampled = EVAL_METRICS["kernel_score"](method, pred, y)
    # ensemble MC over N_EVAL_SAMPLES draws; loose tol covers the sampling error.
    assert torch.isclose(closed, sampled, atol=2e-3)
