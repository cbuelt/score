"""Tests for the multivariate calibration diagnostics.

The load-bearing test is :func:`test_calibrated_forecast_gives_uniform_histograms`:
observations drawn from the predictive itself must produce flat histograms for
all three diagnostics. That is the property the whole construction exists to
check, and it fails for any sign error, off-by-one in the rank, or mismatch
between the closed-form ``a'Sigma a`` and the actual predictive -- so it is
paired with a deliberately mis-specified forecast, to confirm the histograms are
not flat merely because the code is insensitive.
"""

import math

import numpy as np
import pytest
import scipy.stats as sps
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.calibration import (
    DEFAULT_BIN_EDGES_KM,
    default_lags,
    dependence_prerank,
    evaluate_calibration,
    isotropy_prerank,
    mean_prerank,
    randomized_rank,
    rank_to_pit,
    variance_prerank,
    variogram_bins,
    variogram_grid,
)
from src.methods import SampleBased, Diag, LorD, SpectralDiag

EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 50
GAMMA = 1.0
RANK = 2


class _StubDataModule:
    """The smallest thing `evaluate_calibration` accepts: a test split.

    ``x`` is the *processed head output* and the "model" is the identity, so the
    test controls the predictive exactly instead of going through a backbone.
    """

    def __init__(self, pred: torch.Tensor, y: torch.Tensor, batch_size: int = 64):
        self.test_dataset = TensorDataset(pred, y)
        self._batch_size = batch_size

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self._batch_size, shuffle=False
        )


def _identity_model(x):
    return x


class _StubStationDataModule(_StubDataModule):
    """A `_StubDataModule` that also advertises a distance matrix.

    `evaluate_calibration` uses `get_distances` as the discriminator between the
    grid-lag and distance-bin variogram, exactly as `EUPPBenchModule` provides it.
    """

    def __init__(self, pred, y, dist, batch_size: int = 64):
        super().__init__(pred, y, batch_size)
        self._dist = dist

    def get_distances(self):
        return self._dist


def _lora_setup(n_cases: int, C: int, V: int, seed: int = 0):
    """A LorD predictive with a genuinely non-diagonal Sigma."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n_cases, C, V, generator=g, dtype=torch.float64)
    var = torch.rand(n_cases, C, V, generator=g, dtype=torch.float64) + 0.5
    lora = 0.6 * torch.randn(n_cases, C, V, RANK, generator=g, dtype=torch.float64)
    pred = torch.cat([mu.unsqueeze(-1), var.unsqueeze(-1), lora], dim=-1)
    method = LorD(
        "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA, rank=RANK
    )
    method.training = False
    return method, pred


def _draw_truth(method, pred, seed: int, inflate: float = 1.0) -> torch.Tensor:
    """One draw from the predictive per case, optionally over-dispersed.

    ``inflate > 1`` scales the residual, so the *truth* is more variable than the
    forecast claims -- i.e. the forecast is under-dispersed.
    """
    torch.manual_seed(seed)
    sample = method.get_samples(pred, n_samples=1)[..., 0]
    mu = method.get_mean(pred)
    return mu + inflate * (sample - mu)


# ---------------------------------------------------------------------------
# The pre-rank functions in isolation
# ---------------------------------------------------------------------------


def test_preranks_match_a_naive_masked_reference():
    """``rho_loc`` and ``rho_sc`` are the plain mean and 1/d variance over the
    observed coordinates."""
    torch.manual_seed(0)
    B, G, V = 3, 2, 7
    v = torch.randn(B, G, V, dtype=torch.float64)
    mask = torch.ones(B, G, V, dtype=torch.float64)
    mask[0, 0, [1, 4]] = 0.0
    mask[2, 1, [6]] = 0.0
    n_obs = mask.sum(-1)

    got_mean = mean_prerank(v * mask, n_obs, n_spatial=1)
    # `variance_prerank` requires a pre-shifted input; shift by the masked mean.
    centred = (v - got_mean.unsqueeze(-1)) * mask
    got_var = variance_prerank(centred, n_obs, n_spatial=1)

    for b in range(B):
        for g in range(G):
            obs = v[b, g][mask[b, g] > 0].numpy()
            assert got_mean[b, g].item() == pytest.approx(obs.mean(), abs=1e-12)
            assert got_var[b, g].item() == pytest.approx(obs.var(ddof=0), abs=1e-12)


def test_variance_prerank_is_shift_invariant():
    """rho_sc(x + c) = rho_sc(x): what licenses the numerical re-centring."""
    torch.manual_seed(1)
    v = torch.randn(4, 1, 9, dtype=torch.float64)
    n_obs = torch.full((4, 1), 9.0, dtype=torch.float64)
    base = variance_prerank(v - v.mean(-1, keepdim=True), n_obs, n_spatial=1)
    shifted = v + 1000.0
    got = variance_prerank(shifted - shifted.mean(-1, keepdim=True), n_obs, n_spatial=1)
    assert torch.allclose(base, got, atol=1e-12)


def test_randomized_rank_counts_and_breaks_ties():
    """Rank is ``1 + #{members < obs}`` when nothing ties, and stays in 1..M+1."""
    gen = torch.Generator().manual_seed(0)
    obs = torch.tensor([[0.5], [2.5], [-1.0]], dtype=torch.float64)
    members = torch.tensor(
        [[[0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0]]], dtype=torch.float64
    )
    rank = randomized_rank(obs, members, gen)
    assert rank.flatten().tolist() == [2, 4, 1]

    # An observation equal to a member is a genuine tie: the rank must land on
    # either side of it at random, never deterministically below.
    tied_obs = torch.full((4000, 1), 2.0, dtype=torch.float64)
    tied = randomized_rank(tied_obs, members[:1].expand(4000, -1, -1), gen)
    assert set(np.unique(tied.numpy())) == {3, 4}

    # All-tied inputs must spread uniformly over 1..M+1 rather than pin at 1.
    obs = torch.zeros(4000, 1, dtype=torch.float64)
    members = torch.zeros(4000, 1, 3, dtype=torch.float64)
    rank = randomized_rank(obs, members, gen).flatten().numpy()
    assert set(np.unique(rank)) == {1, 2, 3, 4}
    counts = np.bincount(rank, minlength=5)[1:5]
    assert sps.chisquare(counts).pvalue > 0.01


# ---------------------------------------------------------------------------
# Variogram machinery
# ---------------------------------------------------------------------------


def _naive_grid_variogram(v, mask, lag):
    """Brute-force gamma over every in-range, fully-observed pair of one 2-D case."""
    H, W = v.shape
    dy, dx = lag
    num, den = 0.0, 0
    for i in range(H):
        for j in range(W):
            i2, j2 = i + dy, j + dx
            if not (0 <= i2 < H and 0 <= j2 < W):
                continue
            if mask[i, j] == 0 or mask[i2, j2] == 0:
                continue
            num += float(v[i2, j2] - v[i, j]) ** 2
            den += 1
    return num / (2 * den), den


@pytest.mark.parametrize("lag", [(1, 0), (0, 1), (2, 0), (1, 1), (1, -1), (2, -2)])
def test_grid_variogram_matches_a_brute_force_pair_loop(lag):
    """The strided form must equal the definition, including a negative lag."""
    torch.manual_seed(0)
    B, G, H, W = 2, 2, 5, 6
    mask = torch.ones(B, G, H, W, dtype=torch.float64)
    mask[0, 0, 1, 2] = 0.0
    mask[1, 1, 3, 4] = 0.0
    v = torch.randn(B, G, H, W, dtype=torch.float64) * mask

    gamma, count = variogram_grid(v, mask, [lag], n_spatial=2)

    for b in range(B):
        for g in range(G):
            ref, den = _naive_grid_variogram(v[b, g], mask[b, g], lag)
            assert gamma[b, g, 0].item() == pytest.approx(ref, abs=1e-12)
            assert count[b, g, 0].item() == den


def test_variogram_is_symmetric_under_swapping_the_pair_ends():
    """gamma(h,-h) == gamma(-h,h) -- which is why rho_iso may use either."""
    torch.manual_seed(1)
    v = torch.randn(2, 1, 5, 5, dtype=torch.float64)
    mask = torch.ones(2, 1, 5, 5, dtype=torch.float64)
    g, _ = variogram_grid(v, mask, [(1, -1), (-1, 1), (2, -2), (-2, 2)], n_spatial=2)
    assert torch.allclose(g[:, :, 0], g[:, :, 1], atol=1e-12)
    assert torch.allclose(g[:, :, 2], g[:, :, 3], atol=1e-12)


def test_grid_variogram_carries_a_member_axis_unchanged():
    """The member axis must broadcast, so obs and draws share one code path."""
    torch.manual_seed(2)
    v = torch.randn(2, 1, 6, 7, 5, dtype=torch.float64)
    mask = torch.ones(2, 1, 6, 7, dtype=torch.float64)
    mask[0, 0, 2, 3] = 0.0
    v = v * mask.unsqueeze(-1)

    batched, _ = variogram_grid(v, mask, [(1, 0), (0, 2)], n_spatial=2, n_member_axes=1)
    assert batched.shape == (2, 1, 2, 5)
    for m in range(5):
        one, _ = variogram_grid(v[..., m], mask, [(1, 0), (0, 2)], n_spatial=2)
        assert torch.allclose(batched[..., m], one, atol=1e-12)


def test_lag_longer_than_the_axis_is_marked_undefined():
    v = torch.randn(1, 1, 4, dtype=torch.float64)
    mask = torch.ones(1, 1, 4, dtype=torch.float64)
    gamma, count = variogram_grid(v, mask, [(1,), (9,)], n_spatial=1)
    assert torch.isfinite(gamma[0, 0, 0])
    assert torch.isnan(gamma[0, 0, 1]) and count[0, 0, 1] == 0


def test_distance_bin_variogram_matches_a_brute_force_pair_loop():
    torch.manual_seed(3)
    B, N = 2, 9
    mask = torch.ones(B, 1, N, dtype=torch.float64)
    mask[0, 0, 3] = 0.0
    v = torch.randn(B, 1, N, dtype=torch.float64) * mask
    dist = torch.rand(N, N, dtype=torch.float64) * 300
    dist = (dist + dist.T) / 2
    dist.fill_diagonal_(0)

    edges = list(DEFAULT_BIN_EDGES_KM)
    iu = torch.triu_indices(N, N, offset=1)
    bin_of = torch.bucketize(
        dist[iu[0], iu[1]], torch.tensor(edges[1:-1], dtype=torch.float64)
    )
    gamma, count = variogram_bins(v, mask, iu[0], iu[1], bin_of, len(edges) - 1)

    for b in range(B):
        for bi in range(len(edges) - 1):
            num, den = 0.0, 0
            for i in range(N):
                for j in range(i + 1, N):
                    if not edges[bi] <= float(dist[i, j]) < edges[bi + 1]:
                        continue
                    if mask[b, 0, i] == 0 or mask[b, 0, j] == 0:
                        continue
                    num += float(v[b, 0, j] - v[b, 0, i]) ** 2
                    den += 1
            expected = num / (2 * den) if den else 0.0
            assert gamma[b, 0, bi].item() == pytest.approx(expected, abs=1e-12)
            assert count[b, 0, bi].item() == den


def test_pooling_all_pairs_makes_the_dependence_prerank_constant():
    """Why the distance bins exist at all.

    ``sum_{i<j}(x_i-x_j)^2 = d^2 s^2_x``, so an unbinned variogram divided by
    ``s^2_x`` is the constant ``-d/(d-1)`` for every case and every member. Its
    rank would be pure tie-breaking noise and the histogram flat by construction
    -- a diagnostic that cannot fail. This test pins that we never ship it.
    """
    torch.manual_seed(4)
    B, N = 6, 11
    v = torch.randn(B, 1, N, dtype=torch.float64)
    mask = torch.ones(B, 1, N, dtype=torch.float64)
    iu = torch.triu_indices(N, N, offset=1)
    single_bin = torch.zeros(iu.shape[1], dtype=torch.long)

    gamma, _ = variogram_bins(v, mask, iu[0], iu[1], single_bin, 1)
    rho = dependence_prerank(gamma.unsqueeze(2), v.var(dim=-1, unbiased=False))

    assert torch.allclose(rho, torch.full_like(rho, -N / (N - 1)), atol=1e-12)


def test_isotropy_prerank_is_zero_only_for_a_direction_independent_variogram():
    gamma = torch.ones(1, 1, 4, 3, dtype=torch.float64)
    assert torch.allclose(isotropy_prerank(gamma), torch.zeros(1, 1, 3, dtype=torch.float64))

    gamma[0, 0, 1] = 2.0  # axis1 differs from axis0
    iso = isotropy_prerank(gamma)
    assert bool((iso < 0).all())
    # The paper's normalisation is a ratio, so a common rescaling is invisible.
    assert torch.allclose(isotropy_prerank(gamma * 7.0), iso, atol=1e-12)


def test_default_lags_are_dyadic_and_capped():
    assert default_lags(96) == [1, 2, 4, 8, 16]       # cap 24
    assert default_lags(480) == [1, 2, 4, 8, 16, 32, 64]  # cap 120
    assert default_lags(160) == [1, 2, 4, 8, 16, 32]  # cap 40
    assert default_lags(4) == [1]                     # never empty


def test_rank_to_pit_is_midrank_and_flags_undefined():
    ranks = np.array([1, 500, 1000, 0])
    u = rank_to_pit(ranks, n_draws=999)
    assert u[0] == pytest.approx(0.5 / 1000)
    assert u[1] == pytest.approx(499.5 / 1000)
    assert u[2] == pytest.approx(999.5 / 1000)
    assert np.isnan(u[3])


# ---------------------------------------------------------------------------
# End-to-end: uniformity under a calibrated forecast
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibrated_payload():
    n_cases, C, V = 3000, 1, 12
    method, pred = _lora_setup(n_cases, C, V, seed=0)
    y = _draw_truth(method, pred, seed=11)
    dm = _StubDataModule(pred, y, batch_size=128)
    return evaluate_calibration(
        _identity_model, method, dm, n_draws=199, draw_chunk=64, sample_seed=5
    )


def test_calibrated_forecast_gives_uniform_histograms(calibrated_payload):
    """All three diagnostics are flat when the truth comes from the predictive."""
    payload = calibrated_payload

    marg = payload["marg_pit"].ravel()
    assert sps.kstest(marg, "uniform").pvalue > 0.01

    mean_pit = payload["mean_pit"].ravel()
    assert sps.kstest(mean_pit, "uniform").pvalue > 0.01

    # Ranks are discrete on 1..M+1, so a chi-square on coarsened bins, not a KS.
    ranks = payload["var_rank"].ravel()
    assert ranks.min() >= 1 and ranks.max() <= 200
    counts = np.histogram(ranks, bins=20, range=(0.5, 200.5))[0]
    assert sps.chisquare(counts).pvalue > 0.01


def test_the_exact_histogram_counts_every_observed_coordinate(calibrated_payload):
    """`marg_hist_counts` is accumulated over all coordinates, not the stored ones."""
    payload = calibrated_payload
    assert payload["marg_hist_counts"].sum() == payload["marg_pit"].size
    assert payload["marg_pit_mode"] == "full"
    # And it agrees with re-binning the stored array.
    direct = np.histogram(payload["marg_pit"].ravel(), bins=1000, range=(0, 1))[0]
    assert np.array_equal(payload["marg_hist_counts"], direct)


def test_var_hist_counts_agree_with_the_saved_ranks(calibrated_payload):
    payload = calibrated_payload
    rebuilt = np.bincount(payload["var_rank"][:, 0] - 1, minlength=200)
    assert np.array_equal(payload["var_hist_counts"][:, 0], rebuilt)


def test_dependence_prerank_is_uniform_at_every_lag(calibrated_payload):
    """The dependence histogram must be flat at each lag under a calibrated
    forecast -- the property that makes a non-flat one interpretable."""
    payload = calibrated_payload
    ranks = payload["dep_rank"]
    assert ranks.shape == (3000, 1, 1, payload["dep_lags"].shape[0])
    assert str(payload["dep_kind"]) == "grid_lag"

    for li in range(ranks.shape[3]):
        r = ranks[:, 0, 0, li]
        assert r.min() >= 1 and r.max() <= 200
        counts = np.histogram(r, bins=20, range=(0.5, 200.5))[0]
        assert sps.chisquare(counts).pvalue > 0.01, f"lag index {li}"


def test_dependence_histogram_agrees_with_the_saved_ranks(calibrated_payload):
    payload = calibrated_payload
    for li in range(payload["dep_rank"].shape[3]):
        rebuilt = np.bincount(payload["dep_rank"][:, 0, 0, li] - 1, minlength=200)
        assert np.array_equal(payload["dep_hist_counts"][:, 0, 0, li], rebuilt)


def _circulant_spectrum(V: int, width: float) -> torch.Tensor:
    """Conjugate-symmetric spectrum with unit mean; smaller `width` = smoother field.

    A circulant Sigma = F^H diag(s) F has marginal variance mean(s) at every
    coordinate (Parseval), so two spectra normalised to the same mean give
    *identical marginals* and differ only in correlation length.
    """
    k = torch.arange(V, dtype=torch.float64)
    freq = torch.minimum(k, V - k)
    s = torch.exp(-((freq / width) ** 2)) + 1e-6
    return s / s.mean()


def test_dependence_detects_a_misspecified_correlation_length():
    """The case the diagnostic exists for: right marginals, wrong dependence.

    Forecast and truth are both circulant with unit marginal variance, so the
    marginal PIT cannot tell them apart by construction. They differ only in the
    width of the spectrum, i.e. in correlation length -- which is exactly what
    rho_dep is supposed to see.
    """
    n_cases, C, V = 2000, 1, 32
    torch.manual_seed(31)
    mu = torch.randn(n_cases, C, V, dtype=torch.float64)
    fc = _circulant_spectrum(V, width=2.0).expand(n_cases, C, V)   # smooth
    tr = _circulant_spectrum(V, width=8.0).expand(n_cases, C, V)   # rough

    method = SpectralDiag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    method.training = False
    pred = torch.stack([mu, fc], dim=-1)
    truth_pred = torch.stack([mu, tr], dim=-1)

    # Identical marginal sd, so the marginal PIT is blind to the difference.
    assert torch.allclose(
        method.get_marg_std(pred), method.get_marg_std(truth_pred), atol=1e-12
    )
    y = _draw_truth(method, truth_pred, seed=32)

    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 128),
        n_draws=199, sample_seed=7,
    )

    marg = payload["marg_pit"].ravel()
    assert sps.kstest(marg, "uniform").pvalue > 0.01, "marginals should look fine"

    ranks = payload["dep_rank"][:, 0, 0, :]
    lags = payload["dep_lags"].tolist()
    assert lags == [1, 2, 4, 8]

    # Every lag rejects, and the *direction* reverses with scale -- which is the
    # scale-resolved information a single flatness scalar would destroy.
    for li, h in enumerate(lags):
        counts = np.histogram(ranks[:, li], bins=20, range=(0.5, 200.5))[0]
        assert sps.chisquare(counts).pvalue < 1e-20, f"lag {h}"

    # Short range: the truth is rougher, so its variogram is larger relative to
    # its own spread and rho_dep = -gamma/s^2 sits below the members.
    assert ranks[:, 0].mean() < 80, "lag 1 should be left-shifted"
    # Long range: the smooth forecast still has structure at h = 8 while the
    # rough truth has decorrelated, so the ordering flips.
    assert ranks[:, -1].mean() > 110, "lag 8 should be right-shifted"


def test_underdispersed_forecast_is_detected():
    """A forecast whose spread is too small must *not* give flat histograms.

    Guards against the histograms being flat for the trivial reason that the
    diagnostics are insensitive to anything.
    """
    n_cases, C, V = 3000, 1, 12
    method, pred = _lora_setup(n_cases, C, V, seed=0)
    y = _draw_truth(method, pred, seed=11, inflate=1.5)
    dm = _StubDataModule(pred, y, batch_size=128)
    payload = evaluate_calibration(
        _identity_model, method, dm, n_draws=199, draw_chunk=64, sample_seed=5
    )

    # Marginal and location PIT: U-shaped, i.e. excess mass in both tails.
    for key in ("marg_pit", "mean_pit"):
        u = payload[key].ravel()
        tails = np.mean((u < 0.05) | (u > 0.95))
        assert tails > 0.15, f"{key} tail mass {tails:.3f} is not U-shaped"

    # Scale pre-rank: the observed field is more spread out than any member, so
    # the rank piles up at the top rather than at both ends.
    ranks = payload["var_rank"].ravel()
    assert np.mean(ranks > 180) > 0.3


# ---------------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------------


def test_mean_prerank_matches_the_subselected_gaussian_under_masking():
    """With coordinates missing, the closed-form PIT equals the one computed
    directly from the sub-block of a dense Sigma -- exact, not approximate."""
    n_cases, C, V = 6, 1, 10
    method, pred = _lora_setup(n_cases, C, V, seed=3)
    y = _draw_truth(method, pred, seed=7)
    y[0, 0, [2, 5, 9]] = float("nan")
    y[3, 0, [0]] = float("nan")

    dm = _StubDataModule(pred, y, batch_size=4)
    payload = evaluate_calibration(
        _identity_model, method, dm, n_draws=9, draw_chunk=9, sample_seed=1
    )

    # Dense reference Sigma = diag(d) + U U'.
    diag, lora = pred[..., 1], pred[..., 2:]
    U = lora.reshape(n_cases, -1, RANK)
    sigma = torch.diag_embed(diag.flatten(start_dim=1)) + U @ U.transpose(-1, -2)
    mu = method.get_mean(pred).flatten(start_dim=1)

    for b in range(n_cases):
        obs = torch.isfinite(y[b, 0])
        a = (obs.double() / obs.sum()).numpy()
        var = float(a @ sigma[b].numpy() @ a)
        rho_y = float(a @ torch.nan_to_num(y[b, 0], nan=0.0).numpy())
        rho_mu = float(a @ mu[b].numpy())
        expected = sps.norm.cdf((rho_y - rho_mu) / math.sqrt(var))
        assert payload["mean_pit"][b, 0] == pytest.approx(expected, abs=1e-6)

    assert np.array_equal(payload["n_obs"][:, 0], [7, 10, 10, 9, 10, 10])
    assert payload["mask"].shape == (n_cases, V)
    assert not payload["mask"][0, 2]


def test_mask_is_backfilled_when_the_first_gap_comes_late():
    """The mask is only materialized once a gap appears -- it is 200 MB of
    all-True at depth resolution -- so cases seen before the first gap must be
    backfilled rather than dropped."""
    n_cases, C, V = 12, 1, 6
    method, pred = _lora_setup(n_cases, C, V, seed=1)
    y = _draw_truth(method, pred, seed=5)
    y[9, 0, 2] = float("nan")  # last batch only, at batch_size 4

    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 4), n_draws=9, sample_seed=0
    )

    assert payload["mask"].shape == (n_cases, V)
    assert payload["mask"][:8].all()  # backfilled, not dropped
    assert not payload["mask"][9, 2]
    assert payload["mask"].sum() == n_cases * V - 1
    assert payload["marg_hist_counts"].sum() == n_cases * V - 1


def test_mask_is_omitted_when_nothing_is_missing():
    """The mask costs `n_cases * V` bytes; complete data should not pay it."""
    method, pred = _lora_setup(20, 1, 8, seed=4)
    y = _draw_truth(method, pred, seed=2)
    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 8), n_draws=9, sample_seed=0
    )
    assert "mask" not in payload


# ---------------------------------------------------------------------------
# Layout, storage policy and the npz round trip
# ---------------------------------------------------------------------------


def test_channels_are_independent_pre_rank_problems():
    """A multi-channel run yields one diagnostic per channel, d = spatial size."""
    n_cases, C, V = 40, 3, 6
    method, pred = _lora_setup(n_cases, C, V, seed=8)
    y = _draw_truth(method, pred, seed=9)
    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 16), n_draws=19, sample_seed=0
    )

    assert payload["mean_pit"].shape == (n_cases, C)
    assert payload["var_rank"].shape == (n_cases, C)
    assert payload["var_hist_counts"].shape == (20, C)
    assert np.all(payload["n_obs"] == V)  # d is the spatial size, not C * V


def test_large_marginal_pit_falls_back_to_a_seeded_subsample(tmp_path):
    """Above the cap the raw array is subsampled while the histogram stays exact."""
    n_cases, C, V = 64, 1, 300
    method, pred = _lora_setup(n_cases, C, V, seed=6)
    y = _draw_truth(method, pred, seed=1)
    dm = _StubDataModule(pred, y, 16)

    payload = evaluate_calibration(
        _identity_model, method, dm, n_draws=9, raw_cap_mb=1e-4, sample_seed=3
    )
    assert payload["marg_pit_mode"] == "subsampled"
    assert payload["marg_pit"].shape == (n_cases, 300 if V < 512 else 512)
    assert payload["marg_pit_idx"].shape[0] == payload["marg_pit"].shape[1]
    # Exact over every coordinate, not just the retained ones.
    assert payload["marg_hist_counts"].sum() == n_cases * V

    path = tmp_path / "calibration.npz"
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as f:
        assert f["marg_pit"].dtype == np.float32
        assert f["var_rank"].dtype == np.int32
        assert str(f["marg_pit_mode"]) == "subsampled"


def test_two_dimensional_grid_gets_four_directions_plus_isotropy():
    """Depth-shaped input: 4 lag directions + the axial mean, and rho_iso."""
    n_cases, C, H, W = 60, 1, 12, 16
    g = torch.Generator().manual_seed(21)
    mu = torch.randn(n_cases, C, H, W, generator=g, dtype=torch.float64)
    var = torch.rand(n_cases, C, H, W, generator=g, dtype=torch.float64) + 0.5
    pred = torch.stack([mu, var], dim=-1)
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    method.training = False
    y = _draw_truth(method, pred, seed=22)

    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 16), n_draws=49, sample_seed=0
    )

    assert list(payload["dep_directions"]) == ["axis0", "axis1", "diag", "antidiag", "axial_mean"]
    n_lag = payload["dep_lags"].shape[0]
    assert payload["dep_lags"].tolist() == [1, 2]  # min(12,16)/4 = 3
    assert payload["dep_rank"].shape == (n_cases, C, 5, n_lag)
    assert payload["dep_lag_vectors"].shape == (4, n_lag, 2)

    # The axial mean really is the mean of the two axis variograms.
    gam = payload["dep_variogram_obs"]
    assert np.allclose(gam[:, :, 4], 0.5 * (gam[:, :, 0] + gam[:, :, 1]))

    assert payload["iso_rank"].shape == (n_cases, C, n_lag)
    assert payload["iso_hist_counts"].shape == (50, C, n_lag)
    assert list(payload["objectives"]) == [
        "marginal", "mean", "variance", "dependence", "isotropy"
    ]


def test_isotropy_is_absent_for_one_dimensional_events():
    """A station network or a lead-time axis has no directions to contrast."""
    method, pred = _lora_setup(30, 1, 8, seed=23)
    y = _draw_truth(method, pred, seed=24)
    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 10), n_draws=19, sample_seed=0
    )
    assert "iso_rank" not in payload
    assert list(payload["dep_directions"]) == ["axis0"]
    assert list(payload["objectives"]) == ["marginal", "mean", "variance", "dependence"]


def test_isotropy_can_be_switched_off():
    n_cases, C, H, W = 20, 1, 12, 12
    g = torch.Generator().manual_seed(25)
    mu = torch.randn(n_cases, C, H, W, generator=g, dtype=torch.float64)
    var = torch.rand(n_cases, C, H, W, generator=g, dtype=torch.float64) + 0.5
    pred = torch.stack([mu, var], dim=-1)
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    method.training = False
    y = _draw_truth(method, pred, seed=26)
    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 8),
        n_draws=19, sample_seed=0, isotropy=False,
    )
    assert "iso_rank" not in payload
    assert "dep_rank" in payload


def test_station_geometry_switches_to_distance_bins():
    """A datamodule exposing `get_distances` uses geodesic bins, not lags."""
    n_cases, C, N = 40, 1, 12
    method, pred = _lora_setup(n_cases, C, N, seed=27)
    y = _draw_truth(method, pred, seed=28)
    torch.manual_seed(0)
    dist = torch.rand(N, N, dtype=torch.float64) * 300
    dist = (dist + dist.T) / 2
    dist.fill_diagonal_(0)

    payload = evaluate_calibration(
        _identity_model, method, _StubStationDataModule(pred, y, dist, 10),
        n_draws=19, sample_seed=0,
    )

    assert str(payload["dep_kind"]) == "distance_bin"
    assert list(payload["dep_directions"]) == ["distance_bin"]
    assert np.array_equal(payload["dep_bin_edges_km"], np.array(DEFAULT_BIN_EDGES_KM))
    assert payload["dep_rank"].shape == (n_cases, C, 1, len(DEFAULT_BIN_EDGES_KM) - 1)
    assert "dep_lags" not in payload
    assert "iso_rank" not in payload


def test_explicit_lags_override_the_default_ladder():
    method, pred = _lora_setup(20, 1, 96, seed=29)
    y = _draw_truth(method, pred, seed=30)
    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 10),
        n_draws=9, sample_seed=0, lags=[1, 12, 24],
    )
    assert payload["dep_lags"].tolist() == [1, 12, 24]
    assert payload["dep_rank"].shape[3] == 3


def test_needs_refresh_makes_a_plain_rerun_a_safe_resume(tmp_path):
    """Skip only *current* files, so an interrupted sweep resumes correctly.

    Long sweeps can be interrupted, so re-running is the normal path. Keying the
    skip on "file exists" would leave older-schema files in place forever;
    keying it on `schema_version` refreshes exactly the outdated ones.
    """
    from src.evaluation.calibration import SCHEMA_VERSION, needs_refresh

    missing = tmp_path / "absent.npz"
    assert needs_refresh(missing)[0] is True

    current = tmp_path / "current.npz"
    np.savez_compressed(current, schema_version=np.int32(SCHEMA_VERSION))
    assert needs_refresh(current)[0] is False
    assert needs_refresh(current, overwrite=True)[0] is True

    stale = tmp_path / "stale.npz"
    np.savez_compressed(stale, schema_version=np.int32(SCHEMA_VERSION - 1))
    refresh, reason = needs_refresh(stale)
    assert refresh is True and "schema" in reason

    # A sweep killed mid-write leaves a truncated file; it must be redone, not
    # skipped and not allowed to abort the sweep.
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(b"PK\x03\x04 not really a npz")
    assert needs_refresh(truncated)[0] is True


def test_calibration_out_path_collects_results_by_experiment(tmp_path):
    """Diagnostics land in ``<experiment>/calibration/``, not next to the checkpoint.

    The curated folder is a different *set* from the checkpoint tree on purpose:
    a rank sweep trains five ranks and three seeds for the score tables, and one
    of them gets a calibration figure. Two properties matter and are pinned here
    -- the dataset/mode prefix is kept (``whitened_dct_nll`` alone is ambiguous
    across four time-series combinations), and the seed moves from a directory
    into the filename.
    """
    from src.evaluation.calibration import calibration_out_path

    root = tmp_path / "shared"
    (root).mkdir()
    (root / "INDEX.md").write_text("shared tree marker\n")

    def seed_dir(rel):
        d = root / rel
        d.mkdir(parents=True)
        return d

    depth = seed_dir("depth/nyu_bts/whitened_dct_nll/seed42")
    assert calibration_out_path(depth) == (
        root / "depth/calibration/nyu_bts/whitened_dct_nll/calibration_seed42.npz"
    )

    # A `<mode>` level exists only for time series; it must survive, or etth1's
    # multivariate and univariate arms of the same slug would collide.
    ts_m = seed_dir("time_series/etth1/multivariate/whitened_dct_nll/seed42")
    ts_u = seed_dir("time_series/etth1/univariate/whitened_dct_nll/seed42")
    assert calibration_out_path(ts_m) != calibration_out_path(ts_u)
    assert calibration_out_path(ts_m) == (
        root / "time_series/calibration/etth1/multivariate"
             / "whitened_dct_nll/calibration_seed42.npz"
    )

    # Seeds of one arm are siblings, distinguished by filename alone.
    other_seed = seed_dir("depth/nyu_bts/whitened_dct_nll/seed43")
    assert calibration_out_path(other_seed).parent == calibration_out_path(depth).parent
    assert calibration_out_path(other_seed).name == "calibration_seed43.npz"

    # `out_root` replaces the `<experiment>/calibration` prefix and keeps the
    # tail -- the escape hatch for a seed folder whose experiment is read-only.
    elsewhere = calibration_out_path(depth, out_root=tmp_path / "scratch")
    assert elsewhere == (
        tmp_path / "scratch/nyu_bts/whitened_dct_nll/calibration_seed42.npz"
    )

    # Outside the results tree there is no experiment to collect into, so the
    # file is written in place, next to the run.
    raw = tmp_path / "hydra_run"
    raw.mkdir()
    assert calibration_out_path(raw) == raw / "calibration.npz"


def test_members_are_stored_only_on_request():
    method, pred = _lora_setup(12, 1, 5, seed=2)
    y = _draw_truth(method, pred, seed=3)
    dm = _StubDataModule(pred, y, 4)

    default = evaluate_calibration(
        _identity_model, method, dm, n_draws=7, sample_seed=0
    )
    assert "var_prerank_members" not in default

    with_members = evaluate_calibration(
        _identity_model, method, dm, n_draws=7, sample_seed=0, save_members=True
    )
    assert with_members["var_prerank_members"].shape == (12, 1, 7)
    # The saved rank is the rank of the saved observation among the saved members.
    obs = with_members["var_prerank_obs"]
    members = with_members["var_prerank_members"]
    assert np.array_equal(
        with_members["var_rank"], 1 + (members < obs[..., None]).sum(-1)
    )


def test_chunking_does_not_change_the_diagnostics():
    """`draw_chunk` is a memory knob; with the same seed the draws must match."""
    method, pred = _lora_setup(24, 1, 6, seed=5)
    y = _draw_truth(method, pred, seed=4)
    dm = _StubDataModule(pred, y, 8)

    one = evaluate_calibration(
        _identity_model, method, dm, n_draws=32, draw_chunk=32, sample_seed=2
    )
    assert np.array_equal(
        one["var_rank"],
        1
        + (
            evaluate_calibration(
                _identity_model, method, dm, n_draws=32, draw_chunk=32, sample_seed=2,
                save_members=True,
            )["var_prerank_members"]
            < one["var_prerank_obs"][..., None]
        ).sum(-1),
    )


def test_diagonal_family_also_runs():
    """Every family must go through the same path, not just the low-rank one."""
    n_cases, C, V = 500, 1, 10
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(n_cases, C, V, generator=g, dtype=torch.float64)
    var = torch.rand(n_cases, C, V, generator=g, dtype=torch.float64) + 0.5
    pred = torch.stack([mu, var], dim=-1)
    method = Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, C, GAMMA)
    method.training = False
    y = _draw_truth(method, pred, seed=1)

    payload = evaluate_calibration(
        _identity_model, method, _StubDataModule(pred, y, 64), n_draws=99, sample_seed=0
    )
    assert sps.kstest(payload["mean_pit"].ravel(), "uniform").pvalue > 0.01
    # For a diagonal Sigma the mean pre-rank variance is sum(var)/d^2 exactly.
    expected = var.sum(-1).squeeze(-1).numpy() / V**2
    assert np.allclose(payload["mean_prerank_var"][:, 0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# The sampled branch: a method with no closed-form PIT
# ---------------------------------------------------------------------------


class _GenerativeStubModel:
    """A stand-in for a backbone with a noise injector.

    ``evaluate_calibration`` treats ``model`` as a callable; a generative (SB)
    one additionally has to honour ``n_samples``, because for such a method each
    member *is* a forward pass. The stub draws from a per-case Gaussian, so the
    predictive is known exactly and the truth can be drawn from it -- the same
    trick the closed-form fixtures use, with the sampling moved inside the
    model where the real one has it.
    """

    def __init__(self, method, n_eval: int):
        self.method = method
        self.n_eval = n_eval
        self.calls = []

    def __call__(self, x, n_samples=None):
        # `x` is (B, C, V, 2) carrying (mu, sd) per coordinate.
        n_samples = n_samples or self.n_eval
        self.calls.append(n_samples)
        mu, sd = x[..., 0], x[..., 1]
        z = torch.randn(*mu.shape, n_samples, dtype=mu.dtype)
        return mu.unsqueeze(-1) + sd.unsqueeze(-1) * z


@pytest.fixture(scope="module")
def generative_payload():
    n_cases, C, V = 2000, 1, 8
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(n_cases, C, V, generator=g, dtype=torch.float64)
    sd = (torch.rand(n_cases, C, V, generator=g, dtype=torch.float64) + 0.5).sqrt()
    params = torch.stack([mu, sd], dim=-1)
    torch.manual_seed(3)
    y = mu + sd * torch.randn(n_cases, C, V, dtype=torch.float64)

    method = SampleBased(
        loss="energy_score", eps=EPS, n_train_samples=N_TRAIN_SAMPLES,
        n_eval_samples=N_EVAL_SAMPLES, out_channels=C, gamma=GAMMA,
    )
    method.training = False
    model = _GenerativeStubModel(method, N_EVAL_SAMPLES)
    payload = evaluate_calibration(
        model, method, _StubDataModule(params, y, 128),
        n_draws=99, draw_chunk=33, sample_seed=5,
    )
    return payload, model, params, sd


def test_generative_calibrated_forecast_is_uniform_on_every_objective(
    generative_payload,
):
    """The whole protocol still holds when the two location PITs are sampled."""
    payload, _, _, _ = generative_payload
    # Sampled PITs are discrete on the 1/(M+1) grid, so a chi-square on the
    # rank lattice rather than a KS test against a continuous uniform.
    for key in ("marg_pit", "mean_pit"):
        values = payload[key].ravel()
        values = values[np.isfinite(values)]
        assert values.min() >= 0.0 and values.max() <= 1.0
        counts = np.histogram(values, bins=10, range=(0.0, 1.0))[0]
        assert sps.chisquare(counts).pvalue > 0.01, key

    ranks = payload["var_rank"].ravel()
    counts = np.histogram(ranks, bins=10, range=(0.5, 100.5))[0]
    assert sps.chisquare(counts).pvalue > 0.01

    dep = payload["dep_rank"].reshape(-1, payload["dep_rank"].shape[-1])
    for li in range(dep.shape[-1]):
        counts = np.histogram(dep[:, li], bins=10, range=(0.5, 100.5))[0]
        assert sps.chisquare(counts).pvalue > 0.01, li


def test_generative_payload_records_the_sampled_estimator(generative_payload):
    """The npz says how its PITs were made; the key set is otherwise the same as
    for a closed-form method."""
    payload, _, _, _ = generative_payload
    assert str(payload["pit_estimator"]) == "rank"
    # Same objectives, same kinds: only the estimator behind them differs.
    assert list(payload["objectives"]) == ["marginal", "mean", "variance", "dependence"]
    assert list(payload["objective_kinds"]) == ["pit", "pit", "rank", "rank"]
    # Mid-rank PITs live on the (r - 0.5)/(M+1) lattice, r in 1..M+1.
    lattice = (np.arange(1, 101) - 0.5) / 100
    distinct = np.unique(payload["marg_pit"].ravel())
    assert distinct.size <= lattice.size
    assert np.abs(distinct[:, None] - lattice[None, :]).min(axis=1).max() < 1e-6


def test_generative_sampled_mean_variance_matches_the_true_quadratic_form(
    generative_payload,
):
    """``mean_prerank_var`` keeps its meaning: it is still ``a' Sigma a``.

    Estimated from 99 draws rather than computed in closed form, so it is
    checked as an unbiased estimate over the test set rather than case by case.
    """
    payload, _, _, sd = generative_payload
    V = sd.shape[-1]
    exact = (sd.pow(2).sum(-1).squeeze(-1) / V**2).numpy()
    got = payload["mean_prerank_var"][:, 0]
    assert got.mean() == pytest.approx(exact.mean(), rel=0.03)


def test_generative_draws_re_run_the_model_once_per_chunk(generative_payload):
    """Members cost forward passes, so the chunk loop must call the model again."""
    _, model, params, _ = generative_payload
    batches = -(-params.shape[0] // 128)
    # Per batch: one call for the mean, then ceil(99 / 33) = 3 chunked draws.
    assert len(model.calls) == batches * 4
    assert model.calls[:4] == [N_EVAL_SAMPLES, 33, 33, 33]
