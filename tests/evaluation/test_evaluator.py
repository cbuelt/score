"""Tests for the metric registry and the decoupled Evaluator.

The registry's closed-form metrics must agree with the method's own
implementations (single source of truth), the sampling-based metrics must match
the independent references the method tests already use, and the Evaluator must
return batch-size-weighted dataset means for the right (eval) sample count.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import EVAL_METRICS, register
from src.methods import Diag

EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 50
GAMMA = 0.5
OUT_CHANNELS = 1
B, D = 8, 2


@pytest.fixture
def method():
    return Diag("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)


def _pred_and_y():
    torch.manual_seed(0)
    mu = torch.randn(B, OUT_CHANNELS, D)
    var = torch.rand(B, OUT_CHANNELS, D) + 0.5
    pred = torch.stack([mu, var], dim=-1)
    y = torch.randn(B, OUT_CHANNELS, D)
    return pred, y


def test_registry_has_all_metrics():
    for name in (
        "mse",
        "mae",
        "nll",
        "crps",
        "kernel_score",
        "energy_score",
        "variogram_score",
    ):
        assert name in EVAL_METRICS


@pytest.mark.parametrize("name", ["mse", "mae", "nll"])
def test_closed_form_metrics_forward_to_method(method, name):
    """Closed-form registry entries must equal the method's own computation."""
    pred, y = _pred_and_y()
    expected = getattr(method, name)(pred, y)
    result = EVAL_METRICS[name](method, pred, y)
    assert torch.isclose(result, expected)


def test_crps_matches_closed_form(method):
    """Sampled CRPS ≈ the method's closed-form marginal CRPS.

    `crps` is a sampling-based metric (like energy_score/kernel_score), scored on
    the predictive so it also covers future nonparametric methods. For a
    diagonal Gaussian the predictive marginals *are* the closed-form marginals, so
    the ensemble estimate converges to the method's closed form.
    """
    pred, y = _pred_and_y()
    closed_form = method.crps(pred, y)

    method.n_eval_samples = 8000
    torch.manual_seed(0)
    result = EVAL_METRICS["crps"](method, pred, y)
    assert torch.isclose(result, closed_form, atol=0.02, rtol=0.01)


def test_energy_score_matches_reference(method):
    """Sampled energy score ≈ a high-sample sr.es_ensemble reference."""
    import scoringrules as sr

    pred, y = _pred_and_y()
    mu, var = pred[..., 0], pred[..., 1]
    sigma = var.sqrt()

    torch.manual_seed(0)
    n = 4000
    mu_e = mu.unsqueeze(-1).repeat_interleave(n, dim=-1)
    sigma_e = sigma.unsqueeze(-1).repeat_interleave(n, dim=-1)
    samples = mu_e + sigma_e * torch.randn_like(mu_e)
    reference = sr.es_ensemble(
        obs=y.flatten(start_dim=1),
        fct=samples.flatten(start_dim=1, end_dim=-2),
        m_axis=-1,
        v_axis=-2,
        backend="torch",
    ).mean()

    method.n_eval_samples = 4000
    torch.manual_seed(1)
    result = EVAL_METRICS["energy_score"](method, pred, y)
    assert torch.isclose(result, reference, atol=0.05, rtol=0.01)


def test_kernel_score_matches_closed_form(method):
    """Sampled gamma-bandwidth kernel score ≈ the method's closed form."""
    pred, y = _pred_and_y()
    closed_form = method.kernel_score(pred, y)

    method.n_eval_samples = 6000
    torch.manual_seed(0)
    result = EVAL_METRICS["kernel_score"](method, pred, y)
    assert torch.isclose(result, closed_form, atol=0.02, rtol=0.01)


def test_evaluator_returns_all_metrics_weighted(method):
    """Evaluator returns every metric, batch-weighted to the dataset mean."""
    torch.manual_seed(0)
    # 10 rows with batch_size 3 -> a ragged last batch, exercising weighting.
    n = 10
    x = torch.randn(n, OUT_CHANNELS, D)
    y = torch.randn(n, OUT_CHANNELS, D)

    # Forward maps x -> a packed (mu=x, var=1) prediction.
    def forward(batch_x):
        return torch.stack([batch_x, torch.ones_like(batch_x)], dim=-1)

    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    results = Evaluator(method, device="cpu").evaluate(forward, loader)

    assert set(results) == set(EVAL_METRICS)
    # The batch-weighted mse must equal the single-pass mse over the full set.
    pred_all = torch.stack([x, torch.ones_like(x)], dim=-1)
    expected_mse = method.mse(pred_all, y).item()
    assert results["mse"] == pytest.approx(expected_mse, abs=1e-5)


def test_evaluator_uses_eval_sample_count(method):
    """Evaluator flips the method into eval mode before sampling."""
    method.training = True  # pretend we were mid-training
    captured = {}

    orig_get_samples = method.get_samples

    def spy(pred, n_samples=None, flatten=False):
        captured["n"] = method.n_samples
        return orig_get_samples(pred, n_samples=n_samples, flatten=flatten)

    method.get_samples = spy

    x = torch.randn(4, OUT_CHANNELS, D)
    y = torch.randn(4, OUT_CHANNELS, D)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    def forward(bx):
        return torch.stack([bx, torch.ones_like(bx)], dim=-1)

    Evaluator(method, metrics=["energy_score"], device="cpu").evaluate(forward, loader)
    assert captured["n"] == N_EVAL_SAMPLES


def test_evaluator_forwards_metric_kwargs(method):
    """Per-metric kwargs reach the metric callable; metrics without any are
    unaffected (they are still called as `(method, pred, y)`)."""
    from src.evaluation.variogram import variogram_score_p1

    torch.manual_seed(0)
    # A 5-coordinate event, not the module's D=2: with only two coordinates the
    # single pair sits at distance 1, where `inv_dist` and `none` coincide.
    grid = 5
    x = torch.randn(4, OUT_CHANNELS, grid)
    y = torch.randn(4, OUT_CHANNELS, grid)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    def forward(bx):
        return torch.stack([bx, torch.ones_like(bx)], dim=-1)

    weighted = Evaluator(
        method,
        metrics=["variogram_score", "mse"],
        device="cpu",
        metric_kwargs={"variogram_score": {"weight": "inv_dist", "block_size": 1}},
    ).evaluate(forward, loader)

    pred = forward(x)
    expected = variogram_score_p1(method, pred, y, weight="inv_dist").mean().item()
    assert weighted["variogram_score"] == pytest.approx(expected, abs=1e-6)

    # The unweighted default is a different number, so the kwargs really landed.
    plain = Evaluator(method, metrics=["variogram_score"], device="cpu").evaluate(
        forward, loader
    )
    assert plain["variogram_score"] != pytest.approx(weighted["variogram_score"])


def test_evaluator_rejects_kwargs_for_unknown_metric(method):
    with pytest.raises(ValueError, match="metric_kwargs"):
        Evaluator(method, device="cpu", metric_kwargs={"not_a_metric": {"a": 1}})


def test_register_rejects_duplicate():
    with pytest.raises(ValueError):

        @register("mse")
        def _dup(method, pred, y):
            return torch.tensor(0.0)


def test_evaluator_handles_scalar_float_metric():
    """A metric returning a plain float (Deterministic.nll == inf) must not crash.

    The Det method (``Deterministic``) has no density, so its ``nll`` returns the Python
    float ``inf`` rather than a Tensor; the Evaluator must accumulate it without
    calling ``.item()`` on a float.
    """
    from src.methods import Deterministic

    det = Deterministic(
        "mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA
    )

    x = torch.randn(6, OUT_CHANNELS, D)
    y = torch.randn(6, OUT_CHANNELS, D)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    results = Evaluator(det, device="cpu").evaluate(lambda bx: bx, loader)

    assert math.isinf(results["nll"])
    assert math.isfinite(results["mse"])
