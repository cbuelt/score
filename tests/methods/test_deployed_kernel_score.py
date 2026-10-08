"""The stage-2 losses that score the *predictive* Gaussian N(mu, Sigma_P).

`SCORE.kernel_score` scores the bare correlation core ``C = B^T diag(s) B``
against the standardized target ``D_t^{-1}(mu - y)``. On a flat basis that is
sound -- the implied marginal variances are ``Delta(s) = mean(s) = 1`` by the
scale gauge, so ``D_t`` is fixed by stage 1 and the core score is a proper score
of the standardized target. On a localized basis (the graph Fourier basis used
on EUPPBench) the renormalization by the implied marginal variances makes
``D_t^2 = diag / Delta(s)`` a function of the stage-2 parameter ``s``: the
quantity being scored then moves with the parameter, so the objective is not a
proper scoring rule for the predictive covariance ``Sigma_P = D_t C D_t``.

`dense_kernel_score` (exact) and `sample_kernel_score` (fair ensemble estimator)
both score ``Sigma_P`` against ``mu - y`` instead. These tests pin that they are
the same functional, that the dense one is exact, that masking stays exact, and
that the core-level `kernel_score` is unchanged.

Run in float64: the dense equivalences are exact, so their tolerances are
numerical rather than statistical.
"""

import math

import pytest
import torch

from src.methods.base import masked_gaussian_kernel_score
from src.methods.diagonal import Diag
from src.methods.cholesky import Cholesky
from src.methods.lord import LorD
from src.methods.spectral import SpectralDiag
from src.methods.transforms import (
    SeparableOrthogonalTransform,
    graph_laplacian_eigenbasis,
)
from src.methods.score import SCORE

V = 12
BATCH = 4
CHANNELS = 1
GAMMA2 = float(V)


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _graph_basis() -> torch.Tensor:
    """Graph Fourier basis over an irregular point cloud, as EUPPBench uses."""
    g = torch.Generator().manual_seed(7)
    pts = torch.rand(V, 2, generator=g, dtype=torch.float64) * 10.0
    return graph_laplacian_eigenbasis(torch.cdist(pts, pts), knn=4)


def _common(n_samples: int = 4) -> dict:
    return dict(
        eps=1e-9,
        n_train_samples=n_samples,
        n_eval_samples=n_samples,
        out_channels=CHANNELS,
        gamma=[math.sqrt(GAMMA2)],
        loss="kernel_score",
    )


def _factory(kind, n_samples: int = 4):
    """``(builder, head_groups_per_channel)`` for each family."""
    common = _common(n_samples)
    graph = SeparableOrthogonalTransform([_graph_basis()])
    return {
        "diag": (lambda: Diag(**common), 2),
        "lora": (lambda: LorD(rank=2, **common), 2 + 2),
        "cholesky": (lambda: Cholesky(event_dim=V, **common), 2 + (V - 1)),
        "spectral": (lambda: SpectralDiag(**common), 2),
        "whitened_graph": (
            lambda: SCORE(rank=0, transform=graph, **common),
            3,
        ),
        "whitened_graph_lora": (
            lambda: SCORE(rank=2, transform=graph, **common),
            3 + 2,
        ),
    }[kind]


# Families whose own closed form already scores the physical covariance, so
# `dense_kernel_score` must reproduce `kernel_score` exactly.
PHYSICAL_METHODS = ["diag", "lora", "cholesky", "spectral"]
WHITENED_METHODS = ["whitened_graph", "whitened_graph_lora"]


def _sample(kind, seed=0, n_samples=4):
    torch.manual_seed(seed)
    builder, width = _factory(kind, n_samples)
    method = builder()
    pred = method.process_output(torch.randn(BATCH, CHANNELS * width, V))
    y = torch.randn(BATCH, CHANNELS, V)
    return method, pred, y


def _mask():
    mask = torch.ones(BATCH, V)
    mask[0, [1, 4, 7]] = 0.0
    mask[1, [0]] = 0.0
    mask[3, [2, 3, 9, 11]] = 0.0
    return mask


def _dense_reference(sigma: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """Gaussian kernel score of N(0, sigma) at `resid`, from first principles."""
    eye = torch.eye(sigma.shape[-1], dtype=sigma.dtype)
    diff = resid.unsqueeze(-1)
    log_det1 = torch.logdet(eye + 2.0 * sigma / GAMMA2)
    log_det2 = torch.logdet(eye + 4.0 * sigma / GAMMA2)
    maha = (
        diff.transpose(-1, -2) @ torch.linalg.solve(GAMMA2 * eye + 2.0 * sigma, diff)
    ).reshape(-1)
    return 0.5 * (1.0 + torch.exp(-0.5 * log_det2)) - torch.exp(-0.5 * log_det1 - maha)


@pytest.mark.parametrize("kind", PHYSICAL_METHODS)
def test_dense_reproduces_kernel_score_for_physical_families(kind):
    """Where the closed form already scores Sigma_P, the dense loss changes nothing.

    Only SCORE scores a different object (the core), so every other family
    must be able to switch losses without moving a number.
    """
    method, pred, y = _sample(kind)
    assert torch.allclose(
        method.dense_kernel_score(pred, y), method.kernel_score(pred, y), atol=1e-12
    )


@pytest.mark.parametrize("kind", WHITENED_METHODS)
def test_dense_matches_explicit_deployed_gaussian(kind):
    """`dense_kernel_score` is the exact score of N(mu, D_t C D_t) at mu - y."""
    method, pred, y = _sample(kind, seed=1)
    mu, diag, s, lora = method._split(pred)

    basis = method.transform.matrices[0]
    core = torch.einsum("ki,bk,kj->bij", basis, s.flatten(start_dim=1), basis)
    if method.rank > 0:
        factors = lora.flatten(start_dim=1, end_dim=-2)
        core = core + factors @ factors.transpose(-1, -2)
    # D_t^2 = diag / Delta(s) with Delta the implied marginal variances
    # (renormalization on for a localized basis), so the predictive marginal
    # variance is `diag`.
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    scale = (diag.flatten(start_dim=1) / delta).sqrt()
    sigma = scale.unsqueeze(-1) * core * scale.unsqueeze(-2)
    assert torch.allclose(
        torch.diagonal(sigma, dim1=-2, dim2=-1), diag.flatten(start_dim=1), atol=1e-10
    )

    resid = (mu - y).flatten(start_dim=1)
    reference = _dense_reference(sigma, resid).mean()
    assert torch.isclose(method.dense_kernel_score(pred, y), reference, atol=1e-12)


@pytest.mark.parametrize("kind", WHITENED_METHODS)
def test_dense_differs_from_the_core_surrogate(kind):
    """The two are genuinely different functionals on a localized basis.

    A guard against `dense_kernel_score` silently collapsing onto `kernel_score`,
    which would mean `_deployed_dense_terms` had not been overridden.
    """
    method, pred, y = _sample(kind, seed=2)
    assert not torch.isclose(
        method.dense_kernel_score(pred, y), method.kernel_score(pred, y), atol=1e-6
    )


@pytest.mark.parametrize("kind", PHYSICAL_METHODS + WHITENED_METHODS)
def test_masked_equals_subselected_gaussian(kind):
    """Masking must equal scoring the predictive Gaussian on the observed subset."""
    method, pred, y = _sample(kind, seed=3)
    A, r = method._deployed_dense_terms(pred, y)
    mask = _mask()

    got = masked_gaussian_kernel_score(A, r, GAMMA2, mask)
    expected = torch.stack(
        [
            masked_gaussian_kernel_score(
                A[b][mask[b].bool()][:, mask[b].bool()].unsqueeze(0),
                r[b][mask[b].bool()].unsqueeze(0),
                GAMMA2,
                None,
            )[0]
            for b in range(BATCH)
        ]
    )
    assert torch.allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("kind", WHITENED_METHODS)
def test_nan_targets_reach_the_masked_path(kind):
    """With NaNs present the loss equals the explicitly masked dense score."""
    method, pred, y = _sample(kind, seed=4)
    mask = _mask()
    y_nan = y.clone()
    y_nan.flatten(start_dim=1)[mask == 0.0] = float("nan")

    A, r = method._deployed_dense_terms(pred, torch.nan_to_num(y_nan))
    expected = masked_gaussian_kernel_score(A, r, GAMMA2, mask).mean()
    assert torch.isclose(method.dense_kernel_score(pred, y_nan), expected, atol=1e-12)


@pytest.mark.parametrize("kind", WHITENED_METHODS)
def test_sample_estimator_agrees_with_the_exact_score(kind):
    """The fair ensemble estimator converges to the exact predictive score.

    Tolerance is calibrated from the estimator's own spread across independent
    draws rather than hard-coded, so the test is statistically honest and does
    not silently pass on a biased estimator with a generous atol. A wrong
    bandwidth conversion (the sqrt(2)/gamma rescaling into `scoringrules`, whose
    kernel is fixed at exp(-||a-b||^2 / 2)) shifts the mean by far more than
    this.
    """
    method, pred, y = _sample(kind, seed=5, n_samples=400)
    exact = method.dense_kernel_score(pred, y)

    torch.manual_seed(99)
    draws = torch.stack([method.sample_kernel_score(pred, y) for _ in range(60)])
    sem = draws.std() / math.sqrt(draws.numel())
    assert (draws.mean() - exact).abs() < 4.0 * sem


@pytest.mark.parametrize("kind", WHITENED_METHODS)
def test_masked_sample_estimator_agrees_with_the_exact_score(kind):
    """Same, with missing targets: the point-mass fill must be exact, not merely close."""
    method, pred, y = _sample(kind, seed=6, n_samples=400)
    mask = _mask()
    y_nan = y.clone()
    y_nan.flatten(start_dim=1)[mask == 0.0] = float("nan")
    exact = method.dense_kernel_score(pred, y_nan)

    torch.manual_seed(101)
    draws = torch.stack([method.sample_kernel_score(pred, y_nan) for _ in range(60)])
    sem = draws.std() / math.sqrt(draws.numel())
    assert (draws.mean() - exact).abs() < 4.0 * sem


@pytest.mark.parametrize("kind", WHITENED_METHODS)
@pytest.mark.parametrize("loss_name", ["dense_kernel_score", "sample_kernel_score"])
def test_gradients_reach_the_spectrum_head(kind, loss_name):
    """Both losses must train the core: finite, nonzero gradient on the `s` group.

    `sample_kernel_score` is the one that can silently fail here -- `get_samples`
    is @torch.no_grad(), so routing it through the eval sampler would optimize a
    constant.
    """
    torch.manual_seed(11)
    builder, width = _factory(kind)
    method = builder()
    head = torch.randn(BATCH, CHANNELS * width, V, requires_grad=True)
    y = torch.randn(BATCH, CHANNELS, V)

    loss = getattr(method, loss_name)(method.process_output(head), y)
    (grad,) = torch.autograd.grad(loss, head)
    assert torch.isfinite(grad).all()
    s_grad = grad.reshape(BATCH, CHANNELS, width, V)[:, :, 2]
    assert s_grad.abs().max() > 0


def test_eval_sampler_stays_detached():
    """`get_samples` must keep its no-grad contract; only `_draw_samples` carries grad."""
    torch.manual_seed(13)
    builder, width = _factory("whitened_graph")
    method = builder()
    head = torch.randn(BATCH, CHANNELS * width, V, requires_grad=True)
    pred = method.process_output(head)
    assert not method.get_samples(pred, n_samples=3).requires_grad
    assert method._draw_samples(pred, n_samples=3).requires_grad


@pytest.mark.parametrize("kind", PHYSICAL_METHODS)
def test_sample_loss_refuses_families_without_a_differentiable_sampler(kind):
    """Fail loudly rather than train on a zero gradient."""
    method, pred, y = _sample(kind, seed=7)
    with pytest.raises(NotImplementedError, match="differentiable sampler"):
        method.sample_kernel_score(pred, y)


def test_gridded_event_dimensions_are_refused():
    """The dense path must stay unreachable at gridded scale, mask or no mask."""
    torch.manual_seed(17)
    method = Diag(**_common())
    method.dense_mask_max_dim = V - 1
    pred = method.process_output(torch.randn(BATCH, CHANNELS * 2, V))
    with pytest.raises(NotImplementedError, match="dense_mask_max_dim"):
        method.dense_kernel_score(pred, torch.randn(BATCH, CHANNELS, V))
