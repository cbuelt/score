"""Missing-target masking: exactness, and no effect on complete data.

The station datasets have targets missing for a few percent of coordinates and
no fully-reported sample, so the scores must evaluate the predictive *marginal*
on the observed coordinates. For a Gaussian that marginal is exact, and these
tests pin the two properties the design depends on:

1. the masked score equals the score of the explicitly sub-selected Gaussian;
2. on complete data every score is bit-identical to the unmasked implementation,
   so the gridded datasets are untouched.

Run in float64: the equivalences are exact, so the tolerances are numerical, not
statistical, and float32 would hide a real error under its own noise.
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
    dct_matrix,
    graph_laplacian_eigenbasis,
)
from src.methods.score import SCORE

V = 12
BATCH = 4
CHANNELS = 1


def _graph_basis():
    """Graph Fourier basis over an irregular point cloud, as EUPPBench uses.

    The station setting is the one that actually needs masking, so the masked
    equivalences have to be pinned on a graph eigenbasis and not only on the
    analytic transforms -- it is the basis whose rows are localized and whose
    core therefore leans hardest on the renormalization by the implied marginal
    variances.
    """
    g = torch.Generator().manual_seed(7)
    pts = torch.rand(V, 2, generator=g, dtype=torch.float64) * 10.0
    return graph_laplacian_eigenbasis(torch.cdist(pts, pts), knn=4)


def _method(kind, loss="kernel_score"):
    common = dict(
        eps=1e-9,
        n_train_samples=4,
        n_eval_samples=4,
        out_channels=CHANNELS,
        gamma=[math.sqrt(V)],
        loss=loss,
    )
    transform = SeparableOrthogonalTransform([dct_matrix(V)])
    graph = SeparableOrthogonalTransform([_graph_basis()])
    return {
        "diag": (lambda: Diag(**common), 2),
        "lora": (lambda: LorD(rank=2, **common), 2 + 2),
        # The dense family emits one row of L per coordinate, so its head is
        # V-1 groups wider than the diagonal one.
        "cholesky": (lambda: Cholesky(event_dim=V, **common), 2 + (V - 1)),
        "spectral": (lambda: SpectralDiag(**common), 2),
        "whitened": (lambda: SCORE(rank=0, transform=transform, **common), 3),
        "whitened_lora": (
            lambda: SCORE(rank=2, transform=transform, **common),
            3 + 2,
        ),
        "whitened_graph": (
            lambda: SCORE(rank=0, transform=graph, **common),
            3,
        ),
        "whitened_graph_lora": (
            lambda: SCORE(rank=2, transform=graph, **common),
            3 + 2,
        ),
    }[kind]


ALL_METHODS = [
    "diag",
    "lora",
    "cholesky",
    "spectral",
    "whitened",
    "whitened_lora",
    "whitened_graph",
    "whitened_graph_lora",
]


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _sample(kind, seed=0):
    torch.manual_seed(seed)
    factory, width = _method(kind)
    method = factory()
    pred = method.process_output(torch.randn(BATCH, CHANNELS * width, V))
    y = torch.randn(BATCH, CHANNELS, V)
    return method, pred, y


def _mask():
    mask = torch.ones(BATCH, V)
    mask[0, [1, 4, 7]] = 0.0
    mask[1, [0]] = 0.0
    mask[3, [2, 3, 9, 11]] = 0.0
    return mask


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_dense_terms_reproduce_structured_closed_form(kind):
    """The dense (A, r) form must agree with each family's own closed form.

    This is what licenses swapping in the dense path when a mask is present: if
    the two disagreed on complete data, a masked run would not be comparable
    with an unmasked one.
    """
    method, pred, y = _sample(kind)
    A, r = method._dense_score_terms(pred, y)
    reference = method._kernel_score_single(pred, y, reduction="none")
    dense = masked_gaussian_kernel_score(A, r, method.gamma2, None)
    assert torch.allclose(reference, dense, atol=1e-12)


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_masked_equals_subselected_gaussian(kind):
    """Masking must equal scoring the Gaussian restricted to observed coordinates."""
    method, pred, y = _sample(kind)
    A, r = method._dense_score_terms(pred, y)
    mask = _mask()

    got = masked_gaussian_kernel_score(A, r, method.gamma2, mask)
    expected = torch.stack(
        [
            masked_gaussian_kernel_score(
                A[b][mask[b].bool()][:, mask[b].bool()].unsqueeze(0),
                r[b][mask[b].bool()].unsqueeze(0),
                method.gamma2,
                None,
            )[0]
            for b in range(BATCH)
        ]
    )
    assert torch.allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_complete_data_takes_the_unmasked_path(kind):
    """With no NaN present, `kernel_score` must be bit-identical to the unmasked path.

    A regression guard for the gridded datasets, which share this code but have
    no missing targets and must not silently change value.
    """
    method, pred, y = _sample(kind)
    public = method.kernel_score(pred, y)
    structured = method.reduce(method._kernel_score_single(pred, y, reduction="none"), "mean")
    assert public.item() == structured.item()


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_crps_averages_over_observed_coordinates_only(kind):
    """Marginal CRPS must drop masked coordinates and renormalize by the observed count."""
    method, pred, y = _sample(kind)
    y_nan = y.clone()
    y_nan[0, 0, [1, 4, 7]] = float("nan")
    y_nan[1, 0, 0] = float("nan")

    per_coord = method._crps_per_coord(pred, torch.nan_to_num(y_nan)).flatten(start_dim=1)
    mask = torch.isfinite(y_nan).flatten(start_dim=1).to(per_coord.dtype)
    expected = ((per_coord * mask).sum(dim=1) / mask.sum(dim=1)).mean()

    assert torch.allclose(method.crps(pred, y_nan), expected, atol=1e-12)


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_fully_missing_row_stays_finite(kind):
    """A sample with no observations at all must not poison the loss or its gradients.

    The station data has stations that never report, so an all-NaN row is a real
    possibility rather than a contrived edge case.
    """
    torch.manual_seed(1)
    factory, width = _method(kind, loss="crps")
    method = factory()
    head = torch.randn(BATCH, CHANNELS * width, V, requires_grad=True)
    pred = method.process_output(head)
    y = torch.randn(BATCH, CHANNELS, V)
    y[2] = float("nan")  # this sample contributes nothing

    loss = method.crps(pred, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(head.grad).all()


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_masked_nll_equals_the_subselected_gaussian(kind):
    """Masked NLL must equal the likelihood of the marginal on observed coordinates.

    Checked against `torch.distributions.MultivariateNormal` on an explicitly
    sub-selected covariance -- an independent implementation, so this does not
    merely confirm the dense path against itself.
    """
    method, pred, y = _sample(kind)
    rows = torch.arange(V)
    var = method.get_marg_std(pred).flatten(start_dim=1).pow(2)
    cov = 0.5 * (
        var[:, rows].unsqueeze(-1) + var.unsqueeze(-2) - method.pair_diff_var(pred, rows)
    )
    mu = method.get_mean(pred).flatten(start_dim=1)

    mask = _mask()
    y_flat = y.flatten(start_dim=1)
    y_nan = y_flat.clone()
    y_nan[mask == 0] = float("nan")

    got = method.nll(pred, y_nan.reshape(BATCH, CHANNELS, V), reduction="none")
    expected = torch.stack(
        [
            -torch.distributions.MultivariateNormal(
                mu[b][mask[b].bool()],
                cov[b][mask[b].bool()][:, mask[b].bool()],
            ).log_prob(y_flat[b][mask[b].bool()])
            for b in range(BATCH)
        ]
    )
    assert torch.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("kind", ALL_METHODS)
def test_nll_on_complete_data_uses_the_structured_form(kind):
    """No NaN present ⇒ bit-identical to each family's own structured NLL."""
    method, pred, y = _sample(kind)
    assert method.nll(pred, y).item() == method._nll_complete(pred, y).item()


def test_unsupported_losses_reject_missing_targets():
    """The sliced `energy_score` must fail loudly, not return NaN.

    It is the one remaining loss without a masked form: it estimates the score
    from random projections, so masking means restricting the directions to the
    observed subspace per family.
    """
    method, pred, y = _sample("diag")
    y[0, 0, 3] = float("nan")

    assert Diag.MASK_UNSUPPORTED_LOSSES == ("energy_score",)
    for loss in Diag.MASK_UNSUPPORTED_LOSSES:
        method.loss = loss
        with pytest.raises(NotImplementedError, match="masked implementation"):
            method.get_train_loss()(pred, y)


def test_dense_mask_guard_rejects_large_events():
    """The dense path must refuse gridded-scale events instead of trying to allocate."""
    torch.manual_seed(0)
    method = Diag(
        loss="kernel_score", eps=1e-9, n_train_samples=4, n_eval_samples=4,
        out_channels=1, gamma=[1.0],
    )
    big = method.dense_mask_max_dim + 1
    pred = method.process_output(torch.randn(1, 2, big))
    y = torch.randn(1, 1, big)
    y[0, 0, 0] = float("nan")
    with pytest.raises(NotImplementedError, match="dense_mask_max_dim"):
        method.kernel_score(pred, y)
