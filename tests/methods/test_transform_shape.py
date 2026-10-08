"""Analytic transforms on a 1-D event axis, and the deferred (data-shaped) basis.

The event shape belongs to the data -- a time-series forecast horizon has a
*single* spatial axis, an image two -- so a transform configured with
``shape: null`` reads it off the first batch and one method config serves every
task.

Two things therefore need checking: that a deferred transform materializes into
exactly the eager one it stands for, and that the whole SCORE path is correct
on a 1-D field, against dense references built from the explicit DCT matrix.
"""

import math

import pytest
import torch

from src.methods.transforms import (
    SeparableOrthogonalTransform,
    dct_matrix,
    dst_matrix,
    haar_matrix,
    make_separable_transform,
    unbalanced_haar_matrix,
)
from src.methods.score import SCORE

GAMMA = 10.0
OUT_CHANNELS = 1
BATCH_SIZE = 4
L = 12  # forecast horizon: the single spatial axis of a time-series target
H, W = 4, 8  # the 2-D grid, to keep the image tasks covered by the same code
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 100


def _field(seed: int, positive: bool = False, shape=(L,)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH_SIZE, OUT_CHANNELS, *shape, generator=g, dtype=torch.float64)
    return x.abs() + 0.5 if positive else x


def _method(transform, **kwargs) -> SCORE:
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


def _deployed_cov(diag: torch.Tensor, s: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Dense Sigma_P = D_t C D_t with C = B^T diag(s) B and D_t^2 = diag / Delta."""
    B = dct_matrix(L) if B is None else B
    sb = s.reshape(BATCH_SIZE, -1)
    core = B.T @ (sb.unsqueeze(-1) * B)
    delta = torch.diagonal(core, dim1=-2, dim2=-1)
    rt = (diag.reshape(BATCH_SIZE, -1) / delta).sqrt()
    return rt.unsqueeze(-1) * core * rt.unsqueeze(-2)


# ----------------------------------------------------------------------
# The deferred basis
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,builder",
    [("dct", dct_matrix), ("dst", dst_matrix), ("uhaar", unbalanced_haar_matrix)],
)
@pytest.mark.parametrize("shape", [(L,), (H, W)], ids=["1d", "2d"])
def test_deferred_matches_eager(kind, builder, shape):
    """One config, any event shape: the materialized factors are the eager ones."""
    deferred = make_separable_transform(kind=kind)
    eager = make_separable_transform(kind=kind, shape=list(shape))
    x = _field(0, shape=shape)

    assert torch.allclose(deferred.forward(x), eager.forward(x), atol=1e-12)
    assert deferred.spatial_shape == shape
    for got, want in zip(deferred.matrices, [builder(n) for n in shape]):
        assert torch.allclose(got, want, atol=1e-12)


def test_unbalanced_haar_covers_the_forecast_horizons():
    """A *wavelet* core on a time-series target needs the unbalanced basis.

    Classical Haar only exists at powers of two, and no horizon here is one
    (96 for the hourly/10-min tasks, 24 for illness), so `kind: haar` cannot be
    deferred to the data at all -- `uhaar` is what makes the localized basis
    available on those tasks, and it coincides with Haar where Haar exists.
    """
    with pytest.raises(ValueError, match="power of two"):
        haar_matrix(L)
    for n in (L, 24, 96, 104):
        U = unbalanced_haar_matrix(n)
        assert U.shape == (n, n)
        assert torch.allclose(U @ U.T, torch.eye(n, dtype=U.dtype), atol=1e-12)
    for n in (1, 2, 4, 8, 16):
        assert torch.allclose(unbalanced_haar_matrix(n), haar_matrix(n), atol=1e-12)


def test_marg_std_exact_on_1d_uhaar():
    """The renormalization is what a localized basis actually depends on:
    unlike the DCT, a wavelet core's implied variance varies strongly with
    position, so the predictive marginals only equal the stage-1 diag if Delta is
    divided out on the (non-power-of-two) horizon too."""
    transform = make_separable_transform(kind="uhaar")
    method = _method(transform)
    assert method._use_delta

    diag, s = _field(23, positive=True), _field(24, positive=True)
    pred = torch.stack([_field(25), diag, s], dim=-1)
    assert torch.allclose(method.get_marg_std(pred), diag.sqrt(), atol=1e-12)
    Sigma = _deployed_cov(diag, s, unbalanced_haar_matrix(L))
    assert torch.allclose(
        torch.diagonal(Sigma, dim1=-2, dim2=-1),
        diag.reshape(BATCH_SIZE, -1),
        atol=1e-12,
    )


def test_deferred_roundtrips_and_delta_on_1d():
    transform = make_separable_transform(kind="dct")
    x, s = _field(1), _field(2, positive=True)

    assert torch.allclose(transform.inverse(transform.forward(x)), x, atol=1e-12)
    # Delta_i = diag(B^T diag(s) B), the marginal variance the core implies.
    B = dct_matrix(L)
    core = B.T @ (s.reshape(BATCH_SIZE, -1).unsqueeze(-1) * B)
    dense_diag = torch.diagonal(core, dim1=-2, dim2=-1)
    assert torch.allclose(
        transform.delta(s).reshape(BATCH_SIZE, -1), dense_diag, atol=1e-12
    )


def test_deferred_flatness_is_known_before_the_first_batch():
    """The method picks its normalization at construction, i.e. before any data,
    so a deferred transform has to answer `is_flat` by family."""
    dct = make_separable_transform(kind="dct")
    hadamard = make_separable_transform(kind="hadamard")
    assert not dct.is_flat and _method(dct)._use_delta
    assert hadamard.is_flat and not _method(hadamard)._use_delta
    # And the answer survives materialization (checked on the factors there).
    dct.forward(_field(3))
    hadamard.forward(_field(4, shape=(8,)))
    assert not dct.is_flat
    assert hadamard.is_flat


def test_first_batch_fixes_the_shape():
    """Adaptive once, then strict: a later mismatch is a bug, not a resize."""
    transform = make_separable_transform(kind="dct")
    transform.forward(_field(5))
    with pytest.raises(ValueError, match="does not match transform shape"):
        transform.forward(_field(6, shape=(L + 1,)))


def test_deferred_axis_count_must_match_the_data():
    transform = make_separable_transform(kind=["dct", "dct"])
    with pytest.raises(ValueError, match="2 axis kinds"):
        transform.forward(_field(7))


def test_explicit_shape_still_asserts_the_grid():
    """An explicit shape is a check, so a wrong one must still fail loudly."""
    transform = make_separable_transform(kind="dct", shape=[L + 1])
    with pytest.raises(ValueError, match="does not match transform shape"):
        transform.forward(_field(8))
    # Scalar shorthand for the single-axis case.
    assert make_separable_transform(kind="dct", shape=L).spatial_shape == (L,)


def test_unknown_kind_raises_at_construction():
    with pytest.raises(ValueError, match="Unknown transform kind"):
        make_separable_transform(kind="fourier")


def test_matrices_and_kinds_are_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        SeparableOrthogonalTransform([dct_matrix(L)], kinds="dct")
    with pytest.raises(ValueError, match="exactly one"):
        SeparableOrthogonalTransform()


# ----------------------------------------------------------------------
# The SCORE path on a 1-D event axis
# ----------------------------------------------------------------------


def test_marg_std_exact_on_1d():
    """The renormalization holds on a single axis too: the predictive
    marginal variance is the stage-1 diag, for an arbitrary spectrum."""
    method = _method(make_separable_transform(kind="dct"))
    diag, s = _field(9, positive=True), _field(10, positive=True)
    pred = torch.stack([_field(11), diag, s], dim=-1)
    assert torch.allclose(method.get_marg_std(pred), diag.sqrt(), atol=1e-12)
    Sigma = _deployed_cov(diag, s)
    assert torch.allclose(
        torch.diagonal(Sigma, dim1=-2, dim2=-1),
        diag.reshape(BATCH_SIZE, -1),
        atol=1e-12,
    )


def test_nll_matches_dense_on_1d():
    method = _method(make_separable_transform(kind="dct"))
    mu, diag, s, y = (
        _field(12),
        _field(13, positive=True),
        _field(14, positive=True),
        _field(15),
    )
    pred = torch.stack([mu, diag, s], dim=-1)

    Sigma = _deployed_cov(diag, s)
    diff = (y - mu).reshape(BATCH_SIZE, -1, 1)
    maha = (diff.transpose(-1, -2) @ torch.linalg.solve(Sigma, diff)).reshape(-1)
    nll_ref = 0.5 * (L * math.log(2 * math.pi) + torch.logdet(Sigma) + maha)
    assert torch.isclose(method.nll(pred, y), nll_ref.mean(), atol=1e-8)


def test_masked_score_materializes_the_basis():
    """The masked path is the one place that reads the basis matrix directly.

    With a *flat* basis the renormalization is off, so standardization never touches
    the transform and a deferred one would still be unbuilt when the dense core
    is assembled -- it has to be materialized there too.
    """
    n = 8  # Hadamard needs a power of two
    transform = make_separable_transform(kind="hadamard")
    method = _method(transform)
    assert not method._use_delta

    diag, s = _field(19, positive=True, shape=(n,)), _field(20, positive=True, shape=(n,))
    pred = torch.stack([_field(21, shape=(n,)), diag, s], dim=-1)
    y = _field(22, shape=(n,))
    y[:, :, -2:] = float("nan")  # missing observations trigger the masked path
    assert torch.isfinite(method.kernel_score(pred, y))
    assert transform.spatial_shape == (n,)


def test_samples_have_the_deployed_covariance_on_1d():
    method = _method(make_separable_transform(kind="dct"))
    diag, s = _field(16, positive=True), _field(17, positive=True)
    pred = torch.stack([_field(18), diag, s], dim=-1)
    torch.manual_seed(0)
    samples = method.get_samples(pred, n_samples=200_000, flatten=True)  # (B, L, M)
    emp = torch.cov(samples[0])  # (L, L): coordinates are the variables
    assert torch.allclose(emp, _deployed_cov(diag, s)[0], atol=2e-2)
