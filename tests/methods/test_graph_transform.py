"""Graph Fourier basis: orthogonality, determinism, and spatial smoothness.

The SCORE core over an irregular station network is diagonal in the
eigenbasis of the normalized graph Laplacian (the graph Fourier transform), which plays the role the DFT plays on a regular
grid. Three properties have to hold for that substitution to be sound:

1. it is a genuine orthogonal transform, so ``SeparableOrthogonalTransform``
   accepts it and the core stays a correlation matrix;
2. it is a *deterministic* function of the distance matrix -- ``eigh`` pins
   eigenvectors only up to sign, and a checkpoint restored against a
   sign-flipped basis would silently score a different object;
3. low rows are spatially smooth, so a spectrum concentrated at low ``k`` means
   long-range correlation. That is the entire modelling premise, and it is a
   property of the basis rather than of the fit, so it is checked here.

Run in float64: the transform is built and stored in float64, and orthogonality
is exact rather than statistical.
"""

import math

import numpy as np
import pytest
import torch

from src.methods.transforms import (
    SeparableOrthogonalTransform,
    graph_laplacian_eigenbasis,
    make_graph_transform,
)
from src.methods.score import SCORE

N_SIDE = 6
N = N_SIDE * N_SIDE
NORMALIZATIONS = ["sym", "comb", "rw_sym"]


def _grid_points(jitter: float = 0.0, seed: int = 0) -> torch.Tensor:
    """A jittered 2-D point cloud, standing in for the station network."""
    ys, xs = torch.meshgrid(
        torch.arange(N_SIDE, dtype=torch.float64),
        torch.arange(N_SIDE, dtype=torch.float64),
        indexing="ij",
    )
    pts = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)
    if jitter:
        g = torch.Generator().manual_seed(seed)
        pts = pts + jitter * torch.randn(pts.shape, generator=g, dtype=torch.float64)
    return pts


def _dist(jitter: float = 0.0, seed: int = 0) -> torch.Tensor:
    pts = _grid_points(jitter, seed)
    return torch.cdist(pts, pts)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_basis_is_orthogonal(normalization):
    """Rows must be orthonormal -- the contract SeparableOrthogonalTransform checks."""
    U = graph_laplacian_eigenbasis(_dist(), normalization=normalization)
    assert U.shape == (N, N)
    err = (U @ U.T - torch.eye(N, dtype=U.dtype)).abs().max()
    assert err < 1e-10


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_basis_is_deterministic(normalization):
    """Rebuilding from the same distances must give a bit-identical basis.

    `eigh` returns eigenvectors up to sign, so without the sign fix this is the
    test that fails -- and it would fail as a silent scoring difference between a
    training run and its evaluation, not as an error.
    """
    dist = _dist()
    first = graph_laplacian_eigenbasis(dist, normalization=normalization)
    second = graph_laplacian_eigenbasis(dist.clone(), normalization=normalization)
    assert torch.equal(first, second)


def test_eigenvalues_are_ascending_and_graph_is_connected():
    """Row 0 is the smoothest mode; a dense kernel leaves lambda_0 simple.

    Ordering is what makes "low k = large scale" true, which the core's softmax
    spectrum and `temperature` knob both assume.
    """
    dist = _dist()
    U = graph_laplacian_eigenbasis(dist)
    length_scale = float(dist.sort(dim=1).values[:, 10].median())
    W = torch.exp(-(dist**2) / (2 * length_scale**2))
    W.fill_diagonal_(0.0)
    deg = W.sum(dim=1)
    inv_sqrt = deg.rsqrt()
    L = torch.eye(N, dtype=torch.float64) - inv_sqrt.unsqueeze(1) * W * inv_sqrt.unsqueeze(0)

    rayleigh = torch.einsum("ki,ij,kj->k", U, L, U)
    assert torch.all(rayleigh[1:] - rayleigh[:-1] > -1e-10)
    assert rayleigh[0] < 1e-10          # the trivial mode
    assert rayleigh[1] > 1e-6           # simple, i.e. the graph is connected


def test_low_modes_are_spatially_smooth():
    """The premise of the whole construction: low k means long-range structure.

    Measured as the Dirichlet energy of each mode against a *reference* Gaussian
    affinity that the basis was not built from, so this is not circular.
    """
    dist = _dist(jitter=0.15, seed=3)
    U = graph_laplacian_eigenbasis(dist)
    affinity = torch.exp(-(dist**2) / (2 * 1.0**2))
    affinity.fill_diagonal_(0.0)
    # sum_ij a_ij (u_i - u_j)^2 -- small when the mode varies slowly in space.
    energy = torch.stack(
        [(affinity * (u.unsqueeze(0) - u.unsqueeze(1)) ** 2).sum() for u in U]
    )
    # Every one of the 5 smoothest modes is smoother than every one of the 5
    # roughest -- a separation, not just a difference in means.
    assert energy[:5].max() < energy[-5:].min()


def test_correlation_from_a_low_pass_spectrum_decays_with_distance():
    """A spectrum on the smooth modes must imply spatially decaying correlation.

    This is the claim the graph basis exists to support -- that shaping `s` is a
    way of expressing spatial dependence -- and it is what would break first if
    the length scale were badly chosen.
    """
    dist = _dist(jitter=0.1, seed=5)
    U = graph_laplacian_eigenbasis(dist)
    s = torch.exp(-torch.arange(N, dtype=torch.float64) / 3.0)  # low-pass
    cov = U.T @ torch.diag(s) @ U
    scale = cov.diagonal().rsqrt()
    corr = scale.unsqueeze(1) * cov * scale.unsqueeze(0)

    off = ~torch.eye(N, dtype=torch.bool)
    near = (dist < 1.5) & off
    far = dist > 4.0
    assert corr[near].mean() > corr[far].mean()
    assert corr[far].abs().mean() < 0.2


def test_automatic_length_scale_tracks_the_point_spacing():
    """Doubling every distance must double the inferred bandwidth (it is a scale)."""
    dist = _dist()
    base = graph_laplacian_eigenbasis(dist)
    scaled = graph_laplacian_eigenbasis(2.0 * dist)
    # The kernel depends on d/l only, so a pure rescaling leaves the basis fixed.
    assert torch.allclose(base, scaled, atol=1e-8)


def test_tiny_length_scale_warns_about_isolated_nodes():
    """A bandwidth far below the point spacing underflows every edge to zero.

    Under "sym" the isolated nodes land at eigenvalue 1, not 0, so the
    zero-eigenvalue check alone would miss this -- hence the separate degree
    check. Exactly the failure a hard distance cutoff hits, and the reason the
    default kernel is dense; it must be loud rather than silent.
    """
    with pytest.warns(RuntimeWarning, match="effectively isolated nodes"):
        graph_laplacian_eigenbasis(_dist(), length_scale=1e-3)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_disconnected_clusters_warn(normalization):
    """Two well-separated clusters give a degenerate zero eigenvalue.

    The other half of the connectivity check: here every node has neighbours, so
    the degree test passes and only the eigenvalue gap reveals the split.
    """
    pts = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [60.0, 0.0], [61.0, 0.0], [60.0, 1.0]],
        dtype=torch.float64,
    )
    dist = torch.cdist(pts, pts)
    with pytest.warns(RuntimeWarning, match="disconnected clusters"):
        graph_laplacian_eigenbasis(dist, length_scale=1.0, normalization=normalization)


def test_rejects_asymmetric_and_degenerate_input():
    dist = _dist()
    bad = dist.clone()
    bad[0, 1] += 5.0
    with pytest.raises(ValueError, match="symmetric"):
        graph_laplacian_eigenbasis(bad)
    with pytest.raises(ValueError, match="square"):
        graph_laplacian_eigenbasis(dist[:, :3])
    with pytest.raises(ValueError, match="length_scale must be"):
        graph_laplacian_eigenbasis(dist, length_scale=-1.0)
    with pytest.raises(ValueError, match="Unknown normalization"):
        graph_laplacian_eigenbasis(dist, normalization="nope")


def test_transform_is_non_flat_so_delta_renorm_engages():
    """A localized basis needs the renormalization by the implied marginal variances, selected automatically."""
    transform = SeparableOrthogonalTransform([graph_laplacian_eigenbasis(_dist())])
    assert transform.is_flat is False
    assert transform.spatial_shape == (N,)

    method = SCORE(
        loss="kernel_score",
        eps=1e-9,
        n_train_samples=4,
        n_eval_samples=4,
        out_channels=1,
        gamma=[math.sqrt(N)],
        transform=transform,
    )
    assert method._use_delta is True


def test_delta_renorm_restores_the_marginal_variances():
    """diag(Sigma_P) must equal the stage-1 diag exactly, for any spectrum."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        transform = SeparableOrthogonalTransform([graph_laplacian_eigenbasis(_dist())])
        method = SCORE(
            loss="kernel_score",
            eps=1e-9,
            n_train_samples=4,
            n_eval_samples=4,
            out_channels=1,
            gamma=[math.sqrt(N)],
            transform=transform,
        )
        torch.manual_seed(0)
        pred = method.process_output(torch.randn(3, 3, N))
        _, diag, _, _ = method._split(pred)
        assert torch.allclose(method.get_marg_std(pred), diag.sqrt(), atol=1e-10)
    finally:
        torch.set_default_dtype(previous)


def _wsd(**kw):
    transform = kw.pop("transform", None) or SeparableOrthogonalTransform(
        [graph_laplacian_eigenbasis(_dist())],
        eigenvalues=graph_laplacian_eigenbasis(_dist(), return_eigenvalues=True)[1],
    )
    return SCORE(
        loss="kernel_score", eps=1e-9, n_train_samples=4, n_eval_samples=4,
        out_channels=1, gamma=[math.sqrt(N)], transform=transform, **kw
    )


def test_matern_head_emits_three_scalars_not_one_logit_per_mode():
    """The whole point of the parameterization: the head width stops scaling with N.

    A free spectrum needs one output per graph mode; the Whittle-Matern core
    needs ``(sigma, kappa, alpha)`` however many stations there are, which is
    what makes stage 2 small and what lets it transfer to a different station
    set.
    """
    assert _wsd(core_parameterization="matern").group_widths["other"] == 3
    assert _wsd().group_widths["other"] == 1


def test_matern_spectrum_is_monotone_in_the_eigenvalue():
    """``s_k = sigma (kappa^2 + lambda_k)^-2 alpha`` is decreasing in ``lambda_k``.

    This is the structural guarantee the free spectrum could not give. Laplacian
    eigenvectors are oscillatory (mode ``k`` has ~``k`` sign changes), so mass on
    an isolated high mode is a *ringing* correlation -- positive nearby, negative
    at intermediate range. Monotonicity makes that unrepresentable rather than
    merely penalized, so it is checked on random heads, not on a fitted one.
    """
    torch.set_default_dtype(torch.float64)
    try:
        method = _wsd(core_parameterization="matern")
        torch.manual_seed(0)
        # (B, C * 5, N): [mu, diag, sigma, kappa, alpha].
        pred = method.process_output(torch.randn(8, 5, N) * 3)
        _, _, s, _ = method._split(pred)
        s = s.flatten(start_dim=1)
        # Eigenvalues are ascending, so the spectrum must be non-increasing.
        assert (s.diff(dim=1) <= 1e-12).all()
        # And gauge-fixed to mean(s) = 1, like every other core here.
        assert torch.allclose(s.mean(dim=1), torch.ones(s.shape[0]), atol=1e-9)
    finally:
        torch.set_default_dtype(torch.float32)


def test_matern_core_stays_float32_safe():
    """The analytic form bounds the dynamic range, so the core cannot blow up.

    An unconstrained spectrum on this basis can become severely ill-conditioned.
    Here the range is bounded in closed form by
    ``((kappa^2 + lambda_max)/(kappa^2 + lambda_min))^(2 alpha)``, and
    `core_logit_range` caps it outright.
    """
    torch.set_default_dtype(torch.float64)
    try:
        def worst_cond(**kw):
            method = _wsd(core_parameterization="matern", **kw)
            torch.manual_seed(0)
            pred = method.process_output(torch.randn(4, 5, N) * 6)
            A, _ = method._dense_score_terms(pred, torch.randn(4, 1, N))
            ev = torch.linalg.eigvalsh(A)
            return (ev[:, -1] / ev[:, 0]).max().item()

        assert worst_cond(core_logit_range=9.2) <= math.exp(9.2) * 1.01
        assert worst_cond() < 1 / torch.finfo(torch.float32).eps
    finally:
        torch.set_default_dtype(torch.float32)


def test_matern_requires_a_basis_with_eigenvalues():
    """An analytic basis carries no spectral coordinate to evaluate the density at."""
    from src.methods.transforms import dct_matrix

    with pytest.raises(ValueError, match="eigenvalues"):
        _wsd(
            transform=SeparableOrthogonalTransform([dct_matrix(N)]),
            core_parameterization="matern",
        )
    with pytest.raises(NotImplementedError, match="rank=0"):
        _wsd(core_parameterization="matern", rank=2)


def test_logit_cap_is_rejected_on_the_free_spectrum():
    """It clamps the Matern log-spectrum; a free core has none, so it would no-op."""
    with pytest.raises(ValueError, match="core_logit_range"):
        _wsd(core_logit_range=9.2)


def test_make_graph_transform_reads_meta_npz(tmp_path):
    """The factory must consume the same meta.npz the backbone and variogram use."""
    meta = tmp_path / "meta.npz"
    np.savez(meta, dist=_dist().numpy())
    transform = make_graph_transform(str(meta))
    assert isinstance(transform, SeparableOrthogonalTransform)
    assert transform.spatial_shape == (N,)
    assert torch.equal(transform.matrices[0], graph_laplacian_eigenbasis(_dist()))


def test_make_graph_transform_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta.npz"):
        make_graph_transform(str(tmp_path))  # the directory, not the file
    empty = tmp_path / "empty.npz"
    np.savez(empty, lat=np.zeros(3))
    with pytest.raises(KeyError, match="dist"):
        make_graph_transform(str(empty))
