"""Unitary transforms for the SCORE correlation core.

SCORE predicts ``Sigma_P = D corr(B* diag(s) B) D`` (paper Eq. (5)) with a
diagonal core in the B-domain of a fixed unitary transform ``B``. The Fourier
case is handled natively (and fastest) inside :class:`SCORE` /
:class:`SpectralDiag` via the FFT; this module provides the *general* case: any
real-orthogonal, spatially separable transform (Haar wavelets, DCT, per-axis
graph-Laplacian eigenbases, Hadamard, ...).

The key quantity a transform must provide, beyond forward/inverse, is

    Delta_i(s) = sum_k s_k |B_{ki}|^2 = diag(B* diag(s) B),

the implied marginal variance at coordinate ``i``. The correlation core is
then ``corr(B* diag(s) B) = Delta^{-1/2} B* diag(s) B Delta^{-1/2}``, which has
unit diagonal for *every* unitary basis (paper Eq. (10)). For a *flat* basis
(``|B_{ki}|^2 = 1/d`` everywhere, e.g. DFT or Hadamard) ``Delta`` collapses to
``mean(s)``, so the scale gauge ``mean(s) = 1`` (the softmax normalization) already
yields a correlation core and the renormalization is a no-op. Flatness is
detected automatically, so the right normalization is chosen per transform
without configuration.
"""

import math
import os
import warnings
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor


class SeparableOrthogonalTransform:
    """A real-orthogonal transform applied separably over the spatial axes.

    One orthogonal matrix per spatial axis, rows = basis functions; the full
    transform is their Kronecker product ``B = M_1 (x) ... (x) M_N``, applied
    axis-by-axis so it is never materialized. Coefficients are computed as
    ``z = B y`` (:meth:`forward`) and inverted with ``y = B^T z``
    (:meth:`inverse`).

    ``is_flat`` is auto-detected: the Kronecker product is flat iff every
    factor is (``|M[k, i]|^2 = 1/n`` for all entries), in which case
    ``delta(s) = mean(s)`` exactly and the spectral (softmax) normalization
    alone already yields a correlation core.

    Tensors are expected in the methods' layout ``(B, C, d1, ..., dN)`` with
    one matrix per spatial axis ``d1..dN``.

    The matrices can also be *deferred*: pass ``kinds`` (a name from
    ``_MATRIX_BUILDERS``, or one per axis) instead of ``matrices`` and the
    factors are built from the first tensor the transform sees, which is what
    makes an analytic basis dataset-agnostic -- the same DCT config then serves
    a ``(128, 160)`` image grid and a length-96 forecast horizon. The event
    shape is a property of the data, so reading it off the data is both less
    configuration and one less thing that can silently disagree with it. The
    first tensor fixes the shape; later mismatches raise as usual.
    """

    def __init__(
        self,
        matrices: Sequence[Tensor] = None,
        eigenvalues: Tensor = None,
        kinds=None,
    ):
        if (matrices is None) == (kinds is None):
            raise ValueError(
                "Pass exactly one of `matrices` (eager) or `kinds` (deferred, "
                "axis lengths read off the first tensor)."
            )
        # Deferred: one builder name per axis, or a single name applied to every
        # axis of whatever rank the data turns out to have.
        self._kinds = kinds if kinds is None or isinstance(kinds, str) else list(kinds)
        if self._kinds is not None:
            for name in [self._kinds] if isinstance(self._kinds, str) else self._kinds:
                if name not in _MATRIX_BUILDERS:
                    raise ValueError(
                        f"Unknown transform kind {name!r}; expected one of "
                        f"{sorted(_MATRIX_BUILDERS)}."
                    )
        self.matrices = None
        # Per-(device, dtype) cache of the converted matrix lists, so GPU
        # training does not re-transfer the (small) factors every call.
        self._converted: dict = {}
        # Spectral coordinate of each basis function, when the basis comes from
        # an operator that has one (a graph Laplacian). Analytic bases like the
        # DCT leave this None. Needed by a core that *evaluates* a spectral
        # density rather than learning one value per mode -- see
        # `SCORE`'s "matern" parameterization, which sets
        # s_k = g(eigenvalue_k).
        self.eigenvalues = None if eigenvalues is None else torch.as_tensor(eigenvalues).flatten()
        if matrices is not None:
            self._set_matrices(matrices)

    def _set_matrices(self, matrices: Sequence[Tensor]) -> None:
        """Validate and install the per-axis factors (eager, or on first use)."""
        if len(matrices) == 0:
            raise ValueError("Need at least one per-axis matrix.")
        mats: List[Tensor] = []
        for M in matrices:
            M = torch.as_tensor(M)
            if M.is_complex():
                raise ValueError(
                    "SeparableOrthogonalTransform is real-orthogonal only; the "
                    "complex (Fourier) case is handled natively by the FFT path."
                )
            M = M.to(torch.get_default_dtype()) if not M.is_floating_point() else M
            n = M.shape[0]
            if M.shape != (n, n):
                raise ValueError(f"Per-axis matrix must be square, got {tuple(M.shape)}.")
            err = (M @ M.T - torch.eye(n, dtype=M.dtype)).abs().max()
            if err > 1e-6:
                raise ValueError(f"Matrix is not orthogonal (max|MM^T - I| = {err:.2e}).")
            mats.append(M)
        self.matrices = mats
        # |B_ki|^2 factorizes over the axes; cache the per-axis squares for delta.
        self._squares = [M**2 for M in mats]
        self._converted.clear()

    def materialize(self, x: Tensor) -> None:
        """Build the deferred factors for ``x``'s spatial shape ``x.shape[2:]``.

        A no-op for an eager transform and after the first call. Every entry
        point that touches the matrices goes through here, so a deferred
        transform behaves exactly like an eager one from the outside.
        """
        if self.matrices is not None:
            return
        shape = tuple(int(n) for n in x.shape[2:])
        if len(shape) == 0:
            raise ValueError(
                f"Expected a tensor shaped (B, C, *spatial), got {tuple(x.shape)} "
                "with no spatial axes."
            )
        kinds = [self._kinds] * len(shape) if isinstance(self._kinds, str) else self._kinds
        if len(kinds) != len(shape):
            raise ValueError(
                f"Transform was configured with {len(kinds)} axis kinds "
                f"{kinds}, but the data has {len(shape)} spatial axes {shape}."
            )
        self._set_matrices([_MATRIX_BUILDERS[k](n) for k, n in zip(kinds, shape)])

    @property
    def is_flat(self) -> bool:
        """Whether ``|B_ki|^2 = 1/d`` everywhere, so ``delta(s) = mean(s)``.

        Measured on the factors once they exist. A deferred transform must
        answer before it has seen data (the methods read this at construction
        to pick the normalization), so it answers by *family*: only the
        Hadamard basis is flat at every size. The two agree for any axis length
        above two -- at n <= 2 every basis here is trivially flat, and calling
        those non-flat only costs an exactly-cancelling renormalization.
        """
        if self.matrices is None:
            kinds = [self._kinds] if isinstance(self._kinds, str) else self._kinds
            return all(k in _FLAT_KINDS for k in kinds)
        return all((M.shape[0] * M**2 - 1.0).abs().max() < 1e-8 for M in self.matrices)

    @property
    def spatial_shape(self) -> tuple:
        if self.matrices is None:
            raise RuntimeError(
                "Deferred transform has no spatial shape yet; it is fixed by "
                "the first tensor passed to forward/inverse/delta (or by "
                "materialize())."
            )
        return tuple(M.shape[0] for M in self.matrices)

    def _mats_for(self, x: Tensor, which: str) -> List[Tensor]:
        """The matrix list ``which`` converted to ``x``'s device and dtype."""
        key = (which, x.device, x.dtype)
        if key not in self._converted:
            base = {
                "forward": self.matrices,
                "inverse": [M.T for M in self.matrices],
                "delta": [Sq.T for Sq in self._squares],
            }[which]
            self._converted[key] = [
                M.to(device=x.device, dtype=x.dtype) for M in base
            ]
        return self._converted[key]

    def _apply(self, x: Tensor, mats: Sequence[Tensor]) -> Tensor:
        """Contract ``mats[a]`` over spatial axis ``a`` of ``(B, C, *spatial)``."""
        n_spatial = len(mats)
        if x.shape[2 : 2 + n_spatial] != self.spatial_shape:
            raise ValueError(
                f"Spatial shape {tuple(x.shape[2:])} does not match transform "
                f"shape {self.spatial_shape}."
            )
        for axis, M in enumerate(mats):
            x = torch.moveaxis(x, 2 + axis, -1)
            x = x @ M.T
            x = torch.moveaxis(x, -1, 2 + axis)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Coefficients ``z = B x`` over the spatial axes."""
        self.materialize(x)
        return self._apply(x, self._mats_for(x, "forward"))

    def inverse(self, z: Tensor) -> Tensor:
        """Field ``x = B^T z`` over the spatial axes."""
        self.materialize(z)
        return self._apply(z, self._mats_for(z, "inverse"))

    def delta(self, s: Tensor) -> Tensor:
        """Implied marginal variances ``Delta_i = sum_k s_k |B_{ki}|^2``.

        ``s`` is a spectral field shaped ``(B, C, *spatial)``; the result has
        the same shape and equals ``diag(B* diag(s) B)`` reshaped to the grid.
        For a separable transform the weight matrix ``A = |B|^2`` factorizes,
        so this is the transpose-square matrices applied axis-by-axis. For a
        flat basis this returns ``mean(s)`` (exactly), broadcast over the grid.
        """
        self.materialize(s)
        if self.is_flat:
            spatial_dims = list(range(2, s.dim()))
            return s.mean(dim=spatial_dims, keepdim=True).expand_as(s)
        return self._apply(s, self._mats_for(s, "delta"))


def haar_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal Haar wavelet matrix (rows = basis functions).

    Rows are ordered scaling function first, then detail levels coarse to
    fine, each level left to right. ``n`` must be a power of two. Haar is the
    canonical *localized* (non-flat) orthogonal basis: fine-scale rows live on
    supports of two grid points, so a diagonal core in the Haar domain implies
    strongly position-dependent marginal variances whenever the spectrum
    varies within a scale.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of two, got {n}.")
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:
        m = H.shape[0]
        top = torch.kron(H, torch.tensor([1.0, 1.0], dtype=dtype))
        bottom = torch.kron(torch.eye(m, dtype=dtype), torch.tensor([1.0, -1.0], dtype=dtype))
        H = torch.cat([top, bottom]) / math.sqrt(2.0)
    return H


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Normalized Hadamard matrix — the canonical real *flat* basis.

    Every entry is ``±1/sqrt(n)``, so ``|B_ki|^2 = 1/n`` everywhere and the
    spectral normalization ``mean(s) = 1`` alone already yields a correlation
    core (``is_flat`` detects this). ``n`` must be a power of two
    (Sylvester construction).
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of two, got {n}.")
    H = torch.ones(1, 1, dtype=dtype)
    block = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype)
    while H.shape[0] < n:
        H = torch.kron(H, block) / math.sqrt(2.0)
    return H


def dct_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal DCT-II matrix (rows = basis functions).

    ``C[k, i] = c_k sqrt(2/n) cos(pi k (2i + 1) / (2n))`` with ``c_0 = 1/sqrt(2)``,
    ``c_k = 1`` otherwise. Real, orthogonal and defined for every ``n`` (unlike
    Haar/Hadamard). The DCT is *not* flat -- its entries vary in magnitude --
    so a diagonal core in the DCT domain requires the renormalization by the
    implied marginal variances.
    Compared to the DFT it imposes reflective rather than periodic boundary
    conditions.
    """
    k = torch.arange(n, dtype=dtype).reshape(-1, 1)
    i = torch.arange(n, dtype=dtype).reshape(1, -1)
    M = math.sqrt(2.0 / n) * torch.cos(math.pi * k * (2 * i + 1) / (2 * n))
    M[0] = M[0] / math.sqrt(2.0)
    return M


def dst_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal DST-II matrix (rows = basis functions).

    ``S[k, i] = c_k sqrt(2/n) sin(pi (k + 1)(2i + 1) / (2n))`` with
    ``c_{n-1} = 1/sqrt(2)``, ``c_k = 1`` otherwise. Real, orthogonal, any ``n``;
    non-flat like the DCT but with odd (sign-flipping) boundary conditions.
    """
    k = torch.arange(n, dtype=dtype).reshape(-1, 1)
    i = torch.arange(n, dtype=dtype).reshape(1, -1)
    M = math.sqrt(2.0 / n) * torch.sin(math.pi * (k + 1) * (2 * i + 1) / (2 * n))
    M[-1] = M[-1] / math.sqrt(2.0)
    return M


_MATRIX_BUILDERS = {
    "dct": dct_matrix,
    "dst": dst_matrix,
    "haar": haar_matrix,
    "hadamard": hadamard_matrix,
}

# Kinds whose entries all have magnitude 1/sqrt(n), for every n. Used to answer
# `is_flat` on a *deferred* transform, which must pick the normalization before
# it knows the axis lengths -- see SeparableOrthogonalTransform.is_flat.
_FLAT_KINDS = {"hadamard"}


def make_separable_transform(kind, shape=None) -> SeparableOrthogonalTransform:
    """Config-friendly factory: one named orthogonal matrix per spatial axis.

    ``kind`` is either a single name applied to every axis or a list with one
    name per axis. Supported names: ``dct``, ``dst``, ``haar``, ``hadamard``,
    ``uhaar``. Intended as a Hydra ``_target_`` so experiment configs can select
    the transform:

        transform:
          _target_: src.methods.transforms.make_separable_transform
          kind: dct

    ``shape`` is the spatial shape of the target field. Leave it ``null``
    (default) to read it off the data at the first forward pass, which is what
    keeps one method config valid across datasets of different rank -- the
    ``(128, 160)`` depth grid, the ``(721, 1440)`` ERA5 grid and a 1-D forecast
    horizon of length ``pred_len`` all work unchanged. Pass it explicitly
    (``shape: [128, 160]``, or ``shape: 96`` for a single axis) only to assert
    the grid: the matrices are then built up front and a mismatch raises on the
    first batch instead of adapting.
    """
    if shape is None:
        return SeparableOrthogonalTransform(kinds=kind)
    shape = [int(shape)] if isinstance(shape, int) else [int(n) for n in shape]
    kinds = [kind] * len(shape) if isinstance(kind, str) else list(kind)
    if len(kinds) != len(shape):
        raise ValueError(
            f"kind list length {len(kinds)} does not match shape length "
            f"{len(shape)}."
        )
    mats = []
    for name, n in zip(kinds, shape):
        if name not in _MATRIX_BUILDERS:
            raise ValueError(
                f"Unknown transform kind {name!r}; expected one of "
                f"{sorted(_MATRIX_BUILDERS)}."
            )
        mats.append(_MATRIX_BUILDERS[name](n))
    return SeparableOrthogonalTransform(mats)


def graph_laplacian_eigenbasis(
    dist,
    length_scale: float = None,
    knn: int = 10,
    normalization: str = "sym",
    dtype: torch.dtype = torch.float64,
    return_eigenvalues: bool = False,
):
    """Graph-Laplacian eigenbasis over an irregular point set (rows = basis fns).

    The graph Fourier transform: the analogue of the DFT when the coordinates do
    not live on a regular grid, so no FFT exists. Given a matrix of pairwise
    *physical* distances (km for a station network), this builds a weighted
    graph, takes its Laplacian, and returns the eigenvectors ordered by
    eigenvalue -- low rows are smooth, large-scale modes, high rows oscillatory,
    exactly the ordering the DFT convention (and hence the core's softmax
    spectrum and ``temperature`` knob) assumes.

    **The graph is dense, not thresholded.** A cutoff graph ``d <= d_max`` is the
    right structure for message passing, where sparsity is the point, but it is a
    poor Laplacian: cutoffs disconnect stragglers, and every connected component
    contributes its own zero eigenvalue, so the ``lambda = 0`` eigenspace becomes
    degenerate and ``eigh`` resolves it as an arbitrary rotation -- the basis
    would then depend on the LAPACK build. A dense kernel has a simple zero
    eigenvalue and a generically non-degenerate spectrum. Nothing is lost: the
    transform matrix is dense either way, and the eigendecomposition is computed
    once, offline.

    Weights are squared-exponential, ``W_ij = exp(-d_ij^2 / 2 l^2)``, rather than
    inverse-distance: ``1/d`` diverges as ``d -> 0`` and has heavy tails, leaving
    ``W`` dominated by the closest pairs.

    Args:
        dist: ``(n, n)`` symmetric matrix of pairwise distances, zero diagonal.
        length_scale: Kernel bandwidth ``l``, in the units of ``dist``. ``None``
            (default) picks the median over nodes of the distance to the
            ``knn``-th nearest neighbour, which adapts to the sampling density.
        knn: Neighbour rank used for the automatic bandwidth.
        normalization: ``"sym"`` for ``L = I - D^-1/2 W D^-1/2`` (default,
            spectrum in ``[0, 2]``); ``"comb"`` for the combinatorial ``D - W``,
            whose zero mode is exactly the constant vector but whose eigenvectors
            drift toward sparsely sampled nodes; ``"rw_sym"`` for the
            density-normalized diffusion-maps variant (Coifman & Lafon, 2006),
            which replaces ``W`` by ``D^-1 W D^-1`` before forming the symmetric
            normalized Laplacian, so the modes approximate Laplace-Beltrami
            eigenfunctions of the underlying domain rather than of the sample.

    :param return_eigenvalues: also return the ascending Laplacian eigenvalues.
        They are the argument a *spectral density* is evaluated at, so the
        Matern filter core (``core_parameterization="matern"``) needs them
        alongside the basis.
    :return: ``(n, n)`` orthogonal matrix whose **rows** are eigenvectors, in
        ascending eigenvalue order and with a deterministic sign convention;
        or ``(U, eigenvalues)`` when ``return_eigenvalues``.
    """
    D_ = torch.as_tensor(dist, dtype=dtype)
    if D_.dim() != 2 or D_.shape[0] != D_.shape[1]:
        raise ValueError(f"dist must be a square matrix, got {tuple(D_.shape)}.")
    n = D_.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 nodes, got {n}.")
    asym = (D_ - D_.T).abs().max()
    if asym > 1e-6 * max(float(D_.max()), 1.0):
        raise ValueError(f"dist must be symmetric (max|d - d^T| = {asym:.2e}).")

    if length_scale is None:
        # Distance to the k-th nearest *other* node, medianed over nodes. The
        # diagonal is zero, so rank k of the sorted row is the k-th neighbour.
        k = min(int(knn), n - 1)
        neighbour = D_.sort(dim=1).values[:, k]
        length_scale = float(neighbour.median())
        if length_scale <= 0:
            raise ValueError(
                "Automatic length_scale came out non-positive; the distance "
                "matrix has duplicate points. Pass length_scale explicitly."
            )
    length_scale = float(length_scale)
    if length_scale <= 0:
        raise ValueError(f"length_scale must be > 0, got {length_scale}.")

    W = torch.exp(-(D_**2) / (2.0 * length_scale**2))
    W.fill_diagonal_(0.0)

    # A bandwidth far below the point spacing underflows the kernel to zero and
    # leaves nodes with no edges at all. Their eigenvector is then a one-hot, and
    # (under "sym") they pile up at a shared eigenvalue whose eigenspace `eigh`
    # splits arbitrarily -- so the basis stops being a function of the geometry.
    # Checked on the raw kernel degrees, which is where the information is.
    raw_deg = W.sum(dim=1)
    if float(raw_deg.max()) <= 0.0 or float(raw_deg.min()) < 1e-12 * float(raw_deg.max()):
        warnings.warn(
            f"length_scale={length_scale:.3g} leaves effectively isolated nodes "
            f"(min/max kernel degree = {float(raw_deg.min()):.2e} / "
            f"{float(raw_deg.max()):.2e}), so the eigenbasis is not uniquely "
            "determined by the geometry. Increase length_scale.",
            RuntimeWarning,
            stacklevel=2,
        )

    if normalization == "rw_sym":
        # Density normalization: divide out the sampling density before forming
        # the Laplacian, so the operator reflects geometry rather than where the
        # stations happen to be.
        deg = W.sum(dim=1).clamp_min(torch.finfo(dtype).tiny)
        W = W / deg.unsqueeze(1) / deg.unsqueeze(0)

    deg = W.sum(dim=1)
    if normalization == "comb":
        L = torch.diag(deg) - W
    elif normalization in ("sym", "rw_sym"):
        inv_sqrt = deg.clamp_min(torch.finfo(dtype).tiny).rsqrt()
        L = torch.eye(n, dtype=dtype) - inv_sqrt.unsqueeze(1) * W * inv_sqrt.unsqueeze(0)
    else:
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected 'sym', 'comb' "
            "or 'rw_sym'."
        )
    L = 0.5 * (L + L.T)  # kill the asymmetry accumulated in floating point

    evals, evecs = torch.linalg.eigh(L)  # ascending, columns are eigenvectors
    U = evecs.T.contiguous()  # rows are eigenvectors, as the transform expects

    # `eigh` fixes eigenvectors only up to sign; pin it so the basis is a
    # deterministic function of `dist` and a rebuilt model matches a checkpoint.
    pivot = U.abs().argmax(dim=1, keepdim=True)
    U = U * torch.where(U.gather(1, pivot) < 0, -1.0, 1.0).to(dtype)

    # A connected graph has a *simple* zero eigenvalue; multiplicity c means c
    # disconnected clusters, whose zero-eigenspace `eigh` resolves as an
    # arbitrary rotation -- see the docstring. This catches the split-into-
    # clusters case; the degree check above catches fully isolated nodes, which
    # under "sym" land at eigenvalue 1 rather than 0 and would slip past here.
    if float(evals[1]) < 1e-8:
        warnings.warn(
            f"Graph Laplacian has a degenerate zero eigenvalue "
            f"(lambda_1 = {float(evals[1]):.2e}), i.e. the graph splits into "
            f"disconnected clusters, so the basis is not uniquely determined. "
            f"length_scale={length_scale:.3g} is too small for this point set.",
            RuntimeWarning,
            stacklevel=2,
        )
    return (U, evals) if return_eigenvalues else U


def make_graph_transform(
    meta_path: str,
    length_scale: float = None,
    knn: int = 10,
    normalization: str = "sym",
    dist_key: str = "dist",
) -> SeparableOrthogonalTransform:
    """Config-friendly factory: a graph Fourier transform over a point set.

    Reads the pairwise distance matrix from the ``meta.npz`` written by
    ``src/data/build_euppbench.py`` -- the same file the multigraph backbone
    reads for its message-passing edges and the variogram score for its
    inverse-distance weights, so the three consumers of station geometry cannot
    drift apart.

    Note this is a separate factory from :func:`make_separable_transform` rather
    than another entry in its ``kind`` table: a graph basis is determined by a
    distance matrix, not by an axis length, so it does not fit that
    ``(kind, shape)`` signature.

        transform:
          _target_: src.methods.transforms.make_graph_transform
          meta_path: data/euppbench_24h/meta.npz

    :return: A single-axis :class:`SeparableOrthogonalTransform`, i.e. a flat
        event axis of length ``n``. The basis is localized, so ``is_flat`` is
        False and ``SCORE`` selects the renormalization by the implied
        marginal variances automatically.
    """
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"meta_path {meta_path!r} is not a file. It should point at the "
            "meta.npz written by src/data/build_euppbench.py (e.g. "
            "data/euppbench_24h/meta.npz), not at the dataset directory."
        )
    with np.load(meta_path, allow_pickle=False) as f:
        if dist_key not in f:
            raise KeyError(
                f"{meta_path!r} has no {dist_key!r} array; found "
                f"{sorted(f.keys())}."
            )
        dist = np.asarray(f[dist_key], dtype=np.float64)
    U, evals = graph_laplacian_eigenbasis(
        dist,
        length_scale=length_scale,
        knn=knn,
        normalization=normalization,
        return_eigenvalues=True,
    )
    return SeparableOrthogonalTransform([U], eigenvalues=evals)


def unbalanced_haar_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal unbalanced-Haar matrix for arbitrary ``n`` (rows = basis fns).

    Recursive binary splits of the index range into (ceil, floor) halves; each
    split contributes one two-level wavelet row, orthonormal by construction
    (constant ``a`` on the left part, ``-b`` on the right, zero mean, unit
    norm). For ``n`` a power of two this reproduces the classical Haar system.
    Rows are ordered scaling function first, then breadth-first coarse to
    fine. See Fryzlewicz (2007), "Unbalanced Haar technique".
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    rows = [torch.full((n,), 1.0 / math.sqrt(n), dtype=dtype)]
    queue = [(0, n)]
    while queue:
        start, length = queue.pop(0)
        if length < 2:
            continue
        left = (length + 1) // 2
        right = length - left
        a = math.sqrt(right / (left * (left + right)))
        b = math.sqrt(left / (right * (left + right)))
        row = torch.zeros(n, dtype=dtype)
        row[start : start + left] = a
        row[start + left : start + length] = -b
        rows.append(row)
        queue.append((start, left))
        queue.append((start + left, right))
    return torch.stack(rows)


_MATRIX_BUILDERS["uhaar"] = unbalanced_haar_matrix


def dct1_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal DCT-I matrix (rows = basis functions).

    Where the DCT-II mirrors the signal about the half-sample point outside
    each end, the DCT-I mirrors it about the end sample itself: whole-sample
    even symmetry rather than half-sample even. Both are reflective, so this
    isolates *which* reflection the data prefer -- the DCT-II/DCT-I pair
    varies the boundary convention while holding everything else fixed.
    Requires ``n >= 2``.
    """
    if n < 2:
        raise ValueError(f"DCT-I needs n >= 2, got {n}.")
    k = torch.arange(n, dtype=dtype).reshape(-1, 1)
    i = torch.arange(n, dtype=dtype).reshape(1, -1)
    M = torch.cos(math.pi * k * i / (n - 1))
    # Endpoints carry half weight in the DCT-I quadrature; scaling the first
    # and last column by 1/sqrt(2) and normalizing rows makes it orthonormal.
    M[:, 0] /= math.sqrt(2.0)
    M[:, -1] /= math.sqrt(2.0)
    M = M * math.sqrt(2.0 / (n - 1))
    M[0] /= math.sqrt(2.0)
    M[-1] /= math.sqrt(2.0)
    return M


def dct4_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Orthonormal DCT-IV matrix (rows = basis functions).

    Half-sample even at the left boundary and half-sample *odd* at the right:
    the mixed convention, and the one underlying lapped transforms. Defined
    for every ``n`` and self-inverse.
    """
    k = torch.arange(n, dtype=dtype).reshape(-1, 1)
    i = torch.arange(n, dtype=dtype).reshape(1, -1)
    return math.sqrt(2.0 / n) * torch.cos(
        math.pi * (2 * k + 1) * (2 * i + 1) / (4 * n)
    )


def random_orthogonal_matrix(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """A fixed random orthogonal matrix — the structureless null control.

    Unitary but carrying no notion of frequency, locality or boundary. If a
    diagonal core in this basis matched the DCT, the benefit of transforming
    would come from diagonalizing in *some* rotated frame rather than from the
    structure of the chosen one, and the argument for picking a basis would
    collapse. The seed is fixed so the basis is a property of the method and
    not of the run: two runs with the same ``n`` see the same matrix, and the
    seed variation in an experiment stays what it is elsewhere, namely
    initialization and data order.
    """
    generator = torch.Generator().manual_seed(0xC0FFEE + n)
    a = torch.randn(n, n, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Fix the sign convention so QR returns a unique matrix across versions.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(dtype).T.contiguous()


_MATRIX_BUILDERS["dct1"] = dct1_matrix
_MATRIX_BUILDERS["dct4"] = dct4_matrix
_MATRIX_BUILDERS["random"] = random_orthogonal_matrix
