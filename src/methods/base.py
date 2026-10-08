import math
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple, Union

import scoringrules as sr
import torch
from torch import Tensor


def masked_gaussian_kernel_score(
    A: Tensor, r: Tensor, gamma2: float, mask: Optional[Tensor] = None
) -> Tensor:
    """Gaussian kernel score of ``N(0, A)`` against residual ``r``, on a coordinate subset.

    Every parameterization's closed-form kernel score has the same shape,

        fac1 = exp(-0.5 logdet A1 - rᵀ A1⁻¹ r / gamma²),   A1 = I + 2A/gamma²
        fac2 = exp(-0.5 logdet A2),                        A2 = I + 4A/gamma²
        score = 0.5 (1 + fac2) - fac1

    differing only in *which* dense matrix ``A`` and residual ``r`` they feed it
    (see :meth:`PredictiveMethod._dense_score_terms`). Written densely like this
    the score restricted to an observed coordinate subset is one line: masking
    ``A`` as

        A <- (m mᵀ) ⊙ A + diag(1 - m)

    makes it ``blockdiag(A_OO, I)`` under permutation, so the log-determinant
    collapses to ``logdet A_OO`` and, with ``r`` zeroed on the masked
    coordinates, the quadratic form collapses to ``r_Oᵀ A_OO⁻¹ r_O``. The result
    is *exactly* the score of the marginal on the observed coordinates -- which
    is the right target, since a Gaussian's marginal is the Gaussian on the
    submatrix.

    This matters because the structured closed forms cannot express it: a
    principal submatrix of a circulant (or of ``Bᵀ diag(s) B`` for any fixed
    orthogonal ``B``) is not itself of that form, so :class:`SpectralDiag` and
    SCORE have no in-parameterization masking. Going dense costs a
    ``(B, V, V)`` Cholesky, which is negligible at the station-graph scale
    (``V`` in the hundreds) and infeasible at the gridded scale -- hence the
    ``dense_mask_max_dim`` guard on the caller.

    :param A: ``(B, V, V)`` symmetric positive-semidefinite matrix.
    :param r: ``(B, V)`` residual.
    :param gamma2: squared kernel bandwidth.
    :param mask: ``(B, V)`` with 1 on observed coordinates, or ``None`` for the
        fully-observed case (which then reproduces the structured closed form).
    :return: ``(B,)`` per-sample score.
    """
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    if mask is None:
        pair_mask, resid = None, r
    else:
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        resid = r * mask

    def _terms(scale: float, with_quadratic: bool):
        mat = eye + scale * A
        if pair_mask is not None:
            mat = pair_mask * mat + torch.diag_embed(1.0 - mask)
        chol = torch.linalg.cholesky(mat)
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        if not with_quadratic:
            return log_det, None
        solved = torch.cholesky_solve(resid.unsqueeze(-1), chol).squeeze(-1)
        return log_det, (resid * solved).sum(dim=-1)

    log_det1, quadratic = _terms(2.0 / gamma2, True)
    log_det2, _ = _terms(4.0 / gamma2, False)

    fac1 = torch.exp(-0.5 * log_det1 - quadratic / gamma2)
    fac2 = torch.exp(-0.5 * log_det2)
    return 0.5 * (1.0 + fac2) - fac1


def masked_gaussian_nll(
    A: Tensor, r: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    """Negative log-likelihood of ``N(0, A)`` at residual ``r``, on a coordinate subset.

    The same masking identity as :func:`masked_gaussian_kernel_score`, applied to
    the likelihood instead: with ``A <- (m mᵀ) ⊙ A + diag(1 - m)`` the masked
    block contributes a unit determinant and, with ``r`` zeroed there, nothing to
    the quadratic form, so

        -log p(y_O) = 0.5 ( |O| log 2pi + logdet A_OO + r_Oᵀ A_OO⁻¹ r_O )

    falls out with the dimension term counting only the observed coordinates.

    Note this is *less* forgiving numerically than the kernel score, which
    factorizes through ``I + 2A/gamma²`` and is therefore always well
    conditioned. Here ``A_OO`` is inverted directly, so a near-singular
    predictive covariance surfaces as a Cholesky failure rather than a quietly
    large score -- which is the honest behaviour, since the likelihood really
    does diverge there.
    """
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    if mask is None:
        mat, resid = A, r
        dim = torch.full(r.shape[:1], float(A.shape[-1]), device=A.device, dtype=A.dtype)
    else:
        mat = mask.unsqueeze(-1) * mask.unsqueeze(-2) * A + torch.diag_embed(1.0 - mask)
        resid = r * mask
        dim = mask.sum(dim=-1)

    chol = torch.linalg.cholesky(mat)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    solved = torch.cholesky_solve(resid.unsqueeze(-1), chol).squeeze(-1)
    maha = (resid * solved).sum(dim=-1)
    return 0.5 * (dim * math.log(2 * math.pi) + log_det + maha)


def _as_list(x: Union[float, Sequence[float]]) -> List[float]:
    """Normalize a bandwidth spec to a plain ``list[float]``.

    A scalar becomes a one-element list; a list/tuple (or any non-string
    sequence, e.g. an OmegaConf ``ListConfig``) is copied element-wise to plain
    floats. This lets ``gamma``/``gamma_eval`` accept either a single factor or a
    list of factors uniformly.
    """
    if isinstance(x, (list, tuple)) or (
        hasattr(x, "__iter__") and not isinstance(x, (str, bytes))
    ):
        return [float(v) for v in x]
    return [float(x)]

class PredictiveMethod(ABC):

    LOSSES = (
        "mse",
        "mae",
        "nll",
        "kernel_score",
        "dense_kernel_score",
        "sample_kernel_score",
        "energy_score",
        "crps",
        "marginal_kernel_score",
    )

    # Losses with no masked implementation. The coordinate-separable scores mask
    # trivially, and `kernel_score` / `nll` go through the dense path, but the
    # sliced `energy_score` does not: it estimates the score by projecting onto
    # random directions, so masking means drawing those directions from the
    # observed subspace (zero the masked coordinates before normalizing) and
    # taking the dimension constant at the per-sample observed count. That is a
    # per-family change -- each family computes the projected variance vᵀΣv from
    # its own structure -- and it is only reachable by explicitly selecting this
    # loss on a dataset with missing targets, so for now it fails loudly rather
    # than returning NaN. Note the *eval-time* energy score is unaffected: it
    # uses the ensemble estimator, which masks exactly (see
    # src/evaluation/metrics.py).
    MASK_UNSUPPORTED_LOSSES = ("energy_score",)

    # Largest event dimension for which the masked scores may build a dense
    # (B, V, V) matrix. Station networks are in the hundreds; a gridded target
    # (ERA5 is V = 35 200, i.e. 4.6 GiB per matrix) must never reach this path,
    # and since those datasets have no missing targets it never does.
    dense_mask_max_dim = 1024

    def __init__(
        self,
        loss: str,
        out_channels: int,
        eps: float,
        n_train_samples: int,
        n_eval_samples: int,
        gamma: Union[float, Sequence[float]],
        gamma_eval: Optional[Union[float, Sequence[float]]] = None,
    ):
        if loss not in self.LOSSES:
            raise ValueError(
                f"Unknown loss {loss!r}; expected one of {self.LOSSES}"
            )
        self.loss = loss
        self._out_channels = out_channels
        self.eps = eps
        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples
        # Resolved kernel bandwidths, stored as *lists*. `gamma` is used in
        # training mode (train/val); `gamma_eval` in eval mode. They differ
        # because the bandwidth is scaled by the data dimension `d` as
        # gamma = sqrt(gamma_* * d) (see train/evaluate), and the train/eval
        # scaling factors are configured separately. When `gamma_eval` is not
        # given it falls back to `gamma`. A list of bandwidths turns the kernel
        # score into a multi-bandwidth (mixture-kernel) score: the closed-form
        # score is evaluated at each bandwidth and the results are averaged (see
        # `_mean_over_bandwidths`); a single value (or a one-element list)
        # reproduces the plain single-bandwidth score exactly.
        self._gamma_train = _as_list(gamma)
        self._gamma_eval = (
            list(self._gamma_train) if gamma_eval is None else _as_list(gamma_eval)
        )
        # Index of the currently-active bandwidth within the active list. The
        # `_mean_over_bandwidths` loop sweeps it so `gamma`/`gamma2` stay scalar
        # for each per-bandwidth score body; it is otherwise 0.
        self._gamma_idx = 0
        # Mode flag selecting which sample count `n_samples` (and bandwidth list
        # `gamma`/`gamma2`) returns. Defaults to eval mode; the Lightning model
        # flips it to True during train/val steps.
        self.training = False

    @property
    def _active_gammas(self) -> List[float]:
        """The bandwidth list for the current mode (train vs eval)."""
        return self._gamma_train if self.training else self._gamma_eval

    @property
    def gamma(self) -> float:
        """Currently-active kernel bandwidth (a scalar).

        Returns the bandwidth at :attr:`_gamma_idx` within the active
        (train/eval) list, so per-bandwidth score bodies see a single value.
        """
        return self._active_gammas[self._gamma_idx]

    @property
    def gamma2(self) -> float:
        """Squared currently-active kernel bandwidth (a scalar)."""
        return self.gamma**2

    @property
    def gamma_list(self) -> List[float]:
        """All kernel bandwidths for the current mode (train vs eval)."""
        return list(self._active_gammas)

    @property
    def gamma2_list(self) -> List[float]:
        """Squared kernel bandwidths for the current mode."""
        return [g**2 for g in self._active_gammas]

    def _mean_over_bandwidths(self, fn: Callable[..., Tensor], *args, **kwargs) -> Tensor:
        """Average a per-bandwidth score body over all active bandwidths.

        Sweeps :attr:`_gamma_idx` over the active (train/eval) bandwidth list,
        calling ``fn`` once per bandwidth (which reads the scalar
        :attr:`gamma2` for the active index) and averaging the results. With a
        one-element list this is the identity, so single-bandwidth scores are
        reproduced exactly. The index is restored to 0 afterwards.
        """
        gammas = self._active_gammas
        total: Optional[Tensor] = None
        try:
            for i in range(len(gammas)):
                self._gamma_idx = i
                val = fn(*args, **kwargs)
                total = val if total is None else total + val
        finally:
            self._gamma_idx = 0
        return total / len(gammas)

    @property
    def n_samples(self) -> int:
        """Number of Monte-Carlo samples for sampling-based scores.

        Returns the smaller training count while in training mode and the
        larger evaluation count otherwise.
        """
        return self.n_train_samples if self.training else self.n_eval_samples

    @property
    @abstractmethod
    def group_widths(self) -> dict:
        """Absolute channel width of each prediction group, in stacking order.

        Keys are the fixed ordered tuple ``("mu", "sigma", "other")``; a value of
        ``None`` means that group is absent for this method. Widths already
        include the raw dataset channel count ``C`` (``= self._out_channels``),
        e.g. ``{"mu": C, "sigma": C, "other": rank * C}``.

        This is the single source of truth for how the head output is laid out:
        the backbone reads it to decide which heads to build and how wide each
        one is, and :meth:`process_output` un-stacks the channel dim into the
        same groups (group-inner layout, ``reshape(B, C, G, ...)``).
        """
        pass

    @property
    def out_channels(self) -> int:
        """Total channel count the backbone head must emit.

        Derived from :attr:`group_widths` so the flat count and the per-group
        widths can never disagree.
        """
        return sum(w for w in self.group_widths.values() if w is not None)

    @property
    def group_denorm(self) -> dict:
        """How each prediction group transforms when the data is rescaled by ``a``.

        Keyed like :attr:`group_widths`. Only backbones that normalize their input
        and have to undo it on the output read this -- currently
        :class:`~src.models.backbones.patchtst.PatchTST`, whose RevIN layer
        standardizes each input window and must denormalize each group by its own
        rule. Other backbones ignore it.

        * ``"loc"``  -- a location: ``a * x + m`` (the mean).
        * ``"var"``  -- a variance: scales by ``a**2``. The backbone emits
          pre-``softplus`` values, so it applies the rescaling *through* the
          activation the method will apply.
        * ``"scale"`` -- scales linearly by ``a`` (e.g. LorD's low-rank factor
          ``U``, since ``Sigma = diag(d) + U Uᵀ`` must scale by ``a**2``).
        * ``"none"`` -- scale-free (e.g. SCORE's unit-mean spectral
          *correlation* core, whose scale is carried by ``sigma`` instead).

        The default suits every method whose ``"other"`` group is scale-free;
        methods with a linearly-scaling ``"other"`` override it.
        """
        return {"mu": "loc", "sigma": "var", "other": "none"}

    @property
    def other_zero_init(self) -> bool:
        """Whether the backbone's ``"other"`` head may start at exactly zero.

        Zero-initializing the head projections makes the model start at the
        neutral prediction (``mu = 0``, ``diag = softplus(0)``, a flat spectral
        core) -- the right default, but only sound when the loss has a
        non-vanishing gradient there. A group that enters the covariance
        *quadratically* breaks this: for LorD's ``Sigma = diag(d) + U Uᵀ``
        the gradient ``dL/dU`` is proportional to ``U`` itself, so ``U = 0`` is
        an exact stationary point and a zero-init head never receives any
        gradient -- it silently stays dead for the whole run. Such methods
        override this to ``False`` and the backbone breaks the symmetry with a
        small random init instead (small enough that the initial prediction is
        still effectively neutral).
        """
        return True

    @property
    def noise(self) -> Optional[dict]:
        """Latent-noise injection the backbone must build, or ``None``.

        Every Gaussian family here draws its samples *after* the head, from
        parameters the head emits, so the backbone is deterministic and this is
        ``None``. A sample-based generative method has no such parameters: its
        randomness has to enter *inside* the network, so the backbone splices a
        :class:`~src.models.backbones.noise.GaussianNoiseInjector` between its
        trunk and its head. The returned dict is forwarded verbatim as the
        backbone's ``noise`` argument (see
        :class:`~src.methods.sample_based.SampleBased` for the keys).

        Read at construction time alongside :attr:`group_widths`,
        :attr:`group_denorm` and :attr:`other_zero_init` -- the four properties
        by which a method configures its backbone.
        """
        return None

    @property
    def samples_from_backbone(self) -> bool:
        """Whether ``pred`` already carries the ensemble, drawn by the backbone.

        ``False`` for every closed-form family: ``pred`` holds distribution
        *parameters* and :meth:`get_samples` can draw any number of members from
        them, as often as asked, for free.

        ``True`` for the sample-based generative method, where each member costs
        a forward pass through the noise injection and the head. Two things
        follow, and callers that draw samples must honour both:

        * the member count is fixed when the backbone runs, so it is
          :class:`~src.models.gaussian_predictive.LightningPredictiveModel`'s
          ``forward(x, n_samples=...)`` -- not :meth:`get_samples` -- that
          decides ``M``;
        * :meth:`get_samples` can only hand back (a prefix of) what ``pred``
          already holds, so asking it for more members is an error rather than a
          fresh draw.
        """
        return False

    @property
    def has_closed_form_pit(self) -> bool:
        """Whether linear functionals of the predictive have a closed-form PIT.

        ``True`` for every Gaussian family: ``aᵀX`` is univariate Gaussian with
        variance :meth:`quadratic_form`, so the marginal and location
        calibration diagnostics of :mod:`src.evaluation.calibration` are exact
        normal CDFs and need no draws.

        ``False`` for a nonparametric predictive, where the same diagnostics
        must be taken as *ranks* against the ensemble instead. Both routes
        produce a PIT that is uniform under calibration, so the npz payload is
        unchanged; only the estimator differs, and the file records which was
        used.
        """
        return True

    @staticmethod
    def reduce(loss:Tensor, reduction:str = "mean")->Tensor:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    # ---- missing targets ---------------------------------------------------
    #
    # Station observations are missing for a few percent of station-days, with
    # no day fully reported, so incomplete samples cannot simply be dropped.
    # The scores instead evaluate the predictive *marginal* on the observed
    # coordinates, which for a Gaussian is exact. Datasets without missing
    # targets take an early exit and are numerically untouched.

    @staticmethod
    def _mask_of(y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Split a target into ``(y_filled, mask)``.

        Returns ``mask=None`` (and ``y`` unchanged) when nothing is missing, so
        every score keeps its original fast path on complete data. Otherwise the
        NaNs are replaced by zeros -- the scores multiply those coordinates by a
        zero mask, and a finite placeholder keeps ``0 * value`` from becoming
        ``0 * NaN = NaN`` in both the value and the gradient.
        """
        finite = torch.isfinite(y)
        if bool(finite.all()):
            return y, None
        return torch.nan_to_num(y, nan=0.0), finite.flatten(start_dim=1).to(y.dtype)

    @staticmethod
    def _masked_mean(per_coord: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Per-sample mean of a per-coordinate score over the observed coordinates.

        Normalizing by the observed count rather than by ``V`` keeps the score
        on a per-station scale, so it stays comparable across samples (and with
        published numbers) as the number of reporting stations varies.
        """
        flat = per_coord.flatten(start_dim=1)
        if mask is None:
            return flat.mean(dim=1)
        return (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _check_dense_mask(self, mask: Tensor) -> None:
        """Guard the dense masked path against gridded-scale event dimensions."""
        self._check_dense_dim(mask.shape[-1])

    def _check_dense_dim(self, dim: int) -> None:
        """Guard any dense ``(B, V, V)`` path against gridded-scale ``V``.

        Split out of :meth:`_check_dense_mask` because `dense_kernel_score` goes
        dense whether or not there is a mask, so it must check the event
        dimension directly.
        """
        if dim > self.dense_mask_max_dim:
            raise NotImplementedError(
                f"Masked multivariate scores build a dense ({dim}, {dim}) matrix, "
                f"which exceeds dense_mask_max_dim={self.dense_mask_max_dim}. "
                "Missing targets are only supported for small event dimensions "
                "(station networks), not gridded fields."
            )

    def _dense_score_terms(self, pred, y: Tensor) -> Tuple[Tensor, Tensor]:
        """The dense ``(A, r)`` whose kernel score equals this method's closed form.

        ``A`` is the ``(B, V, V)`` matrix the score is taken against and ``r``
        the ``(B, V)`` residual; see :func:`masked_gaussian_kernel_score`. The
        default recovers the physical covariance from the two accessors every
        family already provides for the variogram score,

            Sigma_ij = 0.5 (Sigma_ii + Sigma_jj - Var(X_i - X_j)),

        so a family needs no extra code unless the matrix its closed form scores
        is *not* the physical covariance -- as for SCORE, which scores its
        correlation core against the standardized residual and overrides this
        accordingly.
        """
        flat_y = y.flatten(start_dim=1)
        rows = torch.arange(flat_y.shape[1], device=y.device)
        var = self.get_marg_std(pred).flatten(start_dim=1).pow(2)
        pair = self.pair_diff_var(pred, rows)
        cov = 0.5 * (var[:, rows].unsqueeze(-1) + var.unsqueeze(-2) - pair)
        return cov, (self.get_mean(pred) - y).flatten(start_dim=1)

    def _deployed_dense_terms(self, pred, y: Tensor) -> Tuple[Tensor, Tensor]:
        """The dense ``(A, r)`` of the predictive ``N(mu, Sigma_P)`` itself.

        Same contract as :meth:`_dense_score_terms` -- a ``(B, V, V)`` matrix and
        a ``(B, V)`` residual for :func:`masked_gaussian_kernel_score` -- but
        pinned to the predictive covariance and to the residual in *physical*
        space. For every family whose closed form already scores
        the physical covariance the two coincide, so the default just delegates.
        Only :class:`~src.methods.score.SCORE` differs: its
        `_dense_score_terms` scores the correlation core against the
        standardized residual, and that surrogate is not a proper score of the
        predictive once the standardization depends on the core (see
        `dense_kernel_score`).
        """
        return PredictiveMethod._dense_score_terms(self, pred, y)

    @abstractmethod
    def process_output(self, output_head):
        pass

    @abstractmethod
    def get_samples(self, pred, n_samples: Optional[int] = None, flatten: bool = False):
        """Draw samples from the predictive multivariate Gaussian.

        :param pred: the processed head output (as returned by
            :meth:`process_output`).
        :param n_samples: number of ensemble members ``M`` to draw. Defaults to
            :attr:`n_samples`, which is ``n_eval_samples`` outside training mode
            and ``n_train_samples`` inside it. Note the sampling-based eval
            metrics mask missing targets by zeroing the affected coordinates in
            both the samples and the observation, which is exact for every
            family (a point mass at the observation contributes nothing to any
            norm- or kernel-based score).
        :param flatten: if ``True`` the event axes are collapsed, returning
            ``(B, V, M)`` with ``V = C * prod(spatial)`` (the layout the
            sampling-based scores consume). If ``False`` (the default) samples
            keep the original event shape ``(B, C, D1, ..., DN, M)`` so they can
            be reused directly, e.g. for plotting.

        Samples are returned on ``pred``'s device and dtype.
        """
        pass

    def _draw_samples(self, pred, n_samples: Optional[int] = None, flatten: bool = False):
        """:meth:`get_samples` with the gradient kept: the reparameterized draw.

        Every parametric family decorates `get_samples` with ``@torch.no_grad()``
        -- correct for eval and plotting, fatal for a sampling-based training
        loss, which would then optimize a constant. Families that support such a
        loss expose the undecorated body here (see
        :class:`~src.methods.score.SCORE`); the rest fail loudly
        rather than silently training on a zero gradient.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no differentiable sampler, so the "
            "sampling-based losses are unavailable. Split its `get_samples` "
            "into a `_draw_samples` body plus a @torch.no_grad() wrapper to "
            "enable them."
        )

    @abstractmethod
    def get_marg_std(self, pred) -> Tensor:
        """Per-grid-point marginal standard deviation of the predictive Gaussian.

        Returns the closed-form marginal (pointwise) standard deviation
        ``sqrt(diag(Sigma))`` implied by the processed head output ``pred`` (as
        returned by :meth:`process_output`), i.e. the square root of the
        diagonal of the predictive covariance. Unlike :meth:`get_samples`, this
        is exact -- every method's covariance has a diagonal available in closed
        form (identity for the diagonal method, the Woodbury row-norm for the
        low-rank one, the Parseval mean for the circulant ones, ...).

        The result has the mean's event shape ``(B, C, D1, ..., DN)`` and is
        returned on ``pred``'s device and dtype.
        """
        pass

    def get_mean(self, pred) -> Tensor:
        """Predictive mean ``mu`` with event shape ``(B, C, D1, ..., DN)``.

        Every method but :class:`~src.methods.deterministic.Deterministic` stacks
        its groups on a trailing axis and keeps ``mu`` at index 0; the
        deterministic head emits the mean bare. Routing the access through this
        accessor keeps that layout difference out of the scores.
        """
        return pred[..., 0]

    def pair_diff_var(self, pred, rows: Tensor) -> Tensor:
        """Variance of the pairwise difference ``X_i - X_j``, for a block of ``i``.

        Returns ``Var(X_i - X_j) = Sigma_ii + Sigma_jj - 2 Sigma_ij`` for every
        ``i`` in ``rows`` against *all* ``j``, i.e. shape
        ``(B, len(rows), V)`` with ``V = C * prod(spatial)`` the flattened event
        size and ``rows`` a ``LongTensor`` of flat event indices.

        This is the one quantity the p=1 variogram score needs from the
        covariance (see :mod:`src.evaluation.variogram`): with
        ``X_i - X_j ~ N(mu_i - mu_j, pair_diff_var)`` the score's expectation term
        is a folded-normal mean. Every parameterization here has a closed form
        that avoids ever materializing ``Sigma`` -- a ``d x d`` matrix is
        ~4.6 GiB at ERA5's ``d = 35 200`` -- because the difference variance
        collapses to marginal variances plus a cheap correction: a squared
        distance between low-rank rows, or a lag lookup into the circulant
        autocovariance. Callers block over ``rows`` to bound the ``(len(rows), V)``
        working set.

        The result is returned on ``pred``'s device and dtype, is non-negative,
        and is exactly zero on the self-pairs ``i == j``.
        """
        v = self._pair_diff_var(pred, rows).clamp_min(0.0)
        # `Var(X_i - X_i) = 0`, but the per-family formulas are written for the
        # off-diagonal (e.g. the diagonal method's `d_i + d_j` is `2 d_i` at
        # `i == j`), so pin the self-pairs here rather than in five places. The
        # clamp above absorbs the rounding of otherwise-nonnegative expressions
        # such as `d_i c_0 + d_j c_0 - 2 sqrt(d_i d_j) c_ij`, which is
        # non-negative by Cauchy-Schwarz but can land at -1e-17.
        k = torch.arange(rows.shape[0], device=v.device)
        v[:, k, rows] = 0.0
        return v

    @abstractmethod
    def _pair_diff_var(self, pred, rows: Tensor) -> Tensor:
        """Per-family ``Sigma_ii + Sigma_jj - 2 Sigma_ij``, valid off-diagonal.

        Implemented by each parameterization with its own cheapest algebra; the
        self-pairs and any negative rounding are fixed up by
        :meth:`pair_diff_var`, which is what callers use.
        """
        pass

    def quadratic_form(self, pred, a: Tensor) -> Tensor:
        """The quadratic form ``aᵀ Sigma a``, i.e. ``Var(aᵀX)``, shape ``(B,)``.

        ``a`` carries the *event* shape ``(B, C, D1, ..., DN)`` -- the mean's
        layout, not the flattened one -- because the SpectralDiag and SCORE forms
        contract it with an FFT or a separable transform over the spatial axes
        and would otherwise have to undo a reshape. A leading axis of size 1
        broadcasts over the batch, so a weight vector shared by every case can be
        passed as ``(1, C, D1, ..., DN)``.

        This is what the linear pre-rank functions of
        :mod:`src.evaluation.calibration` need from the covariance: ``aᵀX`` is
        univariate Gaussian with variance ``aᵀ Sigma a``, so its PIT is available
        in closed form. Like :meth:`pair_diff_var` it is implemented per family
        without ever materializing ``Sigma`` -- a ``V x V`` matrix is ~380 GiB at
        depth's ``V = 307 200`` -- and costs ``O(V)`` memory on every path.

        The result is non-negative and returned on ``pred``'s device and dtype.
        """
        mu = self.get_mean(pred)
        if a.shape[1:] != mu.shape[1:]:
            raise ValueError(
                f"`a` has event shape {tuple(a.shape[1:])}, expected "
                f"{tuple(mu.shape[1:])} (the mean's event shape)."
            )
        a = a.to(device=mu.device, dtype=mu.dtype).expand_as(mu)
        # Non-negative by construction, but the per-family algebra sums terms of
        # mixed sign (e.g. `diag/Delta` times a core coefficient), so a form that
        # should be 0 can land at -1e-18. Clamp once here rather than in five
        # places, exactly as `pair_diff_var` does.
        return self._quadratic_form(pred, a).clamp_min(0.0)

    @abstractmethod
    def _quadratic_form(self, pred, a: Tensor) -> Tensor:
        """Per-family ``aᵀ Sigma a`` for an event-shaped, batch-expanded ``a``.

        Shape checking, broadcasting and the non-negativity clamp are handled by
        :meth:`quadratic_form`, which is what callers use.
        """
        pass

    def get_train_loss(self) -> Callable[..., Tensor]:
        """The selected loss, wrapped so unsupported ones reject missing targets.

        Most losses handle missing targets (see :meth:`_mask_of`); those in
        :attr:`MASK_UNSUPPORTED_LOSSES` do not, and silently returning NaN for a
        whole training run is the worst possible failure mode, so they are
        checked on the way in.
        """
        fn = getattr(self, self.loss)
        if self.loss not in self.MASK_UNSUPPORTED_LOSSES:
            return fn

        loss_name = self.loss

        def guarded(pred, y, *args, **kwargs):
            if not bool(torch.isfinite(y).all()):
                raise NotImplementedError(
                    f"Loss {loss_name!r} has no masked implementation, but the "
                    "target contains missing values. Use 'crps', 'mse', 'mae', "
                    "'kernel_score' or 'marginal_kernel_score' on this dataset."
                )
            return fn(pred, y, *args, **kwargs)

        return guarded

    def mse(self, pred, y, reduction: str = "mean") -> Tensor:
        """Mean squared error of the predictive mean, over observed coordinates."""
        y, mask = self._mask_of(y)
        per_coord = (self.get_mean(pred) - y).pow(2)
        return self.reduce(self._masked_mean(per_coord, mask), reduction)

    def mae(self, pred, y, reduction: str = "mean") -> Tensor:
        """Mean absolute error of the predictive mean, over observed coordinates."""
        y, mask = self._mask_of(y)
        per_coord = (self.get_mean(pred) - y).abs()
        return self.reduce(self._masked_mean(per_coord, mask), reduction)

    def crps(self, pred, y, reduction: str = "mean") -> Tensor:
        """Marginal CRPS, averaged over the observed coordinates.

        The per-coordinate values come from :meth:`_crps_per_coord`, which each
        family supplies from its own marginal standard deviation; the masking
        and reduction are shared here.
        """
        y, mask = self._mask_of(y)
        return self.reduce(self._masked_mean(self._crps_per_coord(pred, y), mask), reduction)

    @abstractmethod
    def _crps_per_coord(self, pred, y) -> Tensor:
        """Per-coordinate CRPS, shape ``(B, C, D1, ..., DN)`` (no reduction)."""
        pass

    def nll(self, pred, y, reduction: str = "mean") -> Tensor:
        """Negative log-likelihood, restricted to the observed coordinates.

        On complete data this delegates to the family's own structured form
        (:meth:`_nll_complete`), which exploits the parameterization and never
        materializes ``Sigma``. With missing targets it falls back to the dense
        masked likelihood, since a principal submatrix of a structured covariance
        is not itself structured.

        The dense form needs one thing the kernel score does not: a family
        whose :meth:`_dense_score_terms` works in *standardized* coordinates
        (SCORE) returns the correlation core, not the predictive covariance, so
        the change-of-variables Jacobian of that standardization has to be added
        back --
        :meth:`_dense_logdet_offset`, summed over the observed coordinates only.
        """
        y, mask = self._mask_of(y)
        if mask is None:
            return self._nll_complete(pred, y, reduction)
        self._check_dense_mask(mask)
        A, r = self._dense_score_terms(pred, y)
        nll = masked_gaussian_nll(A, r, mask) + self._dense_logdet_offset(pred, mask)
        return self.reduce(nll, reduction)

    def _nll_complete(self, pred, y, reduction: str = "mean"):
        """Structured NLL over all coordinates, used when nothing is missing.

        Not abstract: a family whose predictive is degenerate (the deterministic
        Dirac) has no density at all and overrides :meth:`nll` outright instead.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement nll")

    def _dense_logdet_offset(self, pred, mask: Optional[Tensor]) -> Tensor:
        """``log|det D̃|`` relating the scored object to the predictive covariance.

        Zero for every family whose :meth:`_dense_score_terms` already returns
        the physical covariance. SCORE overrides it, because its dense terms
        describe the correlation core ``C`` while the predictive covariance is
        ``D̃ C D̃``.
        """
        return pred.new_zeros(pred.shape[0])

    def kernel_score(self, pred, y, reduction: str = "mean"):
        """Multi-bandwidth kernel score: the per-bandwidth closed-form score
        averaged over the active bandwidth list (see `_mean_over_bandwidths`).

        With missing targets the structured per-family closed form is replaced
        by the dense masked one, which evaluates the same score on the observed
        coordinates (see :func:`masked_gaussian_kernel_score`).
        """
        y, mask = self._mask_of(y)
        if mask is None:
            return self._mean_over_bandwidths(
                self._kernel_score_single, pred, y, reduction
            )
        self._check_dense_mask(mask)
        A, r = self._dense_score_terms(pred, y)
        return self._mean_over_bandwidths(
            lambda: self.reduce(
                masked_gaussian_kernel_score(A, r, self.gamma2, mask), reduction
            )
        )

    @abstractmethod
    def _kernel_score_single(self, pred, y, reduction: str = "mean"):
        """Closed-form kernel score at the single active bandwidth `self.gamma2`."""
        pass

    def dense_kernel_score(self, pred, y, reduction: str = "mean"):
        """Multi-bandwidth kernel score of the predictive itself, in closed form.

        Same analytic formula as :meth:`kernel_score`, but always against the
        predictive ``(Sigma_P, mu - y)`` from :meth:`_deployed_dense_terms` rather
        than against whatever matrix a family's structured closed form happens to
        score. For every family except SCORE those are the same object and this
        is simply the dense route to the same number.

        For :class:`~src.methods.score.SCORE` on a localized
        basis they are *not* the same object, and the difference is not cosmetic.
        There `kernel_score` scores the correlation core ``C`` against the
        standardized residual ``D̃⁻¹(mu - y)`` with ``D̃² = tau / Delta(s)``
        (paper Eq. (10)), so the observation being scored moves with the
        stage-2 parameter ``s`` -- which makes it no proper scoring rule at all,
        and leaves its minimizer with no consistency guarantee for the
        predictive ``Sigma_P = D̃ C D̃``. (On a *flat* basis
        ``Delta(s) = mean(s) = 1`` by the scale gauge, so ``D̃`` is fixed by
        stage 1 and the surrogate *is* a proper score of the standardized law --
        which is why the Fourier path keeps `kernel_score`.)

        Goes dense unconditionally, so the ``dense_mask_max_dim`` guard applies
        with or without missing targets: this is a station-network loss, not a
        gridded one.
        """
        y, mask = self._mask_of(y)
        self._check_dense_dim(y.flatten(start_dim=1).shape[-1])
        A, r = self._deployed_dense_terms(pred, y)
        return self._mean_over_bandwidths(
            lambda: self.reduce(
                masked_gaussian_kernel_score(A, r, self.gamma2, mask), reduction
            )
        )

    def sample_kernel_score(self, pred, y, reduction: str = "mean"):
        """Multi-bandwidth Gaussian kernel score of the predictive, from samples.

        The fair (``M(M-1)``) ensemble estimator over ``M = n_samples``
        reparameterized draws,

            score = 0.5 (1 + E[k(X, X')]) - E[k(X, y)],

        i.e. the same functional :meth:`dense_kernel_score` computes exactly and
        the same estimator :func:`src.evaluation.metrics.eval_kernel_score`
        reports -- unbiased in ``M``, so the gradient is unbiased too. Unlike the
        dense form it never builds a ``(V, V)`` matrix, so it is the route that
        survives past station-network event dimensions.

        ``scoringrules`` fixes the Gaussian kernel at ``exp(-||a - b||^2 / 2)``
        and takes no bandwidth argument, but the kernel used here, ``exp(-||a - b||^2 /
        gamma^2)``, is recovered exactly by scaling both arrays by
        ``sqrt(2) / gamma``, so the estimator itself stays library code.

        One sample draw is shared across the bandwidth list: re-drawing per
        bandwidth would only add variance.

        Masking is exact: a missing coordinate is overwritten by the observation
        in every member, making the predictive there a point mass at the truth,
        which contributes to neither ``x_m - y`` nor ``x_m - x_j``.
        """
        y, mask = self._mask_of(y)
        obs = y.flatten(start_dim=1)  # (B, V)
        fct = self._draw_samples(pred, flatten=True)  # (B, V, M)
        if mask is not None:
            fct = fct * mask.unsqueeze(-1) + (obs * (1.0 - mask)).unsqueeze(-1)

        def _at_bandwidth():
            c = math.sqrt(2.0) / self.gamma
            score = sr.gksmv_ensemble(
                obs * c, fct * c, m_axis=-1, v_axis=-2,
                estimator="fair", backend="torch",
            )
            return self.reduce(score, reduction)

        return self._mean_over_bandwidths(_at_bandwidth)

    def marginal_kernel_score(self, pred, y, reduction: str = "mean"):
        """Multi-bandwidth marginal kernel score, averaged over the active
        bandwidth list. Only methods that define `_marginal_kernel_score_per_coord`
        (Diag, SCORE) support this loss.

        Coordinate-separable, so missing targets need only the masked mean."""
        y, mask = self._mask_of(y)
        return self._mean_over_bandwidths(
            lambda: self.reduce(
                self._masked_mean(self._marginal_kernel_score_per_coord(pred, y), mask),
                reduction,
            )
        )

    def _marginal_kernel_score_per_coord(self, pred, y) -> Tensor:
        """Per-coordinate marginal kernel score at the single active bandwidth.

        Not abstract: only some methods provide it (it is one of the selectable
        LOSSES but not universally defined). Methods that support the marginal
        kernel score override this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement marginal_kernel_score"
        )

    @abstractmethod
    def energy_score(self, pred, y, reduction: str = "mean", n_samples: Optional[int] = None):
        """Exact sliced-estimator energy score (random projections on S^{d-1}).

        ``n_samples`` overrides the number of projection directions ``M``;
        defaults to :attr:`n_samples` (``n_eval_samples`` outside training).
        Unlike the O(M^2) ensemble estimator in
        :mod:`src.evaluation.metrics`, this is O(M), so it stays cheap at much
        larger ``M`` (e.g. eval-time reporting uses ``M=500``).
        """
        pass