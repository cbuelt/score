import math
from typing import Optional

import scoringrules as sr
import torch
import torch.nn.functional as F
from torch import Tensor

from src.methods import PredictiveMethod
from src.methods.base import masked_gaussian_kernel_score


class Cholesky(PredictiveMethod):
    """Chol: unconstrained dense Gaussian, parameterized by its Cholesky factor.

    ``Sigma = L Lᵀ`` with ``L`` lower triangular. Unlike every other family here
    this imposes *no* structure at all: the predictive covariance ranges over the
    full SPD cone, so on the tasks where it is affordable it is the reference the
    structured parameterizations (Diag, LorD, SCORE) are approximating. Runs of
    this method are published under the slug ``phys_cholesky``.

    Affordable means small: the closed forms materialize ``(B, V, V)`` matrices,
    which is 4.7 MB at univariate ETT's ``B=128, V=96`` and 0.4 MB at EUPPBench's
    ``B=8, V=115``, but 4.6 GiB *per sample* at ERA5's ``V = 35 200``. The
    :attr:`max_event_dim` guard refuses those outright, mirroring the
    :attr:`~src.methods.base.PredictiveMethod.dense_mask_max_dim` boundary that
    the masked scores rely on.

    Predicting ``L`` rather than ``Sigma`` is what makes this cheap: it is
    positive-definite by construction (no projection, no eigenvalue clipping),
    ``log det Sigma = 2 sum_i log L_ii`` is free, and the Mahalanobis form is one
    triangular solve rather than a factorization.

    **Head layout.** The trailing groups are ``[mu, d, o_1, ..., o_{V-1}]``:
    ``d`` is the squared diagonal of ``L`` and the ``o`` slots carry ``L``'s
    strictly-lower part, *row-indexed* -- coordinate ``i`` emits its own row of
    ``L``, of which only entries ``0..i-1`` survive the ``tril``. That wastes
    half the slots (``V(V-1)`` allocated for ``V(V-1)/2`` parameters), which is
    the price of the framework's per-coordinate head layout: ``group_widths``
    entries must be ``k`` fields over the event grid, and ``V(V-1)/2`` is not of
    that form. Row-indexing is also what makes ``Multigraph``'s default
    ``spectrum_readout="per_station"`` correct here -- row ``i`` genuinely is a
    function of station ``i`` -- as opposed to the mode-indexed SCORE core,
    which needs the ``"global"`` readout.

    ``L`` spans the *whole flattened event* (like :class:`LorD`'s ``U``, and
    unlike the SCORE core, which acts per channel), so it couples channels.
    """

    # Largest event dimension this parameterization will accept. The closed forms
    # are O(V^2) in memory and O(V^3) nowhere (only triangular solves), but a
    # dense (B, V, V) is still infeasible at gridded scale. Matches
    # `dense_mask_max_dim`, the "station networks, not fields" boundary used
    # by the masked scores.
    max_event_dim = 1024

    def __init__(
        self,
        loss: str,
        eps: float,
        n_train_samples: int,
        n_eval_samples: int,
        out_channels: int,
        gamma: float,
        event_dim: Optional[int] = None,
        gamma_eval: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            loss, out_channels, eps, n_train_samples, n_eval_samples, gamma, gamma_eval
        )
        if event_dim is None:
            raise ValueError(
                "Cholesky needs `event_dim`, the total flattened target size "
                "V = C * prod(spatial). It sizes the head (one row of L per "
                "coordinate), so it must be known before the backbone is built. "
                "train.py / evaluate.py / calibration.py inject it from "
                "`datamodule.get_total_dimension()`."
            )
        event_dim = int(event_dim)
        if event_dim > self.max_event_dim:
            raise NotImplementedError(
                f"Cholesky builds a dense ({event_dim}, {event_dim}) Cholesky "
                f"factor, which exceeds max_event_dim={self.max_event_dim}. A full "
                "covariance is only tractable for small event dimensions (station "
                "networks, univariate forecast horizons), not gridded fields -- "
                "use LorD (method=lord) or SCORE there."
            )
        self.event_dim = event_dim

    # ---- head contract -----------------------------------------------------

    @property
    def group_widths(self) -> dict:
        # Mean + the squared diagonal of L + L's strictly-lower part, laid out as
        # V-1 fields over the event grid (row i of L, masked to its first i
        # entries). A one-dimensional target has no off-diagonal at all, so the
        # "other" head is dropped entirely there and this degenerates to Diag.
        C = self._out_channels
        if self.event_dim < 2:
            return {"mu": C, "sigma": C, "other": None}
        return {"mu": C, "sigma": C, "other": (self.event_dim - 1) * C}

    @property
    def group_denorm(self) -> dict:
        # Sigma = L Lᵀ, so under a rescaling of the data by `a` the factor scales
        # linearly, L -> aL. For the strictly-lower group that is the plain
        # "scale" rule. For the diagonal it is why this class stores `d = L_ii²`
        # rather than L_ii itself: d -> a² d is exactly the existing "var" rule,
        # so no backbone needs a new denormalization case.
        return {"mu": "loc", "sigma": "var", "other": "scale"}

    @property
    def other_zero_init(self) -> bool:
        # True, unlike LorD -- and the difference is worth being explicit
        # about, because the two look like the same situation and are not.
        #
        # LorD's Sigma = diag(d) + U Uᵀ is *quadratic* in U, so dL/dU vanishes
        # identically at U = 0 and a zero-init head never trains. Here the
        # diagonal anchors the strictly-lower part: near L_strict = 0 the
        # covariance is Sigma_ik = L_ik L_kk for i > k, so
        # dSigma_ik/dL_ik = L_kk = sqrt(d_k) > 0. The gradient is non-zero at
        # exactly zero, and the neutral start (L diagonal, Sigma = diag(d)) is the
        # right place to begin.
        return True

    def process_output(self, output_head: Tensor) -> Tensor:
        # output_head: (B, C*(2+(V-1)), d1, ..., dN) -> (B, C, d1, ..., dN, 2+(V-1))
        # Group-inner, like every other method: each channel's groups are
        # contiguous (see PredictiveMethod.group_widths).
        B, _, *spatial = output_head.shape
        flat = self._out_channels * math.prod(spatial)
        if flat != self.event_dim:
            raise ValueError(
                f"Cholesky was built with event_dim={self.event_dim} but the "
                f"head output has C={self._out_channels} and spatial shape "
                f"{tuple(spatial)}, i.e. V={flat}. The two must agree -- "
                "`event_dim` comes from `datamodule.get_total_dimension()`."
            )
        n_groups = 2 if self.event_dim < 2 else self.event_dim + 1
        output = output_head.reshape(B, self._out_channels, n_groups, *spatial)
        output = output.movedim(2, -1)
        mu = output[..., 0:1]
        diag = output[..., 1:2]
        diag = F.softplus(diag) + self.eps
        return torch.cat([mu, diag, output[..., 2:]], dim=-1)

    # ---- the factor --------------------------------------------------------

    def _chol(self, pred: Tensor) -> Tensor:
        """The lower-triangular factor ``L``, shape ``(B, V, V)``.

        ``L_ii = sqrt(d_i)`` from the ``sigma`` group and the strictly-lower
        entries from the ``other`` group, row-indexed: coordinate ``i``'s slots
        become row ``i``, padded on the right by one column so the ``tril(-1)``
        drops everything from the diagonal onwards. Built out of place so it is
        safe under autograd.
        """
        diag = pred[..., 1].flatten(start_dim=1)  # (B, V)
        L = torch.diag_embed(diag.sqrt())
        if self.event_dim < 2:
            return L
        # (B, C, *spatial, V-1) -> (B, V, V-1) -> (B, V, V)
        strict = pred[..., 2:].flatten(start_dim=1, end_dim=-2)
        return L + F.pad(strict, (0, 1)).tril(-1)

    def _cov(self, pred: Tensor) -> Tensor:
        """The dense predictive covariance ``Sigma = L Lᵀ``, shape ``(B, V, V)``."""
        L = self._chol(pred)
        return L @ L.transpose(-1, -2)

    # ---- closed forms ------------------------------------------------------

    @torch.no_grad()
    def get_samples(
        self,
        pred: Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> Tensor:
        # X = mu + L z with standard-normal z -- the whole point of carrying the
        # factor rather than the covariance.
        if n_samples is None:
            n_samples = self.n_samples
        mu = pred[..., 0]
        B, C, *spatial = mu.shape
        L = self._chol(pred)
        z = torch.randn(B, self.event_dim, n_samples, device=mu.device, dtype=mu.dtype)
        samples = mu.flatten(start_dim=1).unsqueeze(-1) + L @ z  # (B, V, M)
        if not flatten:
            samples = samples.reshape(B, C, *spatial, n_samples)
        return samples

    def get_marg_std(self, pred: Tensor) -> Tensor:
        # Sigma_ii = ||l_i||², the squared norm of L's i-th row -- NOT the sigma
        # group, which holds only the diagonal entry L_ii² (the conditional
        # variance of coordinate i given 1..i-1).
        mu = pred[..., 0]
        var = self._chol(pred).pow(2).sum(dim=-1)  # (B, V)
        return var.sqrt().reshape(mu.shape)

    def _pair_diff_var(self, pred: Tensor, rows: Tensor) -> Tensor:
        # Sigma_ii + Sigma_jj - 2 Sigma_ij = ||l_i||² + ||l_j||² - 2 l_i·l_j
        #                                  = ||l_i - l_j||².
        # Blocked over `rows` by the caller, so only a (B, len(rows), V) slab of
        # the Gram matrix is formed.
        L = self._chol(pred)
        var = L.pow(2).sum(dim=-1)  # (B, V)
        cross = L[:, rows] @ L.transpose(-1, -2)  # (B, len(rows), V)
        return var[:, rows].unsqueeze(-1) + var.unsqueeze(-2) - 2.0 * cross

    def _quadratic_form(self, pred: Tensor, a: Tensor) -> Tensor:
        # aᵀ Sigma a = aᵀ L Lᵀ a = ||Lᵀ a||².
        L = self._chol(pred)
        a_flat = a.flatten(start_dim=1)  # (B, V)
        return torch.einsum("bi,bij->bj", a_flat, L).pow(2).sum(dim=1)

    def _dense_score_terms(self, pred: Tensor, y: Tensor):
        # The base class reconstructs Sigma from the marginal variances and the
        # pairwise difference variances, which is exact but needs a full
        # `pair_diff_var` pass. Here Sigma is one matmul away, so short-circuit.
        return self._cov(pred), (self.get_mean(pred) - y).flatten(start_dim=1)

    def _nll_complete(
        self, pred: Tensor, y: Tensor, reduction: str = "mean"
    ) -> Tensor:
        L = self._chol(pred)
        resid = (y - pred[..., 0]).flatten(start_dim=1).unsqueeze(-1)  # (B, V, 1)
        # z = L⁻¹ r, so rᵀ Sigma⁻¹ r = ||z||²; and log det Sigma = 2 sum log L_ii.
        z = torch.linalg.solve_triangular(L, resid, upper=False).squeeze(-1)
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        nll = 0.5 * (
            self.event_dim * math.log(2 * math.pi) + log_det + z.pow(2).sum(dim=-1)
        )
        return self.reduce(nll, reduction)

    def _kernel_score_single(
        self, pred: Tensor, y: Tensor, reduction: str = "mean"
    ) -> Tensor:
        # The shared dense closed form in base.py *is* this family's closed form:
        # with mask=None it evaluates the Gaussian kernel score against the plain
        # (B, V, V) covariance, which is what an unstructured Sigma has to do
        # anyway. The structured families need their own algebra only because
        # they must avoid materializing Sigma.
        A, r = self._dense_score_terms(pred, y)
        return self.reduce(
            masked_gaussian_kernel_score(A, r, self.gamma2, None), reduction
        )

    def _marginal_kernel_score_per_coord(self, pred: Tensor, y: Tensor) -> Tensor:
        # Per-coordinate Gaussian kernel score against the marginal N(mu_i,
        # Sigma_ii), with the bandwidth rescaled to the per-coordinate scale
        # exactly as Diag does.
        mu = pred[..., 0]
        v = self.get_marg_std(pred).flatten(start_dim=1).pow(2)
        diff = (mu - y).flatten(start_dim=1)
        gamma2_marg = self.gamma2 / v.shape[-1]

        diag1 = 1.0 + 2.0 * v / gamma2_marg
        diag2 = 1.0 + 4.0 * v / gamma2_marg

        fac1 = torch.exp(-0.5 * torch.log(diag1) - diff.pow(2) / (diag1 * gamma2_marg))
        fac2 = torch.exp(-0.5 * torch.log(diag2))
        return 0.5 * (1.0 + fac2) - fac1

    def energy_score(
        self,
        pred: Tensor,
        y: Tensor,
        reduction: str = "mean",
        n_samples: Optional[int] = None,
    ) -> Tensor:
        mu = pred[..., 0].flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        B, d = mu.shape
        M = self.n_samples if n_samples is None else n_samples
        L = self._chol(pred)

        # M unit directions per batch element, uniform on S^{d-1}
        z = torch.randn(B, d, M, device=mu.device, dtype=mu.dtype)
        v = z / z.norm(dim=1, keepdim=True)  # normalize over coords

        # Pushforward of Sigma along each direction: Var(vᵀX) = ||Lᵀ v||².
        s = torch.einsum("bij,bim->bjm", L, v).pow(2).sum(dim=1).sqrt()
        s = s.clamp_min(self.eps)
        w = torch.einsum("bdm,bd->bm", v, y - mu) / s  # (B, M)

        Phi = 0.5 * (1 + torch.erf(w / math.sqrt(2)))
        phi = torch.exp(-0.5 * w * w) / math.sqrt(2 * math.pi)
        crps = s * (w * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))  # (B, M)

        logCd = 0.5 * math.log(math.pi) + math.lgamma((d + 1) / 2) - math.lgamma(d / 2)
        score = math.exp(logCd) * crps.mean(dim=-1)  # (B,)
        return self.reduce(score, reduction)

    def _crps_per_coord(self, pred: Tensor, y: Tensor) -> Tensor:
        return sr.crps_normal(y, pred[..., 0], self.get_marg_std(pred), backend="torch")
