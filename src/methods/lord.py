from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal
import math
import scoringrules as sr
from src.methods import PredictiveMethod


class LorD(PredictiveMethod):
    """LorD: low-rank-plus-diagonal Gaussian in physical space, Σ = D + UUᵀ.

    The head predicts the mean, a positive per-coordinate variance D and a
    rank-``rank`` factor U per coordinate. Runs of this method are published
    under the slug ``phys_lora_*``.
    """

    def __init__(
        self,
        loss: str,
        eps: float,
        n_train_samples: int,
        n_eval_samples: int,
        out_channels: int,
        gamma: float,
        rank: int,
        gamma_eval: Optional[float] = None,
        nll_float64: bool = False,
        **kwargs,
    ):
        super().__init__(
            loss, out_channels, eps, n_train_samples, n_eval_samples, gamma, gamma_eval
        )
        self.rank = rank
        # Evaluate the log-likelihood in float64 (opt-in, float32 otherwise).
        # Needed at the depth resolution (d ~ 2.4e5): see _nll_complete.
        self.nll_float64 = bool(nll_float64)

    @property
    def group_widths(self) -> dict:
        # Mean + per-pixel diagonal + low-rank factor U (rank columns per
        # channel); U is the customizable "other" head on the backbone.
        C = self._out_channels
        return {"mu": C, "sigma": C, "other": self.rank * C}

    @property
    def group_denorm(self) -> dict:
        # Sigma = diag(d) + U Uᵀ, so under a rescaling of the data by `a` the
        # covariance scales by a**2 and hence U scales linearly by `a` -- unlike
        # the scale-free "other" group the base class assumes.
        return {"mu": "loc", "sigma": "var", "other": "scale"}

    @property
    def other_zero_init(self) -> bool:
        # U enters the covariance quadratically, so dL/dU vanishes identically
        # at U = 0: a zero-init head is an exact stationary point and never
        # trains. The backbone must break the symmetry with a small random init.
        return False

    def process_output(self, output_head: torch.Tensor) -> torch.Tensor:
        # output_head: (B, C*(2+r), d1, ..., dN) -> (B, C, d1, ..., dN, 2+r)
        B, _, *spatial = output_head.shape
        output = output_head.reshape(B, self._out_channels, self.rank + 2, *spatial)
        output = output.movedim(2, -1)
        mu, diag, lora = output[..., 0:1], output[..., 1:2], output[..., 2:]
        diag = F.softplus(diag) + self.eps
        output = torch.cat([mu, diag, lora], dim=-1)
        return output

    @torch.no_grad()
    def get_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # Low-rank + diagonal Gaussian: Sigma = diag(d) + U Uᵀ. Sample with the
        # same LowRankMultivariateNormal used for the NLL. It draws flat samples
        # of shape (M, B, V); we move the member axis last and restore the event
        # shape (B, C, *spatial, M).
        if n_samples is None:
            n_samples = self.n_samples
        mu, diag, lora = pred[..., 0], pred[..., 1], pred[..., 2:]
        B, _, *spatial = mu.shape
        mu_flat = mu.flatten(start_dim=1)
        diag_flat = diag.flatten(start_dim=1)
        U = lora.reshape(B, -1, self.rank)
        dist = LowRankMultivariateNormal(mu_flat, U, diag_flat)
        samples = dist.sample((n_samples,)).movedim(0, -1)  # (B, V, M)
        if not flatten:
            samples = samples.reshape(B, mu.shape[1], *spatial, n_samples)
        return samples

    def get_marg_std(self, pred: torch.Tensor) -> torch.Tensor:
        # Low-rank + diagonal Gaussian: Sigma = diag(d) + U Uᵀ, so the marginal
        # variance at coordinate i is d_i + sum_r U_{i,r}² -- the diagonal plus the
        # squared row-norm of the per-pixel low-rank factor.
        diag, lora = pred[..., 1], pred[..., 2:]
        marg_var = diag + lora.pow(2).sum(dim=-1)
        return marg_var.sqrt()

    def _pair_diff_var(self, pred: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # Sigma = diag(d) + U Uᵀ, so
        #   Sigma_ii + Sigma_jj - 2 Sigma_ij
        #     = d_i + ||u_i||² + d_j + ||u_j||² - 2 u_i·u_j
        #     = d_i + d_j + ||u_i - u_j||².
        # The low-rank part therefore never needs Sigma_ij itself, only the
        # squared distance between the rank-r rows -- an O(r) per-pair `cdist`
        # rather than a d x d lookup. U is reshaped over the *whole* flattened
        # event (as in `get_samples`/`nll`), so this correctly couples channels.
        diag, lora = pred[..., 1], pred[..., 2:]
        B = pred.shape[0]
        d = diag.flatten(start_dim=1)  # (B, V)
        U = lora.reshape(B, -1, self.rank)  # (B, V, r)
        sq_dist = torch.cdist(U[:, rows], U).pow(2)  # (B, len(rows), V)
        return d[:, rows].unsqueeze(-1) + d.unsqueeze(-2) + sq_dist

    def _quadratic_form(self, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Sigma = diag(d) + U Uᵀ, so aᵀ Sigma a = sum_i a_i² d_i + ||Uᵀa||².
        # U spans the whole flattened event (as everywhere else in this class),
        # so cross-channel covariance is picked up correctly.
        diag, lora = pred[..., 1], pred[..., 2:]
        B = pred.shape[0]
        a_flat = a.flatten(start_dim=1)  # (B, V)
        U = lora.reshape(B, -1, self.rank)  # (B, V, r)
        diag_term = (a_flat.pow(2) * diag.flatten(start_dim=1)).sum(dim=1)
        lora_term = torch.einsum("bv,bvr->br", a_flat, U).pow(2).sum(dim=1)
        return diag_term + lora_term

    @staticmethod
    def _logdet_chol(c: float, d: torch.Tensor, U: torch.Tensor):
        """logdet(I + cΣ) for Σ = diag(d) + UUᵀ, plus the pieces of I+cΣ."""
        eye = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype)
        Dc = 1.0 + c * d  # (B, d)
        UtDi = U.transpose(-1, -2) / Dc.unsqueeze(-2)  # (B, r, d)
        C = c * torch.matmul(UtDi, U) + eye  # (B, r, r) SPD
        L = torch.linalg.cholesky(C)
        logdet = Dc.log().sum(-1) + 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return logdet, Dc, L

    def _nll_complete(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        mu, diag, lora = pred[..., 0], pred[..., 1], pred[..., 2:]
        B = pred.shape[0]
        mu = mu.flatten(start_dim=1)
        diag = diag.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        U = lora.reshape(B, -1, self.rank)
        if self.nll_float64:
            # The r x r capacitance I + U^T diag^-1 U is a sum over d = C*H*W
            # terms; at d ~ 2e5 (depth) float32 accumulation over near-eps
            # diagonal entries can lose positive definiteness and break the
            # Cholesky. Evaluating the density in float64 costs O(B d r)
            # doubles; gradients flow through the casts. Enable with
            # method.nll_float64=true.
            dtype = mu.dtype
            lora_normal = LowRankMultivariateNormal(mu.double(), U.double(), diag.double())
            nll = -lora_normal.log_prob(y.double()).to(dtype)
        else:
            lora_normal = LowRankMultivariateNormal(mu, U, diag)
            nll = -lora_normal.log_prob(y)

        return self.reduce(nll, reduction)

    def _kernel_score_single(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        mu, diag, lora = pred[..., 0], pred[..., 1], pred[..., 2:]
        batch_shape = pred.shape[0]
        diff = (mu - y).flatten(start_dim=1)
        d = diag.flatten(start_dim=1)
        U = lora.reshape(batch_shape, -1, self.rank)

        beta = 1.0 / self.gamma2

        # cross term  E[k(X, y)] : scale 2β, plus the Woodbury quadratic form Q
        log_det1, Dc2, L2 = self._logdet_chol(2.0 * beta, d, U)
        w = diff / Dc2  # (B, d)   Dc⁻¹ a
        b = U.transpose(-1, -2) @ w.unsqueeze(-1)  # (B, r, 1) Uᵀ w
        sol = torch.cholesky_solve(b, L2)  # (B, r, 1) C₂⁻¹ b
        maha = (diff * w).sum(-1) - (2.0 * beta) * (b * sol).sum(
            dim=(-2, -1)
        )  # (B,)  Q

        # self term  E[k(X, X')] : scale 4β
        log_det2, _, _ = self._logdet_chol(4.0 * beta, d, U)

        fac1 = torch.exp(-0.5 * log_det1 - maha / self.gamma2)
        fac2 = torch.exp(-0.5 * log_det2)
        score = 0.5 * (1.0 + fac2) - fac1

        return self.reduce(score, reduction)

    def energy_score(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        mu, diag, lora = pred[..., 0], pred[..., 1], pred[..., 2:]
        mu = mu.flatten(start_dim = 1)
        diag = diag.flatten(start_dim = 1)
        y = y.flatten(start_dim = 1)
        B, d = mu.shape
        M = self.n_samples if n_samples is None else n_samples
        U = lora.reshape(B, -1, self.rank)

        # M unit directions per batch element, uniform on S^{d-1}
        z = torch.randn(B, d, M, device=mu.device, dtype=mu.dtype)
        v = z / z.norm(dim=1, keepdim=True)                       # normalize over coords

        # per-slice pushforward stats, contracting the coordinate axis (dim 1)
        s_1 = torch.einsum("bdm,bd->bm", v.pow(2), diag)
        s_2 = torch.einsum("bdr,bdm->brm", U, v).pow(2).sum(dim=1)
        s = torch.sqrt(s_1 + s_2).clamp_min(self.eps)
        w = torch.einsum("bdm,bd->bm", v, y - mu) / s            # (B, M)

        Phi = 0.5 * (1 + torch.erf(w / math.sqrt(2)))
        phi = torch.exp(-0.5 * w * w) / math.sqrt(2 * math.pi)
        crps = s * (w * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))  # (B, M)

        logCd = 0.5 * math.log(math.pi) + math.lgamma((d + 1) / 2) - math.lgamma(d / 2)
        score = math.exp(logCd) * crps.mean(dim=-1)              # (B,)
        return self.reduce(score, reduction)

    def _crps_per_coord(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, diag = pred[..., 0], pred[..., 1]
        return sr.crps_normal(y, mu, torch.sqrt(diag), backend="torch")
