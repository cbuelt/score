from typing import Optional

import torch
import torch.nn.functional as F
import math
import scoringrules as sr
from src.methods import PredictiveMethod


class Diag(PredictiveMethod):
    """Diag: diagonal Gaussian in physical space (independent coordinates).

    Runs of this method are published under the slug ``phys_diag``.
    """

    def __init__(
        self,
        loss: str,
        eps: float,
        n_train_samples: int,
        n_eval_samples: int,
        out_channels: int,
        gamma: float,
        gamma_eval: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            loss, out_channels, eps, n_train_samples, n_eval_samples, gamma, gamma_eval
        )

    @property
    def group_widths(self) -> dict:
        # Mean + per-pixel diagonal variance.
        return {"mu": self._out_channels, "sigma": self._out_channels, "other": None}

    def process_output(self, output_head: torch.Tensor) -> torch.Tensor:
        # output_head: (B, C*2, d1, ..., dN) -> (B, C, d1, ..., dN, 2)
        # Group-inner, matching every other method and the group-inner stacking
        # the backbones emit (see PredictiveMethod.group_widths): each channel's
        # groups are contiguous, i.e. [mu_c0, diag_c0, mu_c1, diag_c1, ...].
        B, _, *spatial = output_head.shape
        output = output_head.reshape(B, self._out_channels, 2, *spatial)
        output = output.movedim(2, -1)
        mu = output[..., 0:1]
        diag = output[..., 1:2]
        diag = F.softplus(diag) + self.eps
        return torch.cat([mu, diag], dim=-1)

    @torch.no_grad()
    def get_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # Diagonal Gaussian: each coordinate is independent N(mu, diag), so a
        # sample is mu + sqrt(diag) * z with standard-normal z. Kept in the
        # original event shape (B, C, *spatial, M).
        if n_samples is None:
            n_samples = self.n_samples
        mu, diag = pred[..., 0], pred[..., 1]
        z = torch.randn(*mu.shape, n_samples, device=mu.device, dtype=mu.dtype)
        samples = mu.unsqueeze(-1) + diag.sqrt().unsqueeze(-1) * z
        if flatten:
            samples = samples.flatten(start_dim=1, end_dim=-2)
        return samples

    def get_marg_std(self, pred: torch.Tensor) -> torch.Tensor:
        # Diagonal Gaussian: Sigma = diag(diag), so the marginal variance at each
        # grid point is just that pixel's diagonal entry.
        return pred[..., 1].sqrt()

    def _pair_diff_var(self, pred: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # Independent coordinates (Sigma_ij = 0 off the diagonal), so the
        # difference variance is just the sum of the two marginal variances.
        d = pred[..., 1].flatten(start_dim=1)  # (B, V)
        return d[:, rows].unsqueeze(-1) + d.unsqueeze(-2)

    def _quadratic_form(self, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Sigma = diag(d), so aᵀ Sigma a = sum_i a_i² d_i.
        return (a.pow(2) * pred[..., 1]).flatten(start_dim=1).sum(dim=1)

    def _nll_complete(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        mu, diag = pred[..., 0], pred[..., 1]
        sigma = torch.sqrt(diag)
        full_nll = (
            torch.log(sigma)
            + math.log(math.sqrt(2 * math.pi))
            + (torch.pow(y - mu, 2) / (2 * diag))
        )
        nll = full_nll.flatten(start_dim=1).sum(dim=1)
        return self.reduce(nll, reduction)

    def _kernel_score_single(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        mu, diag = pred[..., 0], pred[..., 1]
        diag = diag.flatten(start_dim=1)
        diff = (mu - y).flatten(start_dim=1)

        diag1 = 1.0 + 2.0 * diag / self.gamma2
        diag2 = 1.0 + 4.0 * diag / self.gamma2

        log_det1 = torch.sum(torch.log(diag1), dim=-1)
        log_det2 = torch.sum(torch.log(diag2), dim=-1)
        maha = torch.sum(torch.pow(diff, 2) / diag1, dim=-1)

        fac1 = torch.exp(-0.5 * log_det1 - maha / self.gamma2)
        fac2 = torch.exp(-0.5 * log_det2)
        score = 0.5 * (1.0 + fac2) - fac1

        return self.reduce(score, reduction)

    def _marginal_kernel_score_per_coord(
        self, pred: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        mu, diag = pred[..., 0], pred[..., 1]
        v = diag.flatten(start_dim=1)
        diff = (mu - y).flatten(start_dim=1)
        d = v.shape[-1]
        gamma2_marg = self.gamma2 / d

        diag1 = 1.0 + 2.0 * v / gamma2_marg
        diag2 = 1.0 + 4.0 * v / gamma2_marg

        fac1 = torch.exp(-0.5 * torch.log(diag1) - diff.pow(2) / (diag1 * gamma2_marg))
        fac2 = torch.exp(-0.5 * torch.log(diag2))
        return 0.5 * (1.0 + fac2) - fac1

    def energy_score(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        mu, diag = pred[..., 0], pred[..., 1]
        mu = mu.flatten(start_dim = 1)
        diag = diag.flatten(start_dim = 1)
        y = y.flatten(start_dim = 1)
        B, d = mu.shape
        M = self.n_samples if n_samples is None else n_samples

        # M unit directions per batch element, uniform on S^{d-1}
        z = torch.randn(B, d, M, device=mu.device, dtype=mu.dtype)
        v = z / z.norm(dim=1, keepdim=True)                       # normalize over coords

        # per-slice pushforward stats, contracting the coordinate axis (dim 1)
        s = torch.sqrt(torch.einsum("bdm,bd->bm", v.pow(2), diag)).clamp_min(self.eps)
        w = torch.einsum("bdm,bd->bm", v, y - mu) / s            # (B, M)

        Phi = 0.5 * (1 + torch.erf(w / math.sqrt(2)))
        phi = torch.exp(-0.5 * w * w) / math.sqrt(2 * math.pi)
        crps = s * (w * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))  # (B, M)

        logCd = 0.5 * math.log(math.pi) + math.lgamma((d + 1) / 2) - math.lgamma(d / 2)
        score = math.exp(logCd) * crps.mean(dim=-1)              # (B,)
        return self.reduce(score, reduction)

    def _crps_per_coord(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, diag = pred[..., 0], pred[..., 1]
        sigma = torch.sqrt(diag)
        return sr.crps_normal(y, mu, sigma, backend="torch")
