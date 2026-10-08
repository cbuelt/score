from typing import Optional

import torch
import torch.nn.functional as F
from src.methods import PredictiveMethod


class Deterministic(PredictiveMethod):
    """Det: point forecast (a Dirac at the predicted mean), trained with MSE.

    The probabilistic scores are evaluated in their degenerate closed forms so
    that the point forecast can be compared on the same metrics.
    """

    def __init__(self, loss: str, eps:float, n_train_samples:int, n_eval_samples:int, out_channels: int, gamma: float, gamma_eval: Optional[float] = None, **kwargs):
        super().__init__(
            loss, out_channels, eps, n_train_samples, n_eval_samples, gamma, gamma_eval
        )

    @property
    def group_widths(self) -> dict:
        # Mean only.
        return {"mu": self._out_channels, "sigma": None, "other": None}

    def process_output(self, output_head: torch.Tensor) -> torch.Tensor:
        return output_head

    def get_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # A deterministic head is a Dirac at the mean, so every member is the
        # prediction itself. Identical members make the ensemble self-term
        # vanish, reproducing the deterministic closed-form scores.
        samples = pred.unsqueeze(-1).expand(*pred.shape, 2)
        if flatten:
            samples = samples.flatten(start_dim=1, end_dim=-2)
        return samples

    def get_marg_std(self, pred: torch.Tensor) -> torch.Tensor:
        # A deterministic head is a Dirac at the mean: zero covariance, hence
        # zero marginal std everywhere. `pred` is already the mean field.
        return torch.zeros_like(pred)

    def get_mean(self, pred: torch.Tensor) -> torch.Tensor:
        # Unlike every other method, `process_output` is the identity here: there
        # are no groups to un-stack, so `pred` *is* the mean field.
        return pred

    def _quadratic_form(self, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # A Dirac has Sigma = 0, so every quadratic form vanishes.
        return pred.new_zeros(pred.shape[0])

    def _pair_diff_var(self, pred: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # Sigma = 0, so X_i - X_j is degenerate at mu_i - mu_j: zero variance
        # everywhere. The variogram score then reduces to the deterministic
        # `(|y_i - y_j| - |mu_i - mu_j|)^2`.
        B = pred.shape[0]
        V = pred.flatten(start_dim=1).shape[1]
        return torch.zeros(B, rows.shape[0], V, device=pred.device, dtype=pred.dtype)

    @staticmethod
    def nll(
        pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return torch.inf

    def _kernel_score_single(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        diff = pred - y
        sq_norm = torch.sum(torch.pow(diff.flatten(start_dim=1), 2), dim=-1)
        score = -torch.exp(-1 / self.gamma2 * sq_norm) + 1
        return self.reduce(score, reduction)

    def energy_score(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        # A Dirac predictive has no Monte-Carlo variance; n_samples is accepted
        # only so callers can pass it uniformly across method families.
        diff = pred - y
        norm = torch.linalg.vector_norm(diff.flatten(start_dim=1), ord=2, dim=-1)
        return self.reduce(norm, reduction)

    def _crps_per_coord(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # A Dirac predictive's CRPS is the absolute error.
        return torch.abs(pred - y)
