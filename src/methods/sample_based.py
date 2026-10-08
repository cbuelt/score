"""SB: sample-based generative baseline, fitted with the fair Gaussian kernel score.

Every other method in this package is a *parametric* predictive: the head emits
the parameters of a Gaussian and the scores are closed forms in those
parameters. This one is the nonparametric alternative that the generative
forecasting literature uses as its reference --

* Pacchiardi, Adewoyin, Dueben & Dutta, *Probabilistic Forecasting with
  Generative Networks via Scoring Rule Minimization* (JMLR 2024),
  https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting
* the ``torch-naut`` implementation of the same idea,
  https://github.com/proto-n/torch-naut

-- where the network maps ``(input, latent noise) -> one sample``, ``M`` latents
give ``M`` ensemble members, and the parameters are fitted by minimizing a
proper scoring rule of that ensemble against the observation.

The default scoring rule here is the **Gaussian kernel score** at the same
bandwidth ladder as the Gaussian families (:meth:`kernel_score`), which puts this
baseline on the same objective as SCORE rather than on a second one -- the
number it trains on is then the number the results tables report.
``energy_score`` remains available and is what the cited work uses. Runs of this
method are published under the slug ``generative_*`` (e.g.
``generative_kernel_score``).

Either way the estimator is the **fair** (unbiased in ``M``) form, which is what
keeps a finite ``M`` from being rewarded for under-dispersion: the naive
``1/M^2`` spread term is biased low, so a model minimizing it wins by shrinking
its ensemble, while the ``1/(M(M-1))`` correction removes exactly that
incentive and makes a small ``M`` a usable training budget.

The randomness lives in the backbone (see
:mod:`src.models.backbones.noise`), not here: this class is the thin adapter
that tells the backbone to build a noise injector, un-stacks the members the
backbone returns, and supplies the ensemble estimates of the quantities the
evaluation and calibration protocols ask of every method.

**What ``pred`` is.** For every Gaussian family ``pred``'s trailing axis indexes
parameter *groups*; here it indexes ensemble *members*, so ``pred`` has shape
``(B, C, D1, ..., DN, M)`` and *is* the sample. Every consumer that goes through
:meth:`get_samples`, :meth:`get_mean` or :meth:`get_marg_std` is unaffected by
that difference.

**What is not available.** There is no density, so :meth:`nll` reports
``inf`` -- the same convention
:class:`~src.methods.deterministic.Deterministic` uses, which the evaluator
tolerates and the results tables render as an empty cell. Everything else in the
protocol is recycled unchanged: ``crps``, ``energy_score`` and ``kernel_score``
are ensemble estimators over :meth:`get_samples` already, and ``mse`` / ``mae``
only need :meth:`get_mean`.
"""

from typing import Optional

import scoringrules as sr
import torch

from src.methods import PredictiveMethod


class SampleBased(PredictiveMethod):
    """A generative sampler head, trained by fair scoring rule minimization.

    Args:
        loss: Training objective. ``kernel_score`` (the default -- the fair
            ensemble Gaussian kernel score, so this arm optimizes the same
            functional the Gaussian families do), ``energy_score``, or
            ``crps`` / ``mse`` / ``mae``; all are ensemble estimates here.
            ``nll`` and ``marginal_kernel_score`` need a density or closed-form
            Gaussian marginals and are rejected.
        eps: Numerical floor, shared with the base class.
        n_train_samples: Ensemble size ``M`` during train and validation.
        n_eval_samples: Ensemble size ``M`` at evaluation and calibration.
        out_channels: Raw dataset channel count ``C``.
        gamma / gamma_eval: Kernel bandwidths of the Gaussian kernel score --
            the training objective's own ladder, and the eval metric's.
        global_noise_dim: Width of the per-member latent shared across all
            positions -- what gives the samples spatial dependence.
        local_noise_dim: Width of the per-position latent -- what keeps the
            predictive full-rank.
        noise_hidden_dim: Hidden width of the injector's mixer; ``None`` uses
            the trunk's own feature width.
    """

    #: Losses that need either a density or closed-form Gaussian marginals, and
    #: so cannot be estimated from an ensemble at all. The *joint* kernel score
    #: is not among them: :meth:`kernel_score` is the fair ensemble estimator
    #: here, exactly as :meth:`energy_score` is.
    _UNSUPPORTED_LOSSES = ("nll", "marginal_kernel_score")

    # The ensemble estimators mask *exactly* -- a coordinate overwritten by the
    # observation in every member is a point mass at the truth and contributes
    # nothing to any norm, so it drops out of both the cross and the spread
    # term. That is why the base class's blanket ban on a masked `energy_score`
    # (which exists for the *sliced Gaussian* estimator, whose random directions
    # would have to be redrawn from the observed subspace) does not apply here.
    MASK_UNSUPPORTED_LOSSES = ()

    def __init__(
        self,
        loss: str,
        eps: float,
        n_train_samples: int,
        n_eval_samples: int,
        out_channels: int,
        gamma: float,
        gamma_eval: Optional[float] = None,
        global_noise_dim: int = 32,
        local_noise_dim: int = 8,
        noise_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        if loss in self._UNSUPPORTED_LOSSES:
            raise ValueError(
                f"{type(self).__name__} has no density and no closed-form "
                f"covariance, so loss={loss!r} is not available. Use "
                "'kernel_score' (the default), 'energy_score', 'crps', 'mse' "
                "or 'mae'."
            )
        super().__init__(
            loss, out_channels, eps, n_train_samples, n_eval_samples, gamma, gamma_eval
        )
        if n_train_samples < 2 or n_eval_samples < 2:
            raise ValueError(
                "The fair ensemble estimators need at least 2 members; got "
                f"n_train_samples={n_train_samples}, n_eval_samples={n_eval_samples}."
            )
        self.global_noise_dim = global_noise_dim
        self.local_noise_dim = local_noise_dim
        self.noise_hidden_dim = noise_hidden_dim

    # ---- backbone configuration -------------------------------------------

    @property
    def group_widths(self) -> dict:
        # One group: the sample itself. The head emits C channels per member and
        # is replayed once per member, so its width does not grow with M --
        # `out_channels` stays C and only the batch axis is expanded.
        return {"mu": self._out_channels, "sigma": None, "other": None}

    @property
    def group_denorm(self) -> dict:
        # A sample is a draw of the target, so it denormalizes exactly like a
        # location -- the same rule the mean gets, applied to every member.
        return {"mu": "loc", "sigma": "var", "other": "none"}

    @property
    def noise(self) -> dict:
        """The injector spec the backbone builds; see :mod:`src.models.backbones.noise`."""
        return {
            "global_dim": self.global_noise_dim,
            "local_dim": self.local_noise_dim,
            "hidden_dim": self.noise_hidden_dim,
        }

    @property
    def samples_from_backbone(self) -> bool:
        return True

    @property
    def has_closed_form_pit(self) -> bool:
        return False

    # ---- head output -------------------------------------------------------

    def process_output(self, output_head: torch.Tensor) -> torch.Tensor:
        # (B, C*M, d1..dN) -> (B, C, d1..dN, M). Identical to every Gaussian
        # method's un-stacking, with the member axis playing the part of the
        # group axis: the channel axis is group-inner, so each channel's members
        # are contiguous (see `stack_member_groups`).
        batch, stacked, *spatial = output_head.shape
        n_samples, remainder = divmod(stacked, self._out_channels)
        if remainder:
            raise ValueError(
                f"Head emitted {stacked} channels, which is not a multiple of "
                f"the {self._out_channels} target channels."
            )
        output = output_head.reshape(batch, self._out_channels, n_samples, *spatial)
        return output.movedim(2, -1)

    def get_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        """The members already carried by ``pred``.

        Deliberately *not* wrapped in ``torch.no_grad``, unlike every Gaussian
        family: the training loss is a function of these samples, so the graph
        through them is the only thing that trains the model.

        A generative sample costs a forward pass, so unlike the closed-form
        families this cannot conjure more members on demand -- the member count
        was fixed when the backbone ran. Asking for fewer returns a prefix;
        asking for more is an error naming the knob that actually controls it
        (see :attr:`~src.methods.base.PredictiveMethod.samples_from_backbone`).
        """
        available = pred.shape[-1]
        if n_samples is not None:
            if n_samples > available:
                raise ValueError(
                    f"Asked for {n_samples} members but `pred` carries "
                    f"{available}. A generative sample costs a forward pass, so "
                    "the member count is fixed by the backbone: call "
                    f"model(x, n_samples={n_samples}) instead of requesting more "
                    "members here."
                )
            pred = pred[..., :n_samples]
        if flatten:
            pred = pred.flatten(start_dim=1, end_dim=-2)
        return pred

    def get_mean(self, pred: torch.Tensor) -> torch.Tensor:
        # No closed-form mean: the ensemble average is the estimate, and it is
        # what `mse` / `mae` and the calibration centring are taken against.
        return pred.mean(dim=-1)

    # ---- ensemble estimates of the Gaussian accessors ----------------------
    #
    # The evaluation and calibration protocols ask every method for a marginal
    # standard deviation, a pairwise difference variance and a quadratic form.
    # For a nonparametric predictive these are second moments of the sample
    # distribution rather than parameters, so they are estimated from the M
    # members. All three go through the same object -- the centred member matrix
    # A = (x_m - xbar), whose Gram gives the sample covariance
    # Sigma = A Aᵀ / (M - 1) -- and all three exploit its low rank exactly as
    # LorD does, so none of them ever materializes Sigma.

    def _centred(self, pred: torch.Tensor) -> torch.Tensor:
        """Members centred on the ensemble mean, flattened to ``(B, V, M)``."""
        flat = pred.flatten(start_dim=1, end_dim=-2)
        return flat - flat.mean(dim=-1, keepdim=True)

    def get_marg_std(self, pred: torch.Tensor) -> torch.Tensor:
        # Per-coordinate sample standard deviation, reshaped back to the mean's
        # event shape.
        return pred.std(dim=-1, unbiased=True)

    def _pair_diff_var(self, pred: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # Var(X_i - X_j) = Sigma_ii + Sigma_jj - 2 Sigma_ij with
        # Sigma = A Aᵀ / (M - 1). The cross term is one (R, M) x (M, V) matmul,
        # so the working set is O(R*V) -- the same bound every other family's
        # blocked implementation respects.
        a = self._centred(pred)  # (B, V, M)
        denom = a.shape[-1] - 1
        var = a.pow(2).sum(dim=-1) / denom  # (B, V)
        cross = torch.bmm(a[:, rows], a.transpose(1, 2)) / denom  # (B, R, V)
        return var[:, rows].unsqueeze(-1) + var.unsqueeze(-2) - 2.0 * cross

    def _quadratic_form(self, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # aᵀ Sigma a = ||Aᵀ a||² / (M - 1): the projection of every member onto
        # `a`, then its sample variance. O(V*M), no V x V matrix.
        centred = self._centred(pred)  # (B, V, M)
        proj = torch.einsum("bv,bvm->bm", a.flatten(start_dim=1), centred)
        return proj.pow(2).sum(dim=-1) / (centred.shape[-1] - 1)

    # ---- scores ------------------------------------------------------------

    def energy_score(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Fair ensemble energy score (the ``loss=energy_score`` objective).

        ``scoringrules.es_ensemble(..., estimator="fair")`` on the flattened
        event axis, i.e.

            ES = 1/M sum_m ||x_m - y|| - 1/(2 M (M-1)) sum_{i != j} ||x_i - x_j||,

        the same estimator :func:`src.evaluation.metrics.eval_energy_score`
        reports, so the number this trains on and the number the tables report
        are the same functional. (The eval metric keeps its own ``cdist``
        formulation because at ``M = 50`` the ``(M, M, V)`` difference tensor
        scoringrules builds is 3 GB per gridded case; at the training ``M = 10``
        it is 25x smaller. Lower the batch size on ``depth`` if it does not
        fit.)

        Masking is exact: a missing coordinate is overwritten by the
        observation in every member, making the predictive there a point mass at
        the truth, which contributes nothing to either norm.
        """
        y, mask = self._mask_of(y)
        obs = y.flatten(start_dim=1)  # (B, V)
        fct = self.get_samples(pred, n_samples=n_samples, flatten=True)  # (B, V, M)
        if mask is not None:
            fct = fct * mask.unsqueeze(-1) + (obs * (1.0 - mask)).unsqueeze(-1)
        score = sr.es_ensemble(
            obs, fct, m_axis=-1, v_axis=-2, estimator="fair", backend="torch"
        )
        return self.reduce(score, reduction)

    def _crps_per_coord(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Fair ensemble CRPS per coordinate, reshaped to the event shape so the
        # base class's masked mean applies unchanged. Note this materializes an
        # (B, V, M, M) tensor, so it is meant for the training M, not the far
        # larger evaluation one -- the eval `crps` metric has its own chunked
        # estimator.
        fct = self.get_samples(pred, flatten=True)  # (B, V, M)
        crps = sr.crps_ensemble(
            y.flatten(start_dim=1), fct, m_axis=-1, estimator="fair", backend="torch"
        )
        return crps.reshape(y.shape)

    def _draw_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # `get_samples` is already the differentiable draw here (it is not
        # wrapped in no_grad, since the members *are* the model output), so the
        # base class's hook for the sampling-based losses is just that.
        return self.get_samples(pred, n_samples=n_samples, flatten=flatten)

    def kernel_score(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Fair ensemble Gaussian kernel score -- the default training objective.

        Overridden the same way :meth:`energy_score` is: the name keeps its
        meaning across the package while the implementation swaps the Gaussian
        families' closed form for the ensemble estimator, which is the only one
        available without a covariance. That is
        :meth:`~src.methods.base.PredictiveMethod.sample_kernel_score`, whose
        bandwidth ladder and ``M(M-1)`` spread term match
        :func:`src.evaluation.metrics.eval_kernel_score` -- so the number this
        arm trains on and the number the tables report are the same functional.

        Masking is exact, for the reason given on :meth:`energy_score`.
        """
        return self.sample_kernel_score(pred, y, reduction)

    def _kernel_score_single(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form covariance, so the "
            "kernel score has no closed form here. `kernel_score` overrides the "
            "base implementation with the fair ensemble estimator, so this "
            "single-bandwidth closed-form hook is never reached."
        )

    @staticmethod
    def nll(
        pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Not available: a generative sampler has no tractable density.

        Reports ``inf`` rather than raising, following
        :class:`~src.methods.deterministic.Deterministic`. The evaluator
        already tolerates a bare float and the results tables map it to an empty
        cell, so this method can share the ``metrics:`` list of every other
        method without a per-method config.
        """
        return torch.inf
