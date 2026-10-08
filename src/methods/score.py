"""SCORE: Spectral CORrelation Estimation for multivariate Gaussian predictives.

SCORE separates the marginals from the correlation structure. A prediction is
a triple ``(mu, tau, Sigma_hat)``: the mean ``mu``, the marginal variances
``tau`` and a *core* ``Sigma_hat`` in the B-domain (the coordinates ``z = B y``
of a fixed unitary transform ``B``, the DFT by default). The predictive
covariance is the transformed core passed through the correlation operator,

    Sigma_P = D corr(B* Sigma_hat B) D,    D = diag(sqrt(tau)),     (paper Eq. (5))

which pins ``diag(Sigma_P) = tau`` for every core. It is assembled via the
implied marginal variances ``Delta = diag(B* Sigma_hat B)`` as
``Sigma_P = D̃ B* Sigma_hat B D̃`` with ``D̃ = diag(sqrt(tau / Delta))``
(paper Eq. (10)); for the DFT with a diagonal core ``Delta = 1`` and this
renormalization is the identity.

The model is trained with a two-stage fit: stage 1 fits ``mu`` and ``tau`` on
the marginals (CRPS by default); stage 2 freezes them and fits the core to the
standardized target ``Z = D⁻¹ (Y - mu)`` with the Gaussian kernel score, which
has a closed form in the B-domain (paper Appendix C.2.2).
"""

from typing import Optional

import torch
import torch.nn.functional as F
import math
import scoringrules as sr

from src.methods import PredictiveMethod
from src.methods.spectral import SpectralDiag
from src.methods.transforms import SeparableOrthogonalTransform


class SCORE(PredictiveMethod):
    """SCORE predictive with a diagonal (optionally low-rank-corrected) core.

    On the default Fourier path the predictive covariance is
    ``Sigma_P = D Fᴴ diag(s) F D`` with ``D = diag(sqrt(tau))`` the marginal
    scales and ``Fᴴ diag(s) F`` a real circulant *correlation* core (same
    spectral closed forms as :class:`SpectralDiag`). ``D`` carries the marginal
    variances, the core carries the correlation.

    With ``normalize_core`` (the default) the core is a correlation matrix: a
    real circulant has constant diagonal ``mean(s)`` (Parseval), so the scale
    gauge ``mean(s) = 1`` per channel gives ``diag(Sigma_P) = tau`` exactly and
    leaves stage 2 to learn only the *shape* of the spectrum. Setting it to
    ``False`` gives an unconstrained (free-scale) core, whose predictive
    marginal variance is ``tau * mean(s)``.

    The model is fit in two stages (see :func:`src.train.train.train_staged`):

    * **Stage 1** fits the marginals only: the mean ``mu`` and the marginal
      variances ``diag = tau``, scored on the marginals (``crps`` by default,
      or :meth:`marginal_kernel_score`). This fits ``tau`` to
      ``diag(Sigma_Q)`` independently of the core.
    * **Stage 2** fits the core ``s`` to the standardized target
      ``Z = D⁻¹ (y - mu)`` with the Gaussian kernel score, reusing
      :class:`SpectralDiag`'s spectral closed forms. By default ``mu`` and ``D``
      are frozen so the stage-1 marginals stay fixed; with
      ``method.freeze_backbone=false`` stage 2 instead finetunes all weights.

    The head emits three groups per channel -- ``mu``, ``diag`` (the marginal
    variance ``tau``) and ``spectral_diag`` (the spectrum ``s``) -- laid out as
    ``(B, C*3, *spatial)`` and reshaped by :meth:`process_output` to
    ``(B, C, *spatial, 3) = [mu, diag, s]`` (plus ``rank`` factor fields for a
    low-rank correction, paper Eqs. (11)-(12)).
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
        normalize_core: bool = True,
        temperature: float = 1.0,
        core_activation: str = "softmax",
        transform: Optional[SeparableOrthogonalTransform] = None,
        delta_renorm: Optional[bool] = None,
        rank: int = 0,
        core_logit_range: Optional[float] = None,
        core_parameterization: str = "free",
        **kwargs,
    ):
        super().__init__(
            loss, out_channels, eps, n_train_samples, n_eval_samples, gamma, gamma_eval
        )
        self.normalize_core = normalize_core
        # Temperature for the normalized-core softmax (see process_output). Only
        # used when normalize_core is True and core_activation is "softmax".
        self.temperature = temperature
        # How the normalized core maps head outputs to a unit-mean spectrum:
        # "softmax" (default, s = d softmax(h_s), paper Eq. (9)) treats the head
        # outputs as logits -- one coupled map whose per-bin gradients scale
        # with the bin's own probability, so bins driven to very small s see
        # vanishing gradients; "softplus" applies the elementwise softplus and
        # divides by the mean (s <- s / mean(s)) -- the same scale gauge
        # (mean(s) = 1), but elementwise gradients without cross-bin
        # competition. Only consulted when normalize_core is True.
        if core_activation not in ("softmax", "softplus"):
            raise ValueError(
                f"core_activation must be 'softmax' or 'softplus', got "
                f"{core_activation!r}"
            )
        self.core_activation = core_activation
        # Optional non-Fourier transform for the core. None (default) keeps the
        # native FFT path (circulant core); a SeparableOrthogonalTransform
        # switches every score/sampler to the general real-orthogonal path with
        # C = Bᵀ diag(s) B (e.g. the graph Fourier transform on EUPPBench).
        #
        # Whether to renormalize by the implied marginal variances is chosen
        # automatically from the transform: for a *flat* basis (|B_ki|² = 1/d --
        # DFT, Hadamard) the unit-mean spectrum already yields a correlation
        # core; for a localized basis (Haar, graph eigenbases) the diagonal of
        # the core varies, so the predictive covariance is renormalized,
        # Sigma_P = D̃ Bᵀ diag(s) B D̃ with D̃ = diag(sqrt(tau / Δ)) and
        # Δ = diag(Bᵀ diag(s) B) (paper Eq. (10)), which restores
        # diag(Sigma_P) = tau exactly for every spectrum.
        # `delta_renorm` overrides the automatic choice (ablation only).
        self.transform = transform
        # Optional low-rank correction of the core (paper Eqs. (11)-(12)), on
        # the native Fourier path and on any SeparableOrthogonalTransform
        # alike: Sigma_hat = diag(s) + Û Û* in the B-domain, with the rank-r
        # factors parameterized as *real fields in physical space* (on the
        # Fourier path their FFT is then automatically conjugate-symmetric,
        # keeping the core real; a real-orthogonal B needs no constraint). The
        # transformed core is Bᵀ diag(s) B + sum_j u_j u_jᵀ with diagonal
        # Δ_i = Δ_i(s) + sum_j u_j(i)², so the renormalization by Δ is
        # required (and enabled) whenever rank > 0 -- even for flat bases.
        #
        # Hard cap on the spread of the Matern log-spectrum, which bounds the
        # core's condition number by exp(range), so the Cholesky in the NLL and
        # in the masked kernel score stays well conditioned. None disables the
        # cap. Only the "matern" parameterization applies it (see
        # `_matern_spectrum`); there it is a backstop, since the Matern dynamic
        # range is already bounded analytically by
        # ((kappa²+lambda_max)/(kappa²+lambda_min))^(2 alpha).
        self.core_logit_range = None if core_logit_range is None else float(core_logit_range)
        # How the head's output becomes the spectrum.
        #
        # "free" (default): one logit per mode, i.e. d degrees of freedom.
        #
        # "matern": a Matern filter on the graph Fourier basis,
        #     g(lambda) = sigma * (kappa^2 + lambda)^-alpha,   s_k = g(lambda_k)^2,
        # the graph analogue of the Matern/SPDE spectral density -- the SPDE
        # (kappa^2 - L)^(alpha) u = W on the graph (L the graph Laplacian) has
        # exactly this diagonal core in the Laplacian eigenbasis. The head emits
        # *three scalars per case* instead of d logits, so the spectrum is
        # evaluated rather than learned pointwise. Consequences:
        #   * s_k is monotone in lambda_k by construction, so the ringing a free
        #     spectrum can produce on a graph basis is unrepresentable;
        #   * repeated eigenvalues get equal s_k automatically, so the model
        #     does not depend on which rotation `eigh` returns inside a
        #     degenerate eigenspace;
        #   * it transfers to a different station set, since theta does not
        #     carry a per-mode index.
        # NOTE sigma is removed by the scale gauge (s and c*s give the same
        # correlation core), so the identifiable content is (kappa, alpha).
        # The sigma output is kept for faithfulness to the parameterization
        # and simply receives no gradient.
        if core_parameterization not in ("free", "matern"):
            raise ValueError(
                f"core_parameterization must be 'free' or 'matern', got "
                f"{core_parameterization!r}"
            )
        self.core_parameterization = core_parameterization
        if core_parameterization == "matern":
            if getattr(transform, "eigenvalues", None) is None:
                raise ValueError(
                    "core_parameterization='matern' evaluates the spectral "
                    "density at the basis eigenvalues, so it needs a transform "
                    "carrying them (build it with make_graph_transform)."
                )
            if int(rank) != 0:
                raise NotImplementedError(
                    "core_parameterization='matern' is implemented for rank=0; "
                    f"got rank={rank}. The low-rank factors are per-station "
                    "fields and do not fit the 3-scalar head."
                )
        elif self.core_logit_range is not None:
            # The cap is applied to the analytic Matern log-spectrum. A free
            # spectrum is gauge-fixed by the softmax/softplus instead and has no
            # log-spectrum to clamp, so accepting the knob there would silently
            # do nothing.
            raise ValueError(
                "core_logit_range applies to core_parameterization='matern'; "
                f"got {core_parameterization!r}."
            )
        self.rank = int(rank)
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")
        if self.rank > 0:
            self._use_delta = True
        elif delta_renorm is None:
            self._use_delta = transform is not None and not transform.is_flat
        else:
            self._use_delta = bool(delta_renorm) and transform is not None
        # The spectral closed forms (stage-2 kernel_score / nll, paper Eqs.
        # (17), (19)) are reused by composition: this delegate owns the rfft +
        # conjugacy machinery. It is
        # configured with the same bandwidths/eps; its `training` flag is mirrored
        # at call time so it picks the matching gamma. Only used on the Fourier
        # path (transform is None).
        self._spectral = SpectralDiag(
            loss,
            eps,
            n_train_samples,
            n_eval_samples,
            out_channels,
            gamma,
            gamma_eval,
        )

    @property
    def group_widths(self) -> dict:
        # Mean + marginal variance + core spectrum (+ `rank` low-rank factor
        # fields); spectrum and factors share the customizable "other" head on
        # the backbone.
        C = self._out_channels
        if self.core_parameterization == "matern":
            # (sigma, kappa, alpha) rather than one logit per mode. The head
            # still emits them as a field over the event axis; `process_output`
            # pools it, which with a global readout is exactly a 3-output linear
            # map (pooling is linear and composes with the head's own linear map).
            return {"mu": C, "sigma": C, "other": 3 * C}
        return {"mu": C, "sigma": C, "other": C * (1 + self.rank)}

    @property
    def other_zero_init(self) -> bool:
        # The low-rank factors enter the core quadratically, so dL/dU vanishes
        # identically at U = 0 (the same saddle as LorD): the backbone must
        # break the symmetry with a small random init. The diagonal-core head
        # keeps the zero init.
        return self.rank == 0

    def process_output(self, output_head: torch.Tensor) -> torch.Tensor:
        # output_head: (B, C*(3+r), d1, ..., dN) -> (B, C, d1, ..., dN, 3+r)
        # with trailing channels [mu, diag, s, u_1, ..., u_r].
        B, _, *spatial = output_head.shape
        n_groups = 5 if self.core_parameterization == "matern" else 3 + self.rank
        output = output_head.reshape(B, self._out_channels, n_groups, *spatial)
        output = output.movedim(2, -1)
        mu = output[..., 0:1]
        diag = output[..., 1:2]
        diag = F.softplus(diag) + self.eps

        if self.core_parameterization == "matern":
            spectral_diag = self._matern_spectrum(output[..., 2:5])
            return torch.cat([mu, diag, spectral_diag], dim=-1)

        spectral_diag = output[..., 2:3]
        # Low-rank factors stay unconstrained (linear map), like LorD's U.
        lora = output[..., 3:]
        if self.normalize_core:
            spatial_dims = list(range(2, spectral_diag.dim() - 1))
            if self.core_activation == "softmax":
                n_spatial = math.prod(spectral_diag.shape[2:-1])
                # s = d softmax(h_s / T) over the coefficient axes: T > 1
                # flattens the spectrum toward uniform, T < 1 sharpens it. T = 1
                # recovers the plain softmax. The n_spatial rescale keeps
                # mean(s) = 1 for any T.
                logits = spectral_diag / self.temperature
                log_s = logits - torch.logsumexp(
                    logits, dim=spatial_dims, keepdim=True
                )
                spectral_diag = n_spatial * torch.exp(log_s)
            else:
                # Elementwise softplus, explicitly renormalized to mean(s) = 1:
                # the same scale gauge as the softmax but without saturation --
                # each bin's gradient is local instead of scaling with the
                # bin's own probability mass.
                spectral_diag = F.softplus(spectral_diag) + self.eps
                spectral_diag = spectral_diag / spectral_diag.mean(
                    dim=spatial_dims, keepdim=True
                )
            # Floor that keeps the core positive definite (preserves mean 1).
            spectral_diag = (spectral_diag + self.eps) / (1.0 + self.eps)
        else:
            spectral_diag = F.softplus(spectral_diag) + self.eps
        # Fourier path only: the spectrum must be conjugate-symmetric
        # (s_k = s_{d-k}) so Fᴴ diag(s) F is a real circulant. The mirror is a
        # permutation of the
        # frequency axes, so averaging with it preserves mean(s) = 1 above. A
        # real-orthogonal transform needs no such constraint.
        if self.transform is None:
            spectral_diag = SpectralDiag._symmetrize_spectrum(spectral_diag)
        return torch.cat([mu, diag, spectral_diag, lora], dim=-1)

    def _matern_spectrum(self, raw: torch.Tensor) -> torch.Tensor:
        """Whittle-Matern spectrum from three per-case scalars.

        ``raw`` is ``(B, C, *spatial, 3)`` -- the head emits the parameters as a
        field, and they are pooled over the event axes to one triple per case,
        because a spectral density is a property of the forecast day and not of
        any one station. With a global readout the pooling is exact rather than
        lossy: both it and the head are linear, so the composition is precisely a
        3-output linear map on the pooled features.

        Then, with ``lambda_k`` the basis eigenvalues,

            log s_k = 2 log sigma - 2 alpha log(kappa^2 + lambda_k),

        capped in spread (``core_logit_range``) and fixed to the scale gauge
        ``mean(s) = 1``. The gauge removes ``sigma`` entirely, since ``s`` and
        ``c*s`` define the same correlation core and no score can identify the
        ray. ``kappa`` sets the correlation range, ``alpha`` the smoothness.
        """
        spatial_dims = list(range(2, raw.dim() - 1))
        theta = raw.mean(dim=spatial_dims, keepdim=True)  # (B, C, 1.., 3)
        # softplus keeps all three positive; the offsets keep kappa^2 away from 0
        # and alpha away from 0, which is what bounds the dynamic range:
        # max(s)/min(s) = ((kappa^2 + lambda_max) / (kappa^2 + lambda_min))^(2 alpha).
        log_sigma = F.softplus(theta[..., 0:1]).log()
        kappa2 = F.softplus(theta[..., 1:2]) + 1e-3
        alpha = F.softplus(theta[..., 2:3]) + 1e-3

        lam = self.transform.eigenvalues.to(device=raw.device, dtype=raw.dtype)
        # Broadcast the eigenvalues along the event axis: mode k sits at event
        # position k, which is the same convention `_dense_score_terms` uses when
        # it contracts `s` against the basis rows.
        shape = [1, 1] + [1] * len(spatial_dims) + [1]
        shape[2] = lam.numel()
        lam = lam.reshape(shape).clamp_min(0.0)

        log_s = 2.0 * log_sigma - 2.0 * alpha * torch.log(kappa2 + lam)
        if self.core_logit_range is not None:
            spread = log_s.amax(dim=2, keepdim=True) - log_s.amin(dim=2, keepdim=True)
            log_s = log_s * (self.core_logit_range / spread.clamp_min(self.eps)).clamp(max=1.0)
        # Gauge: mean(s) = 1, in log space so the exponent cannot overflow.
        n_spatial = math.prod(raw.shape[2:-1])
        log_s = log_s - torch.logsumexp(log_s, dim=spatial_dims, keepdim=True)
        s = n_spatial * torch.exp(log_s)
        return (s + self.eps) / (1.0 + self.eps)

    # ------------------------------------------------------------------
    # General (non-Fourier) transform path
    # ------------------------------------------------------------------

    def _scale2(
        self,
        diag: torch.Tensor,
        s: torch.Tensor,
        lora: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Effective per-coordinate variance factor ``D̃²`` composing with the core.

        The predictive covariance is ``Sigma_P = D̃ C D̃`` with ``C`` the
        transformed core. On the Fourier path (and without renormalization)
        ``D̃² = diag = tau``; with the renormalization by the implied marginal
        variances active, ``D̃² = tau / Δ`` (paper Eq. (10)) -- diagonal matrices
        commute, so ``D Δ^{-1/2} C Δ^{-1/2} D = D̃ C D̃`` and the standardization
        keeps its structure with a rescaled per-coordinate scale. With low-rank
        factors the core diagonal gains one term per factor,
        ``Δ_i = Δ_i(s) + sum_j u_j(i)²`` (paper Eq. (12)), where the diagonal
        part is ``mean(s)`` on the Fourier path (constant by Parseval) and
        ``transform.delta(s)`` on the general path.
        """
        if not self._use_delta:
            return diag
        if self.rank > 0:
            if self.transform is None:
                spatial_dims = list(range(2, s.dim()))
                base = s.mean(dim=spatial_dims, keepdim=True)
            else:
                base = self.transform.delta(s)
            delta = base + lora.pow(2).sum(dim=-1)
        else:
            delta = self.transform.delta(s)
        return diag / delta.clamp_min(self.eps)

    def _split(self, pred: torch.Tensor):
        """Unpack the processed head into (mu, diag, s, lora-or-None)."""
        mu, diag, s = pred[..., 0], pred[..., 1], pred[..., 2]
        lora = pred[..., 3:] if self.rank > 0 else None
        return mu, diag, s, lora

    def _lora_terms(self, s: torch.Tensor, lora: torch.Tensor, resid: torch.Tensor):
        """B-domain quantities shared by the low-rank closed forms.

        Returns ``(s_flat, y_hat, U_hat)`` where ``y_hat`` / ``U_hat`` are the
        B-domain coefficients of the residual and of the ``r`` factor fields,
        flattened over ``(C, *spatial)``, and ``s_flat`` the matching spectrum.
        On the Fourier path these are the (complex, ortho-normalized) full-grid
        FFTs; on the general path they are real orthogonal coefficients. All
        downstream quantities (Woodbury inner products at shifted spectra)
        derive from these.
        """
        spatial_dims = list(range(2, resid.dim()))
        u_fields = lora.movedim(-1, 0)  # (r, B, C, *spatial)
        if self.transform is None:
            y_hat = torch.fft.fftn(resid, dim=spatial_dims, norm="ortho")
            U_hat = torch.fft.fftn(
                u_fields, dim=[d + 1 for d in spatial_dims], norm="ortho"
            )
        else:
            y_hat = self.transform.forward(resid)
            r = u_fields.shape[0]
            U_hat = self.transform.forward(
                u_fields.reshape(r * resid.shape[0], *resid.shape[1:])
            ).reshape(u_fields.shape)
        s_flat = s.flatten(start_dim=1)
        y_hat = y_hat.flatten(start_dim=1)
        U_hat = U_hat.flatten(start_dim=2)  # (r, B, n)
        return s_flat, y_hat, U_hat

    def _lora_woodbury(
        self,
        s_flat: torch.Tensor,
        y_hat: torch.Tensor,
        U_hat: torch.Tensor,
        alpha: float,
        beta: float,
    ):
        """log det and Mahalanobis of ``alpha I + beta (Bᵀ diag(s) B + U Uᵀ)``.

        Expressed in the B-domain, where the base is diagonal with entries
        ``g = alpha + beta s`` (a spectrum for the DFT, plain orthogonal
        coefficients otherwise); the rank-r part is handled by the matrix
        determinant lemma and the Woodbury identity (paper Eqs. (21)-(22)):

            log det = sum_k log g_k + log det(I_r + beta Uᵀ G⁻¹ U),
            yᵀ(·)⁻¹y = sum_k |ŷ_k|²/g_k − beta wᵀ(I_r + beta Uᵀ G⁻¹ U)⁻¹ w,

        with ``w_j = sum_k conj(û_j) ŷ / g_k``. On the Fourier path these
        inner products are real because the factors are real fields and the
        spectrum is conjugate-symmetric; on the general real-orthogonal path
        every quantity is real to begin with and the conj/.real calls are
        no-ops.
        """
        r = U_hat.shape[0]
        g = alpha + beta * s_flat  # (B, n)
        log_det_circ = torch.log(g).sum(dim=-1)
        maha0 = (y_hat.abs().pow(2) / g).sum(dim=-1)

        # Cross terms: w_j = <u_j, G^-1 y>, M_jl = <u_j, G^-1 u_l> (both real).
        Ug = U_hat / g.unsqueeze(0)  # (r, B, n)
        w = (Ug.conj() * y_hat.unsqueeze(0)).sum(dim=-1).real  # (r, B)
        M = (Ug.conj().unsqueeze(1) * U_hat.unsqueeze(0)).sum(dim=-1).real  # (r, r, B)

        eye = torch.eye(r, device=s_flat.device, dtype=s_flat.dtype)
        cap = eye.unsqueeze(0) + beta * M.permute(2, 0, 1)  # (B, r, r)
        log_det_lr = torch.logdet(cap)
        sol = torch.linalg.solve(cap, w.T.unsqueeze(-1)).squeeze(-1)  # (B, r)
        maha = maha0 - beta * (w.T * sol).sum(dim=-1)
        return log_det_circ + log_det_lr, maha

    def _transform_multi(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply the transform over ``(B, C, *spatial, M)`` by folding M into B."""
        M = x.shape[-1]
        B = x.shape[0]
        xm = torch.movedim(x, -1, 0).reshape(M * B, *x.shape[1:-1])
        z = self.transform.inverse(xm) if inverse else self.transform.forward(xm)
        return torch.movedim(z.reshape(M, B, *x.shape[1:-1]), 0, -1)

    def _transform_terms(self, pred_w: torch.Tensor, y_w: torch.Tensor):
        """Coefficient-space residual and spectrum for the general path.

        The analogue of :meth:`SpectralDiag._spectral_terms` for a real
        orthogonal transform: no half-spectrum, no conjugacy weights — every
        coefficient is one real degree of freedom.
        """
        mu_w, s = pred_w[..., 0], pred_w[..., 1]
        diff_hat = self.transform.forward(mu_w - y_w)
        return diff_hat.flatten(start_dim=1), s.flatten(start_dim=1)

    def _whiten(self, pred: torch.Tensor, y: torch.Tensor):
        """Standardize ``mu`` and ``y`` by the per-coordinate scale ``D̃``.

        Returns ``(pred_spectral, y_w)`` where ``pred_spectral`` is a
        :class:`SpectralDiag`-shaped ``(B, C, *spatial, 2) = [mu_w, s]`` tensor
        (``mu_w = m``, the standardized mean) and ``y_w = D̃⁻¹ y`` (the
        standardized target ``Z``). Since ``D̃⁻¹(mu - y)`` is an ordinary real
        field, scoring ``N(mu_w, Fᴴ diag(s) F)`` against ``y_w`` (a spectral
        score) equals scoring the core against the standardized residual.
        """
        mu, diag, s, lora = self._split(pred)
        sqrt_scale = self._scale2(diag, s, lora).sqrt()
        mu_w = mu / sqrt_scale
        y_w = y / sqrt_scale
        pred_spectral = torch.stack([mu_w, s], dim=-1)
        return pred_spectral, y_w

    @torch.no_grad()
    def get_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # Eval/plotting entry point: the reparameterized draw with the graph
        # detached. `sample_kernel_score` calls `_draw_samples` directly, since a
        # sampling-based training loss under no_grad would optimize a constant.
        return self._draw_samples(pred, n_samples=n_samples, flatten=flatten)

    def _draw_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # Predictive N(mu, D̃ C D̃). Draw a correlation sample c with covariance
        # C = Fᴴ diag(s) F (the SpectralDiag circulant sampler), then scale per
        # coordinate by D̃ and shift by mu (paper Appendix C.2.2, "Sampling").
        # The cost is O(M·N log N).
        if n_samples is None:
            n_samples = self.n_samples
        mu, diag, s, lora = self._split(pred)
        B, _, *spatial = mu.shape
        spatial_dims = list(range(2, 2 + len(spatial)))

        if self.transform is None:
            z = torch.randn(*mu.shape, n_samples, device=mu.device, dtype=mu.dtype)
            Z = torch.fft.fftn(z, dim=spatial_dims, norm="ortho")
            c = torch.fft.ifftn(
                s.sqrt().unsqueeze(-1) * Z, dim=spatial_dims, norm="ortho"
            ).real
        else:
            # White noise stays white under a real orthogonal transform, so the
            # coefficients are sampled directly: c = Bᵀ (sqrt(s) ⊙ eps) has
            # covariance Bᵀ diag(s) B.
            eps_hat = torch.randn(
                *mu.shape, n_samples, device=mu.device, dtype=mu.dtype
            )
            c = self._transform_multi(
                s.unsqueeze(-1).sqrt() * eps_hat, inverse=True
            )
        if self.rank > 0:
            # Core = Bᵀ diag(s) B + sum_j u_j u_jᵀ: the factors are physical
            # fields on every path, so the low-rank component is added with
            # independent standard normal weights per factor & sample.
            eps2 = torch.randn(
                B, self.rank, n_samples, device=mu.device, dtype=mu.dtype
            )
            eps2 = eps2.reshape(B, *([1] * (mu.dim() - 1)), self.rank, n_samples)
            c = c + (lora.unsqueeze(-1) * eps2).sum(dim=-2)
        scale = self._scale2(diag, s, lora).sqrt()
        samples = mu.unsqueeze(-1) + scale.unsqueeze(-1) * c
        if flatten:
            samples = samples.flatten(start_dim=1, end_dim=-2)
        return samples

    def get_marg_std(self, pred: torch.Tensor) -> torch.Tensor:
        # Predictive Sigma_P = D̃ C D̃. On the Fourier path the circulant core
        # has constant diagonal mean_k s_k (Parseval), so the marginal variance
        # at a coordinate is tau * mean_k s_k; with a normalized core
        # (mean(s) = 1) this is tau exactly. On the general path the core
        # diagonal is Δ(s) per coordinate: with the renormalization active it
        # cancels exactly (marginal variance = tau for every spectrum); without
        # it (flat basis or ablation) the marginal variance is tau * Δ(s).
        mu, diag, s, lora = self._split(pred)
        if self.rank > 0:
            # The renormalization is always active with low-rank factors, so
            # the predictive marginal variance is tau exactly.
            return diag.sqrt()
        if self.transform is not None:
            if self._use_delta:
                return diag.sqrt()
            return (diag * self.transform.delta(s)).sqrt()
        spatial_dims = list(range(2, mu.dim()))
        core_var = s.mean(dim=spatial_dims, keepdim=True)
        marg_var = diag * core_var
        return marg_var.sqrt()

    def _transform_cov_block(self, s: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Rows of the raw core ``Bᵀ diag(s) B`` on the general transform path.

        One transform round trip per requested row: the columns are
        ``Bᵀ diag(s) B e_r = inverse(s * forward(e_r))``, with the unit fields
        batched over ``rows``. Channels are independent under the separable
        spatial transform, so cross-channel entries are exactly zero.
        Returns ``(B, len(rows), V)`` like the circulant block.
        """
        B_, C = s.shape[0], s.shape[1]
        spatial = tuple(s.shape[2:])
        R = int(rows.numel())
        E = torch.zeros(R, C, *spatial, device=s.device, dtype=s.dtype)
        E.reshape(R, -1)[torch.arange(R, device=s.device), rows] = 1.0
        F_r = self.transform.forward(E)  # (R, C, *spatial)
        prod = s.unsqueeze(1) * F_r.unsqueeze(0)  # (B, R, C, *spatial)
        out = self.transform.inverse(prod.reshape(B_ * R, C, *spatial))
        return out.reshape(B_, R, -1)

    def _pair_diff_var(self, pred: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # Sigma_P = D̃ (Bᵀ diag(s) B + U Uᵀ) D̃ with D̃² = _scale2 (tau on the
        # plain Fourier path, tau/Δ with the renormalization active), so the
        # off-diagonal is the raw core block rescaled by the two endpoints'
        # effective scales, plus the low-rank outer product; the diagonal is
        # `get_marg_std`² on every path. Only core rows are needed -- never
        # Sigma itself.
        mu, diag, s, lora = self._split(pred)
        var = self.get_marg_std(pred).flatten(start_dim=1).pow(2)  # (B, V)
        scale = self._scale2(diag, s, lora).sqrt().flatten(start_dim=1)  # (B, V)
        if self.transform is None:
            core = SpectralDiag._circulant_cov_block(s, rows)  # (B, R, V)
        else:
            core = self._transform_cov_block(s, rows)
        cov = scale[:, rows].unsqueeze(-1) * core * scale.unsqueeze(-2)
        if lora is not None:
            lf = lora.flatten(start_dim=1, end_dim=-2)  # (B, V, r)
            lf_s = lf * scale.unsqueeze(-1)  # D̃-scaled factors
            cov = cov + torch.einsum("brk,bvk->brv", lf_s[:, rows], lf_s)
        return var[:, rows].unsqueeze(-1) + var.unsqueeze(-2) - 2.0 * cov

    def _quadratic_form(self, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Sigma_P = D̃ (Bᵀ diag(s) B + U Uᵀ) D̃, so with b = D̃ a the form splits
        # into the core and low-rank parts:
        #   aᵀ Sigma_P a = sum_k s_k |(B b)_k|² + ||Uᵀ b||².
        # `B b` is one forward transform (an FFT on the native Fourier path),
        # which is why `a` is taken in event shape: both need the spatial axes.
        # Unlike `_dense_score_terms` this covers *every* path -- native Fourier
        # and multi-axis transforms included -- and never forms the core.
        mu, diag, s, lora = self._split(pred)
        b = a * self._scale2(diag, s, lora).sqrt()
        if self.transform is None:
            spatial_dims = list(range(2, b.dim()))
            b_hat = torch.fft.fftn(b, dim=spatial_dims, norm="ortho")
            core = (s * b_hat.abs().pow(2)).flatten(start_dim=1).sum(dim=1)
        else:
            b_hat = self.transform.forward(b)
            core = (s * b_hat.pow(2)).flatten(start_dim=1).sum(dim=1)
        if lora is None:
            return core
        lf = lora.flatten(start_dim=1, end_dim=-2)  # (B, V, r)
        lora_term = torch.einsum("bv,bvr->br", b.flatten(start_dim=1), lf)
        return core + lora_term.pow(2).sum(dim=1)

    def _nll_complete(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        # Full Gaussian NLL of the predictive N(mu, D̃ C D̃). After the diagonal
        # standardization D̃⁻¹ the Mahalanobis and core log-det reduce to the
        # coefficient-space NLL of the core on the standardized residual; the
        # only extra term is the change-of-variables Jacobian
        # log|det D̃| = sum_i 0.5*log(D̃_i²) (paper Eq. (20)).
        pred_w, y_w = self._whiten(pred, y)
        _, diag, s, lora = self._split(pred)
        if self.rank > 0:
            resid = pred_w[..., 0] - y_w
            s_flat, y_hat, U_hat = self._lora_terms(s, lora, resid)
            n = s_flat.shape[-1]
            log_det, maha = self._lora_woodbury(
                s_flat, y_hat, U_hat, alpha=0.0, beta=1.0
            )
            base = 0.5 * (n * math.log(2 * math.pi) + log_det + maha)
        elif self.transform is None:
            self._spectral.training = self.training
            base = self._spectral.nll(pred_w, y_w, reduction="none")
        else:
            diff_hat, s_flat = self._transform_terms(pred_w, y_w)
            n = diff_hat.shape[-1]
            log_det = torch.log(s_flat).sum(dim=-1)
            maha = (diff_hat.pow(2) / s_flat).sum(dim=-1)
            base = 0.5 * (n * math.log(2 * math.pi) + log_det + maha)
        scale2 = self._scale2(diag, s, lora)
        logdet_D = 0.5 * torch.log(scale2).flatten(start_dim=1).sum(dim=1)
        return self.reduce(base + logdet_D, reduction)

    def _kernel_score_single(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        # Stage-2 training loss: the kernel score of the core at the
        # standardized target (paper Eq. (19)). This is the closed-form score of
        # the *correlation* core, not of the (non-stationary) predictive
        # covariance -- eval-time kernel_score over the predictive uses the
        # ensemble estimator over get_samples instead (see
        # src/evaluation/metrics.py).
        pred_w, y_w = self._whiten(pred, y)
        if self.rank > 0:
            # Low-rank core: the closed form needs log-dets and a Mahalanobis
            # term of I + c * (Bᵀ diag(s) B + U Uᵀ) at c in {2, 4} / gamma²,
            # handled by the diagonal-base Woodbury identities in the B-domain
            # (Fourier and general orthogonal path alike).
            _, _, s, lora = self._split(pred)
            resid = pred_w[..., 0] - y_w
            s_flat, y_hat, U_hat = self._lora_terms(s, lora, resid)
            c1, c2 = 2.0 / self.gamma2, 4.0 / self.gamma2
            log_det1, maha1 = self._lora_woodbury(
                s_flat, y_hat, U_hat, alpha=1.0, beta=c1
            )
            log_det2, _ = self._lora_woodbury(
                s_flat, y_hat, U_hat, alpha=1.0, beta=c2
            )
            fac1 = torch.exp(-0.5 * log_det1 - maha1 / self.gamma2)
            fac2 = torch.exp(-0.5 * log_det2)
            score = 0.5 * (1.0 + fac2) - fac1
            return self.reduce(score, reduction)
        if self.transform is None:
            # The delegate holds its own copy of the bandwidth lists; mirror both
            # the training flag and the active bandwidth index so it scores at the
            # same single bandwidth the base `_mean_over_bandwidths` loop selects.
            self._spectral.training = self.training
            self._spectral._gamma_idx = self._gamma_idx
            return self._spectral._kernel_score_single(pred_w, y_w, reduction)

        # General path: same closed form as the spectral one, but on real
        # orthogonal coefficients -- no half-spectrum, no conjugacy weights.
        diff_hat, s_flat = self._transform_terms(pred_w, y_w)
        sq_diff = diff_hat.pow(2)

        diag1 = 1.0 + 2.0 * s_flat / self.gamma2
        diag2 = 1.0 + 4.0 * s_flat / self.gamma2

        log_det1 = torch.log(diag1).sum(dim=-1)
        log_det2 = torch.log(diag2).sum(dim=-1)
        maha = (sq_diff / diag1).sum(dim=-1)

        fac1 = torch.exp(-0.5 * log_det1 - maha / self.gamma2)
        fac2 = torch.exp(-0.5 * log_det2)
        score = 0.5 * (1.0 + fac2) - fac1
        return self.reduce(score, reduction)

    def _marginal_kernel_score_per_coord(
        self, pred: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Stage-1 loss alternative to crps: the univariate (d=1) Gaussian kernel
        # score per coordinate, averaged over the flattened event (like crps).
        # The bandwidth is the *marginal* (1-D) one: the configured gamma2 is the
        # joint d-dimensional bandwidth gamma2 = factor * d, so dividing by the
        # event size d recovers the per-coordinate factor. Using the joint gamma2
        # would over-smooth a single coordinate and flatten the score.
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

    def _dense_score_terms(self, pred: torch.Tensor, y: torch.Tensor):
        """The dense correlation core and standardized residual, for masked scoring.

        Overrides the physical-covariance default because
        :meth:`_kernel_score_single` here scores the *core* against the
        standardized target, not the predictive ``Sigma_P = D̃ C D̃``. The
        standardization is a diagonal map, so it commutes with selecting a
        coordinate subset -- masking in physical space is masking in
        standardized space, and the same dense masked score applies unchanged.

        The core ``Bᵀ diag(s) B + U Uᵀ`` (the low-rank term absent at ``rank=0``)
        is built explicitly by :meth:`_dense_core`.
        """
        pred_w, y_w = self._whiten(pred, y)
        core = self._dense_core(pred)
        resid = (pred_w[..., 0] - y_w).flatten(start_dim=1)
        return core, resid

    def _dense_core(self, pred: torch.Tensor) -> torch.Tensor:
        """The dense ``(B, V, V)`` correlation core ``Bᵀ diag(s) B + U Uᵀ``.

        Shared by the two dense paths -- :meth:`_dense_score_terms` (the core
        against the standardized residual) and :meth:`_deployed_dense_terms` (the
        same core rescaled to ``Sigma_P``). At the event dimensions that reach
        either, a ``(V, V)`` matmul is cheaper than any structured detour.
        """
        _, _, s, lora = self._split(pred)
        s_flat = s.flatten(start_dim=1)  # (B, V)

        if self.transform is None:
            # The native Fourier path assumes a periodic regular grid, which is
            # exactly the setting that has no missing targets -- and over an
            # irregular station set a circulant core is meaningless anyway (use
            # a graph eigenbasis transform instead). Rather than silently score
            # a different object, refuse.
            raise NotImplementedError(
                "The dense core is not available on the native Fourier path of "
                "SCORE (masked kernel_score, dense_kernel_score). "
                "Configure method.transform (e.g. a graph or DCT basis) for "
                "targets with missing values."
            )
        # A deferred transform (analytic basis, shape read off the data) may not
        # have built its factors yet: the callers only touch the transform when
        # the renormalization by Δ is active. This is the one path that needs
        # the basis matrix itself, so ask for it explicitly.
        self.transform.materialize(s)
        if len(self.transform.matrices) != 1:
            raise NotImplementedError(
                "The dense core supports a single-axis transform only; got "
                f"{len(self.transform.matrices)} axes."
            )
        basis = self.transform.matrices[0].to(device=s.device, dtype=s.dtype)
        core = torch.einsum("ki,bk,kj->bij", basis, s_flat, basis)

        if self.rank > 0:
            factors = lora.flatten(start_dim=1, end_dim=-2)  # (B, V, rank)
            core = core + factors @ factors.transpose(-1, -2)

        return core

    def _deployed_dense_terms(self, pred: torch.Tensor, y: torch.Tensor):
        """The predictive ``Sigma_P = D̃ C D̃`` and the *physical* residual ``mu - y``.

        The override SCORE needs for `dense_kernel_score`: unlike
        :meth:`_dense_score_terms`, which describes the bare correlation core
        against the standardized residual, this is the predictive covariance
        and the residual in the space the data lives in. Built directly
        from :meth:`_dense_core` rather than through the base class's
        ``Sigma_ij = 0.5 (Sigma_ii + Sigma_jj - Var(X_i - X_j))`` identity, which
        would reach the same matrix by a lossier route.

        ``Sigma_P`` is positive definite (``s > 0``) and the kernel score only
        ever factorizes ``I + 2 Sigma / gamma²``, so the Cholesky downstream is
        unconditionally safe.
        """
        mu, diag, s, lora = self._split(pred)
        core = self._dense_core(pred)
        scale = self._scale2(diag, s, lora).sqrt().flatten(start_dim=1)  # (B, V) = D̃
        sigma = scale.unsqueeze(-1) * core * scale.unsqueeze(-2)
        return sigma, (mu - y).flatten(start_dim=1)

    def _dense_logdet_offset(self, pred: torch.Tensor, mask) -> torch.Tensor:
        """``log|det D̃|`` over the observed coordinates.

        :meth:`_dense_score_terms` describes the correlation core ``C`` against
        the standardized residual, but the predictive covariance is
        ``Sigma_P = D̃ C D̃``. The likelihood therefore picks up the
        change-of-variables Jacobian of the standardization,
        ``log|det D̃| = 0.5 * sum_i log D̃_i²`` -- restricted here to
        the observed coordinates, since those are the only ones being scored.
        The kernel score needs no such term: it scores the core directly.
        """
        _, diag, s, lora = self._split(pred)
        per_coord = 0.5 * torch.log(self._scale2(diag, s, lora)).flatten(start_dim=1)
        if mask is None:
            return per_coord.sum(dim=1)
        return (per_coord * mask).sum(dim=1)

    def energy_score(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        # Sliced estimator in physical space for the predictive covariance
        # Sigma_P = D̃ C D̃. The pushforward variance of a direction v is
        # vᵀ Sigma_P v = (D̃ v)ᵀ C (D̃ v) = sum_k s_k |(B (D̃ v))_k|², i.e. the
        # coefficient-space quadratic form with the slice direction pre-scaled
        # by D̃ (= sqrt(tau) on the Fourier path, sqrt(tau/Δ) with the
        # renormalization active).
        mu, diag, s, lora = self._split(pred)
        B, _, *spatial = mu.shape
        spatial_dims = list(range(2, 2 + len(spatial)))
        diff = (mu - y).flatten(start_dim=1)
        d = diff.shape[-1]
        M = self.n_samples if n_samples is None else n_samples

        z = torch.randn(B, d, M, device=mu.device, dtype=mu.dtype)
        v = z / z.norm(dim=1, keepdim=True)  # (B, d, M) unit dirs on S^{d-1}

        # Scale the slice direction by D̃ per coordinate before the transform.
        v_field = v.reshape(B, *mu.shape[1:], M)
        v_field = self._scale2(diag, s, lora).sqrt().unsqueeze(-1) * v_field
        if self.transform is None:
            Fv = torch.fft.fftn(v_field, dim=spatial_dims, norm="ortho")
        else:
            Fv = self._transform_multi(v_field)
        sv2 = (s.unsqueeze(-1) * Fv.abs().pow(2)).sum(
            dim=list(range(1, mu.dim()))
        )  # (B, M)
        if self.rank > 0:
            # Low-rank contribution to the pushforward variance:
            # vᵀ Σ v += sum_j (u_jᵀ D̃ v)².
            lora_flat = lora.flatten(start_dim=1, end_dim=-2)  # (B, d, r)
            v_scaled = v_field.flatten(start_dim=1, end_dim=-2)  # (B, d, M)
            proj = torch.einsum("bdr,bdm->brm", lora_flat, v_scaled)
            sv2 = sv2 + proj.pow(2).sum(dim=1)
        s_slice = sv2.clamp_min(self.eps).sqrt()
        w_slice = torch.einsum("bdm,bd->bm", v, diff) / s_slice  # (B, M)

        Phi = 0.5 * (1 + torch.erf(w_slice / math.sqrt(2)))
        phi = torch.exp(-0.5 * w_slice * w_slice) / math.sqrt(2 * math.pi)
        crps = s_slice * (
            w_slice * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi)
        )  # (B, M)

        logCd = 0.5 * math.log(math.pi) + math.lgamma((d + 1) / 2) - math.lgamma(d / 2)
        score = math.exp(logCd) * crps.mean(dim=-1)  # (B,)
        return self.reduce(score, reduction)

    def _crps_per_coord(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Stage-1 marginal CRPS (paper Eq. (13)): scores mu/diag as the
        # marginals N(mu_i, tau_i), *independently of the core*. Stage 1 fits
        # (and then freezes) mu/diag, so this loss must not depend on s. With a
        # normalized core (mean(s) = 1) these *are* the predictive marginals
        # exactly; with a free core the predictive marginal variance is
        # tau * mean(s). Either way the predictive is scored at eval time by the
        # sample-based `crps` metric (see src/evaluation/metrics.py), which keeps
        # CRPS comparable to the sample-based baseline.
        mu, diag = pred[..., 0], pred[..., 1]
        sigma_marg = torch.sqrt(diag)
        return sr.crps_normal(y, mu, sigma_marg, backend="torch")
