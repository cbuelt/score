from typing import Optional

import torch
import torch.nn.functional as F
import math
import scoringrules as sr
from src.methods import PredictiveMethod


class SpectralDiag(PredictiveMethod):
    """Unnormalized Fourier-diagonal Gaussian, fitted jointly (ablation).

    Per channel the head predicts a physical-space mean ``mu`` and one positive
    per-bin spectral variance ``s_k``. The implied covariance is the real
    circulant ``Sigma = Fᴴ diag(s) F`` (a stationary covariance with constant
    marginal variance ``mean(s)``). Unlike :class:`~src.methods.score.SCORE`
    there are no separate marginal variances and no scale gauge: the spectrum
    carries both marginals and correlation and is fitted in a single stage.
    This class is an ablation and is not part of the main results; SCORE reuses
    its spectral closed forms and samplers.

    All probabilistic losses are evaluated in spectral space using the
    ``rfft`` half-spectrum together with conjugacy weights ``w_k`` (see
    :meth:`_spectral_terms`). A real signal's FFT is Hermitian-symmetric, so the
    full FFT carries only ``n`` real degrees of freedom. Keeping the ``rfftn``
    half and weighting every retained bin by its conjugate multiplicity makes
    every loss equal the physical-space score of ``N(mu, Fᴴ diag(s) F)``
    (Parseval / unitary invariance, with ``norm="ortho"``).
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
        # Mean + per-bin spectral variance (the "sigma" slot holds the spectral
        # diagonal here).
        return {"mu": self._out_channels, "sigma": self._out_channels, "other": None}

    def process_output(self, output_head: torch.Tensor) -> torch.Tensor:
        # output_head: (B, C*2, d1, ..., dN) -> (B, C, d1, ..., dN, 2)
        B, _, *spatial = output_head.shape
        output = output_head.reshape(B, self._out_channels, 2, *spatial)
        output = output.movedim(2, -1)
        mu = output[..., 0:1]
        spectral_diag = output[..., 1:2]
        spectral_diag = F.softplus(spectral_diag) + self.eps
        spectral_diag = self._symmetrize_spectrum(spectral_diag)
        return torch.cat([mu, spectral_diag], dim=-1)

    @staticmethod
    def _symmetrize_spectrum(s: torch.Tensor) -> torch.Tensor:
        """Average ``s`` with its conjugate-mirror so ``s_k = s_{-k}``.

        ``Sigma = Fᴴ diag(s) F`` is a *real* circulant only when ``s`` is
        conjugate-symmetric; for a free positive field it is Hermitian but
        complex, and the three consumers of ``s`` (the sampler, the spectral
        closed forms, and the sliced energy score) then describe three different
        distributions -- the sampler takes ``Re(Sigma)`` (marginal variance
        ``½(mean(s) + mean(√(s_k s_{-k})))``) while ``_spectral_terms`` *assumes*
        the dropped half mirrors the kept one (marginal variance ``mean(s)``).

        Enforcing ``s_k = s_{-k}`` at the source makes ``Sigma`` a genuine real
        circulant (constant marginal variance ``mean(s)``) so all paths agree.
        The mirror ``k -> -k (mod N)`` on each spatial axis is ``flip`` followed
        by ``roll(+1)`` (``flip`` sends index ``i -> N-1-i``; the ``roll``
        shifts that to ``N-i``, i.e. ``-i``), fixing the self-conjugate DC and
        Nyquist bins.

        ``s`` has a trailing singleton axis (the stacked-group axis), so the
        spectral axes are ``2 .. dim-2``.
        """
        spatial_dims = list(range(2, s.dim() - 1))
        mirror = torch.flip(s, dims=spatial_dims)
        mirror = torch.roll(mirror, shifts=(1,) * len(spatial_dims), dims=spatial_dims)
        return 0.5 * (s + mirror)

    @torch.no_grad()
    def get_samples(
        self,
        pred: torch.Tensor,
        n_samples: Optional[int] = None,
        flatten: bool = False,
    ) -> torch.Tensor:
        # Circulant Gaussian: Sigma = Fᴴ diag(s) F with F the unitary DFT. A
        # sample is mu + Fᴴ (sqrt(s) ⊙ white), drawn directly in spectral space
        # and transformed back so the cost is O(M·N log N) rather than the O(N³)
        # dense Cholesky. A real white field stays white under the unitary F, so
        # scaling its spectrum by sqrt(s) and inverting yields covariance
        # Fᴴ diag(s) F; taking the real part gives a real field with that
        # covariance (s is conjugate-symmetric for a real circulant Sigma).
        if n_samples is None:
            n_samples = self.n_samples
        mu, s = pred[..., 0], pred[..., 1]
        B, _, *spatial = mu.shape
        spatial_dims = list(range(2, 2 + len(spatial)))

        z = torch.randn(*mu.shape, n_samples, device=mu.device, dtype=mu.dtype)
        Z = torch.fft.fftn(z, dim=spatial_dims, norm="ortho")
        X = torch.fft.ifftn(
            s.sqrt().unsqueeze(-1) * Z, dim=spatial_dims, norm="ortho"
        )
        samples = mu.unsqueeze(-1) + X.real
        if flatten:
            samples = samples.flatten(start_dim=1, end_dim=-2)
        return samples

    def get_marg_std(self, pred: torch.Tensor) -> torch.Tensor:
        # Circulant Gaussian Sigma = Fᴴ diag(s) F. A unitary DFT has |F_ik|² = 1/N,
        # so every diagonal entry equals mean_k s_k (Parseval): the marginal
        # variance is constant across grid points, per channel. Broadcast that
        # per-channel constant back over the spatial grid.
        mu, s = pred[..., 0], pred[..., 1]
        spatial_dims = list(range(2, mu.dim()))
        marg_var = s.mean(dim=spatial_dims, keepdim=True)
        return marg_var.sqrt().expand_as(mu)

    @staticmethod
    def _circulant_cov_block(s: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """``Sigma_ij`` block of the circulant ``Sigma = Fᴴ diag(s) F``.

        ``s`` is the (conjugate-symmetric) spectral diagonal with event shape
        ``(B, C, *spatial)``; ``rows`` are flat event indices into
        ``V = C * prod(spatial)``. Returns ``(B, len(rows), V)``.

        A circulant covariance depends on ``(i, j)`` only through the *lag*
        ``Delta = (i - j) mod spatial``, and that whole autocovariance function is
        one inverse FFT of the spectrum: with ``F`` the unitary DFT
        (``norm="ortho"``, as everywhere else in this class),

            Sigma_ij = (1/n) sum_k s_k exp(2*pi*i*k.Delta/n) = ifftn(s).real[Delta]

        with the *default* ``norm="backward"`` supplying the ``1/n``. So the
        d x d matrix is never formed: one FFT gives the ``(C, *spatial)``
        autocovariance and each block is a gather at the pair lags. Consistent
        with :meth:`get_marg_std`, since ``ifftn(s).real[0] = mean(s)``.

        The FFT runs over the spatial axes only, so ``Sigma`` is block-diagonal
        across channels: cross-channel entries are exactly zero.
        """
        B, C, *spatial = s.shape
        n = math.prod(spatial)
        spatial_dims = list(range(2, 2 + len(spatial)))
        device = s.device

        # Autocovariance c[Delta], flattened over the spatial axes.
        c = torch.fft.ifftn(s, dim=spatial_dims).real.reshape(B, C, n)

        # Split the flat event index into (channel, spatial position), then
        # accumulate the flat lag axis by axis: for each spatial axis the lag
        # coordinate is (coord_i - coord_j) mod size, weighted by its stride.
        chan_rows, pos_rows = rows // n, rows % n
        pos = torch.arange(n, device=device)
        lag = torch.zeros(rows.shape[0], n, dtype=torch.long, device=device)
        stride = 1
        for size in reversed(spatial):
            ci = (pos_rows // stride) % size
            cj = (pos // stride) % size
            lag = lag + ((ci.unsqueeze(-1) - cj.unsqueeze(0)) % size) * stride
            stride *= size

        # Each row reads its own channel's autocovariance plane at those lags.
        plane = c[:, chan_rows]  # (B, len(rows), n)
        cov = plane.gather(2, lag.unsqueeze(0).expand(B, -1, -1))

        if C == 1:
            return cov
        # C > 1: scatter each row's plane into its own channel slot, leaving the
        # cross-channel columns at zero.
        out = s.new_zeros(B, rows.shape[0], C, n)
        out[:, torch.arange(rows.shape[0], device=device), chan_rows] = cov
        return out.reshape(B, rows.shape[0], C * n)

    def _pair_diff_var(self, pred: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # Sigma_ii + Sigma_jj - 2 Sigma_ij, with the constant circulant diagonal
        # from `get_marg_std` (Parseval: mean(s)) and the off-diagonal from the
        # lag gather above.
        s = pred[..., 1]
        var = self.get_marg_std(pred).flatten(start_dim=1).pow(2)  # (B, V)
        cov = self._circulant_cov_block(s, rows)
        return var[:, rows].unsqueeze(-1) + var.unsqueeze(-2) - 2.0 * cov

    def _quadratic_form(self, pred: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Sigma = Fᴴ diag(s) F with F the unitary DFT, so for a real field `a`
        #   aᵀ Sigma a = (F a)ᴴ diag(s) (F a) = sum_k s_k |(F a)_k|²,
        # one forward FFT rather than the lag gather `_circulant_cov_block` needs
        # for individual entries. The FFT runs over the spatial axes only, so the
        # per-channel blocks stay independent -- matching the block-diagonal
        # structure this covariance has across channels.
        s = pred[..., 1]
        spatial_dims = list(range(2, a.dim()))
        a_hat = torch.fft.fftn(a, dim=spatial_dims, norm="ortho")
        return (s * a_hat.abs().pow(2)).flatten(start_dim=1).sum(dim=1)

    @staticmethod
    def _spectral_terms(pred: torch.Tensor, y: torch.Tensor):
        """Build the conjugacy-weighted half-spectrum terms shared by all losses.

        This is the single place the rfft + conjugacy convention lives.

        ``pred[..., 0]`` is the physical-space mean ``mu`` and ``pred[..., 1]``
        is the spectral diagonal ``s`` (a *variance*, one value per physical
        bin). We take ``rfftn`` of ``mu`` and ``y`` (``norm="ortho"``), keep the
        non-redundant half of ``s`` along the last spatial axis, and form the
        conjugacy weight ``w_k`` over that last axis:

            w_k = 1 for self-conjugate columns (index 0 and, when the last
                  spatial axis is even, its Nyquist bin N/2);
            w_k = 2 otherwise (each retained bin stands in for itself *and* its
                  dropped complex conjugate).

        ``rfftn`` keeps the full extent on every axis but the last, so
        row-conjugate pairs are already present and the weight is a 1-D mask
        over the last axis broadcast over the rest.

        Returns four tensors flattened over ``(C, *half_spatial)``:
            re_diff, im_diff : real / imaginary parts of ``rfft(mu) - rfft(y)``
            s_half           : the half-spectrum variances ``s_k``
            w                : the conjugacy weights ``w_k``
        """
        mu, s = pred[..., 0], pred[..., 1]
        B, _, *spatial = mu.shape
        spatial_dims = list(range(2, 2 + len(spatial)))

        rfft_mu = torch.fft.rfftn(mu, dim=spatial_dims, norm="ortho")
        rfft_y = torch.fft.rfftn(y, dim=spatial_dims, norm="ortho")
        diff = rfft_mu - rfft_y

        # Keep the non-redundant half of s along the last spatial axis.
        n_last = rfft_mu.shape[-1]
        s_half = s[..., :n_last]

        # Conjugacy weights over the last (halved) spatial axis.
        full_last = mu.shape[-1]
        w_col = torch.full((n_last,), 2.0, device=mu.device, dtype=mu.dtype)
        w_col[0] = 1.0
        if full_last % 2 == 0:
            w_col[-1] = 1.0  # Nyquist bin is self-conjugate
        # broadcast over (B, C, *half_spatial_but_last, n_last)
        w = w_col.reshape(*([1] * (diff.dim() - 1)), n_last).expand_as(diff.real)

        re_diff = diff.real.flatten(start_dim=1)
        im_diff = diff.imag.flatten(start_dim=1)
        s_half = s_half.flatten(start_dim=1)
        w = w.flatten(start_dim=1)
        return re_diff, im_diff, s_half, w

    def _nll_complete(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        re_diff, im_diff, s, w = self._spectral_terms(pred, y)
        # n = physical event dimension = C * prod(spatial). Each self-conjugate
        # column contributes its bins once (w=1) and each conjugate-paired column
        # twice (w=2), so the weighted bin count equals n.
        n = (w).sum(dim=-1)

        log_det = torch.sum(w * torch.log(s), dim=-1)
        maha = torch.sum(w * (re_diff.pow(2) + im_diff.pow(2)) / s, dim=-1)
        nll = 0.5 * (n * math.log(2 * math.pi) + log_det + maha)
        return self.reduce(nll, reduction)

    def _kernel_score_single(
        self, pred: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        re_diff, im_diff, s, w = self._spectral_terms(pred, y)
        sq_diff = re_diff.pow(2) + im_diff.pow(2)

        diag1 = 1.0 + 2.0 * s / self.gamma2
        diag2 = 1.0 + 4.0 * s / self.gamma2

        log_det1 = torch.sum(w * torch.log(diag1), dim=-1)
        log_det2 = torch.sum(w * torch.log(diag2), dim=-1)
        maha = torch.sum(w * sq_diff / diag1, dim=-1)

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
        mu, s = pred[..., 0], pred[..., 1]
        B, _, *spatial = mu.shape
        spatial_dims = list(range(2, 2 + len(spatial)))
        diff = (mu - y).flatten(start_dim=1)  # physical residual
        d = diff.shape[-1]  # physical event dimension n = C * prod(spatial)
        M = self.n_samples if n_samples is None else n_samples

        # Sliced estimator in PHYSICAL space (mirrors Diag.energy_score), but
        # the pushforward variance of a direction v uses the spectral structure:
        # for Sigma = Fᴴ diag(s) F,  vᵀ Σ v = sum_k s_k |(F v)_k|².  This is exact
        # and equals the physical energy score of N(mu, Fᴴ diag(s) F).
        z = torch.randn(B, d, M, device=mu.device, dtype=mu.dtype)
        v = z / z.norm(dim=1, keepdim=True)  # (B, d, M) unit dirs on S^{d-1}

        # vᵀ Σ v per slice via the FFT, contracting the spatial/channel axes.
        v_field = v.reshape(B, *mu.shape[1:], M)  # (B, C, *spatial, M)
        Fv = torch.fft.fftn(v_field, dim=spatial_dims, norm="ortho")
        sv2 = (s.unsqueeze(-1) * Fv.abs().pow(2)).sum(
            dim=list(range(1, mu.dim()))
        )  # (B, M)
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
        mu, s = pred[..., 0], pred[..., 1]
        # Marginal CRPS in physical space. The circulant covariance has constant
        # marginal variance sigma^2 = mean(s) (Parseval), shared across locations.
        sigma_marg = torch.sqrt(s.flatten(start_dim=1).mean(dim=1)).reshape(
            -1, *([1] * (mu.dim() - 1))
        )
        sigma_marg = sigma_marg.expand_as(mu)
        return sr.crps_normal(y, mu, sigma_marg, backend="torch")
