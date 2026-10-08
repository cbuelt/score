# --- Project root bootstrap (must run before any `src` import) ---
# Walk up to the `.project-root` marker, put the repo root on sys.path so
# `import src...` works when run as `python src/evaluation/calibration.py`, and
# export PROJECT_ROOT for Hydra's ${oc.env:PROJECT_ROOT} interpolation.
import math
import os
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / ".project-root").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        os.environ.setdefault("PROJECT_ROOT", str(_parent))
        break
# -----------------------------------------------------------------

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

from src.methods import Deterministic, PredictiveMethod
from src.methods.base import _as_list
from src.models.gaussian_predictive import LightningPredictiveModel
from src.utils import RankedLogger
from src.utils.ckpt_compat import install_hydra_target_policy_shim
from src.utils.legacy import upgrade_legacy_targets

__doc__ = """Multivariate calibration diagnostics from a trained checkpoint.

Implements the pre-rank framework of Allen, Ziegel & Ginsbourger, *Assessing the
calibration of multivariate probabilistic forecasts* (QJRMS 2024). A pre-rank
function ``rho: R^d -> R`` collapses a multivariate vector to a scalar; ranking
``rho(y)`` among ``rho(x_1), ..., rho(x_M)`` for members ``x_m ~ F`` yields a
uniform rank whenever ``F`` is auto-calibrated, and different choices of ``rho``
target different aspects of the forecast. Three diagnostics are produced, all on
the *raw* (unstandardised) vector as in the paper:

1. **Marginal PIT** -- ``Phi((y_i - mu_i) / sigma_i)`` per coordinate. Uniform
   iff the marginals are calibrated; says nothing about dependence.
2. **Mean pre-rank** ``rho_loc(x) = (1/d) sum_j x_j``, the paper's location
   diagnostic. Linear, so ``rho_loc(Y)`` is univariate Gaussian with variance
   ``a^T Sigma a`` for ``a = 1/d``: the PIT is closed-form, no sampling.
3. **Variance pre-rank** ``rho_sc(x) = (1/d) sum_j (x_j - xbar)^2``, the paper's
   scale diagnostic. Quadratic in a Gaussian, hence a generalized chi-square, so
   the rank is taken against ``M`` sampled members instead of a closed form.
4. **Dependence pre-rank** ``rho_dep(x; h) = -gamma_x(h) / s^2_x`` (Eq. 2), the
   paper's dependence diagnostic, at a ladder of lags ``h``. Sampled.
5. **Isotropy pre-rank** ``rho_iso(x; h)`` (Eq. 5) on two-dimensional grids only.
   Sampled, and derived from the same variogram table as ``rho_dep``.

On a *regular grid* the empirical variogram at a fixed lag is a shifted
difference, ``gamma_x(h) = mean_j (x_j - x_{j+h})^2 / 2`` over the pairs that
fit -- ``O(V)`` work and ``O(V)`` memory, no pair matrix. This is what makes the
dependence diagnostic affordable where the variogram *score* is not: the score
needs all ``V^2`` weighted pairs (~380 GiB at depth's ``V = 307 200``, hence the
row-blocking in :mod:`src.evaluation.variogram`), the pre-rank needs one lag.

On an *irregular* geometry (EUPPBench's station network) there is no lag
structure, so pairs are binned by geodesic distance instead. Note the pooled,
all-pairs variogram must **not** be used: ``sum_{i,j} (x_i - x_j)^2 = 2 d^2
s^2_x`` exactly, so ``rho_dep`` collapses to a constant (``-d/(d-1)`` with the
unordered-pair normalization used here) for *every* case and member alike. Its
rank would then be decided purely by tie-breaking, giving a flat histogram by
construction -- a diagnostic guaranteed to look calibrated whatever the model
does. Binning by distance is what avoids that degeneracy.

Missing observations are handled by restricting every pre-rank to the reporting
coordinates of that case. This is exact rather than an approximation: a
Gaussian's marginal on a coordinate subset *is* the corresponding sub-block. It
does assume missingness is independent of the observed value given the forecast;
the saved mask lets that be checked after the fact.

Pre-ranks are formed **per channel**, so a time-series run yields ``G = C``
independent diagnostics per case (``d = pred_len``) while the single-channel
datasets yield one (``d = V``). Grouping by channel matches the covariance
structure: only LorD couples channels, so a pre-rank flattened across them
would probe dependence the spectral families cannot represent by construction.

Usage::

    python -m src.evaluation.calibration <run-dir> [<run-dir> ...] --device cuda

where each ``<run-dir>`` is either a seed folder of the results directory
(``paths.results_dir``; ``.../<slug>/seed<N>/`` with ``config.yaml`` and
``stage2.ckpt``) or a raw Hydra run directory (``.hydra/config.yaml`` and
``checkpoints/``).

Inside the results directory, outputs are collected per experiment, next to
``tables/``::

    <experiment>/calibration/<dataset>/[<mode>/]<slug>/calibration_seed<N>.npz

so the folder holds only the configurations chosen for diagnosis (typically one
rank and one seed per model) rather than mirroring the full checkpoint tree. For a
run directory outside the results directory the file is written to
``<run-dir>/calibration.npz``.
"""

log = RankedLogger(__name__, rank_zero_only=True)

#: Bumped whenever the npz key set or their meaning changes.
#: v1: marginal / mean / variance.  v2: adds dependence + isotropy.
SCHEMA_VERSION = 2

#: Coordinates retained per case when the full marginal-PIT array is too large.
SUBSAMPLE_COORDS = 512

#: File that marks the root of a results directory; see :func:`shared_tree_root`.
SHARED_TREE_MARKER = "INDEX.md"

#: Collected diagnostics of one experiment, a sibling of its ``tables/``.
CALIBRATION_DIRNAME = "calibration"

#: Output name for run directories outside a results directory (there is no
#: experiment to collect them into). A file of this name inside a seed folder of
#: a results directory is reported as a leftover.
LEGACY_OUT_NAME = "calibration.npz"

#: Calibration objectives, in the order they are reported. Each entry is
#: ``(name, prefix, kind)``; every objective ``X`` stores ``<prefix>_pit`` (kind
#: ``"pit"``, closed form, continuous on [0, 1]) or ``<prefix>_rank`` (kind
#: ``"rank"``, sampled, integer in ``1..M+1`` with ``0`` marking an undefined
#: case). Every objective but ``mean`` also stores ``<prefix>_hist_counts``;
#: ``mean`` does not, because its array is small and complete, so a plotting
#: script can simply histogram it. Written to the npz as
#: ``objectives`` / ``objective_prefixes`` / ``objective_kinds`` so a plotting
#: script can enumerate what a file contains without hard-coding key names.
#: ``dependence`` and ``isotropy`` carry the extra ``(direction, lag)`` axes;
#: ``isotropy`` is present only for two-dimensional grids.
OBJECTIVES = (
    ("marginal", "marg", "pit"),
    ("mean", "mean", "pit"),
    ("variance", "var", "rank"),
    ("dependence", "dep", "rank"),
    ("isotropy", "iso", "rank"),
)

#: Unit lag directions on a 2-D grid. ``axis0`` is the first spatial axis (rows
#: -- latitude for ERA5, image height for depth), ``axis1`` the second. The two
#: diagonals are at distance ``h*sqrt(2)``, not ``h``, so never compare a
#: diagonal pre-rank against an axial one at the same ``h``; the paper's
#: isotropy contrast only ever pairs directions of equal distance.
DIRECTIONS_2D = (((1, 0), "axis0"), ((0, 1), "axis1"), ((1, 1), "diag"), ((1, -1), "antidiag"))
DIRECTIONS_1D = (((1,), "axis0"),)

#: Default geodesic bin edges (km) for the irregular-geometry variogram. Chosen
#: for the EUPPBench station network: the GNN backbone's message passing cuts off
#: at 100 km, and the free and Matern-filtered cores differ mostly below 50 km
#: and at 100-200 km, so those are the scales worth resolving.
DEFAULT_BIN_EDGES_KM = (0.0, 50.0, 100.0, 200.0, float("inf"))

#: Largest usable lag as a fraction of the axis extent. Past roughly a quarter
#: of the domain the variogram averages few, highly-correlated pairs and is
#: dominated by the boundary -- the usual geostatistical rule of thumb.
DEFAULT_LAG_CAP_FRAC = 0.25


# ---------------------------------------------------------------------------
# Run-directory loading
#
# The analysis notebooks import these helpers from here, so the notebooks and
# this script rebuild a run in exactly the same way.
# ---------------------------------------------------------------------------


def load_run_config(run_dir) -> DictConfig:
    """The resolved config of a run, from either folder layout, with an absolute
    ``data_path``.

    ``config.yaml`` at the top of a synced seed folder, ``.hydra/config.yaml`` in
    a raw Hydra run directory -- the two are the same object, written by the same
    job.
    """
    run_dir = Path(run_dir)
    for candidate in (run_dir / "config.yaml", run_dir / ".hydra" / "config.yaml"):
        if candidate.exists():
            cfg = OmegaConf.load(candidate)
            break
    else:
        raise FileNotFoundError(f"No config.yaml or .hydra/config.yaml under {run_dir}")
    upgrade_legacy_targets(cfg)
    data_path = cfg.datamodule.get("data_path")
    if data_path is not None and not os.path.isabs(data_path):
        cfg.datamodule.data_path = str(Path(os.environ["PROJECT_ROOT"]) / data_path)
    return cfg


def find_checkpoint(run_dir) -> str:
    """The weights that define a run's predictive, in both layouts.

    Synced seed folder: ``stage2.ckpt`` (the final stage of a two-stage fit)
    before ``model.ckpt`` (single stage). Hydra run directory:
    ``checkpoints/stage2/*.ckpt`` before ``checkpoints/*.ckpt``. Never
    ``stage1.ckpt``, which holds only the intermediate stage-1 fit.
    """
    run_dir = Path(run_dir)
    for name in ("stage2.ckpt", "model.ckpt"):
        if (run_dir / name).exists():
            return str(run_dir / name)
    ck = run_dir / "checkpoints"
    found = sorted(ck.glob("stage2/*.ckpt")) or sorted(ck.glob("*.ckpt"))
    if not found:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")
    return str(found[-1])


def run_identity(run_dir, cfg: DictConfig) -> Dict[str, object]:
    """``(dataset, slug, seed)`` of a run.

    A synced seed folder ``<dataset>/[<mode>/][T<horizon>/]<slug>/seed<N>/``
    records its slug and seed in ``metrics.json`` (the slug is derived from the
    method family, basis, core, rank and loss of the run's config), so they are
    read back rather than re-derived. ``dataset`` is the name of the folder two
    levels above the seed folder, which is the dataset itself only when there is
    no ``<mode>``/``T<horizon>`` level in between.
    Raw Hydra run directories have no ``metrics.json``; there the path components
    are the best available answer.
    """
    run_dir = Path(run_dir)
    metrics = run_dir / "metrics.json"
    if metrics.exists():
        payload = json.loads(metrics.read_text())
        if "slug" in payload:
            return {
                "dataset": run_dir.parent.parent.name,
                "slug": str(payload["slug"]),
                "seed": int(payload["seed"]),
            }
    return {
        "dataset": str(cfg.datamodule.get("data_path", "unknown")).rstrip("/").split("/")[-1],
        "slug": run_dir.parent.name,
        "seed": int(cfg.get("seed", -1)),
    }


def build_model_from_run(run_dir, device: str = "cpu", **dm_overrides):
    """Rebuild ``(model, method, datamodule, cfg, ckpt_path)`` for a run folder.

    The backbone and method are excluded from the Lightning module's saved
    hyperparameters, so both must be re-instantiated from the run's own config
    before the state dict can be loaded.
    """
    cfg = load_run_config(run_dir)
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, **dm_overrides
    )
    datamodule.setup("test")

    d = datamodule.get_total_dimension()
    gamma_train = [math.sqrt(f * d) for f in _as_list(cfg.get("gamma_train", [1.0]))]
    gamma_eval = [math.sqrt(f * d) for f in _as_list(cfg.get("gamma_eval", [1.0]))]

    method: PredictiveMethod = hydra.utils.instantiate(
        cfg.method,
        out_channels=datamodule.get_channels(),
        event_dim=d,
        gamma=gamma_train,
        gamma_eval=gamma_eval,
    )
    backbone = hydra.utils.instantiate(
        cfg.backbone,
        in_channels=datamodule.get_input_channels(),
        out_channels=method.out_channels,
        group_widths=method.group_widths,
        group_denorm=method.group_denorm,
        other_zero_init=method.other_zero_init,
        noise=method.noise,
    )

    ckpt = find_checkpoint(run_dir)
    # Only `state_dict` is read -- backbone and method are rebuilt from the run's
    # own config above -- but `torch.load` still unpickles the checkpoint's
    # `hyper_parameters` DictConfig, and a run trained under a newer Hydra names
    # a class this environment does not have. See src/utils/ckpt_compat.py.
    install_hydra_target_policy_shim()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model = LightningPredictiveModel(backbone=backbone, predictive_method=method)
    model.load_state_dict(state)
    model.eval().to(device)
    # Eval mode selects `gamma_eval` and `n_eval_samples`; the flag is a plain
    # bool on the method, not `nn.Module` state, so `model.eval()` does not set it.
    method.training = False
    return model, method, datamodule, cfg, ckpt


# ---------------------------------------------------------------------------
# Pre-rank functions
# ---------------------------------------------------------------------------


def _standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def mean_prerank(
    v: torch.Tensor, n_obs: torch.Tensor, n_spatial: int, n_member_axes: int = 0
) -> torch.Tensor:
    """``rho_loc(v) = (1/d) sum_{j observed} v_j``, the paper's ``xbar``.

    ``v`` is ``(B, G, *spatial[, M])``, already masked to zero on the unobserved
    coordinates; ``n_obs`` is ``(B, G)``. The reduction runs over the spatial
    axes only, so the channel axis ``G`` indexes independent pre-rank problems
    and any trailing member axis is carried through untouched -- one function for
    both the observation and the draws.
    """
    axes = [2 + i for i in range(n_spatial)]
    d = n_obs.reshape(*n_obs.shape, *([1] * n_member_axes))
    return v.sum(dim=axes) / d


def variance_prerank(
    v: torch.Tensor, n_obs: torch.Tensor, n_spatial: int, n_member_axes: int = 0
) -> torch.Tensor:
    """``rho_sc(v) = (1/d) sum_{j observed} (v_j - vbar)^2``, the paper's ``s^2_x``.

    Same layout contract as :func:`mean_prerank`, with one extra requirement:
    ``v`` must already be shifted by a per-case scalar. ``rho_sc`` is invariant
    under adding a constant to every coordinate, and centring near zero first
    keeps the ``E[v^2] - E[v]^2`` form from cancelling catastrophically when the
    field is far from the origin (raw station temperatures sit around 10 K with a
    ~2 K spread, so the uncentred moments differ by two orders of magnitude).

    This is the paper's ``1/d`` sample variance, not the unbiased ``1/(d-1)``
    one: it measures the spread of the given vector rather than estimating an
    unknown population variance, so the bias correction would be meaningless.
    """
    axes = [2 + i for i in range(n_spatial)]
    d = n_obs.reshape(*n_obs.shape, *([1] * n_member_axes))
    s1 = v.sum(dim=axes)
    s2 = v.pow(2).sum(dim=axes)
    return (s2 / d - (s1 / d).pow(2)).clamp_min(0.0)


def default_lags(extent: int, cap_frac: float = DEFAULT_LAG_CAP_FRAC) -> List[int]:
    """Dyadic lag ladder for an axis of length ``extent``, capped at ``cap_frac``.

    The *spacing* is absolute rather than a fraction of the domain, because on a
    lead-time axis the interesting scales are set by the data's periodicities
    (lag 24 is "one day" on hourly data whatever ``pred_len`` is) and a
    horizon-relative rung would mean something different at each horizon. Only
    the *ceiling* is relative -- see :data:`DEFAULT_LAG_CAP_FRAC`. Pass explicit
    lags to probe physical periods directly (e.g. ``1,24,168`` on hourly data).
    """
    cap = max(1, int(extent * cap_frac))
    return [h for h in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if h <= cap]


def _lag_slices(spatial_shape: Tuple[int, ...], lag: Tuple[int, ...]):
    """Index slices selecting the two ends of every pair separated by ``lag``.

    Returns ``(lo, hi)`` such that ``v[hi] - v[lo]`` is the field difference
    across the displacement ``lag``, or ``None`` when the lag does not fit. A
    negative component (the anti-diagonal) walks the other way along that axis.
    """
    lo, hi = [], []
    for shift, n in zip(lag, spatial_shape):
        if abs(shift) >= n:
            return None
        if shift >= 0:
            lo.append(slice(0, n - shift))
            hi.append(slice(shift, n))
        else:
            lo.append(slice(-shift, n))
            hi.append(slice(0, n + shift))
    return tuple(lo), tuple(hi)


def variogram_grid(
    v: torch.Tensor,
    mask: torch.Tensor,
    lag_vectors: List[Tuple[int, ...]],
    n_spatial: int,
    n_member_axes: int = 0,
):
    """Empirical variograms on a regular grid, one per lag vector.

    ``v`` is ``(B, G, *spatial[, M])`` and ``mask`` is ``(B, G, *spatial)``.
    Returns ``(gamma, n_pairs)`` of shapes ``(B, G, L[, M])`` and ``(B, G, L)``
    for ``L = len(lag_vectors)``, with

        gamma_x(h) = 1 / (2 |I(h)|) * sum_{j in I(h)} (x_j - x_{j+h})^2

    and ``I(h)`` the positions where *both* ends are observed. Each lag costs one
    strided subtraction over the field -- ``O(V)`` -- so the whole table is the
    same cost class as the variance pre-rank, not the all-pairs variogram score.
    """
    spatial_shape = tuple(v.shape[2 : 2 + n_spatial])
    axes = [2 + i for i in range(n_spatial)]
    tail = (1,) * n_member_axes
    gammas, counts = [], []
    for lag in lag_vectors:
        sl = _lag_slices(spatial_shape, lag)
        if sl is None:  # lag longer than the axis: no pairs exist
            shape = (v.shape[0], v.shape[1]) + v.shape[2 + n_spatial :]
            gammas.append(torch.full(shape, float("nan"), device=v.device, dtype=v.dtype))
            counts.append(torch.zeros(v.shape[:2], device=v.device, dtype=v.dtype))
            continue
        lo = (slice(None), slice(None)) + sl[0]
        hi = (slice(None), slice(None)) + sl[1]
        pair_mask = mask[lo] * mask[hi]
        diff = v[hi] - v[lo]
        num = (diff.pow(2) * pair_mask.reshape(*pair_mask.shape, *tail)).sum(dim=axes)
        den = pair_mask.sum(dim=axes)
        gammas.append(num / (2.0 * den.clamp_min(1.0)).reshape(*den.shape, *tail))
        counts.append(den)
    return torch.stack(gammas, dim=2), torch.stack(counts, dim=2)


EARTH_RADIUS_KM = 6371.0


def grid_pair_km(lat_deg, lon_step_deg: float, lag: Tuple[int, int]) -> np.ndarray:
    """Great-circle km spanned by ``lag`` starting at each latitude row.

    ``lag`` is ``(dlat, dlon)`` in grid steps. On a lat-lon grid the latitude
    step is a fixed distance but the longitude step is ``cos(phi)`` times as
    long, so a pixel lag is *not* one physical distance: over this project's
    ERA5 domain (35-74.75N) one longitude step runs from 22.8 km in the south to
    7.3 km in the north, a factor of 3.1. Returning the distance per *row* is
    what lets each row's variogram contribution be binned at the separation it
    actually spans.

    Result is length ``n_lat``, padded with ``nan`` for the rows where the lag
    runs off the end (``_lag_slices`` keeps only ``n_lat - |dlat|`` of them, and
    the row index there is the *low* end of the pair).
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    n_lat = lat.size
    dlat, dlon = int(lag[0]), int(lag[1])
    out = np.full(n_lat, np.nan)
    keep = n_lat - abs(dlat)
    if keep <= 0:
        return out
    lat1 = lat[:keep] if dlat >= 0 else lat[-dlat:]
    lat2 = lat[dlat:] if dlat >= 0 else lat[:keep]
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlam = np.deg2rad(dlon * lon_step_deg)
    h = np.sin((phi2 - phi1) / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    out[:keep] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return out


def variogram_grid_rows(
    v: torch.Tensor,
    mask: torch.Tensor,
    lag_vectors: List[Tuple[int, ...]],
    n_member_axes: int = 0,
):
    """Row-resolved variogram numerator/denominator on a 2-D grid.

    The same strided differences as :func:`variogram_grid`, but the *first*
    spatial axis is kept instead of summed out, and the two halves of the
    quotient are returned separately. Both are needed to bin by physical
    distance: which bin a contribution belongs to depends on the row (see
    :func:`grid_pair_km`), and pooling contributions into a bin means summing
    numerators and denominators, not averaging ratios.

    Returns ``(num, den)`` of shapes ``(B, G, L, R[, M])`` and ``(B, G, L, R)``,
    zero-padded to ``R = n_lat``. Summing both over ``R`` and forming
    ``num / (2 * den)`` reproduces :func:`variogram_grid` exactly.
    """
    n_lat, n_lon = int(v.shape[2]), int(v.shape[3])
    tail = (1,) * n_member_axes
    nums, dens = [], []
    for lag in lag_vectors:
        num = v.new_zeros((v.shape[0], v.shape[1], n_lat) + tuple(v.shape[4:]))
        den = mask.new_zeros((mask.shape[0], mask.shape[1], n_lat))
        sl = _lag_slices((n_lat, n_lon), lag)
        if sl is not None:
            lo = (slice(None), slice(None)) + sl[0]
            hi = (slice(None), slice(None)) + sl[1]
            pair_mask = mask[lo] * mask[hi]
            diff = v[hi] - v[lo]
            rows = pair_mask.shape[2]
            num[:, :, :rows] = (
                diff.pow(2) * pair_mask.reshape(*pair_mask.shape, *tail)
            ).sum(dim=3)
            den[:, :, :rows] = pair_mask.sum(dim=3)
        nums.append(num)
        dens.append(den)
    return torch.stack(nums, dim=2), torch.stack(dens, dim=2)


def bin_rowwise_variograms(
    num: torch.Tensor,
    den: torch.Tensor,
    slot: torch.Tensor,
    n_slots: int,
    n_dir: int,
    n_bins: int,
):
    """Pool row-resolved ``(num, den)`` into ``(B, G, D, n_bins[, M])`` gammas.

    ``slot`` is the ``(L, R)`` destination index ``direction * n_bins + bin`` of
    every (lag vector, row) cell, with ``n_slots`` marking cells that fall in no
    bin (a lag that runs off the grid, or a distance past the last edge). One
    ``index_add_`` over the flattened ``(L, R)`` axis does the whole table.

    A bin is the pair-weighted pooling of everything that lands in it, so a
    direction whose lags happen to straddle a bin edge is still summarised by
    the pairs it actually contributed -- and a bin no direction reaches comes
    back ``nan``, which the rank machinery already treats as undefined.
    """
    b, g = num.shape[0], num.shape[1]
    member = tuple(num.shape[4:])
    flat_num = num.flatten(2, 3)
    flat_den = den.flatten(2, 3)
    idx = slot.reshape(-1)
    acc_num = num.new_zeros((b, g, n_slots + 1) + member).index_add_(2, idx, flat_num)
    acc_den = den.new_zeros((b, g, n_slots + 1)).index_add_(2, idx, flat_den)
    acc_num = acc_num[:, :, :n_slots].unflatten(2, (n_dir, n_bins))
    acc_den = acc_den[:, :, :n_slots].unflatten(2, (n_dir, n_bins))
    denom = 2.0 * acc_den
    if member:
        denom = denom.reshape(*denom.shape, *((1,) * len(member)))
    gamma = acc_num / denom
    empty = acc_den == 0
    if member:
        empty = empty.reshape(*empty.shape, *((1,) * len(member)))
    return gamma.masked_fill(empty, float("nan"))


def variogram_bins(
    v: torch.Tensor,
    mask: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    bin_of_pair: torch.Tensor,
    n_bins: int,
    n_member_axes: int = 0,
):
    """Empirical variograms binned by geodesic distance, for irregular geometry.

    The analogue of :func:`variogram_grid` where no lag structure exists: pairs
    are enumerated once (``P = N(N-1)/2``, only 6 555 for 115 stations) and
    accumulated into distance bins. Same return contract, with the bin index
    taking the place of the lag index.
    """
    tail = (1,) * n_member_axes
    # Index the *station* axis explicitly at dim 2: `v[..., idx]` would hit the
    # trailing member axis instead whenever draws are present.
    sq = (v[:, :, pair_j] - v[:, :, pair_i]).pow(2)
    pair_mask = mask[:, :, pair_i] * mask[:, :, pair_j]
    sq = sq * pair_mask.reshape(*pair_mask.shape, *tail)
    gammas, counts = [], []
    for b in range(n_bins):
        sel = bin_of_pair == b
        den = pair_mask[:, :, sel].sum(dim=2)
        num = sq[:, :, sel].sum(dim=2)
        gammas.append(num / (2.0 * den.clamp_min(1.0)).reshape(*den.shape, *tail))
        counts.append(den)
    return torch.stack(gammas, dim=2), torch.stack(counts, dim=2)


def dependence_prerank(gamma: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """``rho_dep(x; h) = -gamma_x(h) / s^2_x`` (Eq. 2).

    ``gamma`` is ``(B, G, D, L[, M])`` and ``var`` the matching ``rho_sc``,
    ``(B, G[, M])``. Dividing by ``s^2_x`` is what makes this a *dependence*
    diagnostic rather than a second scale one: it strips the overall spread and
    leaves the ratio of near-neighbour to overall variation. The minus sign makes
    a larger value mean stronger dependence, matching the reading of the other
    pre-ranks.
    """
    denom = var.unsqueeze(2).unsqueeze(3) if gamma.dim() == var.dim() + 2 else var
    return -gamma / denom.clamp_min(torch.finfo(gamma.dtype).tiny)


def isotropy_prerank(gamma: torch.Tensor) -> torch.Tensor:
    """``rho_iso(x; h)`` (Eq. 5), from the four directions of a 2-D variogram table.

    ``gamma`` is ``(B, G, D, L[, M])`` with the first four directions ordered as
    :data:`DIRECTIONS_2D`; returns ``(B, G, L[, M])``::

        rho_iso = -{ [(g(h,0) - g(0,h)) / (g(h,0) + g(0,h))]^2
                   + [(g(h,h) - g(-h,h)) / (g(h,h) + g(-h,h))]^2 }

    Each bracket contrasts two directions **at the same distance** -- axis vs
    axis at ``h``, diagonal vs anti-diagonal at ``h*sqrt(2)`` -- which is what
    keeps it a statement about the field rather than about the grid. Zero means
    the variogram does not change with direction; more negative means more
    anisotropy. Note ``gamma(h,-h) == gamma(-h,h)`` because a variogram is
    symmetric under swapping the two ends of a pair.

    On a lat-lon grid one step of longitude is ``cos(phi)`` times one step of
    latitude, so this measures the *grid's* anisotropy as much as the field's.
    It is meaningful on square-pixel grids (depth); read it with care on ERA5.
    """
    eps = torch.finfo(gamma.dtype).tiny
    axial = (gamma[:, :, 0] - gamma[:, :, 1]) / (gamma[:, :, 0] + gamma[:, :, 1]).clamp_min(eps)
    diag = (gamma[:, :, 2] - gamma[:, :, 3]) / (gamma[:, :, 2] + gamma[:, :, 3]).clamp_min(eps)
    return -(axial.pow(2) + diag.pow(2))


def rank_to_pit(rank: np.ndarray, n_draws: int) -> np.ndarray:
    """Mid-rank PIT ``u = (R - 0.5) / (M + 1)`` for plotting the empirical CDF.

    The sampled diagnostics give an integer rank in ``1..M+1``; an ECDF needs a
    value in ``[0, 1]``. The mid-rank is deterministic and reproducible, and at
    ``M = 999`` it differs from the exact randomized PIT ``(R - 1 + U)/(M + 1)``
    by at most 5e-4 -- invisible in a figure, and worth it to keep the plot from
    changing between runs. Invalid entries (rank ``0``) map to NaN.
    """
    u = (rank.astype(np.float64) - 0.5) / (n_draws + 1)
    return np.where(rank == 0, np.nan, u)


def rank_from_counts(
    less: torch.Tensor,
    equal: torch.Tensor,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Randomized rank in ``1..M+1`` from tallies of strictly-smaller and tied members.

    Split out of :func:`randomized_rank` so the same randomization serves a
    caller that never holds all ``M`` members at once: the marginal diagnostic
    of a sample-based method ranks *every coordinate* against the draws, so it
    accumulates these two counts chunk by chunk rather than materializing a
    ``(B, V, M)`` member tensor.
    """
    # The generator is a CPU one so the draws are reproducible independently of
    # the accelerator; move the result to wherever the pre-ranks live.
    u = torch.rand(less.shape, generator=generator).to(
        device=less.device, dtype=dtype
    )
    return 1 + less + (u * (equal + 1).to(dtype)).long()


def randomized_rank(
    obs: torch.Tensor, members: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Rank of ``obs`` among itself and ``members``, in ``1..M+1``, ties at random.

    ``obs`` has any leading shape ``(B, ...)`` and ``members`` matches it with a
    trailing member axis ``(B, ..., M)`` -- so the same function serves the
    scalar variance pre-rank ``(B, G)`` and the dependence table
    ``(B, G, D, L)``. Ties have probability zero for a non-degenerate Gaussian
    pre-rank, but a single-coordinate group makes every value identically zero,
    and there the randomization is what keeps the rank uniform instead of pinned
    at 1.
    """
    less = (members < obs.unsqueeze(-1)).sum(dim=-1)
    equal = (members == obs.unsqueeze(-1)).sum(dim=-1)
    return rank_from_counts(less, equal, generator, obs.dtype)


# ---------------------------------------------------------------------------
# The test-set pass
# ---------------------------------------------------------------------------


class _Accumulator:
    """Collects the per-case diagnostics over the test set.

    ``Evaluator`` cannot be reused: it reduces every metric to a scalar and
    ``.item()``s it, whereas these diagnostics are per-case arrays.
    """

    def __init__(self, n_cases: int, event_shape: Tuple[int, ...], n_groups: int,
                 n_draws: int, bins: int, raw_cap_bytes: int, seed: int):
        self.event_shape = event_shape
        self.V = int(np.prod(event_shape))
        self.G = n_groups
        self.n_draws = n_draws
        self.bins = bins
        self.marg_hist = np.zeros(bins, dtype=np.int64)
        # Full marginal-PIT array when it fits, else a fixed seeded subsample of
        # coordinates. The histogram above stays exact either way -- it is
        # accumulated over every observed coordinate before any subsampling.
        full_bytes = n_cases * self.V * 4
        self.marg_mode = "full" if full_bytes <= raw_cap_bytes else "subsampled"
        if self.marg_mode == "full":
            self.marg_idx = None
        else:
            rng = np.random.default_rng(seed)
            k = min(SUBSAMPLE_COORDS, self.V)
            self.marg_idx = np.sort(rng.choice(self.V, size=k, replace=False))
        # Everything else is a plain per-batch list, concatenated at the end.
        # Only the marginal histogram has to be incremental (it is exact over
        # every coordinate even when the stored array is subsampled); the rank
        # histograms are derived from the concatenated ranks in `finalize`.
        self.parts: Dict[str, List[np.ndarray]] = {}
        self.marg: List[np.ndarray] = []
        self.mask: List[np.ndarray] = []
        self.any_missing = False
        self.n_complete = 0  # cases seen before the first gap (see `add_marginal`)

    def add(self, **arrays: np.ndarray) -> None:
        for key, value in arrays.items():
            self.parts.setdefault(key, []).append(value)

    def stacked(self, key: str) -> np.ndarray:
        return np.concatenate(self.parts[key], axis=0)

    def rank_histogram(self, key: str) -> np.ndarray:
        """Counts per rank, ``(M+1, *trailing)``, excluding undefined (rank 0).

        Undefined groups are dropped rather than folded into bin 1, where they
        would fake a left-edge spike that reads as severe miscalibration.
        """
        ranks = self.stacked(key)
        trailing = ranks.shape[1:]
        out = np.zeros((self.n_draws + 1,) + trailing, dtype=np.int64)
        flat = ranks.reshape(ranks.shape[0], -1)
        for col in range(flat.shape[1]):
            keep = flat[:, col]
            keep = keep[keep > 0]
            if keep.size:
                counts = np.bincount(keep - 1, minlength=self.n_draws + 1)
                out.reshape(self.n_draws + 1, -1)[:, col] = counts
        return out

    def add_marginal(self, pit: np.ndarray, finite: np.ndarray) -> None:
        """``pit`` and ``finite`` are ``(B, V)``; NaN where unobserved."""
        observed = pit[finite]
        counts, _ = np.histogram(observed, bins=self.bins, range=(0.0, 1.0))
        self.marg_hist += counts
        if self.marg_mode == "full":
            self.marg.append(pit.astype(np.float32))
        else:
            self.marg.append(pit[:, self.marg_idx].astype(np.float32))

        # The mask is only kept once something is actually missing: it is
        # `n_cases * V` bytes, which is 200 MB of all-True at depth resolution
        # and is not written out at all for a complete dataset. When the first
        # gap does turn up, backfill the all-observed cases seen before it.
        if finite.all() and not self.any_missing:
            self.n_complete += finite.shape[0]
            return
        if not self.any_missing:
            self.any_missing = True
            if self.n_complete:
                self.mask.append(np.ones((self.n_complete, self.V), dtype=bool))
                self.n_complete = 0
        self.mask.append(finite)


@torch.no_grad()
def evaluate_calibration(
    model,
    method: PredictiveMethod,
    datamodule: LightningDataModule,
    device: str = "cpu",
    n_draws: int = 999,
    draw_chunk: int = 64,
    bins: int = 1000,
    raw_cap_mb: float = 100.0,
    sample_seed: int = 0,
    save_members: bool = False,
    lags: Optional[List[int]] = None,
    lag_cap_frac: float = DEFAULT_LAG_CAP_FRAC,
    bin_edges_km: Optional[List[float]] = None,
    isotropy: bool = True,
) -> Dict[str, np.ndarray]:
    """Run all calibration diagnostics over the test set and return the npz payload."""
    # Eval mode -> gamma_eval; set here as well as in `build_model_from_run`, on
    # the `Evaluator` precedent, so a directly-called pass cannot use train-mode
    # bandwidths.
    method.training = False
    loader = datamodule.test_dataloader()
    n_cases = len(datamodule.test_dataset)

    # Event layout from the dataset rather than a probe batch, so no worker pool
    # is spun up and no batch is computed twice.
    event_shape = tuple(datamodule.test_dataset[0][1].shape)
    n_groups = int(event_shape[0])  # channels
    n_spatial = len(event_shape) - 1
    group_size = int(np.prod(event_shape[1:]))

    acc = _Accumulator(
        n_cases=n_cases,
        event_shape=event_shape,
        n_groups=n_groups,
        n_draws=n_draws,
        bins=bins,
        raw_cap_bytes=int(raw_cap_mb * 1024 * 1024),
        seed=sample_seed,
    )
    log.info(
        f"test set: {n_cases} cases, event {event_shape} (V={acc.V}), "
        f"G={n_groups} group(s) of d={group_size}; marginal PIT stored "
        f"'{acc.marg_mode}'"
    )

    # --- variogram geometry ------------------------------------------------
    # Irregular geometry (a station network) has no lag structure, so pairs are
    # binned by geodesic distance; a regular grid uses strided lags. The
    # discriminator is whether the datamodule can supply a distance matrix.
    spatial_shape = tuple(event_shape[1:])
    irregular = n_spatial == 1 and hasattr(datamodule, "get_distances")
    # A 2-D grid whose datamodule can name its latitudes is a *geographic* grid,
    # where a pixel lag is not a physical distance (see `grid_pair_km`). There the
    # strided lags are kept as the sampling device -- an all-pairs distance matrix
    # over 35 200 coordinates would be 619 million pairs -- but every (lag, row)
    # contribution is binned by the km it actually spans, so directions become
    # comparable and the bins read like EUPPBench's.
    latlon = n_spatial == 2 and hasattr(datamodule, "get_grid_coords")
    geom: Dict[str, object] = {}

    if irregular:
        dist = torch.as_tensor(datamodule.get_distances(), dtype=torch.float64)
        edges = list(bin_edges_km) if bin_edges_km else list(DEFAULT_BIN_EDGES_KM)
        iu = torch.triu_indices(dist.shape[0], dist.shape[1], offset=1)
        pair_d = dist[iu[0], iu[1]]
        bin_of_pair = torch.bucketize(pair_d, torch.tensor(edges[1:-1], dtype=torch.float64))
        n_lag = len(edges) - 1
        geom = {
            "kind": "distance_bin",
            "pair_i": iu[0].to(device),
            "pair_j": iu[1].to(device),
            "bin_of_pair": bin_of_pair.to(device),
            "n_bins": n_lag,
            "edges": edges,
            "direction_names": ["distance_bin"],
        }
        per_bin = [int((bin_of_pair == b).sum()) for b in range(n_lag)]
        log.info(
            f"dependence: {n_lag} geodesic bins {edges} km, "
            f"pairs per bin {per_bin} (of {pair_d.numel()})"
        )
        n_dir_raw, n_dir = 1, 1
        do_isotropy = False
    elif latlon:
        lat_deg, lon_deg = datamodule.get_grid_coords()
        lat_deg = np.asarray(lat_deg, dtype=np.float64)
        lon_step = float(np.abs(np.diff(np.asarray(lon_deg, dtype=np.float64))).min())
        units = DIRECTIONS_2D
        extent = min(spatial_shape)
        lag_list = list(lags) if lags else default_lags(extent, lag_cap_frac)
        lag_list = [h for h in lag_list if h < extent]
        if not lag_list:
            raise ValueError(f"no usable lag for spatial shape {spatial_shape}")
        lag_vectors = [tuple(h * u for u in unit) for unit, _ in units for h in lag_list]
        edges = list(bin_edges_km) if bin_edges_km else list(DEFAULT_BIN_EDGES_KM)
        n_lag = len(edges) - 1                      # bins take the lag axis' place
        n_dir_raw = len(units)
        names = [name for _, name in units] + ["pooled"]
        n_dir = len(names)

        # slot = direction * n_bins + bin, or the dump slot for a (lag, row) cell
        # that spans no finite bin. `pooled` is a fifth destination fed by every
        # direction, so it is a genuine pair-weighted pooling rather than a mean
        # of ratios -- the km bin is what makes pooling across directions legal
        # at all.
        n_slots = n_dir * n_lag
        km = np.stack([grid_pair_km(lat_deg, lon_step, lv) for lv in lag_vectors])
        bin_of = np.digitize(km, np.asarray(edges[1:-1], dtype=np.float64))
        valid = np.isfinite(km) & (km >= edges[0]) & (km < edges[-1])
        dir_of = np.repeat(np.arange(n_dir_raw), len(lag_list))[:, None]
        slot = np.where(valid, dir_of * n_lag + bin_of, n_slots)
        slot_pooled = np.where(valid, n_dir_raw * n_lag + bin_of, n_slots)
        geom = {
            "kind": "grid_distance_bin",
            "lag_vectors": lag_vectors,
            "lags": lag_list,
            "edges": edges,
            "direction_names": names,
            "n_bins": n_lag,
            "n_dir": n_dir,
            "n_slots": n_slots,
            "slot": torch.as_tensor(slot, dtype=torch.long, device=device),
            "slot_pooled": torch.as_tensor(slot_pooled, dtype=torch.long, device=device),
            "pair_km": km,
        }
        do_isotropy = isotropy
        with np.errstate(invalid="ignore"):
            spans = [
                (f"{names[d]}: "
                 + "/".join(
                     "-" if not np.any((bin_of[d * len(lag_list):(d + 1) * len(lag_list)] == b)
                                       & valid[d * len(lag_list):(d + 1) * len(lag_list)])
                     else str(int(((bin_of[d * len(lag_list):(d + 1) * len(lag_list)] == b)
                                   & valid[d * len(lag_list):(d + 1) * len(lag_list)]).sum()))
                     for b in range(n_lag)))
                for d in range(n_dir_raw)
            ]
        log.info(
            f"dependence: {n_lag} geodesic bins {edges} km from lags {lag_list} "
            f"on a {lat_deg.min():.2f}-{lat_deg.max():.2f}N grid "
            f"(lon step {lon_step:.3f} deg); (lag,row) cells per bin -- "
            + "; ".join(spans)
            + ("; isotropy on (directions matched by km)" if do_isotropy else "")
        )
    else:
        units = DIRECTIONS_2D if n_spatial == 2 else DIRECTIONS_1D
        extent = min(spatial_shape)
        lag_list = list(lags) if lags else default_lags(extent, lag_cap_frac)
        lag_list = [h for h in lag_list if h < extent]
        if not lag_list:
            raise ValueError(f"no usable lag for spatial shape {spatial_shape}")
        lag_vectors = [
            tuple(h * u for u in unit) for unit, _ in units for h in lag_list
        ]
        n_dir_raw = len(units)
        names = [name for _, name in units]
        # The axial mean is a fifth "direction": both axis lags sit at distance
        # h, so averaging them is legitimate and is the right summary on a
        # square-pixel grid. On a lat-lon grid the two axes are *not* the same
        # physical distance -- use `axis0` (latitude, uniform spacing) there.
        if n_spatial == 2:
            names = names + ["axial_mean"]
        n_dir = len(names)
        n_lag = len(lag_list)
        geom = {
            "kind": "grid_lag",
            "lag_vectors": lag_vectors,
            "lags": lag_list,
            "direction_names": names,
        }
        do_isotropy = isotropy and n_spatial == 2
        log.info(
            f"dependence: lags {lag_list} (cap {int(extent * lag_cap_frac)} of "
            f"{extent}) x directions {names}"
            + ("; isotropy on" if do_isotropy else "")
        )

    def variograms(v: torch.Tensor, m: torch.Tensor, n_member_axes: int) -> torch.Tensor:
        """``(B, G, D, L[, M])`` variogram table for a masked, centred field."""
        if irregular:
            g, _ = variogram_bins(
                v, m, geom["pair_i"], geom["pair_j"], geom["bin_of_pair"],
                geom["n_bins"], n_member_axes,
            )
            return g.unsqueeze(2)  # single "direction"
        if latlon:
            num, den = variogram_grid_rows(v, m, geom["lag_vectors"], n_member_axes)
            per_dir = bin_rowwise_variograms(
                num, den, geom["slot"], geom["n_slots"],
                geom["n_dir"], geom["n_bins"],
            )
            pooled = bin_rowwise_variograms(
                num, den, geom["slot_pooled"], geom["n_slots"],
                geom["n_dir"], geom["n_bins"],
            )
            # `bin_rowwise_variograms` writes into a (n_dir, n_bins) table; the
            # per-direction call fills rows 0..3 and the pooled call row 4.
            return torch.cat([per_dir[:, :, :len(DIRECTIONS_2D)],
                              pooled[:, :, len(DIRECTIONS_2D):]], dim=2)
        g, _ = variogram_grid(v, m, geom["lag_vectors"], n_spatial, n_member_axes)
        g = g.unflatten(2, (n_dir_raw, n_lag))
        if n_spatial == 2:
            axial = 0.5 * (g[:, :, 0:1] + g[:, :, 1:2])
            g = torch.cat([g, axial], dim=2)
        return g

    # --- how the draws and the location diagnostics are obtained -------------
    # A Gaussian family answers the marginal and mean objectives in closed form
    # and can draw any number of members from one forward pass. The sample-based
    # baseline (SB) can do neither: its members *are* forward passes, so each chunk
    # re-runs the model, and its marginal/mean PITs are ranks against those
    # draws instead of normal CDFs. See PredictiveMethod.samples_from_backbone
    # and .has_closed_form_pit.
    closed_form_pit = method.has_closed_form_pit
    backbone_draws = method.samples_from_backbone

    def draw(x: torch.Tensor, pred: torch.Tensor, chunk: int) -> torch.Tensor:
        """``chunk`` fresh members, ``(B, C, *spatial, M)``."""
        if backbone_draws:
            return method.get_samples(model(x, n_samples=chunk), n_samples=chunk)
        return method.get_samples(pred, n_samples=chunk)

    log.info(
        f"draws: {n_draws} in chunks of {draw_chunk}"
        + (" (one backbone pass per chunk)" if backbone_draws else "")
        + f"; marginal/mean PIT {'closed form' if closed_form_pit else 'sampled rank'}"
    )

    # One global seed for the whole pass. The samplers draw from the global RNG
    # and take no `generator=`, so replaying the identical call sequence -- same
    # batch size, same n_draws, same chunking -- is what reproduces the draws.
    # All three are recorded in the output.
    torch.manual_seed(sample_seed)
    rank_gen = torch.Generator(device="cpu").manual_seed(sample_seed + 1)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        mu = method.get_mean(pred)  # (B, C, *spatial)
        finite = torch.isfinite(y)
        mask = finite.to(mu.dtype)
        B = y.shape[0]

        # --- 1. marginal PIT -------------------------------------------------
        # Closed form only where the predictive marginal really is Gaussian.
        # The sampled branch needs the draws, so it runs after the loop below.
        if closed_form_pit:
            sd = method.get_marg_std(pred)
            z = (y - mu) / sd
            pit = torch.where(
                finite, _standard_normal_cdf(z), torch.full_like(z, np.nan)
            )
            acc.add_marginal(
                pit.flatten(start_dim=1).float().cpu().numpy(),
                finite.flatten(start_dim=1).cpu().numpy(),
            )

        y_filled = torch.nan_to_num(y, nan=0.0) * mask
        spatial_axes = list(range(2, y.dim()))
        n_obs = mask.sum(dim=spatial_axes)  # (B, G)
        empty = n_obs == 0
        safe_n = n_obs.clamp_min(1.0)

        def nan_like(t: torch.Tensor) -> torch.Tensor:
            return torch.full_like(t, float("nan"))

        # --- 2. mean pre-rank ------------------------------------------------
        # rho_loc(Y) = a'Y is univariate Gaussian with mean a'mu and variance
        # a'Sigma a, so no sampling is needed. Restricting `a` to the reporting
        # coordinates is exact: a Gaussian's marginal on a coordinate subset is
        # the corresponding sub-block. A nonparametric predictive has no such
        # form and takes the sampled branch after the loop below.
        mean_obs = mean_prerank(y_filled, safe_n, n_spatial)
        if closed_form_pit:
            mean_mu = mean_prerank(mu * mask, safe_n, n_spatial)
            mean_var = mu.new_zeros(B, n_groups)
            for g in range(n_groups):
                # a = 1_O/|O| on channel g, zero elsewhere. Event-shaped, which is
                # the layout `quadratic_form` contracts with the FFT/transform --
                # one call per channel because the groups are separate pre-rank
                # problems.
                a = torch.zeros_like(mu)
                a[:, g] = mask[:, g] / safe_n[:, g].reshape(-1, *([1] * n_spatial))
                mean_var[:, g] = method.quadratic_form(pred, a)
            mean_pit = _standard_normal_cdf(
                (mean_obs - mean_mu)
                / mean_var.clamp_min(torch.finfo(mu.dtype).tiny).sqrt()
            )
            mean_pit = torch.where(empty, nan_like(mean_pit), mean_pit)

        # --- 3. variance pre-rank (sampled) ----------------------------------
        # A quadratic form in a Gaussian is a generalized chi-square, so the rank
        # is taken against draws. rho_sc is invariant under a per-case constant
        # shift, so centre on the observed mean of mu first (see
        # `variance_prerank`).
        shift = mean_prerank(mu * mask, safe_n, n_spatial)  # (B, G)
        shift_e = shift.reshape(B, n_groups, *([1] * n_spatial))
        centred_obs = y_filled - shift_e * mask
        var_obs = variance_prerank(centred_obs, safe_n, n_spatial)

        # --- 4/5. dependence and isotropy pre-ranks (sampled) ----------------
        # Both come off the same variogram table, so the draws and the strided
        # differences are shared: isotropy costs one extra reduction, not an
        # extra pass over the samples.
        gamma_obs = variograms(centred_obs, mask, 0)  # (B, G, D, L)
        dep_obs = dependence_prerank(gamma_obs, var_obs)
        iso_obs = isotropy_prerank(gamma_obs) if do_isotropy else None

        var_members = mu.new_empty(B, n_groups, n_draws)
        dep_members = mu.new_empty(B, n_groups, n_dir, n_lag, n_draws)
        iso_members = (
            mu.new_empty(B, n_groups, n_lag, n_draws) if do_isotropy else None
        )
        # Sampled marginal/mean diagnostics accumulate across chunks rather than
        # holding all M members: two per-coordinate tallies for the marginal
        # rank (see `rank_from_counts`) and the per-member mean pre-rank, which
        # is only (B, G) per draw.
        if not closed_form_pit:
            marg_less = torch.zeros_like(y)
            marg_equal = torch.zeros_like(y)
            mean_members = mu.new_empty(B, n_groups, n_draws)
        drawn = 0
        while drawn < n_draws:
            # Chunk over members and reduce inside the chunk: only the per-member
            # scalars are kept, so the working set is O(V * chunk) rather than
            # O(V * M) -- 999 draws at depth's V would otherwise be 1.2 GB/case.
            chunk = min(draw_chunk, n_draws - drawn)
            samples = draw(x, pred, chunk)  # (B, C, *sp, M)
            if not closed_form_pit:
                obs = y.unsqueeze(-1)
                # NaN compares false both ways, so a missing coordinate tallies
                # 0/0 here and is overwritten with NaN by the `finite` mask below.
                marg_less += (samples < obs).sum(dim=-1)
                marg_equal += (samples == obs).sum(dim=-1)
                mean_members[:, :, drawn:drawn + chunk] = mean_prerank(
                    samples * mask.unsqueeze(-1), safe_n, n_spatial, n_member_axes=1
                )
            centred = (samples - shift_e.unsqueeze(-1)) * mask.unsqueeze(-1)
            sl = slice(drawn, drawn + chunk)
            v_chunk = variance_prerank(centred, safe_n, n_spatial, n_member_axes=1)
            var_members[:, :, sl] = v_chunk
            g_chunk = variograms(centred, mask, 1)  # (B, G, D, L, M)
            dep_members[..., sl] = dependence_prerank(g_chunk, v_chunk)
            if do_isotropy:
                iso_members[..., sl] = isotropy_prerank(g_chunk)
            drawn += chunk

        if not closed_form_pit:
            # A rank in 1..M+1 carries the same information as a closed-form PIT
            # and is uniform under exactly the same null, so it is stored in the
            # same [0, 1] keys via the mid-rank map -- at n_draws = 999 that is
            # 1000 equally spaced values, one per bin of the default histogram.
            # `pit_estimator` in the payload records which route produced them.
            marg_rank = rank_from_counts(
                marg_less.long(), marg_equal.long(), rank_gen, mu.dtype
            )
            pit = torch.where(
                finite,
                (marg_rank.to(mu.dtype) - 0.5) / (n_draws + 1),
                torch.full_like(mu, np.nan),
            )
            acc.add_marginal(
                pit.flatten(start_dim=1).float().cpu().numpy(),
                finite.flatten(start_dim=1).cpu().numpy(),
            )
            mean_rank = randomized_rank(mean_obs, mean_members, rank_gen)
            mean_pit = (mean_rank.to(mu.dtype) - 0.5) / (n_draws + 1)
            mean_pit = torch.where(empty, nan_like(mean_pit), mean_pit)
            # Same quantities as the closed form, estimated from the draws:
            # E[rho_loc(X)] and Var(rho_loc(X)) = a' Sigma a.
            mean_mu = torch.where(
                empty, nan_like(mean_obs), mean_members.mean(dim=-1)
            )
            mean_var = mean_members.var(dim=-1, unbiased=True)

        var_rank = randomized_rank(var_obs, var_members, rank_gen)
        var_rank = torch.where(empty, torch.zeros_like(var_rank), var_rank)
        var_obs = torch.where(empty, nan_like(var_obs), var_obs)

        # A pre-rank is undefined where the group has no observations, and also
        # where a lag/bin caught no observed pair (short test fields, or a
        # station bin emptied by missing reports).
        empty_dl = empty.reshape(B, n_groups, 1, 1) | ~torch.isfinite(dep_obs)
        dep_rank = randomized_rank(dep_obs, dep_members, rank_gen)
        dep_rank = torch.where(empty_dl, torch.zeros_like(dep_rank), dep_rank)
        dep_obs = torch.where(empty_dl, nan_like(dep_obs), dep_obs)

        acc.add(
            n_obs=n_obs.int().cpu().numpy(),
            mean_pit=mean_pit.float().cpu().numpy(),
            mean_prerank_obs=torch.where(empty, nan_like(mean_obs), mean_obs).float().cpu().numpy(),
            mean_prerank_mu=torch.where(empty, nan_like(mean_mu), mean_mu).float().cpu().numpy(),
            mean_prerank_var=mean_var.float().cpu().numpy(),
            var_rank=var_rank.int().cpu().numpy(),
            var_prerank_obs=var_obs.float().cpu().numpy(),
            dep_rank=dep_rank.int().cpu().numpy(),
            dep_prerank_obs=dep_obs.float().cpu().numpy(),
            dep_variogram_obs=gamma_obs.float().cpu().numpy(),
        )
        if do_isotropy:
            empty_l = empty.reshape(B, n_groups, 1) | ~torch.isfinite(iso_obs)
            iso_rank = randomized_rank(iso_obs, iso_members, rank_gen)
            iso_rank = torch.where(empty_l, torch.zeros_like(iso_rank), iso_rank)
            acc.add(
                iso_rank=iso_rank.int().cpu().numpy(),
                iso_prerank_obs=torch.where(empty_l, nan_like(iso_obs), iso_obs)
                .float().cpu().numpy(),
            )
        if save_members:
            acc.add(var_prerank_members=var_members.float().cpu().numpy())

    payload: Dict[str, np.ndarray] = {
        "marg_hist_counts": acc.marg_hist,
        "marg_hist_edges": np.linspace(0.0, 1.0, bins + 1),
        "marg_pit_mode": np.array(acc.marg_mode),
        "marg_pit": np.concatenate(acc.marg, axis=0),
        "var_hist_counts": acc.rank_histogram("var_rank"),
        "dep_hist_counts": acc.rank_histogram("dep_rank"),
        "dep_kind": np.array(geom["kind"]),
        "dep_directions": np.array(geom["direction_names"]),
        # How the `marg_pit` / `mean_pit` arrays were obtained. Both routes give
        # a PIT that is uniform under calibration and live in the same keys, but
        # a sampled one is discrete on the 1/(n_draws+1) grid, so a reader that
        # cares (a Kolmogorov-Smirnov test, say) can tell them apart. A file
        # without this key is to be read as "closed_form".
        "pit_estimator": np.array("closed_form" if closed_form_pit else "rank"),
    }
    for key in (
        "n_obs", "mean_pit", "mean_prerank_obs", "mean_prerank_mu",
        "mean_prerank_var", "var_rank", "var_prerank_obs",
        "dep_rank", "dep_prerank_obs", "dep_variogram_obs",
    ):
        payload[key] = acc.stacked(key)

    if geom["kind"] == "distance_bin":
        payload["dep_bin_edges_km"] = np.array(geom["edges"], dtype=np.float64)
    elif geom["kind"] == "grid_distance_bin":
        # Binned by physical distance like the station network, but reached from
        # strided lags, so the sampling ladder is recorded too: the bins are only
        # as well resolved as the lags feeding them, and `dep_pair_km` is the
        # (lag vector, row) distance table those bins were built from.
        payload["dep_bin_edges_km"] = np.array(geom["edges"], dtype=np.float64)
        payload["dep_lags"] = np.array(geom["lags"], dtype=np.int32)
        payload["dep_lag_vectors"] = np.array(geom["lag_vectors"], dtype=np.int32)
        payload["dep_pair_km"] = np.asarray(geom["pair_km"], dtype=np.float32)
    else:
        payload["dep_lags"] = np.array(geom["lags"], dtype=np.int32)
        payload["dep_lag_vectors"] = np.array(
            geom["lag_vectors"], dtype=np.int32
        ).reshape(n_dir_raw, n_lag, n_spatial)

    if do_isotropy:
        payload["iso_rank"] = acc.stacked("iso_rank")
        payload["iso_prerank_obs"] = acc.stacked("iso_prerank_obs")
        payload["iso_hist_counts"] = acc.rank_histogram("iso_rank")

    # Manifest: which objectives this file actually carries, so a plotting
    # script can enumerate rather than hard-code (isotropy is grid-only). Keyed
    # on the data array, not on `_hist_counts`, which the closed-form `mean`
    # objective has no need of.
    suffix = {"pit": "_pit", "rank": "_rank"}
    present = [o for o in OBJECTIVES if f"{o[1]}{suffix[o[2]]}" in payload]
    payload["objectives"] = np.array([o[0] for o in present])
    payload["objective_prefixes"] = np.array([o[1] for o in present])
    payload["objective_kinds"] = np.array([o[2] for o in present])

    if acc.marg_idx is not None:
        payload["marg_pit_idx"] = acc.marg_idx.astype(np.int32)
    if acc.any_missing:
        payload["mask"] = np.concatenate(acc.mask, axis=0)
    if save_members:
        payload["var_prerank_members"] = acc.stacked("var_prerank_members")
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def shared_tree_root(run_dir) -> Optional[Path]:
    """The root of the results directory above ``run_dir``, or ``None``.

    The root is the nearest ancestor containing :data:`SHARED_TREE_MARKER`, a
    file kept at the top of the results directory and nowhere below it. A raw
    Hydra run directory has no such ancestor, and that is the signal to fall back
    to writing in place.
    """
    for parent in Path(run_dir).resolve().parents:
        if (parent / SHARED_TREE_MARKER).is_file():
            return parent
    return None


def _seed_from_dir_name(run_dir) -> Optional[int]:
    """``42`` from a ``seed42`` folder name, else ``None``.

    In a results directory the path identifies the run, so the folder name is
    authoritative and -- unlike ``run_identity`` -- needs no config load,
    which is what lets the skip decision be made before anything is built.
    """
    name = Path(run_dir).resolve().name
    return int(name[4:]) if name.startswith("seed") and name[4:].isdigit() else None


def calibration_out_path(run_dir, out_root=None) -> Path:
    """Where the diagnostics of ``run_dir`` belong.

    Inside a results directory this is
    ``<experiment>/calibration/<rel>/calibration_seed<N>.npz``, where ``<rel>``
    is the run's own path below the experiment with the seed folder dropped
    (``nyu_bts/whitened_dct_nll``, ``etth1/multivariate/whitened_dct_nll``).
    Mirroring the checkpoint tree's own layout rather than flattening keeps the
    slug unambiguous: ``whitened_dct_nll`` exists under four different
    dataset/mode combinations in ``time_series`` alone.

    ``out_root`` replaces the ``<experiment>/calibration`` prefix, which is how a
    run whose experiment folder is not writable can still be diagnosed.
    """
    run_dir = Path(run_dir).resolve()
    root = shared_tree_root(run_dir)
    seed = _seed_from_dir_name(run_dir)
    if root is None or seed is None:
        return Path(out_root or run_dir) / LEGACY_OUT_NAME
    rel = run_dir.parent.relative_to(root)  # <experiment>/<dataset>/[<mode>/]<slug>
    base = Path(out_root) if out_root is not None else root / rel.parts[0] / CALIBRATION_DIRNAME
    return base.joinpath(*rel.parts[1:]) / f"calibration_seed{seed}.npz"


def needs_refresh(out_path: Path, overwrite: bool = False) -> Tuple[bool, str]:
    """Whether ``out_path`` must be (re)computed, and why.

    Skips only what is already *current*. Treating "file exists" as "done" would
    keep outdated files whenever the schema changes, and long sweeps are often
    interrupted (e.g. when the launching session ends). Checking the stored
    ``schema_version`` instead makes a plain re-run a safe resume:
    unfinished and outdated runs are redone, finished ones are left alone, and
    ``--overwrite`` is reserved for genuinely forcing everything.
    """
    if overwrite:
        return True, "--overwrite"
    if not out_path.exists():
        return True, "no existing file"
    try:
        with np.load(out_path, allow_pickle=False) as existing:
            version = int(existing["schema_version"])
    except Exception as err:  # truncated by a killed sweep, or unreadable
        return True, f"unreadable ({type(err).__name__})"
    if version == SCHEMA_VERSION:
        return False, f"already schema v{version}"
    return True, f"schema v{version} < current v{SCHEMA_VERSION}"


def calibrate_run(run_dir, args) -> Optional[Path]:
    """Run the diagnostics for one run directory and write its npz.

    The destination is :func:`calibration_out_path`, not the run directory.
    """
    run_dir = Path(run_dir)
    out_path = calibration_out_path(run_dir, out_root=args.out_root)
    legacy = run_dir / LEGACY_OUT_NAME
    if legacy.exists() and legacy != out_path:
        log.warning(f"leftover {legacy} predates the calibration/ layout; delete it")
    refresh, reason = needs_refresh(out_path, overwrite=args.overwrite)
    if not refresh:
        log.info(f"skipping {run_dir} ({reason})")
        return None
    if out_path.exists():
        log.info(f"refreshing {run_dir} ({reason})")

    # `data_path` is recorded in each run's own config and is normally right. It is
    # overridable because synced runs may be evaluated on a different machine than
    # the one they were trained on, where the recorded path does not exist; this
    # avoids rewriting the stored config.yaml files.
    dm_overrides = {"num_workers": args.num_workers}
    if args.data_path is not None:
        dm_overrides["data_path"] = str(args.data_path)
    dm_overrides.update(args.dm_override)
    model, method, datamodule, cfg, ckpt = build_model_from_run(
        run_dir, device=args.device, **dm_overrides
    )
    if isinstance(method, Deterministic):
        log.warning(f"skipping {run_dir}: a Dirac has sigma = 0, so no PIT exists")
        return None

    identity = run_identity(run_dir, cfg)
    log.info(f"calibrating {identity['slug']} seed{identity['seed']} from {ckpt}")

    payload = evaluate_calibration(
        model,
        method,
        datamodule,
        device=args.device,
        n_draws=args.n_draws,
        draw_chunk=args.draw_chunk,
        bins=args.bins,
        raw_cap_mb=args.raw_cap_mb,
        sample_seed=args.sample_seed,
        save_members=args.save_members,
        lags=args.lags,
        lag_cap_frac=args.lag_cap_frac,
        bin_edges_km=args.bin_edges_km,
        isotropy=not args.no_isotropy,
    )
    y_shape = tuple(datamodule.test_dataset[0][1].shape)
    payload.update(
        schema_version=np.int32(SCHEMA_VERSION),
        dataset=np.array(identity["dataset"]),
        slug=np.array(identity["slug"]),
        seed=np.int32(identity["seed"]),
        ckpt=np.array(str(ckpt)),
        method_cls=np.array(type(method).__name__),
        n_cases=np.int32(payload["mean_pit"].shape[0]),
        event_shape=np.array(y_shape, dtype=np.int32),
        n_groups=np.int32(payload["mean_pit"].shape[1]),
        n_draws=np.int32(args.n_draws),
        draw_chunk=np.int32(args.draw_chunk),
        batch_size=np.int32(datamodule.test_dataloader().batch_size),
        sample_seed=np.int32(args.sample_seed),
        created=np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )

    # Create files and `calibration/` directories group-writable (umask 002), so a
    # results directory shared by a group stays writable for all of its members.
    # New directories inherit the group (and any setgid bit) from their parent,
    # so only the umask has to be set here.
    previous = os.umask(0o002)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **payload)
    finally:
        os.umask(previous)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info(f"wrote {out_path} ({size_mb:.2f} MB)")
    return out_path


def _int_list(text: str) -> List[int]:
    return [int(t) for t in text.replace(",", " ").split()]


def _float_list(text: str) -> List[float]:
    return [float(t) for t in text.replace(",", " ").split()]


def _key_value(text: str) -> Tuple[str, object]:
    """``KEY=VALUE`` with the value parsed as YAML, as Hydra parses an override.

    So ``eval_batch_size=16`` reaches the datamodule as an int (a DataLoader
    rejects the string ``"16"``), while a path stays the string it was.
    """
    import yaml

    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key.strip(), yaml.safe_load(value.strip())


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.calibration",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="synced seed folders or raw Hydra run directories")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-draws", type=int, default=999,
                   help="members M for the sampled variance pre-rank; ranks are 1..M+1")
    p.add_argument("--draw-chunk", type=int, default=64,
                   help="members drawn at once; bounds the working set to O(V * chunk). "
                        "For the sample-based method (SB) each chunk is also one backbone "
                        "pass, so this trades memory against n_draws/chunk forwards")
    p.add_argument("--bins", type=int, default=1000,
                   help="bins of the exact marginal-PIT histogram")
    p.add_argument("--raw-cap-mb", type=float, default=100.0,
                   help="above this, the raw marginal PIT is stored subsampled")
    p.add_argument("--lags", type=_int_list, default=None,
                   help="comma-separated lags for the dependence pre-rank; default "
                        "is a dyadic ladder capped at --lag-cap-frac of the axis. "
                        "Pass physical periods where they exist, e.g. 1,24,168 on "
                        "hourly data (grid datasets only)")
    p.add_argument("--lag-cap-frac", type=float, default=DEFAULT_LAG_CAP_FRAC,
                   help="largest lag as a fraction of the axis extent")
    p.add_argument("--bin-edges-km", type=_float_list, default=None,
                   help="geodesic bin edges for irregular geometry (EUPPBench); "
                        f"default {list(DEFAULT_BIN_EDGES_KM)}")
    p.add_argument("--no-isotropy", action="store_true",
                   help="skip the isotropy pre-rank on 2-D grids")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-members", action="store_true",
                   help="also store every member's variance pre-rank (large)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--data-path", type=Path, default=None,
                   help="override the datamodule's data_path for every run. The "
                        "path is the one recorded at training time, which is not "
                        "reachable when the checkpoints were trained on another "
                        "machine. Only the "
                        "location changes -- point it at the same dataset, or the "
                        "diagnostics describe a different test set than the name "
                        "on the file claims")
    p.add_argument("--dm-override", type=_key_value, action="append", default=[],
                   metavar="KEY=VALUE", dest="dm_overrides_raw",
                   help="further datamodule keyword overrides, repeatable. Same "
                        "reason as --data-path -- a run config can name files "
                        "that only exist on the training machine -- but not "
                        "restricted to one key. The paper's ERA5 runs need "
                        "statistics_path=<...>/era5_statistics_train1990-2014.npy, "
                        "whose default (<data_path>/era5_statistics.npy) is a "
                        "*different normalization*, not just another location")
    p.add_argument("--out-root", type=Path, default=None,
                   help="replaces the derived <experiment>/calibration prefix; the "
                        "<dataset>/[<mode>/]<slug>/calibration_seed<N>.npz tail is "
                        "kept. Use it when a seed folder's experiment is not "
                        "writable, or to try a sweep out somewhere harmless")
    args = p.parse_args(argv)
    args.dm_override = dict(args.dm_overrides_raw)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    # Unlike `evaluate.py` this is not a Hydra app, so nothing has configured
    # logging: without this the whole run -- including the per-arm exception
    # handler below -- is silent.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    written = 0
    for run_dir in args.run_dirs:
        try:
            if calibrate_run(run_dir, args) is not None:
                written += 1
        except Exception:  # one bad arm must not abort a sweep
            log.exception(f"failed on {run_dir}")
    log.info(f"done: {written} of {len(args.run_dirs)} run(s) written")


if __name__ == "__main__":
    main()
