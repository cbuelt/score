"""Tests for the multi-bandwidth kernel score.

``gamma``/``gamma_eval`` may be a list of bandwidths. The kernel score (and the
marginal kernel score) is then the per-bandwidth closed-form score averaged over
the list. The two invariants exercised here:

* a one-element list ``[g]`` reproduces the scalar-``g`` score exactly (so the
  default ``[1.0]`` config is numerically unchanged), and
* a list ``[a, b, ...]`` equals the mean of the single-bandwidth scores computed
  by separate scalar-``g`` instances (the mean combine rule, and a check that the
  bandwidth-sweep index bookkeeping is correct).

The bandwidth list is also exercised in *training* mode (``method.training =
True``) so the train/eval bandwidth selection and the SCORE delegate's index
mirroring are covered.
"""

import math

import pytest
import torch

from src.methods import Deterministic, Diag, SpectralDiag, SCORE
from src.methods.base import _as_list

EPS = 1e-9
N_TRAIN = 10
N_EVAL = 50
OUT_CHANNELS = 1
BATCH = 8


# --- _as_list normalization -------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.0, [1.0]),
        (2, [2.0]),
        ([0.5, 1.0, 2.0], [0.5, 1.0, 2.0]),
        ((0.5, 2.0), [0.5, 2.0]),
    ],
)
def test_as_list_normalizes(value, expected):
    assert _as_list(value) == expected


def test_scalar_and_singleton_list_store_same_bandwidths():
    """A scalar gamma and a one-element list are stored identically."""
    scalar = Diag("mse", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, 0.5)
    single = Diag("mse", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [0.5])
    assert scalar._gamma_train == single._gamma_train == [0.5]
    assert scalar.gamma == single.gamma == 0.5
    assert scalar.gamma2 == single.gamma2 == 0.25


def test_gamma_list_properties_and_train_eval_switch():
    """gamma_list/gamma2_list expose the active (train vs eval) list, and the
    active scalar gamma follows the _gamma_idx sweep."""
    method = Diag(
        "kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS,
        gamma=[0.5, 2.0], gamma_eval=[1.0],
    )
    # eval mode (default)
    assert method.gamma_list == [1.0]
    assert method.gamma2_list == [1.0]
    # train mode picks the train list
    method.training = True
    assert method.gamma_list == [0.5, 2.0]
    assert method.gamma2_list == [0.25, 4.0]
    # gamma follows the active index; it is restored to 0 after a score call
    assert method.gamma == 0.5
    method._gamma_idx = 1
    assert method.gamma == 2.0


# --- helpers ----------------------------------------------------------------


def _phys_pred(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(BATCH, OUT_CHANNELS, 5, generator=g, dtype=torch.float64)
    var = (
        torch.rand(BATCH, OUT_CHANNELS, 5, generator=g, dtype=torch.float64) + 0.5
    )
    return torch.stack([mu, var], dim=-1)


def _phys_y(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, OUT_CHANNELS, 5, generator=g, dtype=torch.float64)


def _spectral_symmetric_s(seed: int, H: int, W: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(BATCH, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64) + 0.5
    idx_h = (-torch.arange(H)) % H
    idx_w = (-torch.arange(W)) % W
    return 0.5 * (base + base[:, :, idx_h][:, :, :, idx_w])


def _spectral_pred(seed: int, H: int, W: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(BATCH, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)
    s = _spectral_symmetric_s(seed + 1, H, W)
    return torch.stack([mu, s], dim=-1)


def _whitened_pred(seed: int, H: int, W: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(BATCH, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)
    diag = torch.rand(BATCH, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64) + 0.5
    s = _spectral_symmetric_s(seed + 2, H, W)
    return torch.stack([mu, diag, s], dim=-1)


def _spectral_field(seed: int, H: int, W: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, OUT_CHANNELS, H, W, generator=g, dtype=torch.float64)


# --- singleton-list == scalar invariant -------------------------------------


def test_deterministic_singleton_matches_scalar():
    pred = torch.randn(BATCH, OUT_CHANNELS, 5, dtype=torch.float64)
    y = torch.randn(BATCH, OUT_CHANNELS, 5, dtype=torch.float64)
    scalar = Deterministic("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, 1.3)
    single = Deterministic("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [1.3])
    assert torch.allclose(scalar.kernel_score(pred, y), single.kernel_score(pred, y))


def test_phys_singleton_matches_scalar():
    pred, y = _phys_pred(0), _phys_y(1)
    scalar = Diag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, 0.7)
    single = Diag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [0.7])
    assert torch.allclose(scalar.kernel_score(pred, y), single.kernel_score(pred, y))
    assert torch.allclose(
        scalar.marginal_kernel_score(pred, y),
        single.marginal_kernel_score(pred, y),
    )


def test_spectral_singleton_matches_scalar():
    H, W = 4, 6
    pred, y = _spectral_pred(0, H, W), _spectral_field(3, H, W)
    scalar = SpectralDiag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, 10.0)
    single = SpectralDiag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [10.0])
    assert torch.allclose(scalar.kernel_score(pred, y), single.kernel_score(pred, y))


# --- list == mean of single-bandwidth scores (the combine rule) -------------


@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_phys_multi_bandwidth_is_mean(reduction):
    pred, y = _phys_pred(4), _phys_y(5)
    factors = [0.5, 1.0, 2.0]
    multi = Diag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, factors)
    singles = [
        Diag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [g])
        for g in factors
    ]
    got = multi.kernel_score(pred, y, reduction=reduction)
    expected = sum(m.kernel_score(pred, y, reduction=reduction) for m in singles) / len(
        factors
    )
    assert torch.allclose(got, expected)


def test_phys_marginal_multi_bandwidth_is_mean():
    pred, y = _phys_pred(6), _phys_y(7)
    factors = [0.5, 2.0]
    multi = Diag("marginal_kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, factors)
    singles = [
        Diag("marginal_kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [g])
        for g in factors
    ]
    got = multi.marginal_kernel_score(pred, y)
    expected = sum(m.marginal_kernel_score(pred, y) for m in singles) / len(factors)
    assert torch.allclose(got, expected)


def test_spectral_multi_bandwidth_is_mean():
    H, W = 4, 6
    pred, y = _spectral_pred(8, H, W), _spectral_field(9, H, W)
    factors = [5.0, 10.0, 20.0]
    multi = SpectralDiag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, factors)
    singles = [
        SpectralDiag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [g])
        for g in factors
    ]
    got = multi.kernel_score(pred, y)
    expected = sum(m.kernel_score(pred, y) for m in singles) / len(factors)
    assert torch.allclose(got, expected)


def test_whitened_multi_bandwidth_is_mean():
    """Covers the delegate index-mirroring: the SCORE kernel score delegates
    to a SpectralDiag whose active bandwidth index must track the outer sweep."""
    H, W = 4, 6
    pred, y = _whitened_pred(10, H, W), _spectral_field(11, H, W)
    factors = [5.0, 10.0]
    multi = SCORE("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, factors)
    singles = [
        SCORE("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [g])
        for g in factors
    ]
    got = multi.kernel_score(pred, y)
    expected = sum(m.kernel_score(pred, y) for m in singles) / len(factors)
    assert torch.allclose(got, expected)


def test_index_restored_after_score_call():
    """The bandwidth-sweep index is reset to 0 after a kernel_score call so a
    subsequent gamma2 read is deterministic."""
    pred, y = _phys_pred(12), _phys_y(13)
    method = Diag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [0.5, 2.0])
    method.kernel_score(pred, y)
    assert method._gamma_idx == 0


def test_train_mode_uses_train_bandwidths():
    """In training mode the score uses gamma_train, not gamma_eval."""
    pred, y = _phys_pred(14), _phys_y(15)
    method = Diag(
        "kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS,
        gamma=[0.5, 2.0], gamma_eval=[1.0],
    )
    train_ref = Diag("kernel_score", EPS, N_TRAIN, N_EVAL, OUT_CHANNELS, [0.5, 2.0])
    train_ref.training = True
    method.training = True
    assert torch.allclose(method.kernel_score(pred, y), train_ref.kernel_score(pred, y))
