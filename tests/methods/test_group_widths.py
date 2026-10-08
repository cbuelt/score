"""Cross-method tests for the ``group_widths`` head-layout contract.

Every :class:`~src.methods.base.PredictiveMethod` exposes ``group_widths`` -- an
ordered ``{"mu", "sigma", "other"}`` dict of absolute channel widths (``None`` for
an absent group) -- from which the base class derives ``out_channels`` as the sum
of the present widths. The backbone reads ``group_widths`` to decide which output
heads to build; ``process_output`` un-stacks the same groups. These tests pin the
per-method layout and the ``out_channels == sum(group_widths)`` invariant that
keeps the flat count and the per-group widths from ever disagreeing.
"""

import pytest

from src.methods import (
    Deterministic,
    SampleBased,
    Cholesky,
    Diag,
    LorD,
    SpectralDiag,
    SCORE,
)

C = 3  # raw dataset channel count
V = 12  # flattened event size for the dense family (C channels x 4 positions)
EPS = 1e-9
N_TRAIN, N_EVAL = 10, 50
GAMMA = 1.0

_COMMON = dict(eps=EPS, n_train_samples=N_TRAIN, n_eval_samples=N_EVAL, gamma=GAMMA)


def _make(cls, **extra):
    return cls(loss="kernel_score", out_channels=C, **_COMMON, **extra)


# (label, constructed method, expected group_widths)
CASES = [
    (
        "deterministic",
        Deterministic(loss="mse", out_channels=C, **_COMMON),
        {"mu": C, "sigma": None, "other": None},
    ),
    ("phys_diag", _make(Diag), {"mu": C, "sigma": C, "other": None}),
    ("spectral_diag", _make(SpectralDiag), {"mu": C, "sigma": C, "other": None}),
    ("whitened", _make(SCORE), {"mu": C, "sigma": C, "other": C}),
    ("lora_r4", _make(LorD, rank=4), {"mu": C, "sigma": C, "other": 4 * C}),
    (
        "cholesky",
        _make(Cholesky, event_dim=V),
        {"mu": C, "sigma": C, "other": (V - 1) * C},
    ),
    # One group, the sample itself: the head is replayed once per member rather
    # than widened, so the width is C whatever M is.
    (
        "generative",
        SampleBased(loss="energy_score", out_channels=C, **_COMMON),
        {"mu": C, "sigma": None, "other": None},
    ),
]


@pytest.mark.parametrize("label, method, expected", CASES, ids=[c[0] for c in CASES])
def test_group_widths_layout(label, method, expected):
    """Each method reports the documented per-group channel widths."""
    assert method.group_widths == expected


@pytest.mark.parametrize("label, method, expected", CASES, ids=[c[0] for c in CASES])
def test_out_channels_is_sum_of_group_widths(label, method, expected):
    """``out_channels`` is exactly the sum of the present group widths.

    This is the invariant that lets the backbone size its heads from
    ``group_widths`` while the training loop still passes the flat ``out_channels``.
    """
    assert method.out_channels == sum(w for w in expected.values() if w is not None)


def test_group_widths_keys_are_ordered_mu_sigma_other():
    """The key order (mu, sigma, other) is the channel stacking order."""
    for _, method, _ in CASES:
        assert list(method.group_widths.keys()) == ["mu", "sigma", "other"]


def test_lora_other_width_tracks_rank():
    """The LorD ``other`` group is ``rank * C`` (the low-rank factor U)."""
    for rank in (1, 2, 5):
        method = _make(LorD, rank=rank)
        assert method.group_widths["other"] == rank * C
        assert method.out_channels == (rank + 2) * C


def test_cholesky_other_width_tracks_event_dim():
    """The dense ``other`` group is ``(V-1) * C``: one row of ``L`` per coordinate.

    Row ``i`` needs only its first ``i`` entries, but the head layout allocates
    ``k`` fields over the whole event grid, so the widest row sets ``k = V-1``
    and the ``tril`` masks the rest away.
    """
    for event_dim in (2, 8, 20):
        method = _make(Cholesky, event_dim=event_dim)
        assert method.group_widths["other"] == (event_dim - 1) * C
        assert method.out_channels == (event_dim + 1) * C


def test_cholesky_drops_other_head_when_scalar():
    """A one-dimensional target has no off-diagonal, so there is no third head."""
    method = _make(Cholesky, event_dim=1)
    assert method.group_widths == {"mu": C, "sigma": C, "other": None}
    assert method.out_channels == 2 * C


def test_cholesky_refuses_gridded_event_dims():
    """The dense factor is refused above ``max_event_dim`` instead of running out of memory."""
    with pytest.raises(NotImplementedError, match="max_event_dim"):
        _make(Cholesky, event_dim=Cholesky.max_event_dim + 1)


def test_cholesky_requires_event_dim():
    """``event_dim`` is not optional: the head cannot be sized without it."""
    with pytest.raises(ValueError, match="event_dim"):
        _make(Cholesky)
