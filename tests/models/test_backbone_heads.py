"""Tests for the backbone output heads and their channel-stacking contract.

Both the ucast :class:`~src.models.backbones.ucast.DhariwalUNet` and
:class:`~src.models.backbones.depthanything.DepthAnything` stack their per-group
head outputs along the channel dim in **group-inner** order (for each physical
channel ``c`` its groups are contiguous), which is the exact inverse of every
method's ``process_output`` reshape ``(B, C, G, ...)``. With ``C == 1`` group-inner
and group-outer coincide, so the round-trip tests here deliberately use ``C == 2``
to guard the ordering for multi-channel targets.
"""

import pytest
import torch

from src.methods import (
    Deterministic,
    Diag,
    LorD,
    SCORE,
)
from src.models.backbones.depthanything import DepthAnything
from src.models.backbones.ucast import DhariwalUNet

C = 2  # multi-channel: distinguishes group-inner from group-outer
IN_CH = 5
B, H, W = 2, 32, 32

# Small, shallow ucast so the tests stay fast (channel_mult length 2 -> one down).
UCAST_KW = dict(
    img_resolution=32,
    in_channels=IN_CH,
    model_channels=16,
    channel_mult=[1, 2],
    num_blocks=1,
    attn_levels=[],
)


def _ucast(group_widths, **overrides):
    total = sum(w for w in group_widths.values() if w is not None)
    kw = dict(UCAST_KW)
    kw.update(overrides)
    return DhariwalUNet(out_channels=total, group_widths=group_widths, **kw)


# ---------------------------------------------------------------------------
# ucast head shapes: which heads exist per method, and output channel count
# ---------------------------------------------------------------------------

HEAD_CASES = [
    ("deterministic", {"mu": C, "sigma": None, "other": None}, False),
    ("diagonal", {"mu": C, "sigma": C, "other": None}, False),
    ("whitened", {"mu": C, "sigma": C, "other": C}, True),
    ("lora_r3", {"mu": C, "sigma": C, "other": 3 * C}, True),
]


@pytest.mark.parametrize(
    "label, gw, has_other", HEAD_CASES, ids=[c[0] for c in HEAD_CASES]
)
def test_ucast_head_shape_and_presence(label, gw, has_other):
    """The head emits ``out_channels`` channels and builds ``other`` only when needed."""
    net = _ucast(gw).eval()
    total = sum(w for w in gw.values() if w is not None)
    x = torch.randn(B, IN_CH, H, W)
    out = net(x)
    assert out.shape == (B, total, H, W)
    assert net.has_other is has_other
    assert hasattr(net, "other_block") is has_other
    # The output must un-stack cleanly into (B, C, G, H, W) group-inner form.
    out.reshape(B, C, total // C, H, W)


def test_ucast_fallback_single_head_when_group_widths_none():
    """``group_widths=None`` (e.g. config-only instantiate) -> single flat head."""
    net = DhariwalUNet(out_channels=4, group_widths=None, **UCAST_KW).eval()
    x = torch.randn(B, IN_CH, H, W)
    assert net(x).shape == (B, 4, H, W)
    assert net.has_other is False


def test_ucast_shared_head_depth_is_configurable():
    """``head.shared_num_blocks``/``other_num_blocks`` stack that many UNetBlocks."""
    gw = {"mu": C, "sigma": C, "other": C}
    net = _ucast(gw, head={"shared_num_blocks": 3, "other_num_blocks": 2})
    # n_blocks > 1 wraps the blocks in a Sequential of that length.
    assert len(net.shared_block) == 3
    assert len(net.other_block) == 2
    net.eval()
    assert net(torch.randn(B, IN_CH, H, W)).shape == (B, 3 * C, H, W)


# ---------------------------------------------------------------------------
# Group-inner ordering: the headline correctness guard (C = 2)
# ---------------------------------------------------------------------------

def _set_constant_head(conv, values):
    """Make ``conv`` emit a constant ``values[i]`` on its i-th output channel."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.copy_(torch.tensor(values, dtype=conv.bias.dtype))


def test_depthanything_is_group_inner(dinov2_stub_dir):
    """DepthAnything stacks [mu, diag, other] group-inner for C>1.

    Under a group-outer ``cat`` this assertion fails for C=2. Checked on a second
    backbone because the ordering is a *cross-module* contract: it is the exact
    inverse of every method's ``process_output`` reshape, so a backbone that gets
    it wrong mislabels channels silently rather than raising.
    """
    net = DepthAnything(
        dinov2_source_path=dinov2_stub_dir,
        group_widths={"mu": C, "sigma": C, "other": C},
        checkpoint_path=None,
    ).eval()
    head = net.depth_head
    # shared_proj emits [mu(C), sigma(C)] group-inner (index c*2+g); other_proj
    # emits the remaining group.
    shared_vals = []
    for c in range(C):
        shared_vals += [100 + c, 200 + c]  # mu_c, diag_c
    _set_constant_head(head.shared_proj, shared_vals)
    _set_constant_head(head.other_proj, [300 + c for c in range(C)])

    out = net(torch.randn(B, 3, H, W))[0, :, 0, 0]
    # group-inner: [mu0, diag0, other0, mu1, diag1, other1]
    expected = torch.tensor([100.0, 200.0, 300.0, 101.0, 201.0, 301.0])
    assert torch.allclose(out, expected), out.tolist()


def test_ucast_is_group_inner():
    """DhariwalUNet stacks mu/sigma/other group-inner for C>1."""
    gw = {"mu": C, "sigma": C, "other": C}
    net = _ucast(gw).eval()
    # shared_proj emits [mu(C), sigma(C)] in group-inner order (index c*2+g).
    shared_vals = []
    for c in range(C):
        shared_vals += [100 + c, 200 + c]  # mu_c, sigma_c
    _set_constant_head(net.shared_proj, shared_vals)
    _set_constant_head(net.other_proj, [300 + c for c in range(C)])

    r = net(torch.randn(B, IN_CH, H, W)).reshape(B, C, 3, H, W)[0, :, :, 0, 0]
    assert torch.allclose(r[:, 0], torch.tensor([100.0, 101.0]))  # mu
    assert torch.allclose(r[:, 1], torch.tensor([200.0, 201.0]))  # sigma
    assert torch.allclose(r[:, 2], torch.tensor([300.0, 301.0]))  # other


# ---------------------------------------------------------------------------
# Full round-trip: backbone head -> method.process_output recovers the groups
# ---------------------------------------------------------------------------

def test_ucast_whitened_round_trip_through_process_output():
    """ucast head output feeds SCORE.process_output at C=2.

    process_output reshapes (B, 3C, ...) -> (B, C, ..., 3) = [mu, diag, s]. With a
    group-inner head, mu/diag/core each recover the intended per-channel
    constants. A group-outer head would scramble channels 0 and 1.
    """
    gw = {"mu": C, "sigma": C, "other": C}
    net = _ucast(gw).eval()
    shared_vals = []
    for c in range(C):
        shared_vals += [10 + c, 1 + c]  # mu_c, diag_c (diag>0 after softplus)
    _set_constant_head(net.shared_proj, shared_vals)
    _set_constant_head(net.other_proj, [0.5 + c for c in range(C)])

    head_out = net(torch.randn(B, IN_CH, H, W))
    method = SCORE(
        loss="kernel_score", out_channels=C, eps=1e-9,
        n_train_samples=4, n_eval_samples=8, gamma=1.0,
    )
    pred = method.process_output(head_out)  # (B, C, H, W, 3)
    assert pred.shape == (B, C, H, W, 3)
    # mu is passed through untouched; check per-channel means recover the inputs.
    mu = pred[..., 0]
    assert torch.allclose(mu[:, 0], torch.full_like(mu[:, 0], 10.0), atol=1e-4)
    assert torch.allclose(mu[:, 1], torch.full_like(mu[:, 1], 11.0), atol=1e-4)


def test_depthanything_whitened_round_trip_through_process_output(dinov2_stub_dir):
    """Same round-trip for the DepthAnything backbone at C=2."""
    net = DepthAnything(
        dinov2_source_path=dinov2_stub_dir,
        group_widths={"mu": C, "sigma": C, "other": C},
        checkpoint_path=None,
    ).eval()
    head = net.depth_head
    shared_vals = []
    for c in range(C):
        shared_vals += [10 + c, 1 + c]  # mu_c, diag_c (diag>0 after softplus)
    _set_constant_head(head.shared_proj, shared_vals)
    _set_constant_head(head.other_proj, [0.5 + c for c in range(C)])

    head_out = net(torch.randn(B, 3, H, W))
    method = SCORE(
        loss="kernel_score", out_channels=C, eps=1e-9,
        n_train_samples=4, n_eval_samples=8, gamma=1.0,
    )
    pred = method.process_output(head_out)
    mu = pred[..., 0]
    assert torch.allclose(mu[:, 0], torch.full_like(mu[:, 0], 10.0), atol=1e-4)
    assert torch.allclose(mu[:, 1], torch.full_like(mu[:, 1], 11.0), atol=1e-4)


# ---------------------------------------------------------------------------
# ucast stage-2 freezing (two-stage SCORE fit)
# ---------------------------------------------------------------------------

def test_ucast_freeze_for_stage2_leaves_only_other_trainable():
    """After freeze + a subsequent .train(), only the ``other`` head trains and the
    frozen trunk/shared head stay in eval mode (deterministic dropout)."""
    gw = {"mu": C, "sigma": C, "other": C}
    net = _ucast(gw)
    net.train()
    net.freeze_for_stage2()
    net.train()  # Lightning re-calls train() each epoch; must stay frozen.

    trainable = {n.split(".")[0] for n, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"other_block", "other_norm", "other_proj"}
    for module in (net.enc, net.dec, net.shared_block, net.shared_norm, net.shared_proj):
        assert module.training is False
    assert net.other_block.training is True


def test_ucast_freeze_for_stage2_requires_other_head():
    """Freezing without an ``other`` head (e.g. diagonal) is a loud error."""
    net = _ucast({"mu": C, "sigma": C, "other": None})
    with pytest.raises(ValueError):
        net.freeze_for_stage2()
