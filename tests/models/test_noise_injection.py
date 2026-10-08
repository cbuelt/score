"""Tests for the latent-noise injection that turns a backbone into a sampler.

Three things have to hold for every backbone, and they are what these tests
pin:

1. **Layout.** The head is replayed per member and the members are folded back
   into the channel axis *group-inner*, so ``SampleBased.process_output``
   recovers member ``m`` of channel ``c`` exactly. With ``C == 1`` group-inner
   and group-outer coincide, so the round-trips here use ``C == 2``.
2. **Independence.** Members of one case must differ from each other, and cases
   must not leak into one another -- ``repeat_interleave`` versus ``repeat`` is
   a silent one-character bug that scrambles which member belongs to which case.
3. **Cost.** The trunk runs once no matter how many members are drawn; only the
   head is replayed. That is the whole reason the injector sits where it does.
"""

import pytest
import torch

from src.methods import SampleBased, Diag
from src.models.backbones.noise import (
    GaussianNoiseInjector,
    require_n_samples,
    stack_member_groups,
)
from src.models.backbones.patchtst import PatchTST
from src.models.backbones.ucast import DhariwalUNet

C = 2  # multi-channel: distinguishes group-inner from group-outer
IN_CH = 3
B, M = 3, 4
NOISE = {"global_dim": 8, "local_dim": 4, "hidden_dim": 6}

EPS = 1e-9
GAMMA = 1.0


def _method(out_channels: int = C) -> SampleBased:
    return SampleBased(
        loss="energy_score",
        eps=EPS,
        n_train_samples=M,
        n_eval_samples=M,
        out_channels=out_channels,
        gamma=GAMMA,
    )


def _patchtst(method, seq_len=32, pred_len=8, **overrides):
    kw = dict(
        in_channels=C,
        out_channels=method.out_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        group_widths=method.group_widths,
        group_denorm=method.group_denorm,
        other_zero_init=method.other_zero_init,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        patch_len=8,
        stride=4,
        noise=method.noise,
    )
    kw.update(overrides)
    return PatchTST(**kw).eval()


def _ucast(method, resolution=16, **overrides):
    kw = dict(
        img_resolution=resolution,
        in_channels=IN_CH,
        out_channels=method.out_channels,
        group_widths=method.group_widths,
        model_channels=8,
        channel_mult=[1, 2],
        num_blocks=1,
        attn_levels=[],
        other_zero_init=method.other_zero_init,
        noise=method.noise,
    )
    kw.update(overrides)
    return DhariwalUNet(**kw).eval()


# ---------------------------------------------------------------------------
# The injector itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel_axis", [1, -1])
def test_injector_expands_the_batch_member_fastest(channel_axis):
    """``(B, ...) -> (B*M, ...)`` with row ``b*M + m``, and the shape preserved."""
    torch.manual_seed(0)
    feature_dim = 5
    inj = GaussianNoiseInjector(feature_dim, channel_axis=channel_axis, **NOISE)
    shape = (B, feature_dim, 6) if channel_axis == 1 else (B, 6, feature_dim)
    features = torch.randn(*shape)
    out = inj(features, M)
    assert out.shape == (B * M, *shape[1:])

    # The residual branch is small at init, so each member still sits near its
    # own case's features -- which is what identifies the row ordering.
    rows = out.movedim(1, -1) if channel_axis == 1 else out
    src = features.movedim(1, -1) if channel_axis == 1 else features
    for b in range(B):
        for m in range(M):
            nearest = min(
                range(B), key=lambda k: (rows[b * M + m] - src[k]).abs().sum().item()
            )
            assert nearest == b, "members must stay with their own case"


def test_injector_members_are_not_identical():
    """Both latents must actually reach the output, or the ensemble is a Dirac."""
    torch.manual_seed(0)
    inj = GaussianNoiseInjector(5, channel_axis=-1, **NOISE)
    out = inj(torch.randn(B, 6, 5), M).reshape(B, M, 6, 5)
    assert (out.std(dim=1) > 0).all()


def test_injector_needs_both_latents():
    """The two-latent design is load-bearing; a zero width is a config error."""
    with pytest.raises(ValueError, match="global and a local latent"):
        GaussianNoiseInjector(5, global_dim=0, local_dim=4)
    with pytest.raises(ValueError, match="global and a local latent"):
        GaussianNoiseInjector(5, global_dim=4, local_dim=0)


def test_global_latent_is_shared_across_positions_and_local_is_not():
    """The property that separates dependence from marginal spread.

    With the local latent silenced the members of a case differ only through one
    vector shared by every position, so the residual field is perfectly rank-1
    across positions. With the global one silenced it is not.
    """
    torch.manual_seed(0)

    def residual_rank(global_dim, local_dim):
        inj = GaussianNoiseInjector(
            4, channel_axis=-1, global_dim=global_dim, local_dim=local_dim, hidden_dim=8
        )
        # Undo the deliberately tiny init so the residual is measurable.
        with torch.no_grad():
            inj.mix[-1].weight.mul_(1000.0)
        if global_dim == 1:
            with torch.no_grad():  # silence the global latent
                inj.mix[0].weight[:, 4:5].zero_()
        if local_dim == 1:
            with torch.no_grad():  # silence the local latent
                inj.mix[0].weight[:, 5:].zero_()
        features = torch.randn(1, 12, 4)
        out = inj(features, 64).reshape(64, 12, 4)
        resid = (out - features).reshape(64, -1)
        return torch.linalg.matrix_rank(resid - resid.mean(0), rtol=1e-3).item()

    global_only = residual_rank(global_dim=8, local_dim=1)
    local_only = residual_rank(global_dim=1, local_dim=8)
    assert global_only < local_only


def test_stack_member_groups_is_group_inner():
    """``(B*M, C, *spatial) -> (B, C*M, *spatial)`` with each channel contiguous."""
    out = torch.zeros(B * M, C, 5)
    for b in range(B):
        for m in range(M):
            for c in range(C):
                out[b * M + m, c] = b * 1000 + c * 100 + m
    stacked = stack_member_groups(out, M, C, (5,))
    assert stacked.shape == (B, C * M, 5)
    for b in range(B):
        for c in range(C):
            for m in range(M):
                assert torch.all(stacked[b, c * M + m] == b * 1000 + c * 100 + m)


def test_require_n_samples_refuses_to_default():
    """A silent M=1 would divide by M(M-1) = 0 several layers away from the cause."""
    with pytest.raises(ValueError, match="explicit n_samples"):
        require_n_samples(None, "SomeBackbone")
    assert require_n_samples(7, "SomeBackbone") == 7


# ---------------------------------------------------------------------------
# The backbones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["patchtst", "ucast"])
def test_backbone_emits_members_that_process_output_recovers(label):
    """End to end: head output -> ``process_output`` -> ``(B, C, *spatial, M)``."""
    torch.manual_seed(0)
    method = _method()
    if label == "patchtst":
        net, x, spatial = _patchtst(method), torch.randn(B, C, 32), (8,)
    else:
        net, x, spatial = _ucast(method), torch.randn(B, IN_CH, 16, 16), (16, 16)

    raw = net(x, n_samples=M)
    assert raw.shape == (B, C * M, *spatial)
    pred = method.process_output(raw)
    assert pred.shape == (B, C, *spatial, M)
    # Every member is a genuine draw, and the ensemble is not collapsed. (The
    # head is zero-init, so give it a nudge first -- see the injector's own
    # small-init comment.)
    for module in net.modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == method.out_channels:
            torch.nn.init.normal_(module.weight, std=0.1)
    pred = method.process_output(net(x, n_samples=M))
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize("label", ["patchtst", "ucast"])
def test_backbone_without_noise_is_untouched(label):
    """A method with no ``noise`` spec must produce exactly the path without injection."""
    torch.manual_seed(0)
    diag = Diag(
        loss="kernel_score", eps=EPS, n_train_samples=2, n_eval_samples=2,
        out_channels=C, gamma=GAMMA,
    )
    assert diag.noise is None and diag.samples_from_backbone is False
    if label == "patchtst":
        net = _patchtst(diag, noise=diag.noise, out_channels=diag.out_channels,
                        group_widths=diag.group_widths, group_denorm=diag.group_denorm)
        x, spatial = torch.randn(B, C, 32), (8,)
    else:
        net = _ucast(diag, noise=diag.noise, out_channels=diag.out_channels,
                     group_widths=diag.group_widths)
        x, spatial = torch.randn(B, IN_CH, 16, 16), (16, 16)
    assert net.noise_injector is None
    out = net(x)
    assert out.shape == (B, 2 * C, *spatial)
    # Deterministic: two calls agree bit for bit.
    assert torch.equal(out, net(x))


@pytest.mark.parametrize("label", ["patchtst", "ucast"])
def test_backbone_requires_n_samples_when_noise_is_built(label):
    method = _method()
    if label == "patchtst":
        net, x = _patchtst(method), torch.randn(B, C, 32)
    else:
        net, x = _ucast(method), torch.randn(B, IN_CH, 16, 16)
    with pytest.raises(ValueError, match="explicit n_samples"):
        net(x)


def test_patchtst_denormalizes_each_member_by_its_own_case_statistics():
    """RevIN stats are one per case and must be repeated member-wise, not tiled.

    Feeding two cases on wildly different offsets makes the failure visible: a
    ``repeat`` instead of a ``repeat_interleave`` would denormalize case 0's
    members with case 1's window mean, putting members of one case on the wrong
    level entirely.
    """
    torch.manual_seed(0)
    method = _method(out_channels=1)
    net = _patchtst(method, in_channels=1, out_channels=1,
                    group_widths=method.group_widths)
    # Two cases 100 apart; RevIN centres each on its own window mean, so every
    # member of case i must come back near that case's own level.
    x = torch.stack([torch.randn(1, 32), 100.0 + torch.randn(1, 32)])
    pred = method.process_output(net(x, n_samples=M))  # (2, 1, 8, M)
    assert pred[0].abs().max() < 10.0
    assert (pred[1] - 100.0).abs().max() < 10.0


def test_trunk_runs_once_regardless_of_member_count():
    """The injector sits after the trunk, so M scales head cost only."""
    method = _method()
    net = _patchtst(method)
    calls = {"n": 0}
    trunk = net.backbone
    original = trunk.forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    trunk.forward = counting
    net(torch.randn(B, C, 32), n_samples=32)
    assert calls["n"] == 1
