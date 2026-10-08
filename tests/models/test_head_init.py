"""Tests for the ``other_zero_init`` head-initialization contract.

Zero-initializing the head projections makes every model start at the neutral
prediction -- but a group that enters the covariance *quadratically* has no
gradient at zero output: for LorD's ``Sigma = diag(d) + U Uᵀ`` the gradient
``dL/dU`` is proportional to ``U``, so ``U = 0`` is an exact stationary point
and a zero-init head silently never trains. The method therefore owns an
``other_zero_init`` property (like ``group_widths``/``group_denorm``) and the
backbones break the symmetry with a small random init when it is ``False``.
"""

import pytest
import torch

from src.methods import Deterministic, Diag, LorD, SpectralDiag, SCORE
from src.models.backbones.depthanything import DepthAnything
from src.models.backbones.patchtst import PatchTST
from src.models.backbones.ucast import DhariwalUNet

C = 1
METHOD_KW = dict(loss="kernel_score", eps=1e-9, n_train_samples=4,
                 n_eval_samples=4, out_channels=C, gamma=5.0)

# Small, known-working configs (ucast sizes mirror tests/models/test_backbone_heads.py).
UCAST_KW = dict(img_resolution=32, in_channels=3, model_channels=16,
                channel_mult=[1, 2], num_blocks=1, attn_levels=[])
PATCHTST_KW = dict(in_channels=C, out_channels=C, seq_len=32, pred_len=8,
                   patch_len=8, stride=4, n_layers=1, d_model=16, n_heads=4)

BACKBONES = ["ucast", "patchtst", "depthanything"]


def _build(name, method, other_zero_init, dinov2_stub_dir):
    gw = method.group_widths
    total = method.out_channels
    if name == "ucast":
        return DhariwalUNet(out_channels=total, group_widths=gw,
                            other_zero_init=other_zero_init, **UCAST_KW)
    if name == "patchtst":
        return PatchTST(group_widths=gw, group_denorm=method.group_denorm,
                        other_zero_init=other_zero_init, **PATCHTST_KW)
    return DepthAnything(dinov2_source_path=dinov2_stub_dir, group_widths=gw,
                         checkpoint_path=None, other_zero_init=other_zero_init)


def _other_proj(name, backbone):
    return backbone.depth_head.other_proj if name == "depthanything" else backbone.other_proj


def _batch(name):
    if name == "ucast":
        return torch.randn(2, 3, 32, 32), torch.rand(2, C, 32, 32)
    if name == "patchtst":
        return torch.randn(2, C, 32), torch.rand(2, C, 8)
    return torch.rand(2, 3, 28, 28), torch.rand(2, C, 28, 28)


def test_contract_only_lora_forbids_zero_init():
    """The property lives on the method, like group_widths/group_denorm."""
    for cls in (Deterministic, Diag, SpectralDiag, SCORE):
        assert cls(**METHOD_KW).other_zero_init is True
    assert LorD(rank=1, **METHOD_KW).other_zero_init is False


@pytest.mark.parametrize("name", BACKBONES)
def test_other_proj_zero_by_default(name, dinov2_stub_dir):
    """Scale-free "other" groups (the SCORE core) keep the exact zero init."""
    method = SCORE(**METHOD_KW)
    proj = _other_proj(name, _build(name, method, method.other_zero_init, dinov2_stub_dir))
    assert all((p == 0).all() for p in proj.parameters())


@pytest.mark.parametrize("name", BACKBONES)
def test_other_proj_small_random_for_lora(name, dinov2_stub_dir):
    """other_zero_init=False breaks the symmetry, but stays near-neutral."""
    method = LorD(rank=1, **METHOD_KW)
    proj = _other_proj(name, _build(name, method, method.other_zero_init, dinov2_stub_dir))
    weight = next(p for p in proj.parameters() if p.ndim > 1)
    assert weight.abs().max() > 0
    assert weight.abs().max() < 0.05  # symmetry-breaking, not a real prediction


@pytest.mark.parametrize("name", BACKBONES)
def test_lora_head_gradient(name, dinov2_stub_dir):
    """At exact zero init the LorD low-rank head receives
    *identically zero* gradient (stationary point of the quadratic U Uᵀ); with
    the small init it trains."""
    torch.manual_seed(0)
    method = LorD(rank=1, **METHOD_KW)
    method.training = True
    x, y = _batch(name)

    for zero_init, expect_zero_grad in ((True, True), (False, False)):
        backbone = _build(name, method, zero_init, dinov2_stub_dir)
        pred = method.process_output(backbone(x))
        loss = method.get_train_loss()(pred, y)
        loss.backward()
        grads = [p.grad for p in _other_proj(name, backbone).parameters()
                 if p.grad is not None]
        assert grads
        if expect_zero_grad:
            assert all((g == 0).all() for g in grads)
        else:
            assert max(g.abs().max() for g in grads) > 0
