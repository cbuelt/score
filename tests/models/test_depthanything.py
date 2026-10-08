"""Tests for the DepthAnything backbone's checkpoint loading contract.

``load_pretrained`` loads the official Depth Anything checkpoint with
``strict=False`` because two mismatches are *expected*: the new probabilistic
head modules are absent from the checkpoint, and the removed upstream scalar
head (``output_conv2``) is present only there. Anything beyond that means
encoder/decoder weights silently stayed at random init, which must be a loud
error. These tests pin that behaviour using the DINOv2 hub stub from conftest.
"""

import pytest
import torch

from src.models.backbones.depthanything import DepthAnything

GROUP_WIDTHS = {"mu": 1, "sigma": 1, "other": 1}


@pytest.fixture
def backbone(dinov2_stub_dir):
    return DepthAnything(
        dinov2_source_path=dinov2_stub_dir,
        group_widths=GROUP_WIDTHS,
        checkpoint_path=None,
    )


def _official_like_state_dict(backbone):
    """A checkpoint in the official layout: no new head modules, plus the
    removed upstream scalar head."""
    new_modules = (
        "depth_head.shared_block.",
        "depth_head.shared_proj.",
        "depth_head.other_block.",
        "depth_head.other_proj.",
        "depth_head.noise_injector.",
        "image_mean",
        "image_std",
    )
    state = {
        k: v for k, v in backbone.state_dict().items() if not k.startswith(new_modules)
    }
    state["depth_head.scratch.output_conv2.0.weight"] = torch.zeros(1)
    return state


def test_out_channels_matches_group_widths(backbone):
    """``out_channels`` is derived from group_widths (what the head emits)."""
    assert backbone.out_channels == sum(GROUP_WIDTHS.values())


def test_load_pretrained_accepts_official_layout(backbone, tmp_path):
    """The known mismatches (new heads missing, old scalar head extra) load fine."""
    path = tmp_path / "ckpt.pth"
    torch.save(_official_like_state_dict(backbone), path)
    backbone.load_pretrained(str(path))


def test_load_pretrained_rejects_missing_encoder_keys(backbone, tmp_path):
    """A checkpoint that fails to cover the encoder must raise, not silently
    leave those weights at random init."""
    state = _official_like_state_dict(backbone)
    dropped = next(k for k in state if k.startswith("pretrained."))
    del state[dropped]
    path = tmp_path / "ckpt.pth"
    torch.save(state, path)

    with pytest.raises(ValueError, match="does not match"):
        backbone.load_pretrained(str(path))


def test_load_pretrained_rejects_unexpected_keys(backbone, tmp_path):
    """Unknown extra keys (layout drift) must raise as well."""
    state = _official_like_state_dict(backbone)
    state["depth_head.some_renamed_module.weight"] = torch.zeros(1)
    path = tmp_path / "ckpt.pth"
    torch.save(state, path)

    with pytest.raises(ValueError, match="does not match"):
        backbone.load_pretrained(str(path))


def test_load_pretrained_tolerates_the_noise_injector(dinov2_stub_dir, tmp_path):
    """The SB method's noise injector is a *new* module, so it may be absent.

    The allowlist of legitimately-missing keys is spelled out literally, so
    every new module added under ``depth_head`` has to be added to it; otherwise
    the official checkpoint refuses to load for the SB arm on depth -- a failure
    that only appears once a real pretrained checkpoint is involved.
    """
    backbone = DepthAnything(
        dinov2_source_path=dinov2_stub_dir,
        group_widths={"mu": 1, "sigma": None, "other": None},
        checkpoint_path=None,
        noise={"global_dim": 4, "local_dim": 2, "hidden_dim": 4},
    )
    assert backbone.depth_head.noise_injector is not None
    injector_keys = [
        k for k in backbone.state_dict() if "noise_injector" in k
    ]
    assert injector_keys, "the injector must contribute parameters"

    path = tmp_path / "ckpt.pth"
    torch.save(_official_like_state_dict(backbone), path)
    backbone.load_pretrained(str(path))
