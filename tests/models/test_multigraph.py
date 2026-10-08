"""The station-graph backbone: shapes, graph construction, exchangeability."""

import math

import numpy as np
import pytest
import torch

from src.methods.diagonal import Diag
from src.methods.lord import LorD
from src.models.backbones import Multigraph

from tests.conftest import EUPP_N_FEATURES, EUPP_N_STATIONS

MAX_DIST = 100.0


def _method(kind):
    common = dict(
        eps=1e-9, n_train_samples=4, n_eval_samples=4,
        out_channels=1, gamma=[math.sqrt(EUPP_N_STATIONS)],
    )
    if kind == "diag":
        return Diag(loss="crps", **common)
    return LorD(loss="kernel_score", rank=3, **common)


def _backbone(meta_path, method, **kwargs):
    return Multigraph(
        in_channels=EUPP_N_FEATURES,
        out_channels=method.out_channels,
        group_widths=method.group_widths,
        other_zero_init=method.other_zero_init,
        meta_path=f"{meta_path}/meta.npz",
        max_dist=MAX_DIST,
        embedding_dim=4,
        hidden_channels_gnn=8,
        num_layers_gnn=2,
        heads=2,
        hidden_channels_deepset=8,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["diag", "lora"])
def test_output_shape_matches_the_method_head(euppbench_data_dir, kind):
    """The head width must be whatever the predictive method asks for."""
    method = _method(kind)
    backbone = _backbone(euppbench_data_dir, method)
    out = backbone(torch.randn(2, EUPP_N_FEATURES, 3, EUPP_N_STATIONS))

    assert out.shape == (2, method.out_channels, EUPP_N_STATIONS)
    # And `process_output` must be able to un-stack it into the group layout.
    assert method.process_output(out).shape == (
        2, 1, EUPP_N_STATIONS, method.out_channels
    )


def test_edge_count_matches_the_distance_threshold(euppbench_data_dir):
    """Spatial edges follow the distance matrix; inter-member edges are all-pairs.

    Pinning both counts catches the easy mistakes in the graph arithmetic: a
    missing member layer, or a one-directional inter-member connection.
    """
    method = _method("diag")
    backbone = _backbone(euppbench_data_dir, method)
    with np.load(f"{euppbench_data_dir}/meta.npz") as f:
        dist = f["dist"]
    n_spatial = int(((dist > 0) & (dist <= MAX_DIST)).sum())

    members = 3
    edge_index, edge_attr, station_idx = backbone._graph(members, torch.device("cpu"))
    expected = n_spatial * members + EUPP_N_STATIONS * members * (members - 1)

    assert edge_index.shape == (2, expected)
    assert edge_attr.shape == (expected, 1)
    assert edge_index.max() < members * EUPP_N_STATIONS
    # Node j = member * N + station, so the station index cycles per layer.
    assert torch.equal(station_idx, torch.arange(EUPP_N_STATIONS).repeat(members))


def test_invariant_to_ensemble_member_order(euppbench_data_dir):
    """Members are exchangeable, so permuting them must not change the output.

    This is the property that lets a model fitted on 3 (really 11) reforecast
    members be applied to 5 (really 51) forecast members.
    """
    method = _method("diag")
    backbone = _backbone(euppbench_data_dir, method).eval()
    x = torch.randn(2, EUPP_N_FEATURES, 3, EUPP_N_STATIONS)

    with torch.no_grad():
        base = backbone(x)
        permuted = backbone(x[:, :, torch.randperm(3)])
    assert torch.allclose(base, permuted, atol=1e-5)


def test_accepts_a_different_member_count_than_it_was_built_for(euppbench_data_dir):
    """The test split has more members than the training split; both must work."""
    method = _method("diag")
    backbone = _backbone(euppbench_data_dir, method).eval()
    with torch.no_grad():
        assert backbone(torch.randn(2, EUPP_N_FEATURES, 3, EUPP_N_STATIONS)).shape[0] == 2
        assert backbone(torch.randn(2, EUPP_N_FEATURES, 5, EUPP_N_STATIONS)).shape[0] == 2


def test_member_chunking_splits_evenly_and_preserves_shape(euppbench_data_dir):
    """Chunked inference must produce the same shape as an unchunked pass.

    Chunking is mandatory at the real ensemble size (a full 51-member graph
    exceeds GPU memory), so it must not be a shape-changing code path.
    """
    method = _method("diag")
    chunked = _backbone(euppbench_data_dir, method, member_chunk=2).eval()
    with torch.no_grad():
        out = chunked(torch.randn(2, EUPP_N_FEATURES, 5, EUPP_N_STATIONS))
    assert out.shape == (2, method.out_channels, EUPP_N_STATIONS)

    # `tensor_split` keeps the groups even; `chunk` would leave a short tail.
    sizes = [c.shape[0] for c in torch.tensor_split(torch.arange(51), -(-51 // 11))]
    assert sizes == [11, 10, 10, 10, 10]


def test_station_mismatch_is_rejected(euppbench_data_dir):
    """A target width that disagrees with the graph must fail loudly."""
    method = _method("diag")
    backbone = _backbone(euppbench_data_dir, method)
    with pytest.raises(ValueError, match="stations but the graph"):
        backbone(torch.randn(2, EUPP_N_FEATURES, 3, EUPP_N_STATIONS + 1))


def test_gradients_reach_every_parameter(euppbench_data_dir):
    """Including the station embedding, which only sees gradient through the graph."""
    method = _method("diag")
    backbone = _backbone(euppbench_data_dir, method)
    y = torch.randn(2, 1, EUPP_N_STATIONS)
    y[0, 0, 0] = float("nan")

    pred = method.process_output(backbone(torch.randn(2, EUPP_N_FEATURES, 3, EUPP_N_STATIONS)))
    method.crps(pred, y).backward()

    missing = [name for name, p in backbone.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
