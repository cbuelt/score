"""Graph attention backbone for station-based ensemble post-processing.

Port of the "multigraph" GAT model of Feik, Lerch & Stühmer (2024),
*Graph Neural Networks and Spatial Information Learning for Post-Processing
Ensemble Weather Forecasts* (arXiv:2407.11050), reference implementation at
https://github.com/hits-mli/gnn-post-processing (``models/graphensemble/multigraph.py``).

**The graph.** One graph per forecast day. A node ``v_{s,n}`` is station ``s`` as
seen by ensemble member ``n``, so the node set is the ``N x M`` product of
stations and members. Two kinds of edge:

* *spatial* -- stations closer than ``max_dist`` km, within one member's layer,
  carrying the normalized geodesic distance as their edge attribute;
* *inter-member* -- the same station across every pair of members, fully
  connected and bidirectional, carrying a small constant attribute
  (``INTER_MEMBER_ATTR``) so the attention can tell the two kinds apart.

After ``num_layers_gnn`` residual GATv2 blocks the member axis is collapsed by a
permutation-invariant Deep Set aggregation, leaving one feature vector per
station which the head maps to the predictive method's parameter groups.

**Layout.** Unlike the reference, which builds ``torch_geometric.data.Data``
objects and batches them with PyG's loader, this backbone consumes the repo's
ordinary dense batch ``x: (B, F, M, N)``. The graph is identical for every
sample, so it is built once and replicated by index arithmetic. Only
``GATv2Conv`` is borrowed from PyG. Node ``j`` of a sample is
``member * N + station``, matching a ``(B, M, N, F) -> (B*M*N, F)`` flatten.

**Memory.** The edge count grows as ``M**2``: at ``N=115`` and ``max_dist=100``
km there are 1224 spatial edges, so one sample has ``1224*M + 115*M*(M-1)``
edges -- 26k at ``M=11`` but 356k at ``M=51``. Since GATv2 materializes
``(E, heads, hidden)`` per layer, a full 51-member batch needs tens of GiB. Use
``member_chunk`` at evaluation time; Feik et al. do the same (4 groups of 10
plus one of 11), both to bound memory and to match the 11-member ensemble size
the model was trained on.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from src.models.backbones.noise import (
    GaussianNoiseInjector,
    require_n_samples,
    stack_member_groups,
)

# Edge attribute placed on the inter-member edges. The spatial attribute is a
# normalized distance in (0, 1], so a small positive constant keeps the two edge
# kinds distinguishable while staying off the value range the distances use.
# `0.01` follows the reference implementation, which also uses it as the
# `fill_value` for the self-loops GATv2Conv adds.
INTER_MEMBER_ATTR = 0.01


class EmbedStations(nn.Module):
    """Prepend a learned per-station embedding to each node's feature vector.

    The station identity is what lets one jointly-fitted model be locally
    adaptive, playing the same role as the station embeddings of Rasp & Lerch
    (2018). In the reference the station id travels as feature column 0 and is
    swapped for its embedding here; in this dense layout the station is the
    position along the node axis, so the index is passed separately.
    """

    def __init__(self, num_stations: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_stations, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.embed(station_idx), x], dim=-1)


class ResGnn(nn.Module):
    """Stack of GATv2 blocks with residual connections after the first.

    The first block changes the feature width and so cannot be residual; every
    later block adds its (ReLU'd) output to its input, which is what lets the
    stack go several hops deep without the usual attention-stacking
    instabilities. A final linear map sets the output width.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.convolutions = nn.ModuleList()
        for layer in range(num_layers):
            self.convolutions.append(
                GATv2Conv(
                    in_channels if layer == 0 else hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    edge_dim=1,
                    add_self_loops=True,
                    fill_value=INTER_MEMBER_ATTR,
                )
            )
        self.lin = nn.Linear(hidden_channels * heads, out_channels)
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convolutions):
            if i == 0:
                x = self.relu(conv(x, edge_index, edge_attr))
            else:
                x = x + self.relu(conv(x, edge_index, edge_attr))
        return self.lin(x)


class DeepSetAggregator(nn.Module):
    """Permutation-invariant pooling over the ensemble-member axis.

    ``rho(mean_n phi(h_n))`` in the Deep Sets sense (Zaheer et al., 2017). Since
    ensemble members are exchangeable, the aggregation must not depend on their
    order -- and because it is a *mean*, it also transfers across ensemble sizes,
    which is what lets a model fitted on 11 reforecast members be applied to 51
    forecast members.

    Note: the reference implementation calls its second post-aggregation layer
    without assigning the result, so that layer is silently inert. That reads as
    a slip rather than a design, so it is applied here; ``layers_out=1`` recovers
    the reference's effective depth exactly.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers_in: int = 2,
        layers_out: int = 2,
    ):
        super().__init__()
        if layers_in < 1 or layers_out < 1:
            raise ValueError("layers_in and layers_out must both be >= 1")

        def mlp(width_in: int, width_out: int, n_layers: int) -> nn.Sequential:
            layers: list = []
            for i in range(n_layers):
                layers.append(
                    nn.Linear(width_in if i == 0 else hidden_channels,
                              width_out if i == n_layers - 1 else hidden_channels)
                )
                if i < n_layers - 1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.phi = mlp(in_channels, hidden_channels, layers_in)
        self.relu = nn.ReLU()
        self.rho = mlp(hidden_channels, out_channels, layers_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, M, N, C_in)``; returns ``(B, N, C_out)``."""
        x = self.phi(x)
        x = x.mean(dim=1)  # exchangeable members -> plain mean over the member axis
        return self.rho(self.relu(x))


class Multigraph(nn.Module):
    """GATv2 + Deep Set backbone over a station graph.

    Consumes ``x: (B, F, M, N)`` and emits ``(B, out_channels, N)`` laid out in
    group-inner order, so it plugs into any predictive method in
    :mod:`src.methods` without change: ``Diag`` (mu, sigma) reproduces the
    model of Feik et al. (2024), while ``LorD`` / ``SCORE`` turn the same
    backbone into a joint multivariate predictive over the stations.

    Args:
        in_channels: Predictor count ``F`` per node.
        out_channels: Total head width (fallback when ``group_widths is None``).
        group_widths: Per-group channel widths from the predictive method.
        other_zero_init: Whether the ``"other"`` projection may start at zero;
            see :attr:`~src.methods.base.PredictiveMethod.other_zero_init`.
        meta_path: ``meta.npz`` written by ``src/data/build_euppbench.py``,
            supplying the station distance matrix.
        max_dist: Spatial edges are drawn between stations closer than this,
            in km. Feik et al. use 100 km (not tuned; 100 km "delivered good
            results" in preliminary testing).
        embedding_dim: Width of the per-station embedding.
        hidden_channels_gnn: GATv2 hidden width *per head*.
        num_layers_gnn: Number of GATv2 blocks, i.e. message-passing hops.
        heads: Number of attention heads.
        hidden_channels_deepset: Hidden width of the Deep Set MLPs.
        member_chunk: If set, split the member axis into chunks of at most this
            size, run the stack on each and average the head outputs. Required
            at 51 members to bound memory (see the module docstring). Note the
            paper averages the final ``(mu, sigma)``, whereas this averages the
            raw head channels, so the sigma average differs through the
            method's softplus -- a second-order effect.
        spectrum_readout: How the ``"other"`` group is produced.

            ``"per_station"`` (default) applies one shared linear map at every
            station, so output index ``i`` is a function of station ``i``'s
            features. Correct when the group *is* a spatial field -- e.g.
            ``LorD``'s rank factor fields ``u_j(i)``.

            ``"global"`` emits the whole field from the station-pooled feature
            vector, giving every output index its own weight row and bias.
            Required when the group is **not** indexed by station:
            ``SCORE`` reads it as the spectrum ``s_k`` of the core over
            *graph Fourier modes*, and with a shared per-station map the only
            variation across ``k`` is ``w . f_k`` -- the features of the station
            that happens to sit at index ``k``. On a graph basis that
            correspondence is arbitrary (here the station axis is sorted by
            country), and since the bias is a single shared scalar there is no
            per-mode parameter at all, so a static low-pass spectrum cannot be
            represented even in principle; in practice the core then stays
            close to the identity and the implied station correlations are
            negligible.

            At ``rank > 0`` the ``"other"`` group mixes both kinds (a mode-indexed
            spectrum *and* station-indexed factor fields), so neither setting is
            right for every part of it; ``"global"`` is the better compromise
            there because the spectrum is the harder of the two to represent.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        meta_path: str,
        group_widths: Optional[dict] = None,
        other_zero_init: bool = True,
        max_dist: float = 100.0,
        embedding_dim: int = 20,
        hidden_channels_gnn: int = 265,
        num_layers_gnn: int = 2,
        heads: int = 8,
        hidden_channels_deepset: int = 265,
        deepset_layers_in: int = 2,
        deepset_layers_out: int = 2,
        member_chunk: Optional[int] = None,
        spectrum_readout: str = "per_station",
        noise: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.max_dist = float(max_dist)
        self.member_chunk = member_chunk

        # ---- static graph ---------------------------------------------------
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"meta_path {meta_path!r} is not a file. It should point at the "
                "meta.npz written by src/data/build_euppbench.py (e.g. "
                "data/euppbench_24h/meta.npz), not at the dataset directory."
            )
        with np.load(meta_path, allow_pickle=False) as f:
            dist = np.asarray(f["dist"], dtype=np.float64)
        n_stations = dist.shape[0]
        self.n_stations = n_stations

        src, dst = np.nonzero((dist > 0.0) & (dist <= self.max_dist))
        if len(src) == 0:
            raise ValueError(
                f"max_dist={max_dist} km yields no spatial edges for "
                f"{n_stations} stations (min nonzero distance "
                f"{dist[dist > 0].min():.1f} km)."
            )
        # Normalize to (0, 1] so the attribute scale is independent of max_dist.
        attr = dist[src, dst] / dist[src, dst].max()
        self.register_buffer(
            "spatial_edge_index",
            torch.from_numpy(np.stack([src, dst])).long(),
            persistent=False,
        )
        self.register_buffer(
            "spatial_edge_attr",
            torch.from_numpy(attr).float().unsqueeze(-1),
            persistent=False,
        )
        # Per-member-count cache of the expanded single-sample graph. Derived
        # entirely from the buffers above, so it is rebuilt rather than saved.
        self._graph_cache: Dict[Tuple[int, torch.device], tuple] = {}

        # ---- network --------------------------------------------------------
        self.encoder = EmbedStations(n_stations, embedding_dim)
        node_channels = in_channels + embedding_dim

        self.conv = ResGnn(
            in_channels=node_channels,
            hidden_channels=hidden_channels_gnn,
            out_channels=hidden_channels_gnn,
            num_layers=num_layers_gnn,
            heads=heads,
        )
        self.aggr = DeepSetAggregator(
            in_channels=hidden_channels_gnn,
            hidden_channels=hidden_channels_deepset,
            out_channels=hidden_channels_deepset,
            layers_in=deepset_layers_in,
            layers_out=deepset_layers_out,
        )

        # ---- head -----------------------------------------------------------
        # Mirrors the convention of the grid backbones: one projection for the
        # shared mu/sigma groups, an optional second for the "other" group, and
        # group-inner stacking so `process_output` un-stacks it correctly.
        if group_widths is None:
            group_widths = {"mu": out_channels, "sigma": None, "other": None}
        self._num_channels = group_widths["mu"]
        shared_w = group_widths["mu"] + (group_widths["sigma"] or 0)
        other_w = group_widths["other"]

        self.head = nn.Linear(hidden_channels_deepset, shared_w)
        # Start at the neutral prediction (mu = 0, diag = softplus(0)).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Optional latent-noise injection for the sample-based (SB) baseline.
        # It sits on the Deep Set output, i.e. *after* both the message
        # passing and the permutation-invariant collapse of the member axis, so
        # the graph -- by far the most expensive part, its edge count growing as
        # M_ensemble² -- runs once no matter how many samples are drawn, and the
        # latent enters a per-station feature vector that no longer carries any
        # input-member structure to interfere with.
        self.noise_injector = (
            GaussianNoiseInjector(
                feature_dim=hidden_channels_deepset, channel_axis=-1, **noise
            )
            if noise
            else None
        )

        self.has_other = other_w is not None
        if self.has_other:
            if spectrum_readout not in ("per_station", "global"):
                raise ValueError(
                    "spectrum_readout must be 'per_station' or 'global', got "
                    f"{spectrum_readout!r}"
                )
            self.spectrum_readout = spectrum_readout
            # "global" emits the whole `other` field from one pooled vector, so
            # each output index owns a weight row and a bias. "per_station"
            # applies one shared row at every station.
            out_w = n_stations * other_w if spectrum_readout == "global" else other_w
            self.other_proj = nn.Linear(hidden_channels_deepset, out_w)
            nn.init.zeros_(self.other_proj.bias)
            if other_zero_init:
                nn.init.zeros_(self.other_proj.weight)
            else:
                # A group entering the covariance quadratically has no gradient
                # at exactly zero, so break the symmetry with a small init that
                # still leaves the initial prediction effectively neutral.
                nn.init.normal_(self.other_proj.weight, std=1e-3)

    def _graph(self, n_members: int, device: torch.device) -> tuple:
        """Single-sample ``(edge_index, edge_attr, station_idx)`` for ``M`` members."""
        key = (n_members, device)
        if key in self._graph_cache:
            return self._graph_cache[key]

        n = self.n_stations
        m = n_members
        stations = torch.arange(n, device=device)

        # Spatial edges, replicated once per member layer (member-major, so the
        # attribute tiling below lines up).
        offsets = (torch.arange(m, device=device) * n).view(m, 1, 1)
        spatial = (self.spatial_edge_index.to(device).unsqueeze(0) + offsets)
        spatial = spatial.permute(1, 0, 2).reshape(2, -1)
        spatial_attr = self.spatial_edge_attr.to(device).repeat(m, 1)

        # Inter-member edges: every ordered pair of distinct members, per station.
        m1, m2 = torch.meshgrid(
            torch.arange(m, device=device), torch.arange(m, device=device), indexing="ij"
        )
        off_diag = m1 != m2
        inter_src = (m1[off_diag].unsqueeze(-1) * n + stations).reshape(-1)
        inter_dst = (m2[off_diag].unsqueeze(-1) * n + stations).reshape(-1)
        inter = torch.stack([inter_src, inter_dst])
        inter_attr = torch.full(
            (inter.shape[1], 1), INTER_MEMBER_ATTR, device=device, dtype=spatial_attr.dtype
        )

        edge_index = torch.cat([spatial, inter], dim=1)
        edge_attr = torch.cat([spatial_attr, inter_attr], dim=0)
        # Node j = member * N + station, so the station index cycles.
        station_idx = stations.repeat(m)

        self._graph_cache[key] = (edge_index, edge_attr, station_idx)
        return self._graph_cache[key]

    def _forward_members(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full stack on ``x: (B, F, M, N)``, returning ``(B, N, C)``."""
        b, _, m, n = x.shape
        edge_index, edge_attr, station_idx = self._graph(m, x.device)

        # (B, F, M, N) -> (B*M*N, F), node j = member * N + station.
        nodes = x.permute(0, 2, 3, 1).reshape(b * m * n, -1)
        nodes = self.encoder(nodes, station_idx.repeat(b))

        # Replicate the single-sample graph across the batch by node offset.
        batch_offsets = (torch.arange(b, device=x.device) * (m * n)).view(b, 1, 1)
        batch_edge_index = (edge_index.unsqueeze(0) + batch_offsets).permute(1, 0, 2).reshape(2, -1)
        batch_edge_attr = edge_attr.repeat(b, 1)

        nodes = self.conv(nodes, batch_edge_index, batch_edge_attr)
        return self.aggr(nodes.view(b, m, n, -1))

    def forward(self, x: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected x of shape (B, F, M, N), got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.n_stations:
            raise ValueError(
                f"x has {x.shape[-1]} stations but the graph was built for "
                f"{self.n_stations}."
            )

        n_members = x.shape[2]
        if self.member_chunk is None or n_members <= self.member_chunk:
            features = self._forward_members(x)
        else:
            # `tensor_split` (not `chunk`) so the groups come out as even as
            # possible: at 51 members with member_chunk=11 this is
            # [11, 10, 10, 10, 10], exactly Feik et al.'s "4 groups of 10 and a
            # remaining group of 11". `chunk` would instead leave a short tail
            # ([11, 11, 11, 11, 7]), whose aggregate is noisier than the rest.
            n_chunks = -(-n_members // self.member_chunk)
            chunks = torch.tensor_split(x, n_chunks, dim=2)
            features = torch.stack([self._forward_members(c) for c in chunks]).mean(dim=0)

        if self.noise_injector is not None:
            # (B, N, C) -> (B*M, N, C): the head below is replayed once per
            # member, and the members are folded back into the channel axis.
            n_samples = require_n_samples(n_samples, type(self).__name__)
            features = self.noise_injector(features, n_samples)

        b, n, _ = features.shape
        # Group-inner stacking: (B, N, group_w) -> (B, N, C, groups) -> (B, C*G, N),
        # the exact inverse of each method's `process_output` reshape.
        groups = [self.head(features).reshape(b, n, self._num_channels, -1)]
        if self.has_other:
            # "global" reads the whole field off a station-pooled vector; the
            # mean is the same permutation-invariant pooling the Deep Set uses.
            other_in = features.mean(dim=1) if self.spectrum_readout == "global" else features
            groups.append(self.other_proj(other_in).reshape(b, n, self._num_channels, -1))
        out = torch.cat(groups, dim=-1)  # (B, N, C, G)
        out = out.permute(0, 2, 3, 1).reshape(b, self._num_channels * out.shape[-1], n)
        if self.noise_injector is None:
            return out
        # One group per member, so C*G == C here; fold the member axis back into
        # the channel axis group-inner, giving (B, C*M, N).
        return stack_member_groups(out, n_samples, self._num_channels, (n,))

    def freeze_for_stage2(self):
        """Freeze everything but the ``other`` head, for stage 2 of the two-stage fit."""
        if not self.has_other:
            raise ValueError(
                "freeze_for_stage2() requires an 'other' head, but this backbone "
                "was built without one (group_widths['other'] is None)."
            )
        for module in (self.encoder, self.conv, self.aggr, self.head):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        for p in self.other_proj.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen trunk in eval mode once stage 2 froze it."""
        super().train(mode)
        if mode and self.has_other:
            if all(not p.requires_grad for p in self.head.parameters()):
                for module in (self.encoder, self.conv, self.aggr, self.head):
                    module.eval()
        return self
