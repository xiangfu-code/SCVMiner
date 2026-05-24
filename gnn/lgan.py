"""Local-Global Attention Network for smart-contract graph classification.

The implementation follows the LGAN description:

1. Local feature aggregation with attention over each node's neighborhood.
2. Global information interaction with graph-wise self-attention.
3. Sum graph pooling followed by a linear classifier.

Default hyperparameters used here:

- ``in_feats=256``: node embedding dimension from ``scg``.
- ``hidden_feats=128``: hidden node representation dimension.
- ``num_classes=2``: binary vulnerability classification.
- ``k_hop=2``: each local attention layer attends over explicit 2-hop subgraphs.
- ``num_layers=1``: one local-global attention block.
- ``dropout=0.2``: dropout on attention weights and global updates.
- ``negative_slope=0.2``: LeakyReLU slope for local attention scores.
- ``bidirectional_subgraph=True``: use weakly connected k-hop neighborhoods.

The model expects a DGL graph and a node feature tensor.  Batched DGL graphs are
supported; global attention is applied independently inside each graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import dgl
from dgl.nn.functional import edge_softmax
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LGANConfig:
    """Configuration for ``LGAN``.

    Args:
        in_feats: Input node feature size.
        hidden_feats: Hidden node representation size.
        num_classes: Number of output classes.
        k_hop: Explicit local subgraph radius used by local attention.
        num_layers: Number of local/global LGAN blocks.
        dropout: Dropout probability used after local and global updates.
        negative_slope: LeakyReLU slope for local attention scores.
        bidirectional_subgraph: If true, k-hop neighborhoods ignore edge
            direction, matching the paper's distance-based local subgraph view.
    """

    in_feats: int = 256
    hidden_feats: int = 128
    num_classes: int = 2
    k_hop: int = 2
    num_layers: int = 1
    dropout: float = 0.2
    negative_slope: float = 0.2
    bidirectional_subgraph: bool = True


class LocalAttentionLayer(nn.Module):
    """Local attention aggregation corresponding to formulas 3-2 to 3-4."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        k_hop: int = 2,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bidirectional_subgraph: bool = True,
    ) -> None:
        super().__init__()
        if k_hop < 1:
            raise ValueError("k_hop must be at least 1")

        self.k_hop = k_hop
        self.bidirectional_subgraph = bidirectional_subgraph
        self.feature_linear = nn.Linear(in_feats, out_feats, bias=False)
        self.attention_linear = nn.Linear(2 * in_feats, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph: dgl.DGLGraph, node_feats: Tensor) -> Tensor:
        local_graph = self._build_k_hop_attention_graph(graph, node_feats.device)

        with local_graph.local_scope():
            local_graph.ndata["h"] = node_feats

            def edge_attention(edges: Any) -> dict[str, Tensor]:
                pair_feats = torch.cat([edges.dst["h"], edges.src["h"]], dim=-1)
                return {"e": self.leaky_relu(self.attention_linear(pair_feats))}

            def message_func(edges: Any) -> dict[str, Tensor]:
                return {"m": edges.src["z"] * edges.data["alpha"]}

            def reduce_func(nodes: Any) -> dict[str, Tensor]:
                return {"h_next": torch.sum(nodes.mailbox["m"], dim=1)}

            local_graph.apply_edges(edge_attention)
            local_graph.edata["alpha"] = edge_softmax(local_graph, local_graph.edata["e"])
            local_graph.edata["alpha"] = self.dropout(local_graph.edata["alpha"])
            local_graph.ndata["z"] = self.feature_linear(node_feats)

            local_graph.update_all(message_func, reduce_func)
            h_next = cast(Tensor, local_graph.ndata["h_next"])
            return F.relu(h_next)

    def _build_k_hop_attention_graph(self, graph: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
        if graph.batch_size > 1:
            return dgl.batch(
                [self._build_single_k_hop_attention_graph(part, device) for part in dgl.unbatch(graph)]
            )
        return self._build_single_k_hop_attention_graph(graph, device)

    def _build_single_k_hop_attention_graph(self, graph: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
        with graph.local_scope():
            local_graph = graph.cpu()
            if self.bidirectional_subgraph:
                local_graph = dgl.add_reverse_edges(local_graph)
            local_graph = dgl.add_self_loop(dgl.remove_self_loop(local_graph))

            adjacency = local_graph.adj_external(scipy_fmt="coo")
            reachability = adjacency.copy()
            power = adjacency.copy()
            for _ in range(2, self.k_hop + 1):
                power = power @ adjacency
                reachability = reachability + power

            reachability.data[:] = 1.0
            reachability.eliminate_zeros()
            reachability = reachability.tocoo()
            src_ids = torch.as_tensor(reachability.row, dtype=torch.int64, device=device)
            dst_ids = torch.as_tensor(reachability.col, dtype=torch.int64, device=device)

        return dgl.graph((src_ids, dst_ids), num_nodes=graph.num_nodes(), device=device)


class GlobalSelfAttention(nn.Module):
    """Graph-wise full self-attention corresponding to formulas 3-5 and 3-6."""

    def __init__(self, hidden_feats: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_feats, hidden_feats, bias=False)
        self.key = nn.Linear(hidden_feats, hidden_feats, bias=False)
        self.value = nn.Linear(hidden_feats, hidden_feats, bias=False)
        self.output = nn.Linear(hidden_feats, hidden_feats)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_feats**0.5

    def forward(self, graph: dgl.DGLGraph, node_feats: Tensor) -> Tensor:
        graph_node_counts = graph.batch_num_nodes().tolist()
        graph_outputs: list[Tensor] = []

        for graph_feats in torch.split(node_feats, graph_node_counts, dim=0):
            if graph_feats.numel() == 0:
                graph_outputs.append(graph_feats)
                continue

            queries = self.query(graph_feats)
            keys = self.key(graph_feats)
            values = self.value(graph_feats)
            scores = queries @ keys.transpose(0, 1) / self.scale
            attention = self.dropout(torch.softmax(scores, dim=-1))
            graph_outputs.append(self.output(attention @ values))

        return torch.cat(graph_outputs, dim=0)


class LGANBlock(nn.Module):
    """One local-global attention block."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        k_hop: int = 2,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bidirectional_subgraph: bool = True,
    ) -> None:
        super().__init__()
        self.local_attention = LocalAttentionLayer(
            in_feats,
            hidden_feats,
            k_hop,
            dropout,
            negative_slope,
            bidirectional_subgraph,
        )
        self.global_attention = GlobalSelfAttention(hidden_feats, dropout)
        self.norm = nn.LayerNorm(hidden_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, node_feats: Tensor) -> Tensor:
        local_feats = self.local_attention(graph, node_feats)
        global_feats = self.global_attention(graph, local_feats)
        return self.norm(local_feats + self.dropout(global_feats))


class LGAN(nn.Module):
    """Local-Global Attention Network classifier.

    Forward inputs:
        graph: A DGL graph or batched DGL graph.
        node_feats: Tensor of shape ``(num_nodes, in_feats)``.

    Returns:
        Logits of shape ``(batch_size, num_classes)``.
    """

    def __init__(
        self,
        in_feats: int = 256,
        hidden_feats: int = 128,
        num_classes: int = 2,
        k_hop: int = 2,
        num_layers: int = 1,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
        bidirectional_subgraph: bool = True,
    ) -> None:
        super().__init__()
        if k_hop < 1:
            raise ValueError("k_hop must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.config = LGANConfig(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            num_classes=num_classes,
            k_hop=k_hop,
            num_layers=num_layers,
            dropout=dropout,
            negative_slope=negative_slope,
            bidirectional_subgraph=bidirectional_subgraph,
        )
        layer_dims = [in_feats, *([hidden_feats] * num_layers)]
        self.layers = nn.ModuleList(
            LGANBlock(
                layer_dims[index],
                hidden_feats,
                k_hop,
                dropout,
                negative_slope,
                bidirectional_subgraph,
            )
            for index in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_feats, num_classes)

    def forward(self, graph: dgl.DGLGraph, node_feats: Tensor) -> Tensor:
        graph = graph.to(node_feats.device)
        hidden = node_feats.float()

        for layer in self.layers:
            hidden = layer(graph, hidden)

        with graph.local_scope():
            graph.ndata["h"] = hidden
            graph_repr = dgl.sum_nodes(graph, "h")

        return self.classifier(graph_repr)


__all__ = [
    "GlobalSelfAttention",
    "LGAN",
    "LGANBlock",
    "LGANConfig",
    "LocalAttentionLayer",
]
