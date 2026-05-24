from __future__ import annotations

from pathlib import Path
import sys
from collections.abc import Sequence

import dgl
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn.lgan import LGAN
from scg.scg import sc_graph_generator


def main() -> None:
    sc_graph = sc_graph_generator(PROJECT_ROOT / "test_contracts" / "SimpleCall.sol")
    graph, node_features = scgraph_to_dgl(sc_graph.num_nodes, sc_graph.edges, sc_graph.node_embeddings)
    batched_graph = dgl.batch([graph, graph])
    batched_features = torch.cat([node_features, node_features], dim=0)

    model = LGAN()
    model.eval()

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)

    with torch.no_grad():
        single_logits = model(graph, node_features)
        batch_logits = model(batched_graph, batched_features)

    assert single_logits.shape == (1, model.config.num_classes)
    assert batch_logits.shape == (2, model.config.num_classes)

    print(f"model_config = {model.config}")
    print(f"total_parameters = {total_params:,}")
    print(f"trainable_parameters = {trainable_params:,}")
    print(f"single_graph_logits_shape = {tuple(single_logits.shape)}")
    print(f"batched_graph_logits_shape = {tuple(batch_logits.shape)}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def scgraph_to_dgl(
    num_nodes: int,
    edges: Sequence[tuple[int, int]] | Sequence[list[int]],
    node_embeddings: Sequence[Sequence[float]] | Tensor,
) -> tuple[dgl.DGLGraph, Tensor]:
    if len(node_embeddings) != num_nodes:
        raise ValueError("node_embeddings length must match num_nodes")

    if edges:
        src_ids, dst_ids = zip(*edges)
    else:
        src_ids, dst_ids = [], []

    graph = dgl.graph((list(src_ids), list(dst_ids)), num_nodes=num_nodes)
    features = torch.as_tensor(node_embeddings, dtype=torch.float32)
    return graph, features


if __name__ == "__main__":
    main()
