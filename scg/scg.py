"""Smart contract graph generator with node embeddings.

Public interface:
    sc_graph_generator(sol_path, contract_name=None) -> SCGraph

When ``contract_name`` is provided, one contract is converted to one graph.
When ``contract_name`` is omitted, the whole Solidity source file is converted
to one graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scg.gsc import Graph, get_contract_graph, get_solidity_graph
from scg.nre_word2vec import vectorize_code


@dataclass
class SCGraph:
    """Graph data prepared for smart contract vulnerability detection."""

    num_nodes: int
    num_edges: int
    edges: list[tuple[int, int]]
    node_embeddings: list[list[float]]


def sc_graph_generator(sol_path: str | Path, contract_name: str | None = None) -> SCGraph:
    """Generate an ``SCGraph`` for a contract or an entire Solidity file.

    Args:
        sol_path: Path to a Solidity source file.
        contract_name: Optional contract name. If provided, the returned graph
            represents that contract. If omitted, the returned graph represents
            the whole Solidity source file.

    Returns:
        ``SCGraph`` with edge counts and one Word2Vec node embedding per graph
        node.
    """

    graph = _load_graph(sol_path, contract_name)
    node_embeddings = [vectorize_code(node.text) for node in graph.nodes]
    return SCGraph(
        num_nodes=len(graph.nodes),
        num_edges=len(graph.edges),
        edges=graph.edges,
        node_embeddings=node_embeddings,
    )


def _load_graph(sol_path: str | Path, contract_name: str | None) -> Graph:
    if contract_name is None:
        return get_solidity_graph(sol_path)
    return get_contract_graph(sol_path, sol_version=None, contract_name=contract_name)


__all__ = ["SCGraph", "sc_graph_generator"]
