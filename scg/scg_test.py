from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scg.scg import SCGraph, sc_graph_generator


CONTRACT_DIR = PROJECT_ROOT / "test_contracts"


def main() -> None:
    contract_graph = sc_graph_generator(CONTRACT_DIR / "SimpleCall.sol", "SimpleCall")
    solidity_graph = sc_graph_generator(CONTRACT_DIR / "SimpleCall.sol")

    assert isinstance(contract_graph, SCGraph)
    assert isinstance(solidity_graph, SCGraph)
    assert contract_graph.num_nodes > 0
    assert solidity_graph.num_nodes >= contract_graph.num_nodes
    assert contract_graph.num_edges == len(contract_graph.edges)
    assert solidity_graph.num_edges == len(solidity_graph.edges)
    assert len(contract_graph.node_embeddings) == contract_graph.num_nodes
    assert len(solidity_graph.node_embeddings) == solidity_graph.num_nodes

    print(f"contract_graph: nodes={contract_graph.num_nodes}, edges={contract_graph.num_edges}")
    print(f"solidity_graph: nodes={solidity_graph.num_nodes}, edges={solidity_graph.num_edges}")


if __name__ == "__main__":
    main()
