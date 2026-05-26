from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scg.gsc import DEFAULT_SOL_VERSION, Graph, get_contract_graphs, get_solidity_graph
from scg.gsc import _extract_sol_versions_from_file


CONTRACT_DIR = PROJECT_ROOT / "test_contracts"


def main() -> None:
    validate_sol_version_resolution()

    print(f"DEFAULT_SOL_VERSION = {DEFAULT_SOL_VERSION}")
    print()

    for sol_file in sorted(CONTRACT_DIR.glob("*.sol")):
        print(f"== {sol_file.name} ==")
        graphs = get_contract_graphs(sol_file)

        if not graphs:
            print("No contract graph generated.")
            print()
            continue

        for graph in graphs:
            print_graph("Contract", graph)

        print_graph("Solidity", get_solidity_graph(sol_file, None))


def validate_sol_version_resolution() -> None:
    cases = {
        "pragma solidity >=0.4.21 <=0.7.12;": [
            "0.7.12",
            "0.7.6",
            "0.6.12",
            "0.5.17",
            "0.4.26",
            "0.4.21",
        ],
        "pragma solidity 0.4.21<= x <=0.7.12;": [
            "0.7.12",
            "0.7.6",
            "0.6.12",
            "0.5.17",
            "0.4.26",
            "0.4.21",
        ],
        "pragma solidity >=0.4.21 <=0.7.0;": [
            "0.7.0",
            "0.6.12",
            "0.5.17",
            "0.4.26",
            "0.4.21",
        ],
        "pragma solidity >0.5.0;": [
            "0.5.17",
            "0.6.12",
            "0.7.6",
        ],
        "pragma solidity >=0.5.0;": [
            "0.5.0",
            "0.5.17",
            "0.6.12",
            "0.7.6",
        ],
    }

    with TemporaryDirectory() as tmp_dir:
        sol_file = Path(tmp_dir) / "Range.sol"
        for pragma, expected_versions in cases.items():
            sol_file.write_text(f"{pragma}\ncontract C {{}}\n", encoding="utf-8")
            actual_versions = _extract_sol_versions_from_file(sol_file)
            assert actual_versions == expected_versions, actual_versions


def print_graph(label: str, graph: Graph) -> None:
    print(f"{label}: {graph.contract_name}")
    print("Nodes:")
    for node in graph.nodes:
        first_line = node.text.strip().splitlines()[0] if node.text.strip() else ""
        print(f"  [{node.id}] {node.kind}: {node.name} | {first_line}")

    print("Edges:")
    if graph.edges:
        for src_id, dst_id in graph.edges:
            src = graph.nodes[src_id].name
            dst = graph.nodes[dst_id].name
            print(f"  {src_id}->{dst_id}: {src} -> {dst}")
    else:
        print("  <none>")

    print()


if __name__ == "__main__":
    main()
