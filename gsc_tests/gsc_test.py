from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsc import DEFAULT_SOL_VERSION, solidity_to_graph


CONTRACT_DIR = Path(__file__).resolve().parent / "contracts"


def main() -> None:
    print(f"DEFAULT_SOL_VERSION = {DEFAULT_SOL_VERSION}")
    print()

    for sol_file in sorted(CONTRACT_DIR.glob("*.sol")):
        print(f"== {sol_file.name} ==")
        graphs = solidity_to_graph(sol_file)

        if not graphs:
            print("No contract graph generated.")
            print()
            continue

        for graph in graphs:
            print(f"Contract: {graph.contract_name}")
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
