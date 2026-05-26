"""Generate SCGraphs for a labeled Solidity dataset.

By default, this script processes the reentrancy dataset and creates one graph
per Solidity file. Files under ``dependency`` are labeled ``1`` and files under
``undependency`` are labeled ``0``.

Outputs:
    graphs/<dataset_type>_graphs.jsonl
        Graph metadata and structure: file_path, label, num_nodes, num_edges,
        edges.
    graphs/<dataset_type>_node_embeddings.jsonl
        Node embeddings keyed by file_path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scg.scg import sc_graph_generator


DEFAULT_DATASET_TYPE = "reentrancy"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / DEFAULT_DATASET_TYPE
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "graphs"
LABEL_BY_SUBDIR = {
    "dependency": 1,
    "undependency": 0,
}


def iter_labeled_solidity_files(dataset_dir: Path) -> Iterable[tuple[Path, int]]:
    """Yield Solidity files from known label directories in stable order."""

    for subdir, label in LABEL_BY_SUBDIR.items():
        label_dir = dataset_dir / subdir
        if not label_dir.exists():
            raise FileNotFoundError(f"Dataset label directory does not exist: {label_dir}")
        yield from ((sol_file, label) for sol_file in sorted(label_dir.glob("*.sol")))


def generate_graph_files(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataset_type: str = DEFAULT_DATASET_TYPE,
    continue_on_error: bool = True,
) -> tuple[Path, Path]:
    """Generate graph JSONL files for a labeled Solidity dataset."""

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    graph_info_path = output_dir / f"{dataset_type}_graphs.jsonl"
    node_embeddings_path = output_dir / f"{dataset_type}_node_embeddings.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_files = list(iter_labeled_solidity_files(dataset_dir))
    total_count = len(labeled_files)
    success_count = 0
    failures: list[tuple[str, str]] = []

    with (
        graph_info_path.open("w", encoding="utf-8") as graph_info_file,
        node_embeddings_path.open("w", encoding="utf-8") as node_embeddings_file,
    ):
        for cur_count, (sol_file, label) in enumerate(labeled_files, start=1):
            file_path = _portable_file_path(sol_file)
            progress = f"[{cur_count}/{total_count}]"

            try:
                graph = sc_graph_generator(sol_file)
            except Exception as exc:
                if not continue_on_error:
                    raise
                failures.append((file_path, str(exc)))
                print(f"{progress} [failed] {file_path}: {exc}", file=sys.stderr)
                continue

            graph_record = {
                "file_path": file_path,
                "label": label,
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "edges": graph.edges,
            }
            node_embeddings_record = {
                "file_path": file_path,
                "node_embeddings": graph.node_embeddings,
            }

            graph_info_file.write(json.dumps(graph_record, separators=(",", ":")) + "\n")
            node_embeddings_file.write(json.dumps(node_embeddings_record, separators=(",", ":")) + "\n")
            success_count += 1
            print(f"{progress} [ok] {file_path}: nodes={graph.num_nodes}, edges={graph.num_edges}, label={label}")

    print(
        f"Generated {success_count}/{total_count} graphs. "
        f"graph_info={graph_info_path.relative_to(PROJECT_ROOT)}, "
        f"node_embeddings={node_embeddings_path.relative_to(PROJECT_ROOT)}"
    )
    if failures:
        print(f"Skipped {len(failures)} files due to errors.", file=sys.stderr)

    return graph_info_path, node_embeddings_path


def _portable_file_path(path: Path) -> str:
    """Return a stable project-relative path when possible."""

    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graphs for a labeled Solidity dataset.")
    parser.add_argument("--dataset-type", default=DEFAULT_DATASET_TYPE)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first graph generation error instead of skipping failed files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_graph_files(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        continue_on_error=not args.fail_fast,
    )


if __name__ == "__main__":
    main()
