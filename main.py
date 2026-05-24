from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence

import dgl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from gnn.lgan import LGAN


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GRAPHS_DIR = PROJECT_ROOT / "graphs"


@dataclass(frozen=True)
class GraphSample:
    file_path: str
    label: int
    graph: dgl.DGLGraph
    node_features: Tensor


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class SCGraphDataset(Dataset[GraphSample]):
    def __init__(self, samples: Sequence[GraphSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    samples = load_graph_samples(args.data_type, args.graphs_dir)
    folds = make_stratified_folds([sample.label for sample in samples], args.folds, args.seed)

    print(f"data_type = {args.data_type}")
    print(f"samples = {len(samples)}")
    print(f"device = {device}")
    print(
        "hyperparameters = "
        f"in_feats={args.in_feats}, hidden_feats={args.hidden_feats}, k_hop={args.k_hop}, "
        f"num_layers={args.num_layers}, dropout={args.dropout}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, batch_size={args.batch_size}, epochs={args.epochs}"
    )
    print()

    fold_metrics: list[Metrics] = []
    all_indices = list(range(len(samples)))

    for fold_index, val_indices in enumerate(folds, start=1):
        val_index_set = set(val_indices)
        train_indices = [index for index in all_indices if index not in val_index_set]

        train_loader = make_data_loader(samples, train_indices, args.batch_size, shuffle=True)
        val_loader = make_data_loader(samples, val_indices, args.batch_size, shuffle=False)

        model = LGAN(
            in_feats=args.in_feats,
            hidden_feats=args.hidden_feats,
            num_classes=args.num_classes,
            k_hop=args.k_hop,
            num_layers=args.num_layers,
            dropout=args.dropout,
            negative_slope=args.negative_slope,
            bidirectional_subgraph=not args.directed_subgraph,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        print(f"== Fold {fold_index}/{args.folds} ==")
        print(f"train_samples = {len(train_indices)}, val_samples = {len(val_indices)}")
        print(f"model_parameters = {count_parameters(model):,}")

        best_val_f1 = -1.0
        best_metrics: Metrics | None = None

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            if val_metrics.f1 > best_val_f1:
                best_val_f1 = val_metrics.f1
                best_metrics = val_metrics

            print(
                f"epoch {epoch:03d} | "
                f"train loss={train_metrics.loss:.4f} acc={train_metrics.accuracy:.4f} "
                f"f1={train_metrics.f1:.4f} | "
                f"val loss={val_metrics.loss:.4f} acc={val_metrics.accuracy:.4f} "
                f"precision={val_metrics.precision:.4f} recall={val_metrics.recall:.4f} "
                f"f1={val_metrics.f1:.4f}"
            )

        assert best_metrics is not None
        fold_metrics.append(best_metrics)
        print(
            f"best fold {fold_index} | "
            f"loss={best_metrics.loss:.4f} acc={best_metrics.accuracy:.4f} "
            f"precision={best_metrics.precision:.4f} recall={best_metrics.recall:.4f} "
            f"f1={best_metrics.f1:.4f}"
        )
        print()

    mean_metrics = average_metrics(fold_metrics)
    print(f"== {args.folds}-Fold Cross Validation Summary ==")
    print(
        f"loss={mean_metrics.loss:.4f}, accuracy={mean_metrics.accuracy:.4f}, "
        f"precision={mean_metrics.precision:.4f}, recall={mean_metrics.recall:.4f}, "
        f"f1={mean_metrics.f1:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LGAN with k-fold cross validation.")
    parser.add_argument("--data-type", default="reentrancy", help="Dataset prefix under graphs/.")
    parser.add_argument("--graphs-dir", type=Path, default=DEFAULT_GRAPHS_DIR)
    parser.add_argument("--in-feats", type=int, default=256)
    parser.add_argument("--hidden-feats", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--k-hop", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--negative-slope", type=float, default=0.2)
    parser.add_argument("--directed-subgraph", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_graph_samples(data_type: str, graphs_dir: Path) -> list[GraphSample]:
    graph_info_path = graphs_dir / f"{data_type}_graphs.jsonl"
    node_embeddings_path = graphs_dir / f"{data_type}_node_embeddings.jsonl"
    if not graph_info_path.exists():
        raise FileNotFoundError(f"Graph metadata file does not exist: {graph_info_path}")
    if not node_embeddings_path.exists():
        raise FileNotFoundError(f"Node embeddings file does not exist: {node_embeddings_path}")

    embedding_by_file_path: dict[str, list[list[float]]] = {}
    with node_embeddings_path.open("r", encoding="utf-8") as embeddings_file:
        for line in embeddings_file:
            record = json.loads(line)
            embedding_by_file_path[record["file_path"]] = record["node_embeddings"]

    samples: list[GraphSample] = []
    with graph_info_path.open("r", encoding="utf-8") as graph_info_file:
        for line in graph_info_file:
            record = json.loads(line)
            file_path = record["file_path"]
            node_embeddings = embedding_by_file_path[file_path]
            graph, node_features = graph_record_to_dgl(
                num_nodes=record["num_nodes"],
                edges=record["edges"],
                node_embeddings=node_embeddings,
            )
            samples.append(
                GraphSample(
                    file_path=file_path,
                    label=int(record["label"]),
                    graph=graph,
                    node_features=node_features,
                )
            )

    return samples


def graph_record_to_dgl(
    num_nodes: int,
    edges: Sequence[Sequence[int]],
    node_embeddings: Sequence[Sequence[float]],
) -> tuple[dgl.DGLGraph, Tensor]:
    if len(node_embeddings) != num_nodes:
        raise ValueError("node_embeddings length must match num_nodes")

    if edges:
        src_ids, dst_ids = zip(*edges)
    else:
        src_ids, dst_ids = [], []

    graph = dgl.graph((list(src_ids), list(dst_ids)), num_nodes=num_nodes)
    node_features = torch.as_tensor(node_embeddings, dtype=torch.float32)
    return graph, node_features


def make_data_loader(
    samples: Sequence[GraphSample],
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[GraphSample]:
    subset = SCGraphDataset([samples[index] for index in indices])
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_samples)


def collate_graph_samples(batch: Sequence[GraphSample]) -> tuple[dgl.DGLGraph, Tensor, Tensor]:
    graphs = [sample.graph for sample in batch]
    node_features = torch.cat([sample.node_features for sample in batch], dim=0)
    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)
    return dgl.batch(graphs), node_features, labels


def train_one_epoch(
    model: LGAN,
    data_loader: DataLoader[GraphSample],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_samples = 0
    predictions: list[int] = []
    targets: list[int] = []

    for graph, node_features, labels in data_loader:
        graph = graph.to(device)
        node_features = node_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(graph, node_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return build_metrics(total_loss, total_samples, predictions, targets)


@torch.no_grad()
def evaluate(
    model: LGAN,
    data_loader: DataLoader[GraphSample],
    criterion: nn.Module,
    device: torch.device,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions: list[int] = []
    targets: list[int] = []

    for graph, node_features, labels in data_loader:
        graph = graph.to(device)
        node_features = node_features.to(device)
        labels = labels.to(device)

        logits = model(graph, node_features)
        loss = criterion(logits, labels)

        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    return build_metrics(total_loss, total_samples, predictions, targets)


def build_metrics(total_loss: float, total_samples: int, predictions: Sequence[int], targets: Sequence[int]) -> Metrics:
    if total_samples == 0:
        return Metrics(loss=0.0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0)

    true_positive = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    false_positive = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    false_negative = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return Metrics(
        loss=total_loss / total_samples,
        accuracy=correct / total_samples,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def make_stratified_folds(labels: Sequence[int], fold_count: int, seed: int) -> list[list[int]]:
    if fold_count < 2:
        raise ValueError("folds must be at least 2")

    indices_by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(index)

    random_generator = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for label_indices in indices_by_label.values():
        random_generator.shuffle(label_indices)
        for offset, sample_index in enumerate(label_indices):
            folds[offset % fold_count].append(sample_index)

    for fold in folds:
        random_generator.shuffle(fold)

    return folds


def average_metrics(metrics: Sequence[Metrics]) -> Metrics:
    count = len(metrics)
    return Metrics(
        loss=sum(metric.loss for metric in metrics) / count,
        accuracy=sum(metric.accuracy for metric in metrics) / count,
        precision=sum(metric.precision for metric in metrics) / count,
        recall=sum(metric.recall for metric in metrics) / count,
        f1=sum(metric.f1 for metric in metrics) / count,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
