# SCVMiner

SCVMiner is a GNN-based smart contract vulnerability detection framework.

## Framework

It converts Solidity source code into contract graphs, vectorizes graph nodes from their source text, and uses a Local-Global Attention Network (LGAN) to learn graph-level vulnerability representations.

## How to use

### 1. Environment

This project is managed with `uv` and requires Python 3.12 or later.

```bash
uv sync
```

### 2. Smart Contract Graphs

Datasets are organized by vulnerability type and label. For example, the default
reentrancy dataset uses:

```text
datasets/reentrancy/dependency/    # vulnerable samples, label 1
datasets/reentrancy/undependency/  # non-vulnerable samples, label 0
```

Run batch graph generation:

```bash
uv run python scg/generate_dataset_graphs.py
```

The default command processes `datasets/reentrancy/` and writes:

```text
graphs/reentrancy_graphs.jsonl
graphs/reentrancy_node_embeddings.jsonl
```

To process another dataset type:

```bash
uv run python scg/generate_dataset_graphs.py \
  --dataset-type timestamp \
  --dataset-dir datasets/timestamp \
  --output-dir graphs
```

Use `--fail-fast` if you want the batch process to stop at the first graph
generation error.

You can also generate a single graph directly in Python:

```python
from scg.scg import sc_graph_generator

file_graph = sc_graph_generator("path/to/Contract.sol")
contract_graph = sc_graph_generator("path/to/Contract.sol", contract_name="Token")
```

### 3. Train and validate the LGAN

After graph files are generated, train and validate LGAN with k-fold cross
validation:

```bash
uv run python main.py --data-type reentrancy
```

Common options:

```bash
uv run python main.py \
  --data-type reentrancy \
  --graphs-dir graphs \
  --epochs 200 \
  --batch-size 16 \
  --device cpu
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
