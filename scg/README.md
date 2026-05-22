# SC Graph Utilities

This directory contains the graph construction, node embedding, and dataset
batch-generation utilities for SCVMiner.

## High-Level Interface

Use `scg.py` when you need a model-ready graph with node embeddings:

```python
from scg.scg import SCGraph, sc_graph_generator

graph = sc_graph_generator("path/to/Contract.sol")
contract_graph = sc_graph_generator("path/to/Contract.sol", contract_name="Token")
```

If `contract_name` is omitted, one Solidity file becomes one `SCGraph`. If
`contract_name` is provided, only that contract becomes one `SCGraph`.

`SCGraph` fields:

- `num_nodes`: number of graph nodes.
- `num_edges`: number of directed graph edges.
- `edges`: directed edges as `(src_id, dst_id)`.
- `node_embeddings`: one feature vector per node, in node id order.

## Graph Construction

Lower-level graph construction lives in `gsc.py`.

```python
from scg.gsc import get_contract_graph, get_contract_graphs, get_solidity_graph

graphs = get_contract_graphs("path/to/Contract.sol")
contract_graph = get_contract_graph("path/to/Contract.sol", None, "Token")
solidity_graph = get_solidity_graph("path/to/Contract.sol")
```

Nodes are functions, modifiers, and state variables. Edge directions are:

- Function or modifier calls: `caller -> callee`
- State variable writes: `function -> state_variable`
- State variable reads: `state_variable -> function`

`DEFAULT_SOL_VERSION` is used when a file has no concrete Solidity pragma.
Before Slither runs, `gsc.py` checks whether the selected solc version is
installed and installs it with `uv run solc-select install <version>` if needed.

## Node Embeddings

Node feature extraction lives in `nre_word2vec.py`.

```python
from scg.nre_word2vec import vectorize_code

embedding = vectorize_code(node.text)
```

`vectorize_code` tokenizes Solidity code structurally, loads
`models/solidity_word2vec.kv`, looks up known token vectors, and mean-pools them
into one node embedding. Unknown tokens are ignored. If no known tokens are
found, a zero vector is returned.

To train or refresh the Word2Vec model:

```bash
uv run python scg/nre_word2vec.py
```

## Dataset Batch Generation

Use `generate_dataset_graphs.py` to convert a labeled Solidity dataset into
JSONL graph files.

```bash
uv run python scg/generate_dataset_graphs.py
```

Default behavior:

- Dataset: `datasets/reentrancy/`
- Label `1`: Solidity files under `datasets/reentrancy/dependency/`
- Label `0`: Solidity files under `datasets/reentrancy/undependency/`
- Graph metadata output: `graphs/reentrancy_graphs.jsonl`
- Node embedding output: `graphs/reentrancy_node_embeddings.jsonl`

`graphs/<dataset_type>_graphs.jsonl` stores one JSON object per Solidity file:

```json
{"file_path":"datasets/reentrancy/dependency/example.sol","label":1,"num_nodes":10,"num_edges":12,"edges":[[0,1]]}
```

`graphs/<dataset_type>_node_embeddings.jsonl` stores embeddings separately:

```json
{"file_path":"datasets/reentrancy/dependency/example.sol","node_embeddings":[[0.1,0.2]]}
```

The two files align by `file_path`. Generated graph data under `graphs/` is
ignored by git.

Useful options:

```bash
uv run python scg/generate_dataset_graphs.py \
  --dataset-type reentrancy \
  --dataset-dir datasets/reentrancy \
  --output-dir graphs
```

Add `--fail-fast` to stop at the first graph generation error.
