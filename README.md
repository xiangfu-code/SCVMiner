# SCVMiner

SCVMiner: A Smart Contract Vulnerability Detection Method Based on Graph and GNN.

## SC Graph Generator

Use `sc_graph_generator` to convert Solidity code into a model-ready graph:

```python
from scg.scg import sc_graph_generator

file_graph = sc_graph_generator("path/to/Contract.sol")
contract_graph = sc_graph_generator("path/to/Contract.sol", contract_name="Token")
```

Without `contract_name`, one Solidity file becomes one graph. With
`contract_name`, one selected contract becomes one graph. The returned `SCGraph`
contains `num_nodes`, `num_edges`, directed `edges`, and `node_embeddings`.

Nodes are Solidity functions, modifiers, and state variables. Edges represent
function/modifier calls, state-variable writes, and state-variable reads. Node
features are Word2Vec embeddings mean-pooled from each node's Solidity source
text.

Detailed API and dataset instructions are in `scg/README.md`.
