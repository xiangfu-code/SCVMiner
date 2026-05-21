# SCVMiner

SCVMiner: A Smart Contract Vulnerability Detection Method Based on Graph and GNN.

## Graph Structure Construction

```python
from scg.gsc import get_contract_graph, get_contract_graphs, get_solidity_graph

graphs = get_contract_graphs("path/to/Contract.sol", sol_version=None)
graph = get_contract_graph("path/to/Contract.sol", sol_version=None, contract_name="Token")
solidity_graph = get_solidity_graph("path/to/Contract.sol", sol_version=None)
```

Each Solidity contract is converted into a directed graph. Libraries and
interfaces are skipped as top-level graphs, but their functions/modifiers are
included when called by a contract function/modifier.

Use `get_solidity_graph` to convert the whole Solidity source file into one
directed graph. The graph may contain multiple connected components. Contract
functions, modifiers, and state variables may appear as isolated nodes;
library/interface functions and modifiers are included only when connected by a
call edge.

Nodes represent contract members:

- Functions, including constructors and fallback-style functions.
- Modifiers.
- State variables declared in the contract.

Edges encode control/data relationships:

- Function or modifier calls: `caller -> callee`
- State variable writes: `function -> state_variable`
- State variable reads: `state_variable -> function`

## Node Representation Extraction

`scg/nre_word2vec.py` trains Word2Vec embeddings for graph node text. Public
interface:

```python
from scg.nre_word2vec import vectorize_code

node_feature = vectorize_code(node.text)
```

Train or refresh the model:

```bash
uv run python scg/nre_word2vec.py
```

Training data lives in `datasets/` and includes function-like code
blocks plus state variable declarations. Outputs are saved under `models/`.

`vectorize_code` uses the trained `models/solidity_word2vec.kv` by default. It
tokenizes Solidity structurally, maps known tokens to Word2Vec vectors, then
mean-pools them into one node feature vector. The vector length is controlled by
`scg.nre_word2vec.VECTOR_SIZE`.
