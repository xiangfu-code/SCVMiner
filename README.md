# SCVMiner

SCVMiner: A Smart Contract Vulnerability Detection Method Based on Graph and GNN.

## Contract Graph Structure Construction

```python
from gsc import get_contract_graph, get_contract_graphs, get_solidity_graph

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
