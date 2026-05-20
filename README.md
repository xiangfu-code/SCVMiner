# SCVMiner

SCVMiner: A Smart Contract Vulnerability Detection Method Based on Graph and GNN.

## Contract Graph Structure Construction

```python
from gsc import solidity_to_graph

graphs = solidity_to_graph("path/to/Contract.sol", sol_version=None)
```

Each Solidity contract is converted into a directed graph. Libraries and
interfaces are skipped.

Nodes represent contract members:

- Functions, including constructors and fallback-style functions.
- Modifiers.
- State variables declared in the contract.

Edges encode control/data relationships:

- Function or modifier calls: `caller -> callee`
- State variable writes: `function -> state_variable`
- State variable reads: `state_variable -> function`
