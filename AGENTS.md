# AGENTS.md

## Environment

- Python version: `3.12`
- Package manager: `uv`

## scg/gsc.py Interface Notes

`get_contract_graphs(sol_path, sol_version=None)` version resolution:

1. If `sol_version` is provided, use it.
2. Otherwise, extract a concrete version from `pragma solidity ...;`.
3. If no version is found, use `DEFAULT_SOL_VERSION`.
4. Before running Slither, check whether that solc version is installed.
5. If missing, run `uv run solc-select install <version>`.
6. If installation fails, raise an exception with stdout/stderr details.

`get_solidity_graph(sol_path, sol_version=None)` returns one graph for the
whole Solidity source file. The graph may contain multiple connected components.
Contract functions/modifiers/state variables may be isolated nodes.
Library/interface functions and modifiers are included only when connected by a
call edge; uncalled and non-calling library/interface functions/modifiers are
not emitted as standalone nodes.

`get_contract_graphs` and `get_contract_graph` skip libraries/interfaces as
top-level graphs, but include external contract/library/interface
functions/modifiers when directly called by the selected contract's
functions/modifiers.

`Graph` fields:

- `contract_name`: Solidity contract name.
- `nodes`: list of `Node`.
- `edges`: directed edges as `(src_id, dst_id)`.

`Node` fields:

- `id`: stable integer id inside the graph.
- `name`: `ContractName.memberName`.
- `kind`: `"function"`, `"modifier"`, or `"state_variable"`.
- `text`: source text for the node.

Edge semantics:

- Function/modifier calls: `caller -> callee`
- State variable writes: `function -> state_variable`
- State variable reads: `state_variable -> function`

Libraries and interfaces are skipped as top-level graphs.

## Test Fixtures

Solidity examples live in `test_contracts/`.

Current fixtures cover:

- `SimpleCall.sol`: internal calls, modifier use, state variable reads/writes.
- `MixedKinds.sol`: library/interface filtering.
- `NoPragma.sol`: fallback to `DEFAULT_SOL_VERSION`.
- `Legacy0424.sol`: legacy Solidity `0.4.24` support.

## Development Notes

- Prefer editing files with `apply_patch`.
- Keep changes scoped to the requested behavior.
- Do not remove existing test fixtures unless requested.
- If changing graph semantics, update both `scg/gsc.py` documentation and
  `scg/gsc_test.py` output if needed.
- Run the demo/test script after graph logic changes.
