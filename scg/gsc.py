# Graph structure construction

"""Build contract-level graphs from Solidity source files.

Public interface:
    get_contract_graphs(sol_path, sol_version=None) -> list[Graph]
    get_contract_graph(sol_path, sol_version, contract_name) -> Graph
    get_solidity_graph(sol_path, sol_version) -> Graph

Config:
    DEFAULT_SOL_VERSION:
        Fallback compiler version used when sol_version is not provided and no
        version can be extracted from the Solidity file.
    SOLC_SELECT_COMMAND:
        Command used to manage local Solidity compiler versions. Before Slither
        runs, this module checks whether the resolved compiler version is
        installed. If not, it runs:
            uv run solc-select install <version>

Args:
    sol_path:
        Path to a Solidity source file.
    sol_version:
        Optional Solidity compiler version used by solc-select, such as
        "0.8.20". If omitted, this module tries to extract a version from the
        file's "pragma solidity ..." statement. If extraction fails, it uses
        DEFAULT_SOL_VERSION.

Returns:
    A list of Graph objects, one for each contract in the Solidity file.
    Libraries and interfaces are skipped as top-level graphs, but their
    functions/modifiers may be included when called by contract nodes.

Graph fields:
    contract_name:
        Name of the contract represented by this graph.
    nodes:
        Function/modifier/state-variable nodes. The node at g.nodes[i] always
        has id == i.
    edges:
        Directed edges as (src_id, dst_id). Function/modifier call edges point
        from caller to callee. State-variable write edges point from function
        to state variable. State-variable read edges point from state variable
        to function.
Node fields:
    id:
        Stable integer id inside the graph.
    name:
        Node name using "ContractName.memberName".
    kind:
        Node type: "function", "modifier", or "state_variable".
    text:
        Source code block for the function/modifier/state variable.

Example:
    graphs = get_contract_graphs("Token.sol", "0.8.20")
    for graph in graphs:
        print(graph.contract_name, [node.name for node in graph.nodes], graph.edges)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Any

from slither import Slither
from slither.core.declarations import Function, Modifier
from slither.core.variables.state_variable import StateVariable


DEFAULT_SOL_VERSION = "0.8.35"
_PRAGMA_SOLIDITY_PATTERN = re.compile(r"pragma\s+solidity\s+([^;]+);")
_SEMVER_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")
SOLC_SELECT_COMMAND = ("uv", "run", "solc-select")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Node:
    """A function/modifier/state-variable node in a contract graph."""

    id: int
    name: str
    text: str
    kind: str

    @property
    def names(self) -> str:
        """Compatibility alias for the name described in the file header."""
        return self.name

    @property
    def texts(self) -> str:
        """Compatibility alias for the source text described in the file header."""
        return self.text


@dataclass
class Graph:
    """Call graph for one Solidity contract."""

    contract_name: str
    nodes: list[Node]
    edges: list[tuple[int, int]]


def get_contract_graphs(sol_path: str | Path, sol_version: str | None = None) -> list[Graph]:
    """Convert Solidity source code to per-contract graphs.

    Args:
        sol_path: Path to a Solidity source file.
        sol_version: Optional compiler version for solc-select, for example
            ``"0.8.20"``. If omitted, the version is extracted from the file's
            ``pragma solidity`` statement; if no version is found,
            ``DEFAULT_SOL_VERSION`` is used.

    Returns:
        A list of ``Graph`` objects, one for each concrete/abstract contract in
        the source file. Libraries and interfaces are not emitted as top-level
        graphs. Nodes include each contract's functions, modifiers, and state
        variables, plus external contract/library/interface functions and
        modifiers directly called by those contract functions/modifiers.
    """

    _path, slither = _load_slither(sol_path, sol_version)
    graphs: list[Graph] = []

    for contract in slither.contracts:
        if contract.is_library or contract.is_interface:
            continue

        graphs.append(_build_contract_graph(contract))

    return graphs


def get_contract_graph(
    sol_path: str | Path,
    sol_version: str | None,
    contract_name: str,
) -> Graph:
    """Return the graph for a single contract in a Solidity source file.

    Args:
        sol_path: Path to a Solidity source file.
        sol_version: Optional compiler version for solc-select. If ``None``,
            version resolution follows the same rules as ``get_contract_graphs``.
        contract_name: Name of the contract whose graph should be returned.

    Raises:
        ValueError: If no non-library, non-interface graph with ``contract_name``
            exists in the source file.
    """

    for graph in get_contract_graphs(sol_path, sol_version):
        if graph.contract_name == contract_name:
            return graph

    raise ValueError(f"Contract graph not found for {contract_name!r} in {sol_path}")


def get_solidity_graph(sol_path: str | Path, sol_version: str | None = None) -> Graph:
    """Convert one Solidity source file to a single graph.

    The node and edge semantics match ``get_contract_graphs``. The returned graph
    may contain multiple connected components. Concrete/abstract contract
    functions, modifiers, and state variables are included even when isolated.
    Library/interface functions and modifiers are included only when they have a
    call edge to or from another function/modifier in the same source file.
    """

    path, slither = _load_slither(sol_path, sol_version)
    graph_members: list[Function | StateVariable] = []
    contract_callables: list[Function] = []
    external_callables: list[Function] = []

    for contract in slither.contracts:
        callable_nodes = _contract_callable_nodes(contract)
        if contract.is_library or contract.is_interface:
            external_callables.extend(callable_nodes)
            continue

        contract_callables.extend(callable_nodes)
        graph_members.extend(callable_nodes)
        graph_members.extend(contract.state_variables_declared)

    all_callables = contract_callables + external_callables
    callable_set = set(all_callables)
    connected_external_callables: set[Function] = set()

    for source in all_callables:
        for target in _called_functions(source):
            if target not in callable_set:
                continue
            if source in external_callables:
                connected_external_callables.add(source)
            if target in external_callables:
                connected_external_callables.add(target)

    graph_members.extend(
        function for function in external_callables if function in connected_external_callables
    )
    member_to_node_id = {member: index for index, member in enumerate(graph_members)}
    callable_nodes = contract_callables + [
        function for function in external_callables if function in connected_external_callables
    ]

    nodes = [
        Node(
            id=index,
            name=f"{_member_contract_name(member)}.{member.name}",
            text=_source_text(member),
            kind=_node_kind(member),
        )
        for index, member in enumerate(graph_members)
    ]
    edges = _collect_edges(callable_nodes, member_to_node_id)
    return Graph(contract_name=path.stem, nodes=nodes, edges=edges)


def _load_slither(sol_path: str | Path, sol_version: str | None) -> tuple[Path, Slither]:
    path = Path(sol_path)
    if not path.exists():
        raise FileNotFoundError(f"Solidity file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Solidity path is not a file: {path}")

    resolved_sol_version = _resolve_sol_version(path, sol_version)
    _ensure_solc_version_installed(resolved_sol_version)
    slither_kwargs: dict[str, Any] = {
        "solc_disable_warnings": True,
        "solc_solcs_select": resolved_sol_version,
    }
    return path, Slither(str(path), **slither_kwargs)


def _contract_callable_nodes(contract: Any) -> list[Function]:
    return [
        function
        for function in contract.functions_and_modifiers_declared
        if _is_real_function_or_modifier(function)
    ]


def _build_contract_graph(contract: Any) -> Graph:
    callable_nodes = _contract_callable_nodes(contract)
    graph_members: list[Function | StateVariable] = [
        *callable_nodes,
        *contract.state_variables_declared,
        *_called_external_functions(callable_nodes, contract),
    ]
    member_to_node_id = {member: index for index, member in enumerate(graph_members)}
    nodes = [
        Node(
            id=index,
            name=f"{_member_contract_name(member)}.{member.name}",
            text=_source_text(member),
            kind=_node_kind(member),
        )
        for index, member in enumerate(graph_members)
    ]
    edges = _collect_edges(_callable_members(graph_members), member_to_node_id)
    return Graph(contract_name=contract.name, nodes=nodes, edges=edges)


def _called_external_functions(functions: list[Function], contract: Any) -> list[Function]:
    external_functions: list[Function] = []
    seen: set[Function] = set()

    for function in functions:
        for target in _called_functions(function):
            target_contract = getattr(target, "contract_declarer", None) or getattr(target, "contract", None)
            if target_contract is contract or target in seen:
                continue
            seen.add(target)
            external_functions.append(target)

    return external_functions


def _callable_members(members: list[Function | StateVariable]) -> list[Function]:
    return [member for member in members if isinstance(member, Function)]


def _resolve_sol_version(path: Path, sol_version: str | None) -> str:
    if sol_version:
        return sol_version
    return _extract_sol_version_from_file(path) or DEFAULT_SOL_VERSION


def _extract_sol_version_from_file(path: Path) -> str | None:
    source = path.read_text(encoding="utf-8")
    match = _PRAGMA_SOLIDITY_PATTERN.search(source)
    if not match:
        return None

    version_match = _SEMVER_PATTERN.search(match.group(1))
    if not version_match:
        return None
    return version_match.group(0)


def _ensure_solc_version_installed(sol_version: str) -> None:
    versions_result = _run_solc_select("versions")
    if versions_result.returncode != 0:
        raise RuntimeError(_format_solc_select_error("list installed solc versions", versions_result))

    if _solc_version_is_installed(versions_result.stdout, sol_version):
        return

    print(f"solc {sol_version} is not installed. Installing with solc-select...")
    install_result = _run_solc_select("install", sol_version)
    if install_result.returncode != 0:
        raise RuntimeError(_format_solc_select_error(f"install solc {sol_version}", install_result))
    print(f"solc {sol_version} installed successfully.")


def _run_solc_select(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*SOLC_SELECT_COMMAND, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _solc_version_is_installed(versions_output: str, sol_version: str) -> bool:
    version_pattern = re.compile(rf"(^|\s){re.escape(sol_version)}(\s|$)")
    return any(version_pattern.search(line) for line in versions_output.splitlines())


def _format_solc_select_error(action: str, result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"Failed to {action} with command: {' '.join(result.args)}\n"
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def _is_real_function_or_modifier(function: Function) -> bool:
    return not (
        getattr(function, "is_constructor_variables", False)
        or getattr(function, "is_constructor_constant_variables", False)
    )


def _node_kind(member: Function | StateVariable) -> str:
    if isinstance(member, StateVariable):
        return "state_variable"
    if isinstance(member, Modifier):
        return "modifier"
    return "function"


def _source_text(member: Function | StateVariable) -> str:
    source_mapping = getattr(member, "source_mapping", None)
    if source_mapping is None:
        return ""
    return source_mapping.content


def _member_contract_name(member: Function | StateVariable) -> str:
    contract = getattr(member, "contract_declarer", None) or getattr(member, "contract", None)
    return getattr(contract, "name", "")


def _collect_edges(
    functions: list[Function],
    member_to_node_id: dict[Function | StateVariable, int],
) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()

    for source in functions:
        source_id = member_to_node_id[source]

        for target in _called_functions(source):
            target_id = member_to_node_id.get(target)
            if target_id is not None:
                edges.add((source_id, target_id))

        for state_variable in source.state_variables_read:
            target_id = member_to_node_id.get(state_variable)
            if target_id is not None:
                edges.add((target_id, source_id))

        for state_variable in source.state_variables_written:
            target_id = member_to_node_id.get(state_variable)
            if target_id is not None:
                edges.add((source_id, target_id))

    return sorted(edges)


def _called_functions(function: Function) -> list[Function]:
    called: list[Function] = []

    for internal_call in function.internal_calls:
        target = getattr(internal_call, "function", None)
        if isinstance(target, Function):
            called.append(target)

    for _target_contract, high_level_call in function.high_level_calls:
        target = getattr(high_level_call, "function", None)
        if isinstance(target, Function):
            called.append(target)

    for library_call in function.library_calls:
        target = getattr(library_call, "function", None)
        if isinstance(target, Function):
            called.append(target)

    for modifier in function.modifiers:
        if isinstance(modifier, Function):
            called.append(modifier)

    return called
