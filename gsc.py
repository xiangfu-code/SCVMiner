# Graph structure construction

"""Build contract-level graphs from Solidity source files.

Public interface:
    solidity_to_graph(sol_path, sol_version=None) -> list[Graph]

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
    Libraries and interfaces are skipped.

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
    graphs = solidity_to_graph("Token.sol", "0.8.20")
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


def solidity_to_graph(sol_path: str | Path, sol_version: str | None = None) -> list[Graph]:
    """Convert Solidity source code to per-contract graphs.

    Args:
        sol_path: Path to a Solidity source file.
        sol_version: Optional compiler version for solc-select, for example
            ``"0.8.20"``. If omitted, the version is extracted from the file's
            ``pragma solidity`` statement; if no version is found,
            ``DEFAULT_SOL_VERSION`` is used.

    Returns:
        A list of ``Graph`` objects, one for each concrete/abstract contract in
        the source file. Libraries and interfaces are ignored. Nodes include
        functions, modifiers, and state variables.
    """

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

    slither = Slither(str(path), **slither_kwargs)
    graphs: list[Graph] = []

    for contract in slither.contracts:
        if contract.is_library or contract.is_interface:
            continue

        callable_nodes = [
            function
            for function in contract.functions_and_modifiers_declared
            if _is_real_function_or_modifier(function)
        ]
        state_variables = contract.state_variables_declared
        graph_members = callable_nodes + state_variables
        member_to_node_id = {member: index for index, member in enumerate(graph_members)}

        nodes = [
            Node(
                id=index,
                name=f"{contract.name}.{member.name}",
                text=_source_text(member),
                kind=_node_kind(member),
            )
            for index, member in enumerate(graph_members)
        ]
        edges = _collect_edges(callable_nodes, member_to_node_id)
        graphs.append(Graph(contract_name=contract.name, nodes=nodes, edges=edges))

    return graphs


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
        cwd=Path(__file__).resolve().parent,
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


get_contract_graphs = solidity_to_graph
