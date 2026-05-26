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


DEFAULT_SOL_VERSION = "0.4.24"
_PRAGMA_SOLIDITY_PATTERN = re.compile(r"pragma\s+solidity\s+([^;]+);")
_SEMVER_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?")
SOLC_SELECT_COMMAND = ("uv", "run", "solc-select")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LATEST_PATCH_BY_MINOR = {
    (0, 4): "0.4.26",
    (0, 5): "0.5.17",
    (0, 6): "0.6.12",
    (0, 7): "0.7.6",
}


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

    last_error: Exception | None = None
    for resolved_sol_version in _resolve_sol_versions(path, sol_version):
        try:
            _ensure_solc_version_installed(resolved_sol_version)
            slither_kwargs: dict[str, Any] = {
                "solc_disable_warnings": True,
                "solc_solcs_select": resolved_sol_version,
            }
            return path, Slither(str(path), **slither_kwargs)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No Solidity compiler version could be resolved for {path}")


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
    return _resolve_sol_versions(path, sol_version)[0]


def _resolve_sol_versions(path: Path, sol_version: str | None) -> list[str]:
    if sol_version:
        return [sol_version]
    return _extract_sol_versions_from_file(path) or [DEFAULT_SOL_VERSION]


def _extract_sol_version_from_file(path: Path) -> str | None:
    versions = _extract_sol_versions_from_file(path)
    return versions[0] if versions else None


def _extract_sol_versions_from_file(path: Path) -> list[str]:
    source = _strip_solidity_comments(path.read_text(encoding="utf-8", errors="ignore"))
    pragma_specs = [match.group(1) for match in _PRAGMA_SOLIDITY_PATTERN.finditer(source)]
    if not pragma_specs:
        return []

    versions = [
        version
        for spec in pragma_specs
        for version in _SEMVER_PATTERN.findall(spec)
    ]
    if not versions:
        return []

    exact_versions = [
        exact_version
        for spec in pragma_specs
        if (exact_version := _exact_pragma_version(spec)) is not None
    ]
    if exact_versions:
        return [max(exact_versions, key=_version_key)]

    caret_zero_versions = [
        caret_zero_version
        for spec in pragma_specs
        if (caret_zero_version := _caret_zero_pragma_version(spec)) is not None
    ]
    if caret_zero_versions:
        highest_caret_zero = max(caret_zero_versions, key=_version_key)
        major, minor, _patch = _version_key(highest_caret_zero)
        candidates = []
        if fallback_version := _LATEST_PATCH_BY_MINOR.get((major, minor)):
            candidates.append(fallback_version)
        candidates.append(highest_caret_zero)
        return _unique_versions(candidates)

    if range_versions := _range_pragma_versions(pragma_specs):
        return range_versions

    if open_lower_bound_versions := _open_lower_bound_pragma_versions(pragma_specs):
        return open_lower_bound_versions

    lower_bound_versions = [
        version
        for spec in pragma_specs
        for version in _lower_bound_versions(spec)
    ]
    if not lower_bound_versions:
        lower_bound_versions = versions

    highest_lower_bound = max(lower_bound_versions, key=_version_key)
    major, minor, _patch = _version_key(highest_lower_bound)
    if _has_upper_bound_at_or_below(pragma_specs, "0.6.0") and (major, minor) < (0, 6):
        return _unique_versions([_LATEST_PATCH_BY_MINOR[(0, 5)], highest_lower_bound])
    if _has_upper_bound_at_or_below(pragma_specs, "0.7.0") and (major, minor) < (0, 7):
        return _unique_versions([_LATEST_PATCH_BY_MINOR[(0, 6)], highest_lower_bound])
    if (major, minor, _patch) == (0, 4, 99):
        return _unique_versions([_LATEST_PATCH_BY_MINOR[(0, 5)], highest_lower_bound])
    if open_upper_bound_versions := _open_upper_bound_pragma_versions(pragma_specs, highest_lower_bound):
        return open_upper_bound_versions
    return _unique_versions([_LATEST_PATCH_BY_MINOR.get((major, minor), highest_lower_bound), highest_lower_bound])


def _unique_versions(versions: list[str]) -> list[str]:
    unique_versions: list[str] = []
    seen: set[str] = set()
    for version in versions:
        if version in seen:
            continue
        seen.add(version)
        unique_versions.append(version)
    return unique_versions


def _exact_pragma_version(spec: str) -> str | None:
    stripped = spec.strip()
    exact_match = re.match(r"^=?\s*(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)", stripped)
    if not exact_match:
        return None
    if stripped.startswith("=") or re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", stripped):
        return exact_match.group(1)
    return None


def _caret_zero_pragma_version(spec: str) -> str | None:
    stripped = spec.strip()
    caret_match = re.match(r"^\^\s*(0\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)$", stripped)
    if caret_match:
        return caret_match.group(1)
    return None


def _range_pragma_versions(specs: list[str]) -> list[str]:
    lower_bounds: list[str] = []
    upper_bounds: list[str] = []

    for spec in specs:
        for match in _SEMVER_PATTERN.finditer(spec):
            version = match.group(0)
            prefix = spec[: match.start()].rstrip()
            suffix = spec[match.end() :].lstrip()
            if prefix.endswith(">") or prefix.endswith(">=") or suffix.startswith("<") or suffix.startswith("<="):
                lower_bounds.append(version)
            elif prefix.endswith("<") or prefix.endswith("<=") or suffix.startswith(">") or suffix.startswith(">="):
                upper_bounds.append(version)

    if not lower_bounds or not upper_bounds:
        return []

    min_version = max(lower_bounds, key=_version_key)
    max_version = min(upper_bounds, key=_version_key)
    if _version_key(min_version) > _version_key(max_version):
        return []

    candidates = [max_version]
    min_major, min_minor, _min_patch = _version_key(min_version)
    max_major, max_minor, _max_patch = _version_key(max_version)
    for major, minor in sorted(_LATEST_PATCH_BY_MINOR, reverse=True):
        if (major, minor) > (max_major, max_minor) or (major, minor) < (min_major, min_minor):
            continue
        latest_patch = _LATEST_PATCH_BY_MINOR[(major, minor)]
        if _version_key(min_version) <= _version_key(latest_patch) <= _version_key(max_version):
            candidates.append(latest_patch)
    candidates.append(min_version)
    return _unique_versions(candidates)


def _open_lower_bound_pragma_versions(specs: list[str]) -> list[str]:
    lower_bounds: list[str] = []

    for spec in specs:
        has_open_lower_bound = False
        has_upper_bound = False
        for match in _SEMVER_PATTERN.finditer(spec):
            version = match.group(0)
            prefix = spec[: match.start()].rstrip()
            suffix = spec[match.end() :].lstrip()
            if prefix.endswith(">") and not prefix.endswith(">="):
                has_open_lower_bound = True
                lower_bounds.append(version)
            elif prefix.endswith("<") or prefix.endswith("<="):
                has_upper_bound = True
            elif suffix.startswith(">") or suffix.startswith(">="):
                has_upper_bound = True

        if has_open_lower_bound and has_upper_bound:
            return []

    if not lower_bounds:
        return []

    lower_bound = max(lower_bounds, key=_version_key)
    return _open_upper_bound_pragma_versions(specs, lower_bound, include_lower_bound=False)


def _open_upper_bound_pragma_versions(
    specs: list[str],
    lower_bound: str,
    include_lower_bound: bool = True,
) -> list[str]:
    if any(_has_upper_bound(spec) for spec in specs):
        return []

    lower_bound_key = _version_key(lower_bound)
    lower_major, lower_minor, _lower_patch = lower_bound_key
    candidates = [
        lower_bound,
    ] if include_lower_bound else []
    candidates.extend(
        version
        for minor, version in sorted(_LATEST_PATCH_BY_MINOR.items())
        if (lower_major, lower_minor) <= minor <= (0, 7)
        and _version_key(version) > lower_bound_key
    )
    return _unique_versions(candidates)


def _lower_bound_versions(spec: str) -> list[str]:
    lower_bounds: list[str] = []
    for match in _SEMVER_PATTERN.finditer(spec):
        prefix = spec[: match.start()].rstrip()
        if prefix.endswith("<") or prefix.endswith("<="):
            continue
        lower_bounds.append(match.group(0))
    return lower_bounds


def _has_upper_bound(spec: str) -> bool:
    for match in _SEMVER_PATTERN.finditer(spec):
        prefix = spec[: match.start()].rstrip()
        suffix = spec[match.end() :].lstrip()
        if prefix.endswith("<") or prefix.endswith("<="):
            return True
        if suffix.startswith(">") or suffix.startswith(">="):
            return True
    return False


def _has_upper_bound_at_or_below(specs: list[str], version: str) -> bool:
    target = _version_key(version)
    for spec in specs:
        for match in _SEMVER_PATTERN.finditer(spec):
            prefix = spec[: match.start()].rstrip()
            if (prefix.endswith("<") or prefix.endswith("<=")) and _version_key(match.group(0)) <= target:
                return True
    return False


def _version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".", 2)
    patch_match = re.match(r"\d+", patch)
    return int(major), int(minor), int(patch_match.group(0) if patch_match else 0)


def _strip_solidity_comments(source: str) -> str:
    result: list[str] = []
    index = 0
    in_string: str | None = None

    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""

        if in_string:
            result.append(char)
            if char == "\\" and index + 1 < len(source):
                result.append(source[index + 1])
                index += 2
                continue
            if char == in_string:
                in_string = None
            index += 1
            continue

        if char in {'"', "'"}:
            in_string = char
            result.append(char)
            index += 1
            continue

        if char == "/" and next_char == "/":
            while index < len(source) and source[index] != "\n":
                index += 1
            result.append("\n")
            continue

        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(source) and not (source[index] == "*" and source[index + 1] == "/"):
                result.append("\n" if source[index] == "\n" else " ")
                index += 1
            index += 2
            continue

        result.append(char)
        index += 1

    return "".join(result)


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
