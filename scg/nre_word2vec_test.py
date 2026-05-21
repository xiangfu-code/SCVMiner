from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scg.nre_word2vec import vectorize_code


def main() -> None:
    state_variable_vector = vectorize_code("uint256 public totalSupply;")
    function_vector = vectorize_code(
        "function transfer(address to, uint256 value) public returns (bool) { return true; }"
    )

    assert len(state_variable_vector) > 0
    assert len(function_vector) == len(state_variable_vector)
    assert any(value != 0 for value in state_variable_vector)
    assert any(value != 0 for value in function_vector)

    print(f"state_variable_vector_size = {len(state_variable_vector)}")
    print(f"function_vector_size = {len(function_vector)}")


if __name__ == "__main__":
    main()
