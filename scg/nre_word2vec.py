"""Train Word2Vec embeddings for Solidity contract graph node features.

Public interface:
    vectorize_code(code, model_or_keyed_vectors=None) -> list[float]

Training entry point:
    uv run python scg/nre_word2vec.py

Global hyperparameters:
    VECTOR_SIZE:
        Embedding dimension for both Word2Vec token vectors and returned node
        feature vectors.

Training corpus:
    Solidity graph-node source snippets under ``datasets``:
    function-like blocks plus contract-level state variable declarations. This
    matches the ``Node.text`` field produced by ``gsc.py``.

Word2Vec tokenization strategy:
    Solidity code is tokenized structurally instead of splitting on spaces.
    Tokens include identifiers, keywords, numeric literals, hex literals,
    operators, delimiters, brackets, and punctuation. Decimal and integer
    literals are normalized to ``<NUM>``; hexadecimal literals are normalized to
    ``<HEX>``. Comments are removed before training snippets are extracted.

Node vectorization:
    Word2Vec learns vectors for individual Solidity tokens. ``vectorize_code``
    tokenizes one node's source text, looks up known token vectors, and returns
    their mean-pooled vector. Unknown tokens are ignored. If no token is known,
    a zero vector with length ``VECTOR_SIZE`` is returned.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, Sequence, cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "solidity_word2vec.model"
DEFAULT_VECTORS_PATH = PROJECT_ROOT / "models" / "solidity_word2vec.kv"
DEFAULT_STATS_PATH = PROJECT_ROOT / "models" / "solidity_word2vec_stats.json"
VECTOR_SIZE = 256

FUNCTION_LIKE_PATTERN = re.compile(r"\b(function|modifier|constructor|fallback|receive)\b")
TOKEN_PATTERN = re.compile(
    r"""
    0x[a-fA-F0-9]+
    |\d+\.\d+
    |\d+
    |[A-Za-z_][A-Za-z0-9_]*
    |==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|=>|->|\+=|-=|\*=|/=|%=|\*\*
    |[{}()[\].,;:+\-*/%<>=!&|^~?]
    """,
    re.VERBOSE,
)
HEX_LITERAL_PATTERN = re.compile(r"^0x[a-fA-F0-9]+$")
NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
CONTRACT_BODY_PATTERN = re.compile(
    r"\b(contract|library|interface)\s+[A-Za-z_][A-Za-z0-9_]*[^{;]*{"
)
NON_STATE_DECLARATION_PREFIXES = (
    "event",
    "error",
    "function",
    "modifier",
    "constructor",
    "fallback",
    "receive",
    "using",
    "struct",
    "enum",
)

__all__ = ["vectorize_code"]


class KeyedVectorsLike(Protocol):
    """Minimal protocol used by vectorize_code."""

    vector_size: int

    def __contains__(self, key: str) -> bool: ...

    def __getitem__(self, key: str) -> Any: ...


@dataclass(frozen=True)
class TrainingStats:
    """Summary of the training corpus and produced model."""

    source_dir: str
    solidity_files: int
    code_blocks: int
    token_count: int
    vocabulary_size: int
    vector_size: int
    window: int
    min_count: int
    workers: int
    epochs: int
    sg: int
    seed: int
    model_path: str
    vectors_path: str


def iter_solidity_files(source_dir: str | Path) -> list[Path]:
    """Return all Solidity files under ``source_dir`` in stable order."""

    return sorted(Path(source_dir).rglob("*.sol"))


def iter_code_blocks(source_dir: str | Path) -> Iterable[str]:
    """Yield Solidity graph-node source snippets for Word2Vec training."""

    for sol_file in iter_solidity_files(source_dir):
        source = sol_file.read_text(encoding="utf-8", errors="ignore")
        yield from extract_node_code_blocks(source)


def extract_node_code_blocks(source: str) -> list[str]:
    """Extract function-like blocks and state variable declarations."""

    clean_source = strip_comments(source)
    return [
        *extract_function_like_blocks(clean_source, strip_source_comments=False),
        *extract_state_variable_declarations(clean_source, strip_source_comments=False),
    ]


def extract_function_like_blocks(source: str, strip_source_comments: bool = True) -> list[str]:
    """Extract Solidity function-like source blocks that have a body."""

    clean_source = strip_comments(source) if strip_source_comments else source
    blocks: list[str] = []

    for match in FUNCTION_LIKE_PATTERN.finditer(clean_source):
        body_start = _find_next_body_start(clean_source, match.end())
        if body_start is None:
            continue

        body_end = _find_matching_brace(clean_source, body_start)
        if body_end is None:
            continue

        blocks.append(clean_source[match.start() : body_end + 1])

    return blocks


def extract_state_variable_declarations(
    source: str,
    strip_source_comments: bool = True,
) -> list[str]:
    """Extract contract-level state variable declarations."""

    clean_source = strip_comments(source) if strip_source_comments else source
    declarations: list[str] = []

    for match in CONTRACT_BODY_PATTERN.finditer(clean_source):
        body_start = clean_source.find("{", match.start())
        body_end = _find_matching_brace(clean_source, body_start)
        if body_end is None:
            continue

        declarations.extend(_extract_top_level_state_declarations(clean_source[body_start + 1 : body_end]))

    return declarations


def strip_comments(source: str) -> str:
    """Remove Solidity comments while preserving string literal contents."""

    result: list[str] = []
    index = 0
    length = len(source)
    in_string: str | None = None

    while index < length:
        char = source[index]
        next_char = source[index + 1] if index + 1 < length else ""

        if in_string:
            result.append(char)
            if char == "\\" and index + 1 < length:
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
            while index < length and source[index] != "\n":
                index += 1
            result.append("\n")
            continue

        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < length and not (source[index] == "*" and source[index + 1] == "/"):
                result.append("\n" if source[index] == "\n" else " ")
                index += 1
            index += 2
            continue

        result.append(char)
        index += 1

    return "".join(result)


def tokenize_code(code: str) -> list[str]:
    """Tokenize Solidity code into normalized Word2Vec tokens."""

    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(code):
        token = match.group(0)
        if HEX_LITERAL_PATTERN.match(token):
            tokens.append("<HEX>")
        elif NUMBER_PATTERN.match(token):
            tokens.append("<NUM>")
        else:
            tokens.append(token)
    return tokens


def build_training_sentences(source_dir: str | Path) -> list[list[str]]:
    """Build Word2Vec sentences from all Solidity graph-node snippets."""

    sentences = [tokens for block in iter_code_blocks(source_dir) if (tokens := tokenize_code(block))]
    if not sentences:
        raise ValueError(f"No Solidity graph-node code snippets found under {source_dir}")
    return sentences


def train_word2vec(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectors_path: str | Path = DEFAULT_VECTORS_PATH,
    stats_path: str | Path = DEFAULT_STATS_PATH,
    vector_size: int = VECTOR_SIZE,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 20,
    sg: int = 1,
    seed: int = 42,
) -> TrainingStats:
    """Train and save a Word2Vec model for Solidity code tokens."""

    from gensim.models import Word2Vec

    source_dir = Path(source_dir)
    model_path = Path(model_path)
    vectors_path = Path(vectors_path)
    stats_path = Path(stats_path)
    sentences = build_training_sentences(source_dir)

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=seed,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    model.wv.save(str(vectors_path))

    stats = TrainingStats(
        source_dir=str(source_dir),
        solidity_files=len(iter_solidity_files(source_dir)),
        code_blocks=len(sentences),
        token_count=sum(len(sentence) for sentence in sentences),
        vocabulary_size=len(model.wv),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=seed,
        model_path=str(model_path),
        vectors_path=str(vectors_path),
    )
    stats_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    return stats


def vectorize_code(code: str, model_or_keyed_vectors: object | None = None) -> list[float]:
    """Average token vectors for one Solidity source block.

    Args:
        code: Source text from a graph node, such as ``Node.text``.
        model_or_keyed_vectors: Optional gensim ``Word2Vec`` model or
            ``KeyedVectors``. If omitted, vectors are loaded from
            ``DEFAULT_VECTORS_PATH``.

    Returns:
        A Python list of floats. Unknown tokens are ignored. If no known tokens
        are present, a zero vector with the loaded model's vector size is
        returned. For the default trained model, this length is ``VECTOR_SIZE``.
    """

    import numpy as np

    if model_or_keyed_vectors is None:
        model_or_keyed_vectors = _load_keyed_vectors()

    keyed_vectors = cast(
        KeyedVectorsLike,
        getattr(model_or_keyed_vectors, "wv", model_or_keyed_vectors),
    )
    vectors = [keyed_vectors[token] for token in tokenize_code(code) if token in keyed_vectors]
    if not vectors:
        return np.zeros(keyed_vectors.vector_size, dtype=float).tolist()
    return np.mean(vectors, axis=0).astype(float).tolist()


@lru_cache(maxsize=4)
def _load_keyed_vectors(vectors_path: str | Path = DEFAULT_VECTORS_PATH) -> object:
    """Load saved gensim keyed vectors for inference."""

    from gensim.models import KeyedVectors

    return KeyedVectors.load(str(vectors_path))


def _find_next_body_start(source: str, start: int) -> int | None:
    index = start
    while index < len(source):
        char = source[index]
        if char == "{":
            return index
        if char == ";":
            return None
        index += 1
    return None


def _extract_top_level_state_declarations(contract_body: str) -> list[str]:
    declarations: list[str] = []
    statement_start = 0
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    in_string: str | None = None
    index = 0

    while index < len(contract_body):
        char = contract_body[index]

        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == in_string:
                in_string = None
            index += 1
            continue

        if char in {'"', "'"}:
            in_string = char
            index += 1
            continue

        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
            if brace_depth == 0:
                statement_start = index + 1
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == ";" and brace_depth == 0 and paren_depth == 0 and bracket_depth == 0:
            statement = contract_body[statement_start : index + 1].strip()
            statement_start = index + 1
            if _looks_like_state_variable_declaration(statement):
                declarations.append(statement)

        index += 1

    return declarations


def _looks_like_state_variable_declaration(statement: str) -> bool:
    if not statement:
        return False

    first_token_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", statement)
    if not first_token_match:
        return False

    first_token = first_token_match.group(0)
    if first_token in NON_STATE_DECLARATION_PREFIXES:
        return False

    if statement.startswith(("pragma ", "import ")):
        return False

    return True


def _find_matching_brace(source: str, open_brace_index: int) -> int | None:
    depth = 0
    index = open_brace_index
    in_string: str | None = None

    while index < len(source):
        char = source[index]

        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == in_string:
                in_string = None
            index += 1
            continue

        if char in {'"', "'"}:
            in_string = char
            index += 1
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index

        index += 1

    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings from Solidity source code blocks.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vectors-path", type=Path, default=DEFAULT_VECTORS_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cbow", action="store_true", help="Use CBOW instead of skip-gram.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stats = train_word2vec(
        source_dir=args.source_dir,
        model_path=args.model_path,
        vectors_path=args.vectors_path,
        stats_path=args.stats_path,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        sg=0 if args.cbow else 1,
        seed=args.seed,
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
