"""Canonical JSON / JSONL writing, reading and hashing.

Output is byte-stable (sorted keys, fixed separators, six-decimal floats), so two
runs that compute the same values produce identical files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

__all__ = [
    "FLOAT_NDIGITS",
    "sanitize",
    "dumps",
    "loads",
    "dump_json",
    "load_json",
    "write_jsonl",
    "append_jsonl",
    "read_jsonl",
    "iter_jsonl",
    "sha256_text",
    "sha256_json",
    "sha256_file",
]

#: Number of decimal places kept for every float in an artefact.
FLOAT_NDIGITS = 6


def _clean_float(value: float) -> float | None:
    """Round a float to :data:`FLOAT_NDIGITS`, mapping non-finite to ``None``."""
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, FLOAT_NDIGITS)
    if rounded == 0.0:
        return 0.0
    return rounded


def sanitize(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-safe plain Python data."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, Mapping):
        return {str(key): sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitize(item) for item in sorted(obj, key=repr)]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sanitize(dataclasses.asdict(obj))
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return sanitize(to_dict())
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return sanitize(tolist())
    item = getattr(obj, "item", None)
    if callable(item):
        return sanitize(item())
    raise TypeError(f"cannot serialise {type(obj).__name__}: {obj!r}")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Return the canonical JSON text for ``obj``."""
    payload = sanitize(obj)
    if indent is None:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    return json.dumps(
        payload, sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False
    )


def loads(text: str) -> Any:
    """Parse JSON text produced by :func:`dumps`."""
    return json.loads(text)


def dump_json(path: str | Path, obj: Any, *, indent: int | None = 2) -> Path:
    """Write ``obj`` to ``path`` as canonical JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(obj, indent=indent)
    if indent is not None:
        text += "\n"
    target.write_text(text, encoding="utf-8")
    return target


def load_json(path: str | Path) -> Any:
    """Read canonical JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Write ``rows`` as JSONL (one canonical JSON object per line)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(dumps(row))
            handle.write("\n")
    return target


def append_jsonl(path: str | Path, row: Any) -> Path:
    """Append a single row to a JSONL file, creating it if absent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps(row))
        handle.write("\n")
    return target


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    """Yield one parsed object per non-empty line of a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file into a list."""
    return list(iter_jsonl(path))


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 of the canonical JSON encoding of ``obj``."""
    return sha256_text(dumps(obj))


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
