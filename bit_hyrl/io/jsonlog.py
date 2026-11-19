"""Canonical JSON / JSONL writing, reading and hashing.

This is the **only** place in the package allowed to serialise a result object.
Centralising it buys three properties that the paper needs:

1. **Byte-stable output** — ``sort_keys=True`` plus a fixed separator pair means
   two runs that computed the same values produce the same bytes, so a ``diff``
   between two result files is meaningful.
2. **Deterministic floats** — every float is rounded to six decimals and
   ``-0.0`` is normalised to ``0.0``.  Otherwise a value that is numerically
   zero can flip sign between runs and show up as a spurious diff.
3. **Explicit non-finite handling** — ``NaN`` / ``Inf`` become ``null`` and
   ``allow_nan=False`` then guarantees the writer can never emit the invalid
   JSON literals ``NaN`` / ``Infinity`` (which are not valid JSON and break
   every downstream consumer, including the TeX table generator).
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
    """Round a float to :data:`FLOAT_NDIGITS`, mapping non-finite to ``None``.

    Args:
        value: Any float.

    Returns:
        ``None`` for ``NaN`` / ``±Inf``, otherwise the rounded value with
        ``-0.0`` collapsed to ``0.0``.
    """
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, FLOAT_NDIGITS)
    if rounded == 0.0:  # collapses -0.0 -> 0.0
        return 0.0
    return rounded


def sanitize(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-safe plain Python data.

    Conversion rules:

    * ``tuple`` → ``list`` (JSON has no tuple);
    * ``set`` / ``frozenset`` → sorted ``list`` (iteration order of a set is not
      stable across processes, so it must never leak into an artefact);
    * dataclass instances and objects exposing ``to_dict()`` → ``dict``;
    * mapping keys → ``str`` (JSON object keys are strings);
    * ``float`` → rounded to six decimals, ``-0.0`` → ``0.0``,
      ``NaN`` / ``±Inf`` → ``None``;
    * numpy scalars / arrays → Python scalars / lists.

    Args:
        obj: Arbitrary nested structure.

    Returns:
        A structure made only of ``dict``, ``list``, ``str``, ``int``, ``float``,
        ``bool`` and ``None``.

    Raises:
        TypeError: If a value has no defensible JSON representation.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):  # bool already handled above
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
    # numpy scalars and arrays, and anything else exposing ``item`` / ``tolist``
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return sanitize(tolist())
    item = getattr(obj, "item", None)
    if callable(item):
        return sanitize(item())
    raise TypeError(f"无法序列化类型 {type(obj).__name__}: {obj!r}")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Return the canonical JSON text for ``obj``.

    Args:
        obj: Any structure accepted by :func:`sanitize`.
        indent: Optional indentation (``None`` gives one line, as used by JSONL).

    Returns:
        Canonical JSON text: keys sorted, no trailing whitespace, floats
        rounded, non-finite values rendered as ``null``.
    """
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
    """Write ``obj`` to ``path`` as canonical JSON.

    Args:
        path: Destination file; parent directories are created.
        obj: Payload.
        indent: ``None`` for a single line, otherwise the indent width.

    Returns:
        The path written.
    """
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
    """Write ``rows`` as JSONL (one canonical JSON object per line).

    Args:
        path: Destination file; parent directories are created.
        rows: Iterable of payloads.

    Returns:
        The path written.
    """
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
    """Return the SHA-256 of the canonical JSON encoding of ``obj``.

    This is the hash used for edge sequences and attack protocols: it changes
    if and only if the *semantic* content changes, because the encoding is
    canonical.
    """
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
