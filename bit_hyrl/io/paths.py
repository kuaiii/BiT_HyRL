"""Filesystem conventions: where inputs live and where artefacts go.

All paths are derived from the package location, never from the current working
directory, so a run started from anywhere writes to the same place.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "CONFIG_ROOT",
    "RESULTS_ROOT",
    "DEFAULT_CONFIG_FILE",
    "GRAPHS_CONFIG_FILE",
    "SWEEP_CONFIG_FILE",
    "ensure_dir",
    "results_path",
    "fingerprints_file",
    "sweep_dir",
    "records_dir",
    "protocols_dir",
    "checkpoints_dir",
    "atomic_write_text",
    "atomic_write_json",
    "format_c",
    "make_run_id",
    "dataset_path",
]

#: Package root — two levels up from ``bit_hyrl/io/paths.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Pinned input graphs.
DATA_ROOT = PROJECT_ROOT / "dataset" / "testdata"

#: Configuration files.
CONFIG_ROOT = PROJECT_ROOT / "configs"

#: Output root (git-ignored).
RESULTS_ROOT = PROJECT_ROOT / "results"

DEFAULT_CONFIG_FILE = CONFIG_ROOT / "default.json"
GRAPHS_CONFIG_FILE = CONFIG_ROOT / "graphs.json"
SWEEP_CONFIG_FILE = CONFIG_ROOT / "sweep.json"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def results_path(*parts: str | Path) -> Path:
    """Return ``RESULTS_ROOT / *parts``, creating the parent directory."""
    target = RESULTS_ROOT.joinpath(*[str(part) for part in parts])
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def fingerprints_file() -> Path:
    """Path of ``results/fingerprints/graph_fingerprints.json``."""
    return results_path("fingerprints", "graph_fingerprints.json")


def sweep_dir(dataset: str) -> Path:
    """Directory holding the ``c``-sweep artefacts of one dataset."""
    return ensure_dir(RESULTS_ROOT / "sweep" / dataset)


def records_dir() -> Path:
    """Directory holding one JSON file per ``EvaluationRecord``."""
    return ensure_dir(RESULTS_ROOT / "records")


def protocols_dir(dataset: str) -> Path:
    """Directory holding the attack protocols of one dataset."""
    return ensure_dir(RESULTS_ROOT / "protocols" / dataset)


def checkpoints_dir() -> Path:
    """Directory holding PPO checkpoints."""
    return ensure_dir(RESULTS_ROOT / "checkpoints")


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically (write temp file, then rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
    return target


def atomic_write_json(path: str | Path, obj: object, *, indent: int | None = 2) -> Path:
    """Atomically write ``obj`` as canonical JSON (see :mod:`.jsonlog`)."""
    from .jsonlog import dumps

    text = dumps(obj, indent=indent)
    if indent is not None:
        text += "\n"
    return atomic_write_text(path, text)


def format_c(c: float) -> str:
    """Format the hub factor for use inside identifiers."""
    number = float(c)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def make_run_id(dataset: str, topology: str, c: float, policy: str,
                attack: str, seed: int) -> str:
    """Build the canonical run identifier."""
    return f"{dataset}__{topology}__c{format_c(c)}__{policy}__{attack}__s{int(seed)}"


def dataset_path(dataset: str, *, data_root: str | Path | None = None) -> Path:
    """Return the pinned ``.gml`` path of ``dataset``."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    candidate = root / f"{dataset}.gml"
    if not candidate.is_file():
        raise FileNotFoundError(f"graph data not found: {candidate}")
    return candidate
