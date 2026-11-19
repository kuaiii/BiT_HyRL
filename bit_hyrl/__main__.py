"""``python -m bit_hyrl`` — forwards to :func:`bit_hyrl.cli.main`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
