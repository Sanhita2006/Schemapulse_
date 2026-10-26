"""Allow ``python -m schemapulse`` alongside the installed console script."""

from __future__ import annotations

from schemapulse.cli import main

if __name__ == "__main__":
    main()
