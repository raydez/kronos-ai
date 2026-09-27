"""``python -m kronos_ai`` 入口。"""

from __future__ import annotations

import sys

from kronos_ai.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
