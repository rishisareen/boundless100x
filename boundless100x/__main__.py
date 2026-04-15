"""Allow running as: python -m boundless100x"""

import os
import sys
from boundless100x.cli import app

try:
    app()
except SystemExit as e:
    # Typer/click raise SystemExit on --help, bad args, etc.
    code = e.code if isinstance(e.code, int) else 1
    os._exit(code)
except Exception:
    import traceback
    traceback.print_exc()
    os._exit(1)
else:
    # Force exit only on success — jugaad-data / requests can leave background threads alive
    os._exit(0)
