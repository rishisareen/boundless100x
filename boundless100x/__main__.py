"""Allow running as: python -m boundless100x"""

import os
from boundless100x.cli import app

try:
    app()
finally:
    # Force exit — jugaad-data / requests can leave background threads alive
    os._exit(0)
