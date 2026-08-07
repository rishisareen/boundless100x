"""The two things every CLI module shares: one console, one logging setup.

Both live here rather than in `cli.py` because `cli_lifecycle.py` needs them
and `cli.py` imports *it* — a console defined in `cli.py` would make the two
modules circular, and a second console would be worse still. Rich buffers and
wraps per `Console` instance, so two of them means two wrapping widths, two
capture buffers, and a test that captures one while the code writes to the
other.

**One object, imported by name.** `from boundless100x.cli_common import
console` binds the same object in every module, so `console.capture()` taken
in one place sees what any other prints. Rebinding the *name* in one module
does not reach the others, which is why the surfaces that need a wider console
for a test patch each module they are exercising rather than only the one they
called into.
"""

import logging

from rich.console import Console

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
