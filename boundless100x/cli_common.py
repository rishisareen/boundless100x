"""What every CLI module shares: one console, one logging setup, one renderer.

The first two live here rather than in `cli.py` because `cli_lifecycle.py`
needs them and `cli.py` imports *it* — a console defined in `cli.py` would make
the two modules circular, and a second console would be worse still. Rich
buffers and wraps per `Console` instance, so two of them means two wrapping
widths, two capture buffers, and a test that captures one while the code writes
to the other.

**One object, imported by name.** `from boundless100x.cli_common import
console` binds the same object in every module, so `console.capture()` taken
in one place sees what any other prints. Rebinding the *name* in one module
does not reach the others, which is why the surfaces that need a wider console
for a test patch each module they are exercising rather than only the one they
called into.

The third is `ConsoleComponents`, and it is here for the same reason as the
first two: it is shared CLI machinery, and putting it in `cli.py` would put it
somewhere `cli_lifecycle.py` cannot reach without a cycle.

── The console is R14's third surface ────────────────────────────────────

`report_surfaces.py` registers `html` and `markdown` and says plainly that
`console` is the CLI's. This is it, and registering it is what turns
`EXPECTED_SURFACES` from a promise into a check: `@component_surface` refuses a
class missing any `render_<member>` at the line that defines it, and
`missing_members` is then non-vacuous for all three.

**The composition is shared; only the presentation differs.** Every method
below reaches for the same `reading_line` / `metric_cells` / `finding_line` /
`caveat_line` helpers the two document surfaces call, so a phrase added to the
report is a phrase the console gains too. What this surface adds is Rich markup
and one thing neither document needs.

── The one thing a terminal needs and a document does not ────────────────

A document can afford three sentences of interpretation per row; eighty columns
cannot. So `metric_row_line` rebuilds the reading cell with the interpretation
prose clipped to `READING_BUDGET`, and **the two things R12 names are the two
the clip cannot reach**: the figure keeps its unit because it is emitted before
the budget is counted, and the direction of goodness is appended whole
afterwards. A test pins that the console's figure and direction are
byte-identical to the documents' and that only the middle differs — because the
tempting shortening is to cut the tail, and the tail is the direction.

Clipping loses words, which is a real cost and not a free one: the full reason
is in the research note the same run writes, and the ellipsis is what says to go
and look.

── Nothing here escapes, and that is upstream's doing ────────────────────

Rich reads `[` as the start of a style tag, so an unescaped `[` in a component
would silently swallow the rest of the line — the failure `_evidence_cell`
exists to prevent for text that never went through a component. Component text
cannot do it: `report_components._MARKUP_SHAPES` refuses brackets at
construction, with the reason spelled out as "a bracket, which Rich reads as
console markup". That rule was written for this surface. Text arriving from
anywhere else — a store, an exception, a scraped sector name — still needs
`rich.markup.escape`, and every existing call site keeps it.
"""

import logging

from rich.console import Console

from boundless100x.output.report_components import (
    Caveat,
    Finding,
    MetricRow,
    ReadingLine,
    Unknown,
    component_surface,
)
from boundless100x.output.report_surfaces import (
    caveat_line,
    finding_line,
    is_body,
    metric_cells,
    reading_line,
)
from boundless100x.output.report_vocabulary import (
    DISCLOSURE_LINK_TEXT,
    NO_FIGURE_LABEL,
)

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


# ── The console surface (R14) ─────────────────────────────────────────────

# How much interpretation prose one metric row gets before it is clipped.
#
# Measured against the shapes that actually arrive rather than chosen round: a
# label runs to about 25 characters, a figure with its unit to about 12, and the
# longest shipped direction phrase — "a middle range is best; both ends are
# worse" — to 43. At 88 the whole row lands inside two wrapped lines on the
# 80-column default, which is what keeps a 57-metric listing skimmable. Raise it
# and the listing doubles in height; lower it and a reason stops being a
# sentence.
READING_BUDGET = 88

# What a clipped line ends on. One character, so it costs nothing, and it is
# the reader's cue that the research note has the rest.
ELLIPSIS = "…"

# Where a cut reads as a decision rather than as a rendering fault, best first
# by position rather than by kind — the *latest* boundary inside the budget
# wins, because preferring a full stop at column 20 over a clause break at
# column 70 throws away fifty characters to obey a ranking nobody asked for.
_BOUNDARIES = (". ", " — ", "; ", ", ")


def clip(text: str, budget: int = READING_BUDGET) -> str:
    """Prose, shortened to a budget — never a figure and never a direction.

    Only ever called on the *interpretation* half of a row and on R18's
    coverage clause. Both are written as a lead statement followed by an
    elaboration, so cutting at the last sentence or clause boundary inside the
    budget usually keeps a whole thought: a sector-mismatch reason keeps
    "A lender's assets are the loan book it earns on, not plant it has to
    sweat", and a coverage clause keeps its share and the bar it missed.

    Falling back to a word boundary, and only then to the raw budget, because
    cutting mid-word reads as a rendering bug rather than as a shortening — and
    a reader who cannot tell those apart stops trusting the ones that are
    deliberate.
    """
    value = str(text or "")
    if len(value) <= budget:
        return value

    window = value[:budget]
    # Half the budget as the floor: a boundary in the first few characters
    # leaves a fragment shorter than the ellipsis is worth.
    cut = max(window.rfind(marker) for marker in _BOUNDARIES)
    if cut < budget // 2:
        cut = window.rfind(" ")
    head = window[:cut].rstrip(" ,;:—-") if cut > 0 else ""
    return f"{head}{ELLIPSIS}" if head else f"{window}{ELLIPSIS}"


def metric_row_line(component: MetricRow) -> tuple[str, str, str]:
    """`metric_cells`'s three cells, with the middle one fitted to a terminal.

    Name and contribution come from the shared helper untouched. The reading is
    rebuilt rather than borrowed, and that rebuild is the single place the three
    surfaces are allowed to differ — see the module docstring for why a document
    and an eighty-column terminal cannot carry the same amount of prose.

    The order is the documents' order, deliberately: figure, then what it means,
    then which way is better. R12 is about those three being seen together, and
    a console that reordered them to save space would be solving the width
    problem by recreating the layout the whole plan exists to replace.
    """
    name, _document_reading, contribution = metric_cells(component)
    value = component.value or NO_FIGURE_LABEL
    body = component.reading if component.known else component.unknown.reason
    reading = f"{value} — {clip(body)}"
    if component.direction:
        reading = f"{reading} ({component.direction})"
    return name, reading, contribution


@component_surface("console")
class ConsoleComponents:
    """The component set as lines of Rich markup.

    Compare each method with its HTML and Markdown siblings in
    `report_surfaces.py`: the strings come from the same helpers and what
    differs is punctuation and style tags. That is R14 as something a reader of
    these two files can check.

    Every method returns one printable chunk, never a whole screen. Headings,
    tables, blank lines and ordering are the command's, exactly as the frame is
    the template's on the two document surfaces.
    """

    def render_reading(self, component: ReadingLine) -> str:
        """A section's score and its one line (R5), with R18's clause beneath.

        The subject is omitted for the documents' reason — the frame has
        already printed it as a heading or a table cell, and printing it twice
        would put "Size" immediately above "Size".
        """
        headline, body, qualifier = reading_line(component)
        tone = "" if component.known else "dim "
        lead = f"[bold]{headline}[/bold] — " if headline else ""
        out = f"[{tone}white]{lead}{body}[/{tone}white]" if tone else f"{lead}{body}"
        if qualifier:
            # Clipped for `metric_row_line`'s reason, and safe for the same
            # one: R18's clause opens on the share and the bar it missed, so
            # the two figures survive any budget that leaves a clause at all.
            out = f"{out}\n[yellow]{clip(qualifier)}[/yellow]"
        return out

    def render_finding(self, component: Finding) -> str:
        headline, body = finding_line(component)
        marker, colour = {
            "good": ("✓", "green"), "bad": ("✗", "red"),
        }.get(component.sentiment, ("•", "yellow"))
        line = f"[{colour}]{marker} {headline}[/{colour}]"
        return f"{line} [dim]— {body}[/dim]" if body else line

    def render_metric_row(self, component: MetricRow) -> str:
        """One metric, as one line.

        A line rather than a table row, and the choice is R12's rather than
        Rich's: `metric_cells` puts the figure and its interpretation in one
        cell precisely because "a figure in one column and its interpretation
        three columns away is the layout this whole plan exists to replace",
        and a console table would put them back in different columns as soon as
        the terminal was narrow enough to wrap.
        """
        name, reading, contribution = metric_row_line(component)
        style = "cyan" if component.known else "dim"
        return (
            f"[{style}]{name}[/{style}] — {reading} "
            f"[dim]· {contribution}[/dim]"
        )

    def render_unknown(self, component: Unknown) -> str:
        return f"[dim]? {component.subject} — {component.reason}[/dim]"

    def render_caveat(self, component: Caveat) -> str:
        colour = "yellow" if component.severity == "warning" else "dim"
        return f"[{colour}]! {caveat_line(component)}[/{colour}]"

    def render_disclosure(self, component) -> str:
        """R3 on a surface with nowhere to link to.

        A terminal has no anchors, so the reference cannot be a link — but R3's
        requirement is that the explanation is *reachable* and never inline,
        and both halves still hold: the flow gets the shared phrase and the name
        of the document the body is in, and the body itself renders only where a
        command deliberately asks for it.
        """
        if not is_body(component):
            return f"[dim]({DISCLOSURE_LINK_TEXT}: see the research note)[/dim]"
        return f"[bold]{component.title}[/bold]\n[dim]{component.body}[/dim]"
