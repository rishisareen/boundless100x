"""Markup for the closed component set — one renderer per surface (R13, R14).

U9 built the components as *data*: a finding, a metric row, a reading, a
disclosure, an unknown-with-reason, a caveat, each carrying no markup at all.
This module is where markup is finally added, twice, and the shape of the file
is the argument for R14 holding.

**The composition is shared; only the markup differs.** `metric_cells`,
`reading_line` and `finding_line` below turn a component into the strings a
reader sees, and both renderer classes call them. Neither class decides what a
row *says* — it decides whether the row is a `<tr>` or a `| … |`. That is the
difference between "R14 is a property of the code" and "R14 is two templates
somebody matched carefully once": a phrase added to one surface and forgotten
on the other is not a diff you can make here, because there is one place to
add it.

**Registration is enforced at import time.** `@component_surface` refuses a
class missing any `render_<member>`, so a seventh component — or a member
somebody forgot — fails on the line defining the class rather than as a section
that silently exists in one report and not the other. Only `html` and
`markdown` register here; `console` is the CLI's, and U11 owns it.

**Escaping is per surface and asymmetric on purpose.** HTML escapes everything,
because a company name really can contain an ampersand. Markdown escapes
nothing, because `guard_text` already refuses every character that would
matter — pipes, backticks, brackets, bold runs, leading block markers, control
characters — at the point the component was constructed. A table cell here
cannot break its table, and the reason is upstream rather than in this file.
"""

from __future__ import annotations

from markupsafe import escape

from boundless100x.output.report_components import (
    Caveat,
    Disclosure,
    Finding,
    MetricRow,
    ReadingLine,
    Unknown,
    component_surface,
)
from boundless100x.output.report_vocabulary import (
    DISCLOSURE_LINK_TEXT,
    NO_FIGURE_LABEL,
    UNWEIGHTED_CONTRIBUTION,
)

# How an anchor becomes an id. Prefixed so a metric id can never collide with
# an id the page already uses for something else.
ANCHOR_PREFIX = "explain-"

# The three columns a metric row occupies, in order. Named here so both
# surfaces head their tables identically and a test can assert they do.
ROW_HEADERS: tuple[str, str, str] = ("Metric", "Reading", "Contribution")


def anchor_id(anchor: str) -> str:
    return f"{ANCHOR_PREFIX}{anchor}"


# ── What a component says, before anything says it in markup ──────────────


def reading_line(component: ReadingLine) -> tuple[str, str, str]:
    """`(headline, body, qualifier)` — a section's one line (R5), R18 included.

    The headline is the score, the body is the reading or the reason there is
    none, and the qualifier is R18's coverage clause or an empty string. The
    `subject` is deliberately absent: it is the section's title, which the
    surrounding frame has already printed as a heading on both surfaces, and
    printing it twice would put "Size" on the page immediately above "Size".
    """
    body = component.text if component.known else component.unknown.sentence
    return component.headline, body, component.qualifier


def metric_cells(component: MetricRow) -> tuple[str, str, str]:
    """`(name, reading, contribution)` — the three things a row has to say.

    Every one of the three is non-empty by construction, which is R4 surviving
    contact with a table. The two cells that could have been blank are the ones
    that get a phrase instead of a dash: a metric that produced nothing says
    so, and a metric that carries no weight says that rather than leaving the
    reader to infer it from white space.

    The value and the reading share a cell because R12 is about them being seen
    together — a figure in one column and its interpretation three columns away
    is the layout this whole plan exists to replace.
    """
    value = component.value or NO_FIGURE_LABEL
    body = component.reading if component.known else component.unknown.reason
    reading = f"{value} — {body}"
    if component.direction:
        reading = f"{reading} ({component.direction})"

    scored = [part for part in (component.score, component.weight) if part]
    contribution = " · ".join(scored) if scored else UNWEIGHTED_CONTRIBUTION
    return component.label, reading, contribution


def finding_line(component: Finding) -> tuple[str, str]:
    """`(headline, body)`. A finding with no body is a badge, which is allowed."""
    return component.headline, component.text


def caveat_line(component: Caveat) -> str:
    if component.subject:
        return f"{component.subject} — {component.text}"
    return component.text


def is_body(component) -> bool:
    """Whether this disclosure carries the explanation or only points at it.

    `Disclosure` and `DisclosureRef` share a kind (they are the same member of
    the closed set seen from two sides), so one handler renders both and this
    is how it tells them apart.
    """
    return isinstance(component, Disclosure)


# ── HTML ──────────────────────────────────────────────────────────────────


@component_surface("html")
class HtmlComponents:
    """The component set as HTML fragments.

    Every method returns a fragment, never a whole document: the frame — head,
    headings, table wrappers, the appendix — is the template's, and what goes
    inside a section is this class's. Splitting it there is what keeps the
    template thin enough that its Markdown twin is recognisably the same file.
    """

    def render_reading(self, component: ReadingLine) -> str:
        headline, body, qualifier = reading_line(component)
        state = "reading" if component.known else "reading unknown"
        lead = (
            f'<span class="headline">{escape(headline)}</span> — '
            if headline else ""
        )
        out = [f'<p class="{state}">{lead}{escape(body)}</p>']
        if qualifier:
            out.append(f'<p class="qualifier">{escape(qualifier)}</p>')
        return "\n".join(out)

    def render_finding(self, component: Finding) -> str:
        headline, body = finding_line(component)
        detail = f'<p class="detail">{escape(body)}</p>' if body else ""
        return (
            f'<div class="finding {component.sentiment}">'
            f'<p class="headline">{escape(headline)}</p>{detail}</div>'
        )

    def render_metric_row(self, component: MetricRow) -> str:
        name, reading, contribution = metric_cells(component)
        ref = (
            self.render_disclosure(component.disclosure)
            if component.disclosure else ""
        )
        state = "known" if component.known else "unknown"
        return (
            f'<tr class="{state}">'
            f'<th scope="row">{escape(name)}{ref}</th>'
            f'<td>{escape(reading)}</td>'
            f'<td class="contribution">{escape(contribution)}</td></tr>'
        )

    def render_unknown(self, component: Unknown) -> str:
        return (
            f'<li class="unknown"><strong>{escape(component.subject)}</strong>'
            f' — {escape(component.reason)}</li>'
        )

    def render_caveat(self, component: Caveat) -> str:
        return (
            f'<p class="caveat {component.severity}">'
            f'{escape(caveat_line(component))}</p>'
        )

    def render_disclosure(self, component) -> str:
        """A body in the deferred section, or a link to one from the flow (R3).

        The link carries the title as its tooltip and the shared phrase as its
        text, so the reading flow never contains a word of the explanation —
        only a way to reach it.
        """
        target = anchor_id(component.anchor)
        if not is_body(component):
            return (
                f' <a class="explain" href="#{target}" '
                f'title="{escape(component.title)}">{DISCLOSURE_LINK_TEXT}</a>'
            )
        return (
            f'<div class="disclosure" id="{target}">'
            f'<h4>{escape(component.title)}</h4>'
            f'<p>{escape(component.body)}</p></div>'
        )


# ── Markdown ──────────────────────────────────────────────────────────────


@component_surface("markdown")
class MarkdownComponents:
    """The same components, as Markdown.

    Compare each method with its HTML sibling: the strings come from the same
    three helpers above, and what differs is punctuation. That is R14 — "the
    same content from the same declarations" — as something a reader of this
    file can check rather than a claim in a docstring.
    """

    def render_reading(self, component: ReadingLine) -> str:
        headline, body, qualifier = reading_line(component)
        lead = f"**{headline}** — " if headline else ""
        out = [f"{lead}{body}"]
        if qualifier:
            out.append("")
            out.append(f"*{qualifier}*")
        return "\n".join(out)

    def render_finding(self, component: Finding) -> str:
        headline, body = finding_line(component)
        marker = {"good": "✓", "bad": "✗"}.get(component.sentiment, "•")
        return f"- {marker} **{headline}**" + (f" — {body}" if body else "")

    def render_metric_row(self, component: MetricRow) -> str:
        name, reading, contribution = metric_cells(component)
        if component.disclosure:
            target = anchor_id(component.disclosure.anchor)
            name = f"{name} ([{DISCLOSURE_LINK_TEXT}](#{target}))"
        return f"| {name} | {reading} | {contribution} |"

    def render_unknown(self, component: Unknown) -> str:
        return f"- **{component.subject}** — {component.reason}"

    def render_caveat(self, component: Caveat) -> str:
        return f"> {caveat_line(component)}"

    def render_disclosure(self, component) -> str:
        target = anchor_id(component.anchor)
        if not is_body(component):
            return f"([{DISCLOSURE_LINK_TEXT}](#{target}))"
        # A raw anchor tag, because Markdown has no syntax for one and the link
        # a row carries has to land somewhere. Every renderer of this file
        # passes HTML through, and a reader of the plain text sees an empty
        # tag rather than a broken sentence.
        return (
            f'<a id="{target}"></a>\n'
            f"**{component.title}** — {component.body}"
        )


__all__ = [
    "ANCHOR_PREFIX",
    "ROW_HEADERS",
    "HtmlComponents",
    "MarkdownComponents",
    "anchor_id",
    "caveat_line",
    "finding_line",
    "is_body",
    "metric_cells",
    "reading_line",
]
