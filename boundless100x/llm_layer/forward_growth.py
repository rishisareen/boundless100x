"""Forward-growth extraction: the content gate, the boundary validator, the cache.

Guidance, capacity plans and market-size statements live in annual-report
prose, and no amount of arithmetic over the financials will produce them. This
module is what turns that prose into structured data the compute engine can
read offline — and, mostly, what stops it turning the *wrong* prose into
confident nonsense.

Three guards stand between a PDF and a stored forward signal, and each catches
something the others cannot.

**Provenance gates the input** (KTD4). A section tagged `fallback` holds
first-N-pages text rather than the section it names, so it is never submitted.
Cheaper, and no tokens spent on a chairman's letter that cannot answer the
question. Provenance is tagged per *section* while a report year usually
carries a mix — 16 of 29 real report-years do — so every stored entry records
the section it came from, or a promises-kept reading could be built from a
chairman's letter while MD&A was never read.

**A content gate stands behind provenance** (KTD9), because `found` is a claim
rather than a guarantee. Heading detection over arbitrary filer layouts will
never be exact: a review of the corpus found 8 of 18 `found` MD&A slices were
actually auditor's reports, governance, CSR or HR text. Phase 0 fixed the root
cause — a heading must now open its line and not be continued in lowercase —
and precision rose from ~56% to ~85%, but at least one residual survives: a
bare "Management Discussion and Analysis" line immediately followed by
governance prose, structurally indistinguishable from a real heading. Phase 0
could tolerate that residue because its fallback text was only Pass 1
background. This cannot: the extractor mines whatever it is handed, and
governance prose yields well-formed, confident, wrong guidance. So a `found`
section must additionally *look like* the section it claims to be, or it is
downgraded to **`suspect`** and treated exactly as `fallback`. The bucket is
reported rather than folded into `fallback`, because how often the tag was
wrong is the number that tells you whether the gate is working.

**Grounding stands behind shape** (KTD3). Type-checking cannot distinguish a
well-formed reading from a well-formed fabrication — every field is the right
type either way, and the result becomes a forward signal. Because each entry
must carry the verbatim source sentence, the validator additionally verifies
that sentence is a literal substring of the text actually submitted for that
year, and that the entry's own value and period appear inside it. One string
search turns "auditable if a reader bothers" into "verified before storage",
and closes fabrication and any instruction-like text embedded in a filing in
the same move.

Note what none of these can do for the other: KTD3's grounding *passes* on a
suspect slice, because the quoted sentence really is in the submitted text —
the submitted text is genuinely the auditor's report. Only content can tell
those apart, which is why KTD9 exists as its own guard.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

from boundless100x import forward_growth_schema as schema

logger = logging.getLogger(__name__)

# Re-exported from the shared contract so this module reads self-contained;
# the compute engine reads the same three values without importing llm_layer.
FOUND = schema.FOUND
SUSPECT = schema.SUSPECT
FALLBACK = schema.FALLBACK

PROMPT_NAME = "forward_growth_extraction.txt"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Only sections some sub-metric can actually read are worth paying for.
# `governance` is extracted by Phase 0 and answers none of the three questions
# below, so it is gated for reporting but never submitted.
EXTRACTABLE_SECTIONS = tuple(sorted(
    {name for names in schema.REQUIRED_SECTIONS.values() for name in names}
))

DEFAULT_CHAR_BUDGET = 6000
# The gate reads the opening of a slice. A wrong-section slice is wrong from
# its first line; reading further would only let a passing mention of "outlook"
# ten pages in rescue an auditor's report.
DEFAULT_SCAN_CHARS = 3000
# How many distinct markers a slice must show. MD&A has a mandated structure
# (SEBI LODR Schedule V(B)) so two is a low bar for a genuine one; a chairman's
# letter is looser prose, but one marker would admit almost anything.
DEFAULT_MIN_MARKERS = {"mdna": 2, "chairman": 2, "governance": 2}

# Positive markers, written against what each section is *required* to contain
# rather than against what the false positives happened to look like — the
# latter only ever catches the failures already seen.
SECTION_MARKERS: dict[str, tuple[re.Pattern, ...]] = {
    "mdna": tuple(re.compile(p, re.I) for p in (
        r"industry\s+structure\s+and\s+development",
        r"opportunit\w+\s+and\s+threats",
        r"segment[\s–—-]*wise",
        r"\boutlook\b",
        r"risks?\s+and\s+concerns",
        r"internal\s+control\s+systems?",
        r"econom(?:y|ic)\s+(?:review|overview|scenario)",
        r"industry\s+(?:review|overview|scenario)",
        r"key\s+financial\s+ratios",
        r"financial\s+performance\s+with\s+respect\s+to",
    )),
    "chairman": tuple(re.compile(p, re.I) for p in (
        r"dear\s+(?:share\s*holders?|members|stakeholders|friends)",
        r"it\s+gives\s+me|i\s+am\s+(?:pleased|delighted|happy)",
        r"on\s+behalf\s+of\s+the\s+board",
        r"your\s+company",
        r"yours\s+sincerely|warm\s+regards",
    )),
    "governance": tuple(re.compile(p, re.I) for p in (
        r"philosophy\s+on\s+corporate\s+governance",
        r"composition\s+of\s+the\s+board",
        r"board\s+of\s+directors",
        r"audit\s+committee",
        r"nomination\s+and\s+remuneration",
        r"board\s+met\b",
        r"general\s+(?:body\s+)?meeting",
    )),
}

# Text that, near the start of a slice, means it is something else entirely —
# whatever else the slice may contain. These are exactly the categories the
# corpus review found masquerading as MD&A.
_AUDIT_AND_GOVERNANCE = tuple(re.compile(p, re.I) for p in (
    r"terms\s+of\s+reference\s+of\s+the\s+\w+\s+committee",
    r"key\s+audit\s+matters",
    r"we\s+have\s+audited",
    r"independent\s+auditor",
    r"auditor'?s?\s+report",
    r"report\s+on\s+the\s+audit\s+of",
    r"corporate\s+social\s+responsibility\s+(?:policy|committee|activities)",
))

SECTION_DISQUALIFIERS: dict[str, tuple[re.Pattern, ...]] = {
    "mdna": _AUDIT_AND_GOVERNANCE,
    "chairman": _AUDIT_AND_GOVERNANCE,
    # A governance report legitimately discusses committee terms of reference;
    # only an auditor's report masquerading as one is disqualifying.
    "governance": tuple(re.compile(p, re.I) for p in (
        r"key\s+audit\s+matters", r"we\s+have\s+audited",
    )),
}


# ── The content gate ───────────────────────────────────────────────────────


def _gate_one(name: str, text: str, scan_chars: int, min_markers: int) -> tuple[bool, str]:
    """Whether a slice looks like the section it claims to be, and why not."""
    opening = (text or "")[:scan_chars]
    if not opening.strip():
        return False, "empty slice"

    blocked = [
        pattern.pattern
        for pattern in SECTION_DISQUALIFIERS.get(name, ())
        if pattern.search(opening)
    ]
    if blocked:
        return False, f"opens with {name}-disqualifying text: {', '.join(blocked)}"

    hits = [
        pattern.pattern
        for pattern in SECTION_MARKERS.get(name, ())
        if pattern.search(opening)
    ]
    if len(hits) < min_markers:
        return False, (
            f"only {len(hits)} of the {min_markers} markers a {name} section "
            f"should show in its opening {scan_chars} chars"
        )
    return True, ""


def gate_sections_with_reasons(
    sections_by_year: dict,
    scan_chars: int = DEFAULT_SCAN_CHARS,
    min_markers: dict | None = None,
    enabled: bool = True,
) -> tuple[dict, dict]:
    """Three-valued provenance per section, plus why anything was downgraded.

    Runs on the extraction path rather than in the fetcher on purpose: Phase 0's
    `.sections.json` sidecars stay exactly as they are, and the gate can be
    tuned — or switched off — without refetching twenty annual reports.
    """
    minimums = {**DEFAULT_MIN_MARKERS, **(min_markers or {})}

    gated: dict[str, dict] = {}
    reasons: dict[str, dict] = {}
    for year, sections in (sections_by_year or {}).items():
        if not isinstance(sections, dict):
            continue
        gated[year], reasons[year] = {}, {}
        for name, section in sections.items():
            provenance = (section or {}).get("provenance", FALLBACK)
            if provenance != FOUND or not enabled:
                # Only a `found` tag is a claim. A `fallback` slot never
                # claimed to be the section, so there is nothing to disprove.
                gated[year][name] = provenance
                continue

            genuine, why = _gate_one(
                name, (section or {}).get("text", ""), scan_chars,
                minimums.get(name, 2),
            )
            gated[year][name] = FOUND if genuine else SUSPECT
            if not genuine:
                reasons[year][name] = why
                logger.info(
                    f"Content gate: {year} {name} downgraded found -> suspect ({why})"
                )
    return gated, reasons


def gate_sections(sections_by_year: dict, **kwargs) -> dict:
    """Three-valued provenance per section. See `gate_sections_with_reasons`."""
    return gate_sections_with_reasons(sections_by_year, **kwargs)[0]


def build_submission(
    sections_by_year: dict,
    gated: dict,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> dict:
    """`{year: {section: text}}` — only what may be sent, capped.

    A year with no usable section is absent rather than present-and-empty, so
    "nothing to submit" is a property of the payload rather than something
    every caller has to check for separately.
    """
    payload: dict[str, dict] = {}
    for year, sections in (sections_by_year or {}).items():
        if not isinstance(sections, dict):
            continue
        usable = {
            name: (sections[name] or {}).get("text", "")[:char_budget]
            for name in EXTRACTABLE_SECTIONS
            if name in sections and gated.get(year, {}).get(name) == FOUND
        }
        usable = {name: text for name, text in usable.items() if text.strip()}
        if usable:
            payload[year] = usable
    return payload


# ── Prompt assembly ────────────────────────────────────────────────────────


def vocabulary_prompt_block() -> str:
    """The closed field list, rendered for the prompt.

    Phase 1's lesson, applied: asked for an id without a menu, a model invents
    plausible ones. Sending the vocabulary is what makes boundary validation
    something other than a rejection machine.
    """
    lines = ["Entry kinds and their fields:"]
    for kind in schema.ENTRY_KINDS:
        fields = schema.FIELDS[kind]
        lines.append(f"  {kind}:")
        lines.append(f"    required: {', '.join(fields['required'])}")
        if fields["optional"]:
            lines.append(f"    optional: {', '.join(fields['optional'])}")

    lines.append("")
    lines.append("Valid values for guidance.metric (nothing else is accepted):")
    for metric_id, spec in sorted(schema.GUIDANCE_METRICS.items()):
        lines.append(f"  {metric_id} — reported as {spec['unit']}")
    return "\n".join(lines)


def render_report_text(submission: dict) -> str:
    """The submitted sections, labelled so the model can attribute each entry."""
    blocks = []
    for year in sorted(submission, reverse=True):
        blocks.append(f"=== REPORT YEAR {year} ===")
        for name, text in sorted(submission[year].items()):
            blocks.append(f"--- SECTION: {name} ---")
            blocks.append(text)
    return "\n".join(blocks)


def prompt_template() -> str:
    return (PROMPTS_DIR / PROMPT_NAME).read_text()


def prompt_digest() -> str:
    """Fingerprint of the prompt file.

    Part of the sidecar version: a prompt change that silently reused stale
    extractions would be indistinguishable from the new prompt working.
    """
    try:
        return hashlib.sha256(prompt_template().encode()).hexdigest()[:12]
    except OSError:
        return "unavailable"


# ── Boundary validation ────────────────────────────────────────────────────


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number_appears(sentence: str, value) -> bool:
    """Whether a number is written somewhere in its own quoted sentence.

    Filings write `1,500` and models return `1500`; both are the same claim.
    Grouping separators are stripped from the haystack rather than guessed at
    in the needle, and both the integer and decimal renderings are tried.
    """
    haystack = re.sub(r"(?<=\d)[,\s](?=\d)", "", sentence or "")
    candidates = {f"{float(value):.10g}"}
    if float(value).is_integer():
        candidates.add(str(int(value)))
    return any(candidate in haystack for candidate in candidates)


def _period_appears(sentence: str, period) -> bool:
    """Whether a target period's year is named in its own quoted sentence.

    Grounds on the year rather than the exact string: a filing writes `FY26`,
    `FY2026`, or `financial year 2026` for one period, and demanding a literal
    match would discard correct readings far more often than fabricated ones.
    """
    text = str(period or "")
    years = re.findall(r"\d{2,4}", text)
    if not years:
        return False
    haystack = str(sentence or "")
    for year in years:
        if year in haystack:
            return True
        if len(year) == 4 and year[2:] in haystack:
            return True
    return False


def _validate_entry(
    kind: str, entry, year: str, submitted_sections: dict, discarded: list
) -> dict | None:
    """One entry, or None with the reason recorded."""
    where = f"{year}.{kind}"

    def drop(reason):
        discarded.append({"where": where, "reason": reason})
        logger.warning(f"Forward-growth entry discarded ({where}): {reason}")
        return None

    if not isinstance(entry, dict):
        return drop(f"entry is {type(entry).__name__}, expected an object")

    fields = schema.FIELDS[kind]
    allowed = set(fields["required"]) | set(fields["optional"])
    unknown = sorted(set(entry) - allowed)
    if unknown:
        discarded.append({
            "where": where,
            "reason": f"field(s) outside the declared set stripped: {', '.join(unknown)}",
        })
        logger.warning(f"Forward-growth fields stripped ({where}): {unknown}")
    kept = {key: value for key, value in entry.items() if key in allowed}

    missing = [name for name in fields["required"] if kept.get(name) is None]
    if missing:
        return drop(f"missing required field(s): {', '.join(missing)}")

    section = kept["section"]
    if section not in submitted_sections:
        return drop(
            f"section {section!r} was never submitted for {year} "
            f"(sent: {', '.join(sorted(submitted_sections)) or 'none'})"
        )

    sentence = kept["source_sentence"]
    if not isinstance(sentence, str) or sentence not in submitted_sections[section]:
        return drop(
            "source_sentence is not a literal substring of the submitted "
            f"{section} text for {year} — the claim is not in the document"
        )

    if kind == schema.GUIDANCE:
        if kept["metric"] not in schema.GUIDANCE_METRICS:
            return drop(
                f"guidance metric {kept['metric']!r} is outside the closed set "
                f"({', '.join(sorted(schema.GUIDANCE_METRICS))})"
            )
        if not _is_number(kept["target_value"]):
            return drop(f"target_value {kept['target_value']!r} is not a number")
        if "target_value_high" in kept and not _is_number(kept["target_value_high"]):
            return drop(f"target_value_high {kept['target_value_high']!r} is not a number")
        if not _number_appears(sentence, kept["target_value"]):
            return drop("target_value does not appear in its own source_sentence")
        if not _period_appears(sentence, kept["target_period"]):
            return drop("target_period does not appear in its own source_sentence")

    elif kind == schema.CAPEX:
        if not _is_number(kept["amount_inr_cr"]):
            return drop(f"amount_inr_cr {kept['amount_inr_cr']!r} is not a number")
        if not _number_appears(sentence, kept["amount_inr_cr"]):
            return drop("amount_inr_cr does not appear in its own source_sentence")
        if not _period_appears(sentence, kept["commissioning_year"]):
            return drop("commissioning_year does not appear in its own source_sentence")

    elif kind == schema.TAM:
        if not _is_number(kept["market_size_inr_cr"]):
            return drop(f"market_size_inr_cr {kept['market_size_inr_cr']!r} is not a number")
        if not _number_appears(sentence, kept["market_size_inr_cr"]):
            return drop("market_size_inr_cr does not appear in its own source_sentence")

    return kept


def validate_extraction(raw, submission: dict, gated: dict) -> dict:
    """Turn a raw model response into storable entries, or into nothing.

    Deliberately defensive throughout, exactly as `checkpoints.record_from_pass2`
    is and for the same reason: `_parse_json_response` performs no schema
    validation of any kind, so a malformed, truncated, or simply older response
    reaches here unchecked and every shape must degrade rather than raise.

    Returns `{"years": {year: {"sections", "guidance", "capex", "tam"}},
    "discarded": [...]}`. Every submitted year is present even when it yielded
    nothing, carrying the gated provenance of each of its sections — that is
    what lets a sub-metric say *why* it is indeterminate rather than only that
    it is.
    """
    discarded: list[dict] = []

    years: dict[str, dict] = {}
    for year, sections in submission.items():
        years[year] = {
            "sections": dict(gated.get(year, {})),
            **{kind: [] for kind in schema.ENTRY_KINDS},
        }

    payload = raw.get("years") if isinstance(raw, dict) else None
    if not isinstance(payload, dict):
        if raw is not None:
            logger.warning(
                "Forward-growth response carried no 'years' mapping — "
                "treating the extraction as empty"
            )
        return {"years": years, "discarded": discarded}

    for year, kinds in payload.items():
        year = str(year)
        if year not in years:
            discarded.append({
                "where": year,
                "reason": f"report year {year} was never submitted",
            })
            logger.warning(f"Forward-growth: response named unsubmitted year {year}")
            continue
        if not isinstance(kinds, dict):
            discarded.append({
                "where": year,
                "reason": f"year payload is {type(kinds).__name__}, expected an object",
            })
            continue

        for kind, entries in kinds.items():
            if kind not in schema.ENTRY_KINDS:
                discarded.append({
                    "where": year,
                    "reason": f"entry kind {kind!r} is outside the declared set "
                              f"({', '.join(schema.ENTRY_KINDS)})",
                })
                logger.warning(f"Forward-growth: unknown entry kind {kind!r} in {year}")
                continue
            if not isinstance(entries, list):
                discarded.append({
                    "where": f"{year}.{kind}",
                    "reason": f"expected a list, got {type(entries).__name__}",
                })
                continue

            for entry in entries:
                validated = _validate_entry(
                    kind, entry, year, submission[year], discarded
                )
                if validated is not None:
                    years[year][kind].append(validated)

    kept = sum(len(years[y][k]) for y in years for k in schema.ENTRY_KINDS)
    logger.info(
        f"Forward-growth extraction: {kept} entries kept, {len(discarded)} discarded"
    )
    return {"years": years, "discarded": discarded}


# ── Sidecar cache ──────────────────────────────────────────────────────────


def _source_digest(submission: dict) -> dict:
    """Per-year fingerprint of the exact text submitted.

    Digesting the submitted payload rather than the raw section text means the
    char budget and the gate's own verdict are covered too: raise the budget or
    tighten the gate and the cache invalidates, because a different question
    was asked.
    """
    digests = {}
    for year, sections in submission.items():
        canonical = json.dumps(sections, sort_keys=True)
        digests[year] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return digests


def _version_block(submission: dict, model: str) -> dict:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "prompt_digest": prompt_digest(),
        "model": model,
        "source_digest": _source_digest(submission),
    }


def read_sidecar(path, submission: dict, model: str) -> dict | None:
    """Cached extraction for exactly this question, or None.

    An annual report does not change after filing, so without this the
    corpus-wide validation sweep and every re-analysis pay again for identical
    text. But "identical" has to mean the whole question — source text, field
    schema, prompt, and model id — or a prompt change would silently reuse the
    old prompt's answers.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Ignoring unreadable forward-growth sidecar {target.name}: {e}")
        return None

    if not isinstance(stored, dict) or stored.get("version") != _version_block(
        submission, model
    ):
        logger.info(
            f"Forward-growth sidecar {target.name} is stale — re-extraction required"
        )
        return None
    years = stored.get("years")
    return years if isinstance(years, dict) else None


def write_sidecar(path, years: dict, submission: dict, model: str) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"version": _version_block(submission, model), "years": years},
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning(f"Could not cache forward-growth extraction at {target}: {e}")
