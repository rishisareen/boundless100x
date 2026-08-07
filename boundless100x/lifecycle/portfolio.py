"""Concentration guardrails, counted in names because there are no rupees.

v05 §8.1 asks for per-name and per-sector limits, and §14.1 for a sleeve split
and per-lane tranche sizes. Both are stated in the roadmap as percentages of
capital — 10–15% per core name, 5% per re-rating name, 25–30% per sector — and
**this system has no capital to take a percentage of** (KTD8). The watchlist
records lanes, states, evidence and dates; it has never recorded an invested
amount, a tranche, or a cost basis, and this phase does not add one.

So the guardrail that can actually be checked is a **count**: how many
positioned names a lane holds, and how many of those sit in the same sector.
A count is a proxy for the roadmap's percentages, and an honest one only
because the sizing rules are themselves roughly equal-weight — eight core names
at 10–15% each is the same statement as "no more than eight". A percentage
computed from nothing would read as a measurement, which is the failure this
module exists to avoid: a fabricated denominator is worse than a coarse rule,
because only one of the two admits what it does not know.

Everything here therefore states its basis. `BASIS_COUNTS` travels in every
reading and every rendered line says "counts", so no surface can present these
as shares of a portfolio.

**Sector is resolved once per run and never stored.** It is read off each
ticker's already-fetched `metadata.sector` during `advance()`, exactly as the
deployment-pace modulator resolves its corpus median once before the ticker
loop. Persisting it to the watchlist would add a field that goes stale between
fetches and would then have to be reconciled with the corpus — a second source
of truth for a fact `raw_data/` already holds.

Two gaps are kept distinct, in the house style:

  * a lane with **no configured cap** counts its names and says the cap is
    unknown, rather than reporting a pass it cannot justify;
  * a name with **no sector reading** (a fetch predating the breadcrumb fix)
    counts toward its lane and is excluded from sector grouping, listed by
    name. It is never grouped with the other unknowns — that would either
    invent a correlation between unrelated companies or hide a real one inside
    a bucket named for a missing field.
"""

import logging
import math

from boundless100x.lifecycle import states as lifecycle_states

logger = logging.getLogger(__name__)

# ── Owner policy (v05 §8.1, §14.1) ──
# STARTING POINTS awaiting Phase 5 simulator evidence, mirroring `config.yaml`'s
# `portfolio:` block so a caller that supplies no config — a test, a future
# simulator — reads the same numbers the CLI does.

# §14.1's split between the compounder sleeve and the re-rating sleeve. Read
# here so the block has exactly one parser; nothing consumes it yet, because
# allocating between sleeves needs the rupee amounts this system does not hold.
DEFAULT_SLEEVE_SPLIT = {"core": 0.7, "rerating": 0.3}

# §4.4's per-lane tranche sizes: a core position built in thirds, a re-rating
# position in halves. Same status — declared, parsed, awaiting a consumer.
DEFAULT_TRANCHE_SIZE_PCT = {"core": 0.33, "rerating": 0.5}

# The count proxies for §4.4's 10–15% and 5% per-name caps, and §8.1's 25–30%
# sector cap. Counts, not percentages — see the module docstring.
DEFAULT_MAX_POSITIONED_PER_LANE = {"core": 8, "rerating": 5}
DEFAULT_MAX_POSITIONED_PER_SECTOR = 3

# Stated in every reading, so a renderer cannot present these as capital
# shares even by accident.
BASIS_COUNTS = "counts"

# Below this, a sector holds one name, and one name is not a concentration.
_MIN_GROUP = 2

# A lane the watchlist would never write. Defensive only — `WatchlistManager`
# validates the lane on load — but a positioned name whose lane could not be
# read must still be counted somewhere visible rather than dropped.
_UNKNOWN_LANE = "unknown"


def config_from(config: dict | None) -> dict:
    """Owner settings for the portfolio block, with the shipped defaults.

    Accepts either the whole pipeline config (`config_from(service.config)`) or
    the `portfolio:` block on its own — the `friction.config_from` idiom, for
    the same reason: both call sites are natural, and a caller who passes the
    wrong one would otherwise get silent defaults presented as the owner's own
    caps.
    """
    config = config or {}
    section = config.get("portfolio") if "portfolio" in config else config
    section = section or {}
    return {
        "sleeve_split": _mapping(
            section.get("sleeve_split"), DEFAULT_SLEEVE_SPLIT, "sleeve_split"
        ),
        "tranche_size_pct": _mapping(
            section.get("tranche_size_pct"),
            DEFAULT_TRANCHE_SIZE_PCT,
            "tranche_size_pct",
        ),
        "max_positioned_per_lane": _mapping(
            section.get("max_positioned_per_lane"),
            DEFAULT_MAX_POSITIONED_PER_LANE,
            "max_positioned_per_lane",
        ),
        "max_positioned_per_sector": _cap(
            section.get("max_positioned_per_sector"),
            DEFAULT_MAX_POSITIONED_PER_SECTOR,
            "max_positioned_per_sector",
        ),
    }


def _mapping(value, default: dict, name: str) -> dict:
    """A per-lane setting, or the shipped default with a warning.

    A scalar where a mapping belongs (`max_positioned_per_lane: 5`) is an easy
    edit to make and would otherwise raise deep inside the counting loop, taking
    the reading down for a config typo.
    """
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        logger.warning(
            f"Portfolio: {name} {value!r} is not a per-lane mapping — using "
            f"{default}"
        )
        return dict(default)
    return dict(value)


def _cap(value, default, name: str):
    """A whole-number cap, or the shipped default with a warning.

    `bool` is excluded because it is an `int` in Python, and `True` as a cap
    would silently mean "one name". A negative cap is meaningless and a
    fractional one is a percentage that wandered into a count field — the exact
    confusion this module is built to prevent — so both fall back loudly.
    Zero is allowed: "hold nothing in this lane" is a real instruction.
    """
    if value is None:
        return None if default is None else default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value != int(value)
    ):
        logger.warning(
            f"Portfolio: {name} {value!r} is not a whole-number count of names "
            f"— using {default}. These caps count POSITIONS, not percentages."
        )
        return default
    return int(value)


def unavailable(reason: str) -> dict:
    """The one shape for "the reading could not be built, and here is why".

    A missing concentration reading must never render as an empty one: zero
    breaches and no reading look identical in a summary line, and only one of
    them means the caps were checked.
    """
    return {"available": False, "reason": reason, "basis": BASIS_COUNTS}


def _sector_key(sector) -> str:
    """A grouping key that survives how Screener's breadcrumb happens to be written.

    Sector text comes from a scraped breadcrumb, so "Chemicals", "chemicals" and
    " Chemicals " are the same industry written three ways. Case and surrounding
    whitespace are folded; nothing else is, because a stemmer here would start
    merging industries on its own judgement.
    """
    if not isinstance(sector, str):
        return ""
    return " ".join(sector.split()).casefold()


def check_concentration(entries_with_sector, config: dict | None = None) -> dict:
    """Positioned-name counts per lane and per sector, against the configured caps.

    `entries_with_sector` is a list of `{ticker, lane, state, sector}` dicts the
    caller assembles — `advance()` seeds it from the **watchlist** and overlays
    the sector each successful analysis reported. That direction is the load
    bearing part: seeded from the run's outcomes instead, a ticker whose fetch
    failed would vanish from its lane's total, and a full lane would read as
    having room on the one day its data went missing.

    Only `probe` and `scale` count, read straight off `states.POSITIONED` rather
    than re-listed here, so the lifecycle keeps a single definition of which
    states hold capital.

    Returns a reading whose every number is a count of names:

        available, basis, positioned, positioned_tickers,
        lanes    {lane: {positioned, tickers, max, breach, note}}
        sectors  [{sector, tickers, count, max, breach}]  — groups of 2+ only
        unknown_sector, breaches, notes

    `sectors` reports every group of two or more because correlation is worth
    saying out loud before it is worth stopping; `breach` marks the subset that
    exceeds the configured cap.
    """
    settings = config_from(config)
    lane_caps = settings["max_positioned_per_lane"]
    sector_cap = settings["max_positioned_per_sector"]

    rows = []
    for raw in entries_with_sector or []:
        if not isinstance(raw, dict):
            logger.warning(f"Portfolio: skipping malformed entry {raw!r}")
            continue
        if raw.get("state") not in lifecycle_states.POSITIONED:
            continue
        rows.append({
            "ticker": str(raw.get("ticker") or "").strip(),
            "lane": raw.get("lane") or _UNKNOWN_LANE,
            "sector": raw.get("sector"),
        })

    lanes = _lane_counts(rows, lane_caps)
    sectors, unknown_sector = _sector_groups(rows, sector_cap)

    breaches = [lane["breach_note"] for lane in lanes.values() if lane["breach"]]
    breaches += [group["note"] for group in sectors if group["breach"]]

    notes = list(breaches)
    notes += [group["note"] for group in sectors if not group["breach"]]
    notes += [
        lane["note"] for lane in lanes.values()
        if lane["max"] is None and lane["positioned"]
    ]
    if unknown_sector:
        # Logged as well as reported: the fix is a refetch, and the owner can
        # only run one if the gap is visible from the run that noticed it.
        note = (
            f"{len(unknown_sector)} positioned name(s) carry no sector reading "
            f"({', '.join(unknown_sector)}) — excluded from the sector check. "
            f"Refetch to pick up the metadata."
        )
        logger.info(f"Portfolio: {note}")
        notes.append(note)

    return {
        "available": True,
        # Counts of names, never shares of capital (KTD8).
        "basis": BASIS_COUNTS,
        "positioned": len(rows),
        "positioned_tickers": sorted(row["ticker"] for row in rows),
        "lanes": lanes,
        "sectors": sectors,
        "unknown_sector": unknown_sector,
        "breaches": breaches,
        "notes": notes,
    }


def _lane_counts(rows: list[dict], lane_caps: dict) -> dict:
    """Positioned names per lane, against each lane's configured cap.

    Every lane with a configured cap appears even at zero: headroom is a reading
    too, and a lane missing from the table is indistinguishable from a lane
    nobody checked.
    """
    lanes = {}
    for lane in sorted(set(lane_caps) | {row["lane"] for row in rows}):
        tickers = sorted(row["ticker"] for row in rows if row["lane"] == lane)
        cap = _cap(lane_caps.get(lane), None, f"max_positioned_per_lane[{lane}]")
        breach = cap is not None and len(tickers) > cap
        lanes[lane] = {
            "lane": lane,
            "positioned": len(tickers),
            "tickers": tickers,
            "max": cap,
            "breach": breach,
            "note": (
                f"{lane} lane holds {len(tickers)} positioned name(s) and has "
                f"no cap configured — counted, not checked"
                if cap is None
                else f"{lane} lane holds {len(tickers)} of a maximum "
                     f"{cap} positioned name(s)"
            ),
            "breach_note": (
                f"{lane} lane holds {len(tickers)} positioned name(s) against a "
                f"cap of {cap} ({', '.join(tickers)}) — counts of names, not "
                f"a share of capital"
            ) if breach else "",
        }
    return lanes


def _sector_groups(rows: list[dict], sector_cap) -> tuple[list[dict], list[str]]:
    """Sectors holding two or more positioned names, and the names with no sector.

    Grouped across lanes on purpose: a correlated pair is correlated whichever
    sleeve bought it, and splitting the count by lane would let the same sector
    sit twice under two caps that each read as satisfied.
    """
    groups: dict[str, dict] = {}
    unknown: list[str] = []

    for row in rows:
        key = _sector_key(row["sector"])
        if not key:
            unknown.append(row["ticker"])
            continue
        group = groups.setdefault(
            key, {"sector": " ".join(str(row["sector"]).split()), "tickers": []}
        )
        group["tickers"].append(row["ticker"])

    reported = []
    for group in groups.values():
        tickers = sorted(group["tickers"])
        if len(tickers) < _MIN_GROUP:
            continue
        breach = sector_cap is not None and len(tickers) > sector_cap
        reported.append({
            "sector": group["sector"],
            "tickers": tickers,
            "count": len(tickers),
            "max": sector_cap,
            "breach": breach,
            "note": (
                f"{len(tickers)} positioned names in the {group['sector']} sector "
                f"({', '.join(tickers)}) exceeds the cap of {sector_cap} "
                f"— counts of names, not a share of capital"
                if breach
                else f"{len(tickers)} positioned names share the "
                     f"{group['sector']} sector ({', '.join(tickers)}) — "
                     f"correlated, within the cap of {sector_cap}"
            ),
        })

    # Largest group first: the one closest to a cap is the one worth reading.
    reported.sort(key=lambda g: (-g["count"], g["sector"].casefold()))
    return reported, sorted(unknown)


def would_breach(lane: str, sector, reading: dict | None) -> list[str]:
    """Why adding one more positioned name to this lane or sector breaks a cap.

    Empty means there is room. **The one statement of the question**, asked
    identically by the two places that need it: `reinvestment.propose_routing`
    deciding whether to advise deploying into a candidate, and `advance`
    deciding whether it may apply a transition that would take a position. Two
    copies would eventually disagree, and the disagreement would be invisible —
    a router that skips a candidate the transition path is happy to buy reads
    as a ranking quirk, not as a guardrail with two minds.

    Every figure consulted is a **count of positioned names**, never a share of
    capital; the module docstring argues why that is the only honest guardrail
    this system can compute.

    Three fail-closed cases, all saying the same thing in different words:
    absence must not read as headroom.

      * **A reading that could not be built blocks everything.** The
        alternative is committing capital into a lane whose occupancy is
        unknown.
      * **A lane the reading does not describe blocks.** Its occupancy is not
        zero, it is unmeasured.
      * **A lane with no configured cap blocks.** `_lane_counts` reports it
        honestly — `max: None`, "counted, not checked" — and that honesty is
        precisely what must not be read as room, or the one lane nobody had got
        round to configuring becomes the one lane capital can always flow into.
        Zero is a cap, not a gap: `_cap` allows it because "hold nothing in
        this lane" is a real instruction, and it blocks on the cap it breaches
        rather than on missing configuration.

    The sector half is deliberately partial and says so. `check_concentration`
    reports groups of two or more, so a candidate joining a sector that
    currently holds one positioned name is invisible here — that is the group
    size the cap is nowhere near. A name whose sector could not be read is
    invisible on the other side, for the reason the module docstring gives.
    """
    if not isinstance(reading, dict) or not reading.get("available"):
        detail = (reading or {}).get("reason", "no reading was produced")
        return [
            f"the concentration reading is unavailable ({detail}), so the "
            f"{lane} lane cannot be shown to have room — refused rather than "
            f"assumed"
        ]

    reasons = []
    lane_row = (reading.get("lanes") or {}).get(lane)
    if not isinstance(lane_row, dict):
        reasons.append(
            f"the concentration reading describes no {lane!r} lane, so its "
            f"occupancy is unknown — refused rather than assumed"
        )
    else:
        cap = lane_row.get("max")
        held = lane_row.get("positioned", 0)
        if cap is None:
            reasons.append(
                f"the {lane} lane holds {held} positioned name(s) and has no cap "
                f"configured (portfolio.max_positioned_per_lane[{lane}]) — there "
                f"is no limit to check one more against, so it is refused rather "
                f"than assumed"
            )
        elif held + 1 > cap:
            reasons.append(
                f"the {lane} lane already holds {held} of a maximum {cap} "
                f"positioned name(s) — one more would breach the cap "
                f"(counts of names, not a share of capital)"
            )

    key = _sector_key(sector)
    if key:
        for group in reading.get("sectors") or []:
            cap = group.get("max")
            if _sector_key(group.get("sector")) != key or cap is None:
                continue
            if group.get("count", 0) + 1 > cap:
                reasons.append(
                    f"the {group['sector']} sector already holds "
                    f"{group['count']} positioned name(s) against a cap of "
                    f"{cap} ({', '.join(group.get('tickers') or [])}) — counts "
                    f"of names, not a share of capital"
                )
    return reasons


def describe(reading: dict | None) -> str:
    """One line a person can read, with the basis attached.

    The basis is in the string rather than in a caption beside it, for the
    reason `friction.describe` states: this line gets copied into logs and
    summaries where a caption does not follow it, and a bare "core 3/8" would
    read as a percentage to anyone who met it there.
    """
    if not reading:
        return "concentration: no reading"
    if not reading.get("available"):
        return (
            f"concentration: unavailable — "
            f"{reading.get('reason', 'no reason given')}"
        )

    lanes = ", ".join(
        f"{lane['lane']} {lane['positioned']}/"
        f"{'—' if lane['max'] is None else lane['max']}"
        for lane in reading["lanes"].values()
    ) or "no positions"
    breaches = reading.get("breaches") or []
    verdict = (
        f"{len(breaches)} cap breach(es)" if breaches else "no cap breached"
    )
    return f"concentration by positioned-name counts: {lanes} — {verdict}"
