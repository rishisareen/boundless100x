"""Boundless100x CLI — SQGLP Financial Research System."""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
import typer

load_dotenv()
from rich.table import Table

from boundless100x.cli_common import console, setup_logging

app = typer.Typer(
    name="boundless100x",
    help="SQGLP Financial Research System for Indian Markets",
    no_args_is_help=True,
)
logger = logging.getLogger(__name__)

# The lifecycle surface — `watchlist`, its `queue` subgroup, and the helpers
# they render through — lives in `cli_lifecycle.py`. Registered here the way
# `corpus` is, and the helpers are re-exported below so that every name this
# module and the test suite already published still resolves from it.
from boundless100x.cli_lifecycle import (  # noqa: E402
    _evidence_cell,
    _lane_context_if_tracked,
    _print_capped_transitions,
    _print_concentration,
    _print_exit_friction,
    _print_lane_status,
    _print_routing_result,
    _print_routing_snapshot,
    _record_checkpoints_if_tracked,
    _snapshot_age,
    watchlist_app,
)

app.add_typer(watchlist_app, name="watchlist")


@app.command()
def analyze(
    ticker: str = typer.Argument(help="NSE symbol (e.g., ASTRAL)"),
    bse_code: str = typer.Option(None, help="BSE scrip code"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM analysis"),
    deep: bool = typer.Option(False, "--deep", help="Use Opus for Pass 1 & 2 (~5x LLM cost, deeper analysis)"),
    formats: str = typer.Option("html,md,json", help="Output formats (comma-separated)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
):
    """Run full SQGLP analysis pipeline for a company."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService
    from boundless100x.output.report_generator import ReportGenerator

    mode = " [bold magenta](DEEP — Opus)[/bold magenta]" if deep else ""
    console.print(f"\n[bold blue]Boundless100x SQGLP Analysis: {ticker}[/bold blue]{mode}\n")

    svc = Boundless100xService()
    result = svc.analyze(
        ticker=ticker,
        bse_code=bse_code,
        use_llm=not no_llm,
        deep=deep,
    )

    # Print summary to console
    _print_scores(result, svc)

    if result.llm_analysis and not result.llm_analysis.get("skipped"):
        _print_llm_summary(result)

    # A thesis is only worth writing down if something later checks it. When
    # this company is tracked, the structured monitorables Pass 2 produced
    # become the checkpoints `watchlist advance` tests each quarter.
    _record_checkpoints_if_tracked(ticker, result)

    # A tracked company's report carries its lane, its gates and what a
    # position is modeled to keep after friction; an untracked one renders
    # exactly as it did before that section existed (KTD9).
    lane_context = _lane_context_if_tracked(ticker, result, svc)
    _print_lane_status(lane_context)

    # Generate reports
    fmt_list = [f.strip() for f in formats.split(",")]
    generator = ReportGenerator()
    report_dir = generator.generate(
        result, formats=fmt_list, lane_context=lane_context
    )

    console.print(f"\n[bold green]Reports saved to:[/bold green] {report_dir}")

    if result.errors:
        console.print(f"\n[bold yellow]Warnings ({len(result.errors)}):[/bold yellow]")
        for e in result.errors:
            console.print(f"  [yellow]! {e}[/yellow]")


@app.command()
def compute(
    ticker: str = typer.Argument(help="NSE symbol"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Compute metrics only (no peers, no LLM, no reports)."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService

    console.print(f"\n[bold blue]Computing metrics for {ticker}[/bold blue]\n")

    svc = Boundless100xService()
    result = svc.analyze_quick(ticker)

    _print_scores(result, svc)

    # Print all metrics
    table = Table(title="Computed Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Flags", style="yellow")

    for mid, mr in sorted(result.metrics.items()):
        if mr.ok:
            val = mr.value
            if isinstance(val, float):
                val = f"{val:.2f}"
            flags = ", ".join(mr.flags) if mr.flags else ""
            table.add_row(mid, str(val), flags)
        else:
            table.add_row(mid, f"[red]ERR: {mr.error}[/red]", "")

    console.print(table)


@app.command()
def backtest(
    output_dir: str = typer.Option(
        str(Path(__file__).parent / "output" / "backtests"),
        help="Where to write the report",
    ),
    min_years: int = typer.Option(8, help="Minimum years of financials to qualify"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Score companies on the first half of their history, check the second half."""
    setup_logging(verbose)

    from datetime import date

    from boundless100x.compute_engine.backtest import WalkForwardBacktest
    from boundless100x.service import Boundless100xService

    svc = Boundless100xService()

    console.print("\n[bold blue]Walk-forward backtest[/bold blue]")
    report = WalkForwardBacktest(
        svc.suite.raw_data_dir, svc.engine, svc.scorer, svc.eligibility,
        min_total_years=min_years,
    ).run()

    companies = report["companies"]
    if companies:
        table = Table(title="Score then vs. return since")
        table.add_column("Ticker", style="bold")
        table.add_column("Scored on", justify="right")
        table.add_column("Composite", justify="right")
        table.add_column("Fwd yrs", justify="right")
        table.add_column("Realized CAGR", justify="right")
        for row in sorted(companies, key=lambda r: r["realized_cagr_pct"], reverse=True):
            cagr = row["realized_cagr_pct"]
            colour = "green" if cagr > 15 else "yellow" if cagr > 0 else "red"
            table.add_row(
                row["ticker"], row["truncation_date"],
                f"{row['composite_then']}/10", str(row["forward_span"]["years"]),
                f"[{colour}]{cagr:+.1f}%[/{colour}]",
            )
        console.print(table)

        correlations = report["correlations"]
        console.print(
            f"\nSpearman (composite vs return): "
            f"[bold]{correlations['composite_vs_return']}[/bold] over n={correlations['n']}"
        )
        for element, value in correlations.get("elements_vs_return", {}).items():
            console.print(f"   {element:20} {value}")

        cohorts = report.get("eligibility_cohorts")
        if cohorts:
            console.print("\n[bold blue]Forward return by 100x-eligibility verdict[/bold blue]")
            for verdict, stats in cohorts.items():
                console.print(
                    f"   {verdict:15} n={stats['n']:<3} "
                    f"mean={stats['mean_cagr_pct']:+.1f}%  median={stats['median_cagr_pct']:+.1f}%  "
                    f"range=[{stats['min_cagr_pct']:+.1f}%, {stats['max_cagr_pct']:+.1f}%]"
                )
    else:
        console.print("[yellow]No company qualified.[/yellow]")

    if report["skipped"]:
        console.print(f"\n[dim]Skipped {len(report['skipped'])}:[/dim]")
        for entry in report["skipped"]:
            console.print(f"   [dim]{entry['ticker']}: {entry['reason']}[/dim]")

    if report["excluded_metrics"]:
        console.print(
            f"\n[dim]Metrics excluded to prevent look-ahead leakage "
            f"({len(report['excluded_metrics'])}):[/dim]"
        )
        for entry in report["excluded_metrics"][:10]:
            console.print(f"   [dim]{entry['metric']} ({entry['tickers_affected']} tickers)[/dim]")

    console.print(f"\n[yellow]{report['limitations']['verdict']}[/yellow]")

    out = Path(output_dir) / date.today().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps(report, indent=2, default=str))
    console.print(f"[green]Report written to {out / 'backtest.json'}[/green]")


@app.command()
def screen(
    tickers: str = typer.Argument(help="Comma-separated NSE symbols to screen"),
    preset: str = typer.Option("compounders", help="Screening preset name"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Screen a list of companies using preset criteria."""
    setup_logging(verbose)

    from boundless100x.service import Boundless100xService
    from boundless100x.compute_engine.screener import Screener

    ticker_list = [t.strip() for t in tickers.split(",")]
    console.print(
        f"\n[bold blue]Screening {len(ticker_list)} companies "
        f"with preset: {preset}[/bold blue]\n"
    )

    svc = Boundless100xService()
    screener = Screener()

    # Show preset info
    preset_info = screener.presets.get(preset)
    if preset_info:
        console.print(f"[dim]{preset_info.get('name', preset)}[/dim]")
        console.print(f"[dim]{preset_info.get('description', '').strip()}[/dim]\n")

    survivors = screener.screen_quick(ticker_list, svc, preset=preset)

    if not survivors:
        console.print("[yellow]No companies passed the screening criteria.[/yellow]")
        return

    table = Table(title=f"Screening Results — {preset}")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Composite", justify="right")

    # Add filter metric columns
    filter_metrics = list(preset_info.get("filters", {}).keys()) if preset_info else []
    for mid in filter_metrics:
        table.add_column(mid, justify="right")

    for entry in survivors:
        row = [
            str(entry.get("rank", "")),
            entry["ticker"],
            f"{entry.get('sqglp_composite', 'N/A')}/10" if entry.get("sqglp_composite") else "N/A",
        ]
        for mid in filter_metrics:
            val = entry.get(mid)
            if val is not None:
                row.append(f"{val:.2f}")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[green]{len(survivors)} companies passed screening[/green]")


@app.command()
def sweep(
    tickers: str = typer.Option(
        None, help="Comma-separated symbols to extract forward growth from"
    ),
    all_tickers: bool = typer.Option(
        False, "--all", help="Every ticker with a gated-found extractable section"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Price the sweep without calling the API"
    ),
    pilot: int = typer.Option(
        None, help="Run only the first N tickers, and name the rest as deferred"
    ),
    ceiling: float = typer.Option(
        None, help="Stop once this much (USD) has been spent"
    ),
    out: str = typer.Option(None, help="Write the full report as JSON here"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Extract forward growth across chosen tickers, with the cost known first."""
    setup_logging(verbose)

    from boundless100x.llm_layer import sweep as sweep_module
    from boundless100x.service import Boundless100xService

    svc = Boundless100xService()
    requested = [t.strip() for t in tickers.split(",")] if tickers else None

    console.print("\n[bold blue]Forward-growth extraction sweep[/bold blue]\n")
    try:
        report = sweep_module.sweep(
            svc, tickers=requested, all_tickers=all_tickers, dry_run=dry_run,
            cost_ceiling_usd=ceiling, limit=pilot,
        )
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    priced = [p for p in report["plans"] if not p.get("skipped")]
    if priced:
        table = Table(title="What would be submitted")
        table.add_column("Ticker", style="cyan bold")
        table.add_column("Years", style="dim")
        table.add_column("Sections", style="dim")
        table.add_column("Chars", justify="right")
        table.add_column("Est. tokens", justify="right")
        table.add_column("Est. $", justify="right")
        for plan in priced:
            table.add_row(
                plan["ticker"], ", ".join(plan["years"]),
                ", ".join(plan["sections"]), f"{plan['submission_chars']:,}",
                f"{plan['estimated_input_tokens']:,}",
                f"{plan['estimated_cost_usd']:.4f}",
            )
        console.print(table)

    estimate = report["estimate"]
    console.print(
        f"[bold]Estimate:[/bold] {estimate['tickers']} ticker(s), "
        f"~${estimate['usd']:.4f} (worst case ${estimate['usd_max']:.4f})"
    )

    for entry in report["skipped"]:
        console.print(f"[dim]skipped {entry['ticker']}: {entry['reason']}[/dim]")
    if report["deferred"]:
        console.print(
            f"[dim]deferred to a later batch: {', '.join(report['deferred'])}[/dim]"
        )

    if report["dry_run"]:
        console.print("\n[yellow]Dry run — no API call was made.[/yellow]")
    else:
        results = Table(title="Extraction results")
        results.add_column("Ticker", style="cyan bold")
        results.add_column("Status")
        results.add_column("Kept", justify="right")
        results.add_column("Discarded", justify="right")
        results.add_column("$", justify="right")
        for result in report["results"]:
            colour = "green" if result["status"] == "ok" else "red"
            results.add_row(
                result["ticker"], f"[{colour}]{result['status']}[/{colour}]",
                str(result["kept"]), str(len(result["discarded"])),
                f"{result.get('cost_usd', 0):.4f}",
            )
        console.print(results)

        if report.get("discard_summary"):
            console.print("\n[bold]Why entries were discarded[/bold]")
            for reason, count in report["discard_summary"].items():
                console.print(f"  {count:>4}  {reason}")

        if report["not_reached"]:
            console.print(
                f"\n[yellow]Cost ceiling reached — not run: "
                f"{', '.join(report['not_reached'])}[/yellow]"
            )
        actual = report["actual"]
        console.print(
            f"\n[bold]Actual:[/bold] ${actual['usd']:.4f} "
            f"({actual['input_tokens']:,} in + {actual['output_tokens']:,} out)"
        )

    _write_report(out, report)


# ── Corpus Commands ──

corpus_app = typer.Typer(
    help="Snapshot, refetch and audit the cached corpus in raw_data/"
)
app.add_typer(corpus_app, name="corpus")


def _load_config() -> dict:
    from boundless100x.service import load_config

    return load_config()


def _raw_data_dir() -> Path:
    """The corpus the fetchers actually write to.

    Read off the suite rather than recomputed. `corpus restore` *deletes* its
    destination, and snapshot/audit/restore deriving the path independently of
    the fetchers would mean that if the location ever moved, restore would
    delete a stale tree while the fetchers wrote elsewhere — the exact failure
    the audit exists to catch, in the audit itself.
    """
    from boundless100x.data_fetcher.suite import DataFetcherSuite

    return Path(DataFetcherSuite(_load_config()).raw_data_dir)


def _resolve_snapshot(explicit: str | None, config: dict):
    """An explicit snapshot path, or the newest — and a clear error if neither."""
    from boundless100x.data_fetcher import corpus_snapshot

    chosen = Path(explicit) if explicit else corpus_snapshot.latest_snapshot(
        config=config
    )
    if chosen is None:
        console.print(
            f"[red]No snapshot found under "
            f"{corpus_snapshot.snapshot_root(config)}[/red]"
        )
        raise typer.Exit(1)
    return chosen


def _write_report(out: str | None, report: dict) -> None:
    if not out:
        return
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[dim]Full report written to {target}[/dim]")


@corpus_app.command("snapshot")
def corpus_snapshot_cmd(
    destination: str = typer.Option(
        None, help="Where to write the snapshot (default: corpus_snapshot.dir)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Copy raw_data/ somewhere safe before anything overwrites it."""
    setup_logging(verbose)

    from boundless100x.data_fetcher import corpus_snapshot

    try:
        made = corpus_snapshot.snapshot(
            _raw_data_dir(), destination=destination, config=_load_config()
        )
    except corpus_snapshot.SnapshotError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    totals = made["manifest"]["totals"]
    console.print(
        f"[green]Snapshot written to {made['path']}[/green]\n"
        f"[dim]{totals['directories']} directories, {totals['files']} files, "
        f"{totals['bytes'] / 1e6:.0f}MB[/dim]"
    )
    console.print(
        f"\n[dim]Restore with: python -m boundless100x corpus restore "
        f"--snapshot {made['path']}[/dim]"
    )


@corpus_app.command("restore")
def corpus_restore_cmd(
    snapshot: str = typer.Option(None, help="Snapshot directory (default: newest)"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Put a snapshot back, replacing the current raw_data/ entirely."""
    setup_logging(verbose)

    from boundless100x.data_fetcher import corpus_snapshot

    config = _load_config()
    chosen = _resolve_snapshot(snapshot, config)

    target = _raw_data_dir()
    console.print(
        f"[yellow]This replaces {target} entirely with {chosen}.[/yellow]\n"
        f"[dim]Restore replaces rather than merges — a half-restored corpus is "
        f"worse than either state.[/dim]"
    )
    if not yes and not typer.confirm("Proceed?"):
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit(1)

    try:
        restored = corpus_snapshot.restore(chosen, target)
    except corpus_snapshot.SnapshotError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    totals = restored["totals"]
    console.print(
        f"[green]Restored {totals['files']} files "
        f"({totals['bytes'] / 1e6:.0f}MB) to {target}[/green]"
    )


@corpus_app.command("refetch")
def corpus_refetch_cmd(
    tickers: str = typer.Option(
        None, help="Comma-separated symbols (default: every cached ticker)"
    ),
    no_cache_bypass: bool = typer.Option(
        False, "--no-cache-bypass",
        help="Serve fresh cache entries instead of reaching the network",
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Refetch tickers the run log records as done"
    ),
    force: bool = typer.Option(
        False, "--force", help="Start even with no corpus snapshot present"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Refresh every cached ticker from the network, one at a time."""
    setup_logging(verbose)

    from boundless100x.data_fetcher import corpus_snapshot, refetch as refetch_module
    from boundless100x.service import Boundless100xService

    config = _load_config()
    svc = Boundless100xService(config=config)
    requested = [t.strip() for t in tickers.split(",")] if tickers else None

    console.print("\n[bold blue]Corpus refetch[/bold blue]\n")
    try:
        report = refetch_module.refetch(
            svc.suite,
            tickers=requested,
            bypass_cache=not no_cache_bypass,
            resume=not no_resume,
            require_snapshot=not force,
            snapshot_config=config,
        )
    except corpus_snapshot.SnapshotError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    for entry in report["skipped"]:
        console.print(f"[dim]skipped {entry['name']}: {entry['reason']}[/dim]")
    if report["resumed"]:
        console.print(
            f"[dim]resumed: {len(report['resumed'])} ticker(s) already complete "
            f"({', '.join(report['resumed'])})[/dim]"
        )

    table = Table(title="Refetch outcomes")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Status")
    table.add_column("Seconds", justify="right")
    table.add_column("Detail", style="dim", max_width=60)
    for outcome in report["outcomes"]:
        colour = "green" if outcome["status"] == "ok" else "red"
        table.add_row(
            outcome["ticker"], f"[{colour}]{outcome['status']}[/{colour}]",
            f"{outcome['seconds']:.0f}", outcome["detail"],
        )
    console.print(table)

    failed = [o for o in report["outcomes"] if o["status"] != "ok"]
    console.print(
        f"\n[green]{len(report['outcomes']) - len(failed)} refetched[/green]"
        + (f", [red]{len(failed)} failed[/red]" if failed else "")
    )
    console.print(f"[dim]Run log: {report['run_log']}[/dim]")
    console.print(
        "[dim]Now run: python -m boundless100x corpus audit[/dim]"
    )


@corpus_app.command("audit")
def corpus_audit_cmd(
    snapshot: str = typer.Option(None, help="Snapshot to compare against (default: newest)"),
    out: str = typer.Option(None, help="Write the full report as JSON here"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Say what the refetch changed, counted off the corpus on disk."""
    setup_logging(verbose)

    from boundless100x.data_fetcher import corpus_audit, corpus_snapshot

    config = _load_config()
    chosen = _resolve_snapshot(snapshot, config)

    try:
        report = corpus_audit.audit_against_snapshot(_raw_data_dir(), chosen)
    except corpus_snapshot.SnapshotError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    head = report["headline"]
    console.print(
        f"\n[bold blue]Corpus audit[/bold blue] "
        f"[dim](before: {report['before']['created_at']} — {chosen.name})[/dim]\n"
    )

    table = Table(title="Headline")
    table.add_column("Measure", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Which", style="dim", max_width=70)
    for label, key in (
        ("Tickers gained quarterly.csv", "gained_quarterly"),
        ("Tickers still without one", "still_without_quarterly"),
        ("Tickers gained adj_close", "gained_adj_close"),
        ("Tickers still without it", "still_without_adj_close"),
        ("Directories gained report years", "gained_report_years"),
        ("Codes with 2+ MD&A years, pre-gate (before)", "two_or_more_mdna_years_before"),
        ("Codes with 2+ MD&A years, pre-gate (after)", "two_or_more_mdna_years_after"),
    ):
        names = head[key]
        table.add_row(label, str(len(names)), ", ".join(names))
    table.add_row("Annual report years added", str(head["report_years_added"]), "")
    console.print(table)

    if report["regressions"]:
        console.print(
            f"\n[bold red]{len(report['regressions'])} regression(s) — "
            f"investigate before discarding the snapshot[/bold red]"
        )
        for entry in report["regressions"]:
            console.print(
                f"  [red]{entry['directory']}[/red] {entry['kind']}: {entry['detail']}"
            )
    else:
        console.print("\n[green]No regressions: nothing shrank or disappeared.[/green]")

    _write_report(out, report)


# ── Display Helpers ──

def _print_scores(result, svc):
    summary = svc.get_element_summary(result)

    table = Table(title=f"SQGLP Scores — {result.ticker}")
    table.add_column("Element", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right", style="dim")

    element_order = [
        "size", "quality_business", "quality_management",
        "growth", "longevity", "price",
    ]
    element_names = {
        "size": "Size (S)",
        "quality_business": "Quality - Business (Q)",
        "quality_management": "Quality - Mgmt (Q)",
        "growth": "Growth (G)",
        "longevity": "Longevity (L)",
        "price": "Price (P)",
    }

    for el in element_order:
        info = summary.get(el, {})
        score = info.get("score")
        weight = info.get("weight", "")
        if score is not None:
            color = "green" if score >= 7 else "yellow" if score >= 4 else "red"
            table.add_row(element_names.get(el, el), f"[{color}]{score:.1f}/10[/{color}]", weight)
        else:
            table.add_row(element_names.get(el, el), "[dim]N/A[/dim]", weight)

    composite = summary.get("composite")
    table.add_section()
    table.add_row("[bold]COMPOSITE[/bold]", f"[bold]{composite}/10[/bold]", "100%")

    console.print(table)
    _print_coverage(result)
    _print_eligibility(result)


def _print_coverage(result):
    """Say how much evidence the composite rests on — a renormalised score
    otherwise looks identical to a fully measured one."""
    coverage = (result.scores or {}).get("coverage") or {}
    composite = coverage.get("composite")
    if composite is None or composite >= 0.999:
        return

    colour = "red" if composite < 0.85 else "yellow"
    console.print(
        f"[{colour}]Scored on {composite * 100:.0f}% of metric weight[/{colour}]"
    )
    thin = [
        f"{el} {cov * 100:.0f}%"
        for el, cov in (coverage.get("elements") or {}).items()
        if cov is not None and cov < 0.85
    ]
    if thin:
        console.print(f"  [dim]thin elements: {', '.join(thin)}[/dim]")
    unscored = coverage.get("unscored") or []
    if unscored:
        shown = ", ".join(unscored[:6])
        more = f" (+{len(unscored) - 6} more)" if len(unscored) > 6 else ""
        console.print(f"  [dim]unscored: {shown}{more}[/dim]")


def _print_eligibility(result):
    """The 100x verdict — necessary conditions the weighted composite cannot express."""
    eligibility = getattr(result, "eligibility", None)
    if not eligibility:
        return

    verdict = eligibility.get("verdict")
    style, label = {
        "eligible": ("green", "100x CANDIDATE"),
        "not_eligible": ("red", "NOT A 100x CANDIDATE"),
    }.get(verdict, ("yellow", "ELIGIBILITY UNKNOWN"))

    console.print(f"\n[{style}][bold]{label}[/bold][/{style}]")
    for gate_id, detail in eligibility.get("gates", {}).items():
        mark, colour = {
            True: ("PASS", "green"),
            False: ("FAIL", "red"),
        }.get(detail.get("passed"), ("????", "yellow"))
        console.print(f"  [{colour}]{mark}[/{colour}]  {detail.get('reason', gate_id)}")


def _print_llm_summary(result):
    llm = result.llm_analysis
    if not llm:
        return

    p2 = llm.get("pass2", {})
    if p2 and not p2.get("error") and not p2.get("skipped"):
        # The guarded action, never the raw p2 one — the console is a decision
        # surface too, and it prints the eligibility gates just above this.
        # Recomputed rather than read off result.final_action, so a stale or
        # unset field cannot fall back to the unchecked model action here.
        from boundless100x.action_policy import resolve_for_result

        decision = resolve_for_result(result) or {}
        action = decision.get("action") or "N/A"

        console.print("\n[bold]Investment Thesis:[/bold]")
        console.print(f"  {p2.get('thesis', 'N/A')}")
        console.print(
            f"  Conviction: [bold]{p2.get('conviction_level', 'N/A')}[/bold] | "
            f"Action: [bold]{action}[/bold] | "
            f"Period: {p2.get('target_holding_period', 'N/A')}"
        )
        if decision.get("constraints"):
            if decision.get("capped"):
                console.print(
                    f"  [yellow]Capped from {decision['llm_action']} "
                    f"to {decision['ceiling']}:[/yellow]"
                )
            for reason in decision["constraints"]:
                console.print(f"    [yellow]- {reason}[/yellow]")

    usage = llm.get("usage", {})
    if usage:
        console.print(
            f"\n[dim]LLM: {usage.get('total_tokens', 0)} tokens | "
            f"~${usage.get('estimated_cost_usd', 0):.4f} | "
            f"{usage.get('total_seconds', 0)}s[/dim]"
        )


if __name__ == "__main__":
    import os
    try:
        app()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        os._exit(code)
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
    else:
        os._exit(0)
