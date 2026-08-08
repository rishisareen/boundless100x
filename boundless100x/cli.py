"""Boundless100x CLI — SQGLP Financial Research System."""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
import typer

load_dotenv()
from rich.table import Table

from boundless100x.cli_common import ConsoleComponents, console, setup_logging
# The two legal provider literals, single-sourced from the transport that
# implements them so a flag value and a config value are never two vocabularies.
from boundless100x.llm_layer.transport import COST_BASIS_ESTIMATED, LLMProvider
# Words a reader meets, taken from the report's vocabulary rather than spelled
# here. The console is a decision surface — it prints the eligibility gates and
# the action — so an enum leaking onto it is the same defect as an enum leaking
# into the report, and one table is what keeps the two saying the same thing.
from boundless100x.output.report_vocabulary import (  # noqa: E402
    ACTION_LABELS,
    ACTION_UNKNOWN_LABEL,
    COMPOSITE_TITLE as _COMPOSITE_TITLE,
)

app = typer.Typer(
    name="boundless100x",
    help="SQGLP Financial Research System for Indian Markets",
    no_args_is_help=True,
)
logger = logging.getLogger(__name__)

# A gate that recorded neither a reason nor a label. Stated rather than filled
# in with the gate id, which is what the fallback here used to do.
_GATE_UNNAMED = "a gate this run recorded no reason for"

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


def _config_with_provider(llm_provider: LLMProvider | None) -> dict:
    """This run's config, with `--llm-provider` applied when it was given.

    Threading the flag as a composition-root config override rather than a new
    service argument is deliberate: the provider is *already* a config key, and
    a second way of saying the same thing is a second thing to keep in
    agreement. `Boundless100xService` already accepts an injected dict.
    """
    config = _load_config()
    if llm_provider is not None:
        config.setdefault("llm", {})["provider"] = llm_provider.value
    return config


def _resolved_provider(config: dict) -> str:
    """Which transport this run will use, read off the config the flag landed in.

    The one place the question is answered. Asking a *result* instead — the
    `provider` a finished run reports — is only available once something has
    run, which is exactly the condition a dry run does not meet.
    """
    return (config.get("llm") or {}).get("provider", LLMProvider.ANTHROPIC.value)


def _provider_banner(config: dict) -> str:
    """Name the transport in a command's header when it is not the API default."""
    if _resolved_provider(config) != LLMProvider.CLAUDE_CLI.value:
        return ""
    return " [bold cyan](claude_cli — subscription-billed)[/bold cyan]"


def _print_cli_cost_caveat(config: dict, *, dry_run: bool) -> None:
    """What `claude_cli` does to a dollar figure, printed beside that figure.

    Both of the sweep's cost footers go through here, and the provider test
    lives inside rather than at either call site, because the caveat has
    already been unreachable once from exactly that shape: the live branch
    guarded on `report["actual"]["provider"]`, which a dry run leaves `None`
    (`sweep()` returns before any transport runs), so the one path whose entire
    job is to inform a spending decision was the one path that could not warn.
    A banner naming `claude_cli` beside a wrong number reads as confirmation
    of the number.
    """
    if _resolved_provider(config) != LLMProvider.CLAUDE_CLI.value:
        return

    if dry_run:
        # Yellow, not dim: on a dry run this *is* the decision-relevant content
        # — the figure above it is priced off `MODEL_PRICING`, which is an API
        # price table, and there is no claude_cli one. Two separate corrections
        # apply, and only the first is a multiplier: Claude Code writes every
        # prompt token at 1-hour-TTL cache-write rates (2x standard input),
        # which converges to ~1.7–1.8x the API path per company; and each call
        # additionally pays a measured ~$0.033 of harness prefix that the
        # estimate has no term for at all, and that does not amortize across
        # the sweep's independent sessions. Both figures are from
        # docs/plans/2026-08-08-008-feat-llm-provider-claude-cli-plan.md.
        console.print(
            "[yellow]This estimate prices the API path — MODEL_PRICING is API "
            "pricing and there is no claude_cli table. The claude_cli bill "
            "converges to ~1.7–1.8x the figure above (every prompt token is "
            "written at 2x cache-write rates), plus a fixed ~$0.033 per call "
            "of harness prefix the estimate does not model. Set any --ceiling "
            "against that, not against the number printed here.[/yellow]"
        )
    else:
        # Most of what a CLI call bills has nothing to do with the ticker,
        # so the per-ticker column above is not a per-ticker fact — and a
        # `--ceiling` calibrated against API pricing trips far sooner here.
        console.print(
            "[dim]claude_cli bills real dollars including a fixed per-call "
            "harness overhead, so the per-ticker figures are not comparable "
            "to the API path and any --ceiling binds sooner.[/dim]"
        )


def _service(config: dict | None = None):
    """This run's service, or a legible refusal in place of a traceback.

    Every command starts by building one, and the build validates the metric
    registry. That validation is fatal by design and stays that way — a metric
    missing its `presentation:` block must stop the run rather than reach a
    reader as a bare number — but the thing it guards against is a hand-edited
    YAML file in the documented `custom/` drop-in directory, and a stack trace
    is the wrong answer to a typo. `RegistryValidationError` names the file,
    the metric and the key, so all this has to do is print it and stop.

    `config=None` is not a special case: `Boundless100xService` loads the
    default config for exactly that, so the sites that pass a provider-adjusted
    config and the sites that pass nothing share one construction.
    """
    from rich.markup import escape

    from boundless100x.service import Boundless100xService, RegistryValidationError

    try:
        return Boundless100xService(config=config)
    except RegistryValidationError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    ticker: str = typer.Argument(help="NSE symbol (e.g., ASTRAL)"),
    bse_code: str = typer.Option(None, help="BSE scrip code"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM analysis"),
    deep: bool = typer.Option(False, "--deep", help="Use Opus for Pass 1 & 2 (~1.7x LLM cost, deeper analysis)"),
    llm_provider: LLMProvider = typer.Option(
        None, "--llm-provider",
        help="Which transport carries the LLM calls (default: config's llm.provider)",
    ),
    # `clarity` is the research note (U10). This call site passes `formats=`
    # explicitly, so the generator's own default never reaches it — the token
    # has to be here or an `analyze` run produces every report except the new
    # one (KTD3).
    formats: str = typer.Option(
        "html,md,clarity,json", help="Output formats (comma-separated)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
):
    """Run full SQGLP analysis pipeline for a company."""
    setup_logging(verbose)

    from boundless100x.output.report_components import caveat_from_run_error
    from boundless100x.output.report_generator import ReportGenerator

    config = _config_with_provider(llm_provider)
    mode = " [bold magenta](DEEP — Opus)[/bold magenta]" if deep else ""
    # A run's billing path in its first line of output: the CLI path spends the
    # subscription's headless pool rather than API credits, and that is not a
    # difference anyone should have to infer from the usage block afterwards.
    mode += _provider_banner(config)
    console.print(f"\n[bold blue]Boundless100x SQGLP Analysis: {ticker}[/bold blue]{mode}\n")

    svc = _service(config)
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
        # `service.py` writes `f"Data fetch failed: {e}"` here, and this block
        # used to print the whole thing — the raw `str(exc)` reaching the reader
        # verbatim, which is the fifth of the problem frame's five routes around
        # the vocabulary layer and the one R15 names outright.
        # `caveat_from_run_error` keeps the authored clause in front of the
        # colon and sends the rest to the log, where it is useful and where
        # nobody reads it as prose.
        surface = ConsoleComponents()
        console.print(f"\n[bold yellow]Warnings ({len(result.errors)}):[/bold yellow]")
        for error in result.errors:
            console.print(f"  {surface.render_caveat(caveat_from_run_error(error))}")


@app.command()
def compute(
    ticker: str = typer.Argument(help="NSE symbol"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Compute metrics only (no peers, no LLM, no reports)."""
    setup_logging(verbose)

    console.print(f"\n[bold blue]Computing metrics for {ticker}[/bold blue]\n")

    svc = _service()
    result = svc.analyze_quick(ticker)

    _print_scores(result, svc)
    _print_metrics(result, svc)


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

    svc = _service()

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


def _coerce_cli_value(raw: str):
    """A `--set key=value` value, type-coerced from the string the shell
    handed us.

    `simulate(config, overrides)`'s dot-path values reach real config
    consumers — a trading-day lag becomes an argument to
    `pandas.bdate_range(periods=n+1)`, a cap becomes an argument compared
    with `>` against a count — so `"0"` must become the int `0`, not the
    string `"0"` (which is truthy and not an int, and would raise or
    silently misbehave three frames downstream rather than at the CLI
    boundary where the mistake is easiest to see). Bool before int/float,
    because `int("true")` raises and `"true"` should not fall through to a
    string; int before float, so `"5"` stays exactly `5` rather than
    becoming `5.0`. Anything that coerces to neither is left as the string
    it arrived as — a lane name, a posture, a trigger id.
    """
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@app.command()
def simulate(
    tickers: str = typer.Option(
        None, help="Comma-separated NSE symbols to restrict the replay to"
    ),
    start: str = typer.Option(None, help="Replay window start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, help="Replay window end date (YYYY-MM-DD)"),
    set_overrides: list[str] = typer.Option(
        None, "--set",
        help="Override a config value: dotted.path.to.key=value (repeatable)",
    ),
    output_dir: str = typer.Option(
        str(Path(__file__).parent / "output" / "simulations"),
        help="Where to write the report",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Replay the production lifecycle rules over the corpus's own history.

    Truncates every active ticker to each replay date, scores and evaluates
    it through the same production evaluators `watchlist advance` uses, and
    hands money-moving proposals to a simulated owner (`config.yaml`'s
    `simulator:` block) rather than a person — see
    `docs/plans/2026-08-07-007-feat-phase4-strategy-simulator-plan.md`.
    """
    setup_logging(verbose)

    from datetime import date

    from boundless100x.simulator import outputs as outputs_module
    from boundless100x.simulator import simulate as run_simulation

    overrides = {}
    for item in set_overrides or []:
        if "=" not in item:
            console.print(f"[red]--set {item!r} is not key=value — skipped[/red]")
            continue
        key, _, raw_value = item.partition("=")
        overrides[key.strip()] = _coerce_cli_value(raw_value.strip())

    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None

    console.print("\n[bold blue]Strategy simulator[/bold blue]\n")
    result = run_simulation(
        None, overrides or None, tickers=ticker_list, start=start, end=end,
    )

    console.print(outputs_module.render_summary(result))

    if result.get("errors"):
        console.print(f"\n[bold yellow]Errors ({len(result['errors'])}):[/bold yellow]")
        for entry in result["errors"][:10]:
            console.print(
                f"  [yellow]! {entry.get('ticker')} ({entry.get('date')}): "
                f"{entry.get('error')}[/yellow]"
            )

    out = Path(output_dir) / date.today().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    (out / "simulation.json").write_text(json.dumps(result, indent=2, default=str))
    console.print(f"\n[green]Report written to {out / 'simulation.json'}[/green]")


@app.command()
def screen(
    tickers: str = typer.Argument(help="Comma-separated NSE symbols to screen"),
    preset: str = typer.Option("compounders", help="Screening preset name"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Screen a list of companies using preset criteria."""
    setup_logging(verbose)

    from boundless100x.compute_engine.screener import Screener

    ticker_list = [t.strip() for t in tickers.split(",")]
    console.print(
        f"\n[bold blue]Screening {len(ticker_list)} companies "
        f"with preset: {preset}[/bold blue]\n"
    )

    svc = _service()
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
    llm_provider: LLMProvider = typer.Option(
        None, "--llm-provider",
        help="Which transport carries the extraction calls "
             "(default: config's llm.provider)",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Extract forward growth across chosen tickers, with the cost known first."""
    setup_logging(verbose)

    from boundless100x.llm_layer import sweep as sweep_module

    config = _config_with_provider(llm_provider)
    svc = _service(config)
    requested = [t.strip() for t in tickers.split(",")] if tickers else None

    console.print(
        f"\n[bold blue]Forward-growth extraction sweep[/bold blue]"
        f"{_provider_banner(config)}\n"
    )
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
        # The dry run is what stands between a mistyped flag and a corpus-wide
        # spend, so the caveat about what the estimate is *not* pricing belongs
        # here more than anywhere else it appears.
        _print_cli_cost_caveat(config, dry_run=True)
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
        # `input_tokens` is the transport's own count, which on the claude_cli
        # path excludes everything the cache served or absorbed — so without
        # this the footer prints a few thousand tokens beside a real dollar
        # figure and the arithmetic looks impossible.
        cached = actual.get("cached_input_tokens")
        cached_note = f" (+{cached:,} cached)" if cached is not None else ""
        console.print(
            f"\n[bold]Actual:[/bold] ${actual['usd']:.4f} "
            f"[dim]({actual.get('cost_basis', COST_BASIS_ESTIMATED)}, "
            f"{actual.get('provider') or 'unknown provider'})[/dim] — "
            f"{actual['input_tokens']:,} in + {actual['output_tokens']:,} out"
            f"{cached_note}"
        )
        # Config, not `actual["provider"]`, for the same reason the dry-run
        # branch has to use it: one resolution of "which transport is this",
        # asked identically on both branches. The two cannot disagree here
        # anyway — the config is what built the transport that reported it.
        _print_cli_cost_caveat(config, dry_run=False)

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

    config = _load_config()
    svc = _service(config)
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
#
# Everything below renders through the same two layers the research note does:
# `report_reading` turns a declaration plus a computed value into a reading,
# and `report_components` turns that into one of six components which
# `ConsoleComponents` finally marks up. That is R14 — "the same content from the
# same declarations" — as a property of where the content is built rather than
# of how carefully three surfaces were matched by hand.
#
# It is also where three of the problem frame's five routes around the
# vocabulary layer were: the element-label map this file kept, the raw metric
# ids in the `unscored:` line, and the raw exception strings reaching the
# Warnings block. Each is now a lookup or a component, and a lookup fails
# loudly where a derived label succeeds on everything.


def _declaration_view(result, svc):
    """`(vocabulary, readings, configs)` — the declarations, read for this run.

    The same three inputs `_clarity_context` assembles from: the registry's
    `presentation:` blocks, the sector-applicability table, and this run's own
    `MetricResult`s. Built here rather than passed in because both display
    helpers need it and building it twice is how two surfaces come to disagree
    about one company.

    The registry comes off the **service's own engine**, not a fresh one: the
    values on screen were computed by that engine, and reading the declarations
    from a second instance would be reading a registry nothing on this screen
    was scored against.

    The applicability table is fenced, and the fence is not defensive
    housekeeping. `sector_applicability.yaml` is hand-maintained, expected to
    grow by manual edits, and validated **nowhere else** — the engine never
    reads it — so a bad edit surfaces first at `SectorApplicability`'s
    constructor, which is here. Here is after the fetch, the compute and the
    whole LLM spend, and before a single report has been written: unfenced, a
    typo in a *display* table aborts `analyze` with a traceback and leaves
    nothing on disk to show for the run. `ReportGenerator.generate` already
    fences the identical construction inside its clarity block, appending to
    `result.errors` rather than raising; this is the same decision on the
    console's side of the boundary.
    """
    from boundless100x.compute_engine.sector import SectorApplicability
    from boundless100x.output.report_components import Vocabulary
    from boundless100x.output.report_reading import read_metrics

    configs = getattr(getattr(svc, "engine", None), "metrics", None) or {}
    metadata = (getattr(result, "data", None) or {}).get("metadata") or {}

    # `None` is not a silent pass. `read_metrics` reads an absent table as
    # indeterminate-because-nobody-asked for every metric — the same answer an
    # unreviewed sector already gets — so the console loses the applicability
    # note and keeps everything else, which is the trade the log line states.
    applicability = None
    if configs:
        try:
            applicability = SectorApplicability(configs.keys())
        except ValueError as exc:
            logger.warning(
                f"Sector applicability is unavailable this run, so no metric "
                f"is reported as applying or not applying to this company's "
                f"sector: {exc}"
            )

    readings = read_metrics(
        configs,
        result.metrics or {},
        sector=metadata.get("sector"),
        applicability=applicability,
    )
    return Vocabulary(configs), readings, configs


def _shown_score(line) -> float | None:
    """The figure a reading puts in front of the reader, read back off the line.

    `ReadingLine.headline` is `f"{shown:.1f} / 10"`, and `shown` is the score
    *after* the one rounding both builders do before they band it. Recovering
    it is therefore not a re-derivation — it is the same number the words were
    built from, which is the whole point of asking the line rather than the
    caller.

    `None` for a line with no headline, which is the line with no score.
    """
    headline = (getattr(line, "headline", "") or "").split("/")[0].strip()
    try:
        return float(headline)
    except ValueError:
        return None


def _score_cell(line, rendered: str) -> str:
    """A rendered reading, coloured by the band its own words were built from.

    The colour is the console's addition and nothing else's — the words are
    already the note's — and the two have to agree about the same number. They
    did not. This banded the **raw** score while `composite_reading` and
    `section_reading` both round to one decimal before banding, so a real
    composite of 6.97 rendered a yellow cell around the words `7.0 / 10 —
    Reads strong`: the exact figure-against-interpretation disagreement the
    single builder was introduced to end, reappearing one layer out where the
    builder could not reach it. Element rows were spared only because
    `get_element_summary` happens to pre-round.

    So the band is read off the built line rather than asked a second time of
    a number the line no longer shows. One banding call per row, on the figure
    the reader is looking at.
    """
    from boundless100x.output.report_components import score_band

    band = score_band(_shown_score(line)) if getattr(line, "known", True) else None
    colour = {"strong": "green", "middling": "yellow", "weak": "red"}.get(band, "dim")
    return f"[{colour}]{rendered}[/{colour}]"


def _print_scores(result, svc):
    """The six element scores, each with the reading the report gives it.

    The labels come from `ELEMENT_CONFIG` — the report's own — rather than from
    the map this function used to keep. That map spelled three of the six
    differently (`Quality - Business` against `Quality — Business`,
    `Quality - Mgmt` against `Quality — Management`) with nothing keeping the
    two in step, which is the plainest version of the defect this unit exists
    to remove: two surfaces naming the same thing two ways.

    The `Score` column is gone and the score is now the headline *inside* the
    reading, for R12's reason. A column of bare figures beside a column of
    words invites reading one without the other, and `4.7/10` on its own says
    nothing about whether that is a problem.
    """
    from boundless100x.output.report_reading import read_element_coverages
    from boundless100x.output.report_vocabulary import ELEMENT_CONFIG

    surface = ConsoleComponents()
    vocabulary, _readings, _configs = _declaration_view(result, svc)
    summary = svc.get_element_summary(result)
    coverages = read_element_coverages((result.scores or {}).get("coverage"))

    table = Table(title=f"SQGLP Scores — {result.ticker}")
    table.add_column("Element", style="bold")
    table.add_column("Reading")
    table.add_column("Weight", justify="right", style="dim")

    for element, config in ELEMENT_CONFIG.items():
        score = (summary.get(element) or {}).get("score")
        line = _element_reading(element, vocabulary, score, coverages.get(element))
        table.add_row(
            str(config["label"]),
            _score_cell(line, surface.render_reading(line)),
            str(config["weight"]),
        )

    table.add_section()
    # Built once and both rendered and coloured from it, for `_score_cell`'s
    # reason: the cell's colour comes off the line, so the line has to be the
    # same object the reader is shown.
    composite = _composite_reading(summary.get("composite"))
    table.add_row(
        f"[bold]{_COMPOSITE_TITLE}[/bold]",
        _score_cell(composite, surface.render_reading(composite)),
        "100%",
    )

    console.print(table)
    _print_coverage(result, vocabulary)
    _print_eligibility(result, vocabulary)


def _element_reading(element, vocabulary, score, coverage):
    """One element's collapsed line (R5) with R18's clause, or the reason there
    is none. `section_reading` is the note's builder, called unchanged."""
    from boundless100x.output.report_components import section_reading

    return section_reading(element, vocabulary, score=score, coverage=coverage)


def _composite_reading(composite):
    """The composite as a `ReadingLine`, built by hand for `_clarity_lead`'s
    reason: it is not an element, so no element-shaped builder covers it.

    The score is rounded **before** it is banded, which is the rule
    `section_reading` states: banding the raw figure while the headline rounds
    it produced `7.0 / 10 — Reads middling` against a band boundary at exactly
    seven, and a number disagreeing with its own interpretation on the same
    line is the defect this whole report exists to remove.
    """
    from boundless100x.output.report_components import (
        composite_reading,
    )

    return composite_reading(composite, subject=_COMPOSITE_TITLE)


def _print_coverage(result, vocabulary):
    """How much evidence the composite rests on, and what is missing by name.

    A renormalised score otherwise looks identical to a fully measured one.

    Two things this used to print are gone. The **per-element** line read
    `thin elements: quality_business 32%` — the raw element key R15 keeps off
    the page — and it is not relabelled but removed: R18's coverage clause now
    renders inside that element's own row above, which is where a reader meets
    the score it qualifies rather than four lines below it.

    The **unscored list** printed nineteen metric ids. Every metric the engine
    discovers carries a registered `name`, so this is a lookup and not a
    fallback: a metric with none is counted and named as such rather than
    quietly humanised, because a derived label is the leak with better
    typography that `finding_from_flag` refuses for exactly the same reason.
    """
    coverage = (result.scores or {}).get("coverage") or {}
    composite = coverage.get("composite")
    if composite is None or composite >= 0.999:
        return

    colour = "red" if composite < 0.85 else "yellow"
    console.print(
        f"[{colour}]Scored on {composite * 100:.0f}% of metric weight[/{colour}]"
    )

    unscored = coverage.get("unscored") or []
    if not unscored:
        return
    named, unnamed = [], []
    for metric_id in unscored:
        name = vocabulary.metric_name(metric_id)
        (named if name else unnamed).append(name or metric_id)
    for metric_id in unnamed:
        logger.warning(
            f"Metric {metric_id!r} has no registered display name, so the "
            f"unscored line counts it rather than naming it (R15)"
        )

    shown = ", ".join(named[:6])
    more = [f"+{len(named) - 6} more"] if len(named) > 6 else []
    if unnamed:
        more.append(f"{len(unnamed)} this report has no name for")
    suffix = f" ({', '.join(more)})" if more else ""
    console.print(f"  [dim]Not scored: {shown}{suffix}[/dim]")


def _print_eligibility(result, vocabulary):
    """The 100x verdict — necessary conditions the weighted composite cannot express.

    The gate reason is **narrated**: `EligibilityEvaluator._summarise` writes
    `market_cap 138604.00 lte 3000`, and `Vocabulary.narrate` swaps each
    registered id for its display name. That is a lookup, so an id nobody
    registered is left alone and visible rather than silently prettified — and
    the comparator words (`lte`, `gte`) survive it, which is a residual this
    unit does not fix: they are the evaluator's own sentence, shared with
    `action_policy`'s constraints, and rewording them there would move text
    three other surfaces render.

    A gate with no reason recorded falls back to its declared label, never to
    its id, which is what the `detail.get('reason', gate_id)` here used to do.
    """
    eligibility = getattr(result, "eligibility", None)
    if not eligibility:
        return

    verdict = eligibility.get("verdict")
    style, label = {
        "eligible": ("green", "100x CANDIDATE"),
        "not_eligible": ("red", "NOT A 100x CANDIDATE"),
    }.get(verdict, ("yellow", "ELIGIBILITY UNKNOWN"))

    console.print(f"\n[{style}][bold]{label}[/bold][/{style}]")
    for detail in eligibility.get("gates", {}).values():
        mark, colour = {
            True: ("PASS", "green"),
            False: ("FAIL", "red"),
        }.get(detail.get("passed"), ("????", "yellow"))
        reason = vocabulary.narrate(detail.get("reason") or "").strip()
        console.print(
            f"  [{colour}]{mark}[/{colour}]  "
            f"{reason or detail.get('label') or _GATE_UNNAMED}"
        )


def _print_metrics(result, svc):
    """Every computed metric, grouped by element and read rather than listed.

    This replaced a three-column table of `metric_id | 25.7 | flag_id`, which
    was R12 and R15's failure in one place: a bare figure with no unit beside
    another with a different one, under a raw id, next to a raw flag — and an
    errored metric rendered `ERR: 'borrowings'`, a `KeyError` reaching the
    reader verbatim.

    A line per metric rather than a row, per `render_metric_row`'s note: the
    figure and its interpretation have to be seen together, and a table puts
    them in different columns as soon as the terminal is narrow enough to wrap.

    Order inside a group is `Section.flow`'s — findings, then rows, then the
    unknowns — so the console walks a section the way both document surfaces
    do, without needing the expansion decision that decides a *report's* shape
    and has nothing to say about a listing that shows everything anyway.
    """
    from boundless100x.output.report_components import (
        Unknown,
        finding_from_flag,
        metric_row,
    )
    from boundless100x.output.report_vocabulary import (
        ELEMENT_CONFIG,
        UNSCORED_SECTION_TITLE,
    )

    surface = ConsoleComponents()
    vocabulary, readings, configs = _declaration_view(result, svc)
    shares = _weight_shares(configs)
    details = (result.scores or {}).get("details") or {}

    groups: dict[str, list[str]] = {}
    for metric_id in sorted(readings, key=lambda mid: vocabulary.metric_name(mid) or mid):
        element = (configs.get(metric_id) or {}).get("element", "")
        groups.setdefault(
            element if element in ELEMENT_CONFIG else "", []
        ).append(metric_id)

    console.print(f"\n[bold]Metrics — {result.ticker}[/bold]")
    ordered = [
        (str(ELEMENT_CONFIG[element]["label"]), members)
        for element, members in (
            (element, groups.get(element, [])) for element in ELEMENT_CONFIG
        ) if members
    ]
    if groups.get(""):
        ordered.append((UNSCORED_SECTION_TITLE, groups[""]))

    for title, members in ordered:
        console.print(f"\n[bold]{title}[/bold]")
        rows, unknowns, findings = [], [], []
        for metric_id in members:
            detail = details.get(metric_id) or {}
            built = metric_row(
                metric_id, readings[metric_id], vocabulary,
                score=detail.get("score") if isinstance(detail, dict) else None,
                weight_share=shares.get(metric_id),
            )
            (unknowns if isinstance(built, Unknown) else rows).append(built)
            for flag in getattr(result.metrics.get(metric_id), "flags", None) or []:
                findings.append(finding_from_flag(flag))
        for finding in findings:
            renderer = (
                surface.render_unknown if isinstance(finding, Unknown)
                else surface.render_finding
            )
            console.print(f"  {renderer(finding)}")
        for row in rows:
            console.print(f"  {surface.render_metric_row(row)}")
        for unknown in unknowns:
            console.print(f"  {surface.render_unknown(unknown)}")


def _weight_shares(configs) -> dict:
    """Each metric's share of its own element's declared weight.

    Read off `report_expansion`, which is where that arithmetic is declared and
    where a test already pins it against the scorer's. Restating it here would
    be a third copy of `_declared_weights`, and the note's own comment says the
    risk in restating is drift — a contribution cell reading `25% of this
    element` in the report and something else on the console is the drift
    nobody would see.

    It calls the module function rather than `ExpansionDecider`, which is the
    same arithmetic without the collaborators. Going through the decider meant
    validating the whole contradiction table and globbing and parsing every
    `reports/*/scores.json` on disk — neither of which `weight_share` reads —
    on every `compute`, for a directory that only ever grows.
    """
    if not configs:
        return {}

    from boundless100x.output.report_expansion import declared_weight_shares

    return declared_weight_shares(configs)


def _action_label(action) -> str:
    """An action, in the reader's words. Absent is absent, not "N/A".

    `ACTION_UNKNOWN_LABEL` rather than the key itself for an action outside the
    vocabulary: an unrecognised action is already capped by `action_policy`, so
    printing its raw name here would put a string nobody chose beside a
    decision somebody did.
    """
    if not action:
        return "none — the model produced no action"
    return ACTION_LABELS.get(action, ACTION_UNKNOWN_LABEL)


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
        # The label, never the enum. `strong_buy` is a key of the action
        # vocabulary and R15 keeps keys off the page; `_action_label` is the
        # same lookup the research note's action finding makes, so the two
        # decision surfaces name one decision one way.
        action = _action_label(decision.get("action"))

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
                    f"  [yellow]Capped from "
                    f"{_action_label(decision['llm_action'])} "
                    f"to {_action_label(decision['ceiling'])}:[/yellow]"
                )
            for reason in decision["constraints"]:
                console.print(f"    [yellow]- {reason}[/yellow]")

    usage = llm.get("usage", {})
    if usage:
        # `estimated_cost_usd` holds *actual* metered cost on the claude_cli
        # path, so the basis is printed rather than a bare `~$` that would
        # describe a real bill as a guess.
        provider = usage.get("provider")
        via = f" via {provider}" if provider else ""
        # The token count above is the transport's, and on the claude_cli path
        # that number **excludes** everything served from or written to cache: a
        # two-pass run that moved ~35K tokens reports ~1.6K. Printed bare beside
        # an API-path report's honest 34,000, the CLI path reads as forty times
        # more token-efficient at twice the price. The correction has to be
        # rendered, not merely recorded.
        cached = usage.get("total_cached_input_tokens")
        cached_note = f" (+{cached:,} cached)" if cached is not None else ""
        # The totals are short by this many calls' tokens — see `_log_failed_call`.
        failed = usage.get("failed_calls")
        failed_note = (
            f" | {failed} failed call{'s' if failed > 1 else ''} (tokens unknown)"
            if failed
            else ""
        )
        console.print(
            f"\n[dim]LLM: {usage.get('total_tokens', 0)} tokens{cached_note} | "
            f"{usage.get('cost_basis', COST_BASIS_ESTIMATED)} "
            f"${usage.get('estimated_cost_usd', 0):.4f}{via} | "
            f"{usage.get('total_seconds', 0)}s{failed_note}[/dim]"
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
