"""``schemapulse`` — the command line surface.

Six commands, in the order you would actually use them:

    schemapulse doctor       is this machine ready?
    schemapulse inspect      deterministic baseline analysis (no model at all)
    schemapulse diagnose     the full loop: Gemma → validate → shadow experiment
    schemapulse demo         bring the environment up and run the golden path
    schemapulse report       render the HTML evidence report
    schemapulse provider     which Gemma adapter is active, and is it live?

``inspect`` deliberately works with no model and no network. That is what makes
it a useful *fallback* as well as a useful first step: if Gemma is unreachable
during judging, the deterministic half of the product still demonstrates.
"""

from __future__ import annotations

import codecs
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from schemapulse import __version__
from schemapulse.config import Settings, load_dotenv_if_present, reload_settings
from schemapulse.demo.runner import (
    compose_down,
    docker_status,
    ensure_demo_environment,
    load_golden_query,
)
from schemapulse.errors import (
    ConfigurationError,
    DatabaseError,
    DatabaseUnavailableError,
    ModelError,
    SafetyError,
    SchemaPulseError,
)
from schemapulse.plans.comparator import Verdict

app = typer.Typer(
    name="schemapulse",
    help="Execution-Verified PostgreSQL Optimization Agent. Gemma proposes; PostgreSQL decides.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

BANNER = "[bold]SchemaPulse[/bold] v{version} — execution-verified PostgreSQL optimization"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _force_utf8_stdio() -> None:
    """Make the console safe for the box-drawing and arrow glyphs we print.

    A stock Windows console handle is still cp1252, and a single character it
    cannot encode aborts the whole command with ``UnicodeEncodeError`` long
    after the useful work finished.  Ask for UTF-8, and fall back to replacement
    characters if even that is refused, so the tool degrades instead of dying.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None) or ""
        try:
            needs_change = codecs.lookup(encoding).name != "utf-8"
        except LookupError:
            needs_change = True
        if needs_change and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic console
                pass


def _settings(**overrides: Any) -> Settings:
    load_dotenv_if_present()
    base = reload_settings()
    if overrides:
        return base.model_copy(update=overrides)
    return base


def _read_query(path: Path) -> str:
    if not path.is_file():
        raise typer.BadParameter(f"query file not found: {path}")
    return path.read_text(encoding="utf-8")


def _agent(settings: Settings) -> Any:
    from schemapulse.engine import SchemaPulse

    return SchemaPulse(settings=settings)


def _fail(exc: Exception, hint: str = "") -> None:
    err_console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
    if hint:
        err_console.print(f"[yellow]hint:[/yellow] {hint}")
    raise typer.Exit(code=1)


def _verdict_style(verdict: str) -> str:
    return {
        "verified": "bold green",
        "rejected": "bold red",
        "inconclusive": "bold yellow",
        "failed": "bold magenta",
        "blocked": "bold cyan",
    }.get(verdict, "white")


def _print_plan_tree(plan: Any, title: str) -> None:
    """Render the plan as a real tree, with self time per node."""
    if not plan.nodes:
        console.print(f"[dim]{title}: no nodes[/dim]")
        return
    tree = Tree(f"[bold]{title}[/bold]")

    def add(node: Any, parent: Any) -> None:
        label = f"[cyan]{node.node_type}[/cyan]"
        if node.relation:
            label += f" [dim]on[/dim] [white]{node.relation}[/white]"
        if node.index_name:
            label += f" [dim]using[/dim] [magenta]{node.index_name}[/magenta]"
        meta = f"{node.self_time_ms:.2f}ms self | {node.actual_total_rows:,} rows"
        if node.rows_removed_total:
            meta += f" | -{node.rows_removed_total:,} filtered"
        if node.filter:
            meta += f" | filter {node.filter}"
        if node.index_condition:
            meta += f" | cond {node.index_condition}"
        branch = parent.add(f"{label} [dim]({meta})[/dim]")
        for child in node.children:
            add(child, branch)

    add(plan.nodes[0], tree)
    console.print(tree)


def _print_insight(insight: Any) -> None:
    plan = insight.plan
    console.print(
        Panel(
            f"[bold]{insight.query.strip().splitlines()[0]}[/bold]\n"
            f"[dim]{insight.database} · {insight.server_version} · "
            f"tables: {', '.join(insight.tables) or 'n/a'}[/dim]",
            title="query under analysis",
            border_style="blue",
        )
    )

    stats = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    stats.add_column("metric")
    stats.add_column("value", justify="right")
    stats.add_row("execution time", f"[bold]{plan.execution_time_ms:.2f} ms[/bold]")
    stats.add_row("planning time", f"{plan.planning_time_ms:.2f} ms")
    stats.add_row("rows examined", f"{plan.total_rows_examined:,}")
    stats.add_row("rows removed by filter", f"{plan.rows_removed_by_filter:,}")
    stats.add_row(
        "rows returned",
        f"{insight.rows_returned:,}" if insight.rows_returned is not None else "n/a",
    )
    stats.add_row("buffers (hit/read)", f"{plan.shared_hit_blocks:,} / {plan.shared_read_blocks:,}")
    stats.add_row("indexes used", ", ".join(plan.indexes_used) or "[red]none[/red]")
    console.print(Panel(stats, title="baseline", border_style="cyan"))

    _print_plan_tree(plan, "execution plan")

    if insight.hotspots:
        hot = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        hot.add_column("")
        hot.add_column("finding")
        hot.add_column("evidence", overflow="fold")
        for h in insight.hotspots:
            mark = "[yellow]▲[/yellow]" if h.severity == "warn" else "[dim]·[/dim]"
            hot.add_row(mark, h.title, f"[dim]{h.detail}[/dim]")
        console.print(
            Panel(hot, title="deterministic findings (no model involved)", border_style="yellow")
        )

    if insight.existing_indexes:
        console.print(
            Panel(
                "\n".join(f"[dim]·[/dim] {d}" for d in insight.existing_indexes),
                title="existing indexes",
                border_style="grey37",
            )
        )


def _progress_printer(stage: str, info: dict[str, Any]) -> None:
    """Live progress for the demo. Every line corresponds to real work."""
    if stage == "stage":
        console.print(
            f"[bold blue]▸ {info['stage']}[/bold blue] [dim]{info.get('detail', '')}[/dim]"
        )
    elif stage == "model_request":
        console.print(
            f"[magenta]  Gemma[/magenta] {info['provider']}/{info['model']} "
            f"attempt {info['attempt']}"
            + (
                f" [dim](with {info['feedback_items']} measured result(s) as feedback)[/dim]"
                if info.get("feedback_items")
                else ""
            )
        )
    elif stage == "model_response":
        live = "live" if info.get("is_live") else "[yellow]FIXTURE (no model call)[/yellow]"
        note = (
            f" [yellow]repaired: {info['repair_note']}[/yellow]" if info.get("repair_note") else ""
        )
        console.print(
            f"[magenta]  → response[/magenta] {info['latency_ms']:.0f} ms · "
            f"{info['candidates']} candidate(s) · {live}{note}"
        )
        if info.get("root_cause"):
            console.print(f"    [dim]root cause: {info['root_cause']}[/dim]")
    elif stage == "candidate_validated":
        if info["ok"]:
            console.print(
                f"[green]    ✓ passed validation[/green] [dim]{info['candidate_sql'][:110]}[/dim]"
            )
        else:
            console.print(
                f"[red]    ✗ rejected[/red] [dim]{'; '.join(info['reasons'])[:130]}[/dim]"
            )
    elif stage == "index_created":
        console.print(
            f"[dim]    {info.get('action', 'created')}: {info['index']}"
            f"{'' if 'ms' in info else ' [' + str(info.get('create_ms', '')) + ' ms]'}[/dim]"
        )
    elif stage == "analyzed":
        console.print(f"[dim]    ANALYZE {info['table']}[/dim]")
    elif stage == "baseline_run":
        status = f"[red]{info['error']}[/red]" if info.get("error") else f"{info['ms']:.2f} ms"
        console.print(f"    [dim]baseline   run {info['run']}/{info['total']}:[/dim] {status}")
    elif stage == "candidate_run":
        status = f"[red]{info['error']}[/red]" if info.get("error") else f"{info['ms']:.2f} ms"
        console.print(f"    [dim]candidate  run {info['run']}/{info['total']}:[/dim] {status}")
    elif stage == "baseline_done":
        console.print(
            f"[cyan]    baseline median[/cyan] {info['median_ms']:.2f} ms "
            f"[dim](spread {info['spread_pct']:.0f}%)[/dim]"
        )
    elif stage == "experiment_complete":
        style = _verdict_style(info["verdict"])
        sp = f" · {info['speedup']:.2f}x" if info.get("speedup") else ""
        console.print(
            f"    [{style}]{info['verdict'].upper()}[/{style}]{sp} [dim]{info['reason'][:90]}[/dim]"
        )
    elif stage == "index_dropped":
        console.print(f"[dim]    cleaned up {info['index']}[/dim]")


def _print_result(result: Any, report_path: Path | None = None) -> None:
    verdict = result.verdict.value
    speedup = result.speedup

    if verdict == Verdict.VERIFIED.value and result.comparison and speedup:
        head = (
            f"[bold green]VERIFIED[/bold green]  "
            f"{result.comparison.baseline_median_ms:.2f} ms → "
            f"[bold]{result.comparison.candidate_median_ms:.2f} ms[/bold]  "
            f"[bold green]{speedup:.2f}x faster[/bold green]"
        )
    else:
        head = f"[{_verdict_style(verdict)}]{verdict.upper()}[/{_verdict_style(verdict)}]"

    console.print()
    console.print(
        Panel(head, border_style=_verdict_style(verdict).replace("bold ", ""), expand=False)
    )

    if result.comparison:
        ev = Table(show_header=False, box=None, pad_edge=False)
        for line in result.comparison.evidence_lines:
            ev.add_row(f"[dim]·[/dim] {line}")
        console.print(
            Panel(
                ev,
                title="measured evidence",
                border_style="green" if verdict == "verified" else "yellow",
            )
        )

    if result.verified:
        console.print(
            Panel(
                Syntax(
                    result.verified.statement, "sql", theme="monokai", background_color="default"
                ),
                title="verified candidate (executed only on the shadow database)",
                border_style="green",
            )
        )

    if result.recommendation:
        rec = Table(show_header=False, box=None, pad_edge=False)
        rec.add_row("[bold]suggested production change (never auto-applied)[/bold]")
        rec.add_row(
            Syntax(
                result.recommendation.production_sql.strip(),
                "sql",
                theme="monokai",
                background_color="default",
            )
        )
        for w in result.recommendation.warnings:
            rec.add_row(f"[yellow]![/yellow] [dim]{w}[/dim]")
        console.print(Panel(rec, title="recommendation", border_style="blue"))

    if len(result.attempts) > 1 or verdict != "verified":
        att = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        att.add_column("#")
        att.add_column("verdict")
        att.add_column("candidate SQL", overflow="fold", max_width=44)
        att.add_column("speedup")
        att.add_column("reason", overflow="fold")
        for i, a in enumerate(result.attempts, start=1):
            sp = f"{a.speedup:.2f}x" if a.speedup else "—"
            sql = a.candidate.candidate_sql if a.candidate is not None else ""
            att.add_row(
                f"{a.attempt}.{i}" if len(result.attempts) > 1 else str(a.attempt),
                f"[{_verdict_style(a.verdict.value)}]{a.verdict.value.upper()}[/{_verdict_style(a.verdict.value)}]",
                sql,
                sp,
                a.reason,
            )
        console.print(
            Panel(att, title="every attempt, including rejected ones", border_style="grey37")
        )

    provider_note = (
        f"live model: {result.provider}/{result.model_id}"
        if result.provider_is_live
        else f"[yellow]deterministic baseline ({result.provider}) — no model was called[/yellow]"
    )
    footer = (
        f"{provider_note} · {len(result.attempts)} attempt(s) · {result.elapsed_ms:.0f} ms total"
    )
    if report_path:
        footer += f" · report: {report_path}"
    console.print(f"[dim]{footer}[/dim]")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"schemapulse {__version__}")


@app.command()
def provider(
    provider_name: Annotated[
        str | None, typer.Option("--provider", "-p", help="gemini | ollama | fixture")
    ] = None,
    probe: Annotated[
        bool, typer.Option("--probe", help="Contact the provider to test reachability")
    ] = False,
) -> None:
    """Show which Gemma adapter is active and whether it is live."""
    from schemapulse.agent.providers import resolve_provider

    settings = _settings()
    try:
        resolved = resolve_provider(settings, allow_fallback=False)
    except ModelError as exc:
        _fail(
            exc,
            f"set GEMINI_API_KEY, or run Ollama and `ollama pull {settings.ollama_model}`",
        )
        return

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("[bold]provider[/bold]", resolved.provider.name)
    table.add_row("[bold]model[/bold]", resolved.provider.model_id)
    table.add_row(
        "[bold]status[/bold]",
        "[green]live — a real model call will be made[/green]"
        if resolved.is_live
        else "[yellow]offline baseline — NO model will be called[/yellow]",
    )
    if resolved.downgraded_from:
        table.add_row("[bold]downgraded from[/bold]", resolved.downgraded_from)
        table.add_row("[bold]reason[/bold]", resolved.reason)
    if probe:
        health = resolved.provider.health()
        for key, value in health.items():
            table.add_row(f"[dim]{key}[/dim]", str(value))
    console.print(Panel(table, title=f"{BANNER.format(version=__version__)}", border_style="blue"))
    if not resolved.is_live:
        err_console.print(
            "[yellow]note:[/yellow] fixture output is always labelled in the CLI and the "
            "HTML report so it is never presented as a model result."
        )


@app.command()
def doctor(
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Check Python, Docker, database connectivity and the Gemma provider."""
    settings = _settings()
    report: dict[str, Any] = {"version": __version__, "python": sys.version.split()[0]}

    docker = docker_status()
    report["docker"] = {
        "available": docker.available,
        "version": docker.version,
        "compose": docker.compose,
        "error": docker.error,
    }

    from schemapulse.agent.providers import resolve_provider
    from schemapulse.engine import SchemaPulse

    try:
        agent = SchemaPulse(settings=settings)
        health = agent.health()
    except ConfigurationError as exc:
        _fail(exc, "SCHEMAPULSE_SHADOW_DATABASE_URL must differ from SCHEMAPULSE_DATABASE_URL")
        return
    report["databases"] = {"target": health["target"], "shadow": health["shadow"]}

    try:
        resolved = resolve_provider(settings, allow_fallback=True)
        provider_health = resolved.provider.health()
        report["provider"] = {
            **resolved.describe(),
            "health": provider_health,
        }
    except ModelError as exc:
        report["provider"] = {"error": str(exc), "is_live": False}

    if json_out:
        console.print_json(json.dumps(report, default=str))
        return

    rows = Table(show_header=False, box=None, pad_edge=False)
    rows.add_column("check", style="bold")
    rows.add_column("status")
    rows.add_column("detail", overflow="fold")

    rows.add_row("python", "[green]ok[/green]", report["python"])

    if docker.available:
        rows.add_row(
            "docker",
            "[green]ok[/green]",
            f"server {docker.version}, compose {'yes' if docker.compose else 'no'}",
        )
    else:
        rows.add_row("docker", "[yellow]unavailable[/yellow]", docker.error)

    for key, label in (("target", "target database"), ("shadow", "shadow database")):
        info = report["databases"][key]
        if info.get("reachable"):
            rows.add_row(
                label,
                "[green]ok[/green]",
                f"{info['dsn']} · {info.get('server_version')} · {info.get('database_size', '')}",
            )
        else:
            rows.add_row(label, "[red]unreachable[/red]", info.get("error", "unknown"))

    provider_info = report.get("provider", {})
    if provider_info.get("error"):
        rows.add_row("gemma provider", "[red]unavailable[/red]", provider_info["error"])
    else:
        live = provider_info.get("is_live")
        health = provider_info.get("health", {})
        rows.add_row(
            "gemma provider",
            "[green]ok[/green]" if live else "[yellow]fixture[/yellow]",
            f"{provider_info['provider']}/{provider_info['model']} — "
            f"{health.get('detail') or ('live model' if live else 'deterministic baseline')}",
        )

    demo = Path(__file__).resolve().parents[2] / "demo" / "slow_orders.sql"
    rows.add_row(
        "demo fixture",
        "[green]ok[/green]" if demo.is_file() else "[yellow]missing[/yellow]",
        str(demo) if demo.is_file() else "run from a checkout",
    )

    console.print(
        Panel(
            rows, title=f"{BANNER.format(version=__version__)}", border_style="blue", expand=False
        )
    )
    ok = report["databases"]["target"].get("reachable") and report["databases"]["shadow"].get(
        "reachable"
    )
    if ok:
        console.print(
            "[green]ready.[/green] try: [bold]schemapulse inspect demo/slow_orders.sql[/bold]"
        )
    else:
        console.print(
            "[yellow]databases are not reachable.[/yellow] try: [bold]docker compose up -d --wait[/bold] "
            "then [bold]schemapulse demo --seed-only[/bold]"
        )


@app.command()
def inspect(
    query: Annotated[Path, typer.Argument(help="Path to a .sql file containing a single SELECT")],
    count_rows: Annotated[
        bool, typer.Option("--rows/--no-rows", help="Also count returned rows")
    ] = True,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Deterministic baseline analysis. Works with NO model and NO network."""
    settings = _settings()
    sql = _read_query(query)
    try:
        agent = _agent(settings)
        insight = agent.inspect(sql, count_rows=count_rows)
    except SafetyError as exc:
        _fail(exc, "SchemaPulse analyses SELECT statements only in the MVP")
        return
    except DatabaseUnavailableError as exc:
        _fail(exc, "docker compose up -d --wait   then   schemapulse demo --seed-only")
        return
    except (DatabaseError, ConfigurationError) as exc:
        _fail(exc)
        return

    if json_out:
        console.print_json(json.dumps(insight.to_dict(), default=str))
        return

    console.print(
        Panel(
            f"[bold]{BANNER.format(version=__version__)}[/bold]", border_style="blue", expand=False
        )
    )
    console.print("[dim]mode: deterministic analysis only — no model is contacted[/dim]\n")
    _print_insight(insight)


@app.command()
def diagnose(
    query: Annotated[Path, typer.Argument(help="Path to a .sql file containing a single SELECT")],
    provider_name: Annotated[
        str | None, typer.Option("--provider", "-p", help="gemini | ollama | fixture")
    ] = None,
    require_live: Annotated[
        bool,
        typer.Option("--require-live/--allow-fixture", help="Fail instead of using the fixture"),
    ] = False,
    min_speedup: Annotated[
        float | None, typer.Option("--min-speedup", help="Acceptance threshold")
    ] = None,
    runs: Annotated[int | None, typer.Option("--runs", "-n", help="Timed runs per arm")] = None,
    attempts: Annotated[
        int | None, typer.Option("--attempts", help="Max experiment attempts")
    ] = None,
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Write the HTML report")
    ] = True,
    json_out: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Run the full loop: Gemma hypothesis → validation → shadow experiment → verdict."""
    settings = _settings()
    overrides: dict[str, Any] = {}
    if min_speedup:
        overrides["min_speedup"] = min_speedup
    if runs:
        overrides["benchmark_runs"] = runs
    if attempts:
        overrides["max_candidate_attempts"] = attempts
    if provider_name:
        overrides["model_provider"] = provider_name
    if overrides:
        settings = settings.model_copy(update=overrides)

    sql = _read_query(query)
    console.print(
        Panel(
            f"[bold]{BANNER.format(version=__version__)}[/bold]", border_style="blue", expand=False
        )
    )
    console.print(
        f"[dim]threshold: {settings.min_speedup}x · runs/arm: {settings.benchmark_runs} · "
        f"max attempts: {settings.max_candidate_attempts}[/dim]\n"
    )

    try:
        agent = _agent(settings)
        result = agent.diagnose(sql, require_live_model=require_live, progress=_progress_printer)
    except SafetyError as exc:
        _fail(exc, "SchemaPulse analyses SELECT statements only in the MVP")
        return
    except ModelError as exc:
        _fail(
            exc,
            "try --provider ollama (local Gemma) or --provider fixture (deterministic baseline)",
        )
        return
    except DatabaseUnavailableError as exc:
        _fail(exc, "docker compose up -d --wait   then   schemapulse demo --seed-only")
        return
    except (DatabaseError, ConfigurationError) as exc:
        _fail(exc)
        return

    report_path: Path | None = None
    if report:
        from schemapulse.reporting.report import render_report

        try:
            report_path = render_report(result, Path(settings.artifacts_dir))
        except Exception as exc:  # reporting must never break a diagnosis
            err_console.print(f"[yellow]could not write report: {exc}[/yellow]")

    if json_out:
        console.print_json(json.dumps(result.to_dict(), default=str))
    else:
        _print_result(result, report_path)

    raise typer.Exit(code=0 if result.verdict is Verdict.VERIFIED else 2)


@app.command()
def demo(
    seed_only: Annotated[
        bool, typer.Option("--seed-only", help="Only bring up + seed, then exit")
    ] = False,
    down: Annotated[bool, typer.Option("--down", help="Tear the demo environment down")] = False,
    provider_name: Annotated[
        str | None, typer.Option("--provider", "-p", help="gemini | ollama | fixture")
    ] = None,
    orders: Annotated[
        int | None, typer.Option("--orders", help="Order rows in the fixture")
    ] = None,
    skip_docker: Annotated[
        bool, typer.Option("--skip-docker", help="Do not run docker compose")
    ] = False,
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Write the HTML report")
    ] = True,
) -> None:
    """The golden path, end to end: environment → seed → inspect → diagnose → report."""
    if down:
        ok, message = compose_down()
        console.print(("[green]" if ok else "[red]") + f"docker compose down: {message}[/]")
        return

    settings = _settings()
    if provider_name:
        settings = settings.model_copy(update={"model_provider": provider_name})

    console.print(
        Panel(
            f"[bold]{BANNER.format(version=__version__)}[/bold]", border_style="blue", expand=False
        )
    )

    console.print("[bold blue]▸ environment[/bold blue]")
    seed_report, notes = ensure_demo_environment(
        settings,
        start_containers=not skip_docker,
        seed_data=True,
        orders=orders,
    )
    for note in notes:
        console.print(f"  [dim]·[/dim] {note}")
    if seed_report.orders:
        console.print(
            f"  [green]fixture ready[/green] · {seed_report.orders:,} orders · "
            f"{seed_report.customers:,} customers · {seed_report.pending_orders:,} pending "
            f"({seed_report.pending_pct:.2f}%) · {seed_report.orders_size} · "
            f"{seed_report.extra_order_indexes} non-primary indexes on orders"
        )
    failed = [n for n in seed_report.notes if "FAILED" in n]
    if failed:
        for f in failed:
            err_console.print(f"[red]{f}[/red]")
        err_console.print("[yellow]hint:[/yellow] docker compose up -d --wait")
        raise typer.Exit(code=1)

    if seed_only:
        console.print(
            "\n[green]seeded.[/green] next: [bold]schemapulse inspect demo/slow_orders.sql[/bold]"
        )
        return

    sql = load_golden_query()

    console.print("\n[bold blue]▸ baseline (no model)[/bold blue]")
    agent = _agent(settings)
    try:
        insight = agent.inspect(sql)
    except SchemaPulseError as exc:
        _fail(exc, "docker compose up -d --wait")
        return
    _print_insight(insight)

    console.print("\n[bold blue]▸ optimization loop[/bold blue]")
    try:
        result = agent.diagnose(sql, progress=_progress_printer)
    except ModelError as exc:
        _fail(exc, "fall back with: schemapulse demo --provider fixture")
        return
    except SchemaPulseError as exc:
        _fail(exc)
        return

    report_path: Path | None = None
    if report:
        from schemapulse.reporting.report import render_report

        try:
            report_path = render_report(result, Path(settings.artifacts_dir))
        except Exception as exc:
            err_console.print(f"[yellow]could not write report: {exc}[/yellow]")

    _print_result(result, report_path)

    console.print(
        "\n[dim]The database accepted nothing on trust. Every candidate was created on a "
        "disposable shadow database and measured.[/dim]"
    )


@app.command()
def report(
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open in the default browser")
    ] = False,
) -> None:
    """Render the HTML evidence report from the most recent diagnosis."""
    import webbrowser

    from schemapulse.reporting.report import ResultStore

    settings = _settings()
    store = ResultStore.from_settings(settings)
    try:
        path = store.render()
    except FileNotFoundError as exc:
        _fail(exc, "run `schemapulse diagnose demo/slow_orders.sql` first")
        return

    payload = store.load()
    console.print(
        Panel(
            f"[bold]report written[/bold]\n{path}\n\n"
            f"verdict: {payload.get('verdict')} · "
            f"speedup: {payload.get('speedup')} · "
            f"provider: {payload.get('provider')} "
            f"({'live' if payload.get('provider_is_live') else 'deterministic baseline'})",
            border_style="green",
            expand=False,
        )
    )
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


def main() -> None:
    _force_utf8_stdio()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
