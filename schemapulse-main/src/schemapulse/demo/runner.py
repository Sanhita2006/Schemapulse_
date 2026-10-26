"""Demo environment bootstrap: containers, schema, deterministic data.

``schemapulse demo`` must work on a laptop that has never seen this repository.
That means one command has to bring up two PostgreSQL servers and load a fixture
into both.  Everything here is idempotent, because a demo that only works once is
not a demo.

The fixture is generated in pure SQL (``generate_series`` plus modular
arithmetic, no ``random()``) so:
  * the baseline and shadow databases are guaranteed byte-identical,
  * a recorded demo replays,
  * the test suite can shrink the dataset without changing its shape.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemapulse.config import Settings
from schemapulse.db.connection import DbTarget, execute, session, target_from_dsn

log = logging.getLogger("schemapulse.demo")


def _find_demo_dir() -> Path:
    """Locate the ``demo/`` directory.

    Works for an editable checkout (``src/schemapulse/demo/runner.py`` -> project
    root), for a wheel installed next to the package, and for any working
    directory inside the repository.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "demo"
        if (candidate / "init.sql").is_file():
            return candidate
    cwd_candidate = Path.cwd() / "demo"
    if (cwd_candidate / "init.sql").is_file():
        return cwd_candidate
    packaged = Path(__file__).resolve().parent / "fixtures"
    if (packaged / "init.sql").is_file():
        return packaged
    return here.parents[3] / "demo"


DEMO_DIR = _find_demo_dir()
INIT_SQL = DEMO_DIR / "init.sql"
SEED_SQL = DEMO_DIR / "seed.sql"
SLOW_QUERY = DEMO_DIR / "slow_orders.sql"
EXPECTED = DEMO_DIR / "expected" / "golden_result.json"


@dataclass(slots=True)
class SeedReport:
    """What the fixture loader actually did."""

    schema_applied: bool = False
    data_applied: bool = False
    orders: int = 0
    customers: int = 0
    pending_orders: int = 0
    pending_pct: float = 0.0
    orders_size: str = ""
    extra_order_indexes: int = 0
    elapsed_s: float = 0.0
    targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_applied": self.schema_applied,
            "data_applied": self.data_applied,
            "orders": self.orders,
            "customers": self.customers,
            "pending_orders": self.pending_orders,
            "pending_pct": self.pending_pct,
            "orders_size": self.orders_size,
            "extra_order_indexes": self.extra_order_indexes,
            "elapsed_s": round(self.elapsed_s, 2),
            "targets": self.targets,
            "notes": self.notes,
        }


@dataclass(slots=True)
class DockerStatus:
    available: bool
    version: str | None = None
    compose: bool = False
    error: str = ""
    services: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# docker
# --------------------------------------------------------------------------- #
def docker_status() -> DockerStatus:
    """Is a usable Docker daemon reachable?"""
    if shutil.which("docker") is None:
        return DockerStatus(available=False, error="docker executable not found on PATH")
    try:
        version = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return DockerStatus(available=False, error=f"could not run docker: {exc}")
    if version.returncode != 0:
        detail = (version.stderr or version.stdout).strip().splitlines()
        return DockerStatus(
            available=False,
            error=detail[-1] if detail else "docker daemon is not responding",
        )
    try:
        compose = (
            subprocess.run(
                ["docker", "compose", "version", "--short"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        compose = False
    return DockerStatus(available=True, version=version.stdout.strip(), compose=compose)


def compose_up(wait_seconds: int = 90) -> tuple[bool, str]:
    """Start the demo services. Returns ``(ok, message)``."""
    status = docker_status()
    if not status.available:
        return False, status.error
    cmd = (
        ["docker", "compose", "up", "-d", "--wait"]
        if status.compose
        else ["docker-compose", "up", "-d"]
    )
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=wait_seconds, check=False
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"docker compose failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        return False, " | ".join(tail)
    return True, "demo services started"


def compose_down(remove_volumes: bool = True) -> tuple[bool, str]:
    status = docker_status()
    if not status.available:
        return False, status.error
    cmd = ["docker", "compose", "down"]
    if remove_volumes:
        cmd.append("-v")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"docker compose down failed: {exc}"
    return proc.returncode == 0, (proc.stdout or proc.stderr).strip()[-300:]


def container_running(name: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return name in (proc.stdout or "").split()


# --------------------------------------------------------------------------- #
# schema + data
# --------------------------------------------------------------------------- #
def _demo_targets(settings: Settings) -> tuple[DbTarget, DbTarget]:
    return (
        target_from_dsn("baseline", settings.database_url, read_only=False),
        target_from_dsn("shadow", settings.shadow_database_url, read_only=False),
    )


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"demo file not found: {path}")
    # ``utf-8-sig`` also accepts a plain UTF-8 file, but tolerates the BOM that
    # Windows editors like to add.  The server rejects a BOM in the first
    # statement, so this is load-bearing rather than cosmetic.
    return path.read_text(encoding="utf-8-sig")


#: psql meta-commands start with a backslash.  They are meaningful to the
#: ``psql`` client only, so they are stripped before we hand the file to psycopg.
_PSQL_META = re.compile(r"^\s*\\[^\n]*$", re.MULTILINE)


def _strip_psql_meta(sql: str) -> str:
    """Drop ``\\set`` / ``\\timing`` style directives.

    SchemaPulse deliberately does not shell out to ``psql`` -- the demo has to
    work on a machine with only a Python venv and Docker.  Leaving these lines
    in would send backslash commands to the server as SQL.
    """
    return _PSQL_META.sub("", sql)


def _substitute_vars(sql: str, *, orders: int, customers: int) -> str:
    """Replace psql ``:var`` placeholders with literals."""
    return (
        _strip_psql_meta(sql)
        .replace(":orders", str(int(orders)))
        .replace(":customers", str(int(customers)))
    )


def _apply_schema(conn: Any) -> None:
    execute(conn, _substitute_vars(_read(INIT_SQL), orders=0, customers=0))


def _apply_data(conn: Any, *, orders: int, customers: int) -> None:
    """Load the fixture.

    The file is a multi-statement script, so it must go through psycopg with no
    bound parameters (the simple query protocol is the only one that accepts
    more than one statement per call).  Counts come back from
    :func:`_fixture_summary`, so we ignore the final result set here.
    """
    execute(conn, _substitute_vars(_read(SEED_SQL), orders=orders, customers=customers))


def seed(
    settings: Settings,
    *,
    orders: int | None = None,
    customers: int | None = None,
    force: bool = False,
    targets: tuple[DbTarget, ...] | None = None,
) -> SeedReport:
    """Create the schema and load the deterministic fixture on both servers."""
    started = time.perf_counter()
    orders = orders if orders is not None else settings.demo_orders
    customers = customers if customers is not None else settings.demo_customers
    report = SeedReport(orders=orders, customers=customers)

    demo_targets = targets if targets is not None else _demo_targets(settings)
    for target in demo_targets:
        report.targets.append(target.safe_dsn)
        try:
            with session(target, timeout_ms=0) as conn:  # 0 => no statement_timeout
                needs_seed = force or _needs_seed(conn, orders=orders, customers=customers)
                _apply_schema(conn)
                report.schema_applied = True
                if not needs_seed:
                    report.notes.append(f"{target.name}: already seeded, left untouched")
                else:
                    _apply_data(conn, orders=orders, customers=customers)
                    report.data_applied = True
                summary = _fixture_summary(conn)
                report.orders = int(summary.get("orders", 0))
                report.customers = int(summary.get("customers", 0))
                report.pending_orders = int(summary.get("pending_orders", 0))
                report.pending_pct = float(summary.get("pending_pct") or 0.0)
                report.orders_size = str(summary.get("orders_size", ""))
                report.extra_order_indexes = int(summary.get("extra_order_indexes", 0))
        except Exception as exc:
            report.notes.append(f"{target.name}: FAILED — {type(exc).__name__}: {exc}")

    report.elapsed_s = time.perf_counter() - started
    return report


def _needs_seed(conn: Any, *, orders: int, customers: int) -> bool:
    try:
        rows = execute(
            conn,
            """
            SELECT
                (SELECT count(*) FROM customers) AS c,
                (SELECT count(*) FROM orders)    AS o
            """,
        )
    except Exception:
        return True
    if not rows:
        return True
    return int(rows[0]["c"]) != customers or int(rows[0]["o"]) != orders


def _fixture_summary(conn: Any) -> dict[str, Any]:
    rows = execute(
        conn,
        """
        SELECT
            (SELECT count(*) FROM customers)                        AS customers,
            (SELECT count(*) FROM orders)                           AS orders,
            (SELECT count(*) FROM orders WHERE status = 'pending')  AS pending_orders,
            round(
                100.0 * (SELECT count(*) FROM orders WHERE status = 'pending')
                / NULLIF((SELECT count(*) FROM orders), 0), 3)      AS pending_pct,
            pg_size_pretty(pg_total_relation_size('orders'))       AS orders_size,
            (SELECT count(*) FROM pg_indexes
              WHERE schemaname = 'public' AND tablename = 'orders'
                AND indexname NOT LIKE '%_pkey')                   AS extra_order_indexes
        """,
    )
    return dict(rows[0]) if rows else {}


def load_golden_query() -> str:
    """The demo query with its comment header stripped."""
    sql = _read(SLOW_QUERY)
    return "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--")).strip()


def ensure_demo_environment(
    settings: Settings,
    *,
    start_containers: bool = True,
    seed_data: bool = True,
    orders: int | None = None,
    customers: int | None = None,
) -> tuple[SeedReport, list[str]]:
    """Bring the demo up and report honestly what happened."""
    notes: list[str] = []
    status = docker_status()

    if status.available and start_containers:
        ok, message = compose_up()
        notes.append(message)
        if not ok:
            notes.append(
                "could not start containers; assuming they are already running or "
                "pointing SCHEMAPULSE_*_DATABASE_URL at your own servers"
            )
    elif not status.available:
        notes.append(f"Docker unavailable ({status.error}); using the configured databases")

    report = SeedReport()
    if seed_data:
        report = seed(settings, orders=orders, customers=customers)
        notes.extend(report.notes)
    return report, notes
