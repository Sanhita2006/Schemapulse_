"""Evidence reports.

Two outputs, both generated from the same measured data:

* ``<artifacts>/last-result.json`` — the full machine-readable trace, so results
  can be diffed, asserted on in CI, or fed to another tool.
* ``<artifacts>/report.html`` — a single self-contained file with the query, the
  before/after plan trees, the measured timings, the model's hypothesis and the
  verdict.  No JavaScript framework, no build step, no CDN: it opens from
  ``file://`` on a plane.

The report deliberately shows rejected candidates too.  A report that only
prints successes is marketing; a report that prints the rejected attempts is
evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from schemapulse.models import DiagnosisResult

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
LAST_JSON = "last-result.json"
REPORT_HTML = "report.html"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@dataclass(slots=True)
class ResultStore:
    """Where artifacts live and how the latest result is found."""

    directory: Path

    @classmethod
    def from_settings(cls, settings: Any) -> ResultStore:
        return cls(directory=Path(settings.artifacts_dir))

    @property
    def last_json(self) -> Path:
        return self.directory / LAST_JSON

    @property
    def report_html(self) -> Path:
        return self.directory / REPORT_HTML

    def ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: DiagnosisResult) -> Path:
        self.ensure()
        payload = result.to_dict()
        payload["generated_at"] = datetime.now(UTC).isoformat()
        payload["schemapulse_version"] = _version()
        self.last_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self.last_json

    def load(self) -> dict[str, Any]:
        if not self.last_json.is_file():
            raise FileNotFoundError(
                f"no saved result at {self.last_json}. Run `schemapulse diagnose` or "
                f"`schemapulse demo` first."
            )
        return json.loads(self.last_json.read_text(encoding="utf-8"))

    def render(self) -> Path:
        payload = self.load()
        # The template is written against the flattened view model, not the raw
        # stored payload, and runs with StrictUndefined so a missing key fails
        # loudly instead of rendering an empty <td>.
        view = build_view_model(payload)
        html = (
            _env()
            .get_template("report.html")
            .render(
                data=view,
                generated_at=payload.get("generated_at", ""),
                version=payload.get("schemapulse_version", ""),
            )
        )
        self.ensure()
        self.report_html.write_text(html, encoding="utf-8")
        return self.report_html


def _version() -> str:
    from schemapulse import __version__

    return __version__


# --------------------------------------------------------------------------- #
# view model
# --------------------------------------------------------------------------- #
def render_report(result: DiagnosisResult, directory: Path) -> Path:
    """Persist a result and render its HTML report. Returns the HTML path."""
    store = ResultStore(directory=directory)
    store.save(result)
    return store.render()


def write_report(directory: Path) -> Path:
    """Render HTML from the most recently saved result."""
    return ResultStore(directory=directory).render()


def build_view_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a stored result into what the template needs."""
    insight = payload.get("insight", {})
    plan = insight.get("plan", {})
    comparison = payload.get("comparison") or {}

    baseline_ms = comparison.get("baseline_median_ms")
    candidate_ms = comparison.get("candidate_median_ms")
    speedup = comparison.get("speedup")

    return {
        "verdict": (payload.get("verdict") or "unknown").upper(),
        "speedup": speedup,
        "speedup_display": f"{speedup:.1f}x" if isinstance(speedup, int | float) else "n/a",
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "min_speedup": payload.get("min_speedup"),
        "provider": payload.get("provider"),
        "is_live": payload.get("provider_is_live"),
        "model_id": payload.get("model_id"),
        "query": insight.get("query"),
        "database": insight.get("database"),
        "tables": insight.get("tables", []),
        "rows_examined": insight.get("rows_examined"),
        "rows_returned": insight.get("rows_returned"),
        "plan_shape": plan.get("plan_shape"),
        "nodes": plan.get("nodes", []),
        "hotspots": insight.get("hotspots", []),
        "evidence": comparison.get("evidence", []),
        "attempts": payload.get("attempts", []),
        "verified_candidate": payload.get("verified_candidate"),
        "recommendation": payload.get("recommendation"),
        "trace": payload.get("trace", []),
    }


def load_rendered_result(directory: Path) -> dict[str, Any]:
    store = ResultStore(directory=directory)
    return build_view_model(store.load())
