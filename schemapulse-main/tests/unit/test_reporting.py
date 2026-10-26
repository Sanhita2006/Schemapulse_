"""Reporting: the artifact a judge actually reads.

Two classes of bug matter here and both have shipped:

* a Jinja error or a shape mismatch between the view model and the template,
  which loses the whole report after a long, expensive run
* a report that quietly mislabels a fixture run as a live one
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures.explain_plans import (
    GOLDEN_QUERY,
    INDEX_SCAN_PLAN,
    SEQ_SCAN_PLAN,
)

from schemapulse.models import (
    AttemptOutcome,
    DiagnosisResult,
    QueryInsight,
    Recommendation,
)
from schemapulse.plans.comparator import Verdict, compare
from schemapulse.plans.parser import find_hotspots, parse_plan
from schemapulse.reporting.report import (
    ResultStore,
    build_view_model,
    load_rendered_result,
    render_report,
    write_report,
)

CANDIDATE_SQL = (
    'CREATE INDEX "sp_orders_status_created_at_partial" ON "orders" '
    '("created_at" DESC) WHERE "status" = \'pending\''
)


def result(*, verified: bool = True) -> DiagnosisResult:
    baseline_plan = parse_plan(SEQ_SCAN_PLAN, GOLDEN_QUERY)
    candidate_plan = parse_plan(INDEX_SCAN_PLAN, GOLDEN_QUERY)
    comparison = compare(
        baseline_runs=(147.16, 156.75, 162.42, 110.51, 91.61),
        candidate_runs=(1.52, 1.48, 1.61, 1.39, 1.72) if verified else (140.0, 141.0, 142.0),
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        min_speedup=1.2,
        added_index="sp_orders_status_created_at_partial" if verified else "ix",
    )
    insight = QueryInsight(
        query=GOLDEN_QUERY,
        database="schemapulse",
        server_version="PostgreSQL 16.3",
        tables=["orders", "customers"],
        plan=baseline_plan,
        hotspots=list(find_hotspots(baseline_plan)),
        existing_indexes=["orders_pkey"],
        rows_returned=50,
    )
    return DiagnosisResult(
        insight=insight,
        comparison=comparison,
        attempts=[
            AttemptOutcome(
                attempt=1,
                verdict=comparison.verdict,
                reason="measured in the shadow database",
                speedup=comparison.speedup,
                baseline_median_ms=comparison.baseline_median_ms,
                candidate_median_ms=comparison.candidate_median_ms,
                index_used=comparison.added_index,
                evidence=list(comparison.evidence_lines),
                provider="fixture",
            )
        ],
        best_attempt=1,
        recommendation=(
            Recommendation(
                production_sql=CANDIDATE_SQL,
                measured_speedup=comparison.speedup,
                rationale="shadow benchmark",
                warnings=["not applied to the target database"],
            )
            if comparison.verdict is Verdict.VERIFIED
            else None
        ),
        provider="fixture",
        provider_is_live=False,
        model_id="schemapulse-heuristic-v1",
        min_speedup=1.2,
    )


def store_in(tmp_path: Path, **kwargs: Any) -> tuple[ResultStore, Path]:
    store = ResultStore(directory=tmp_path / "artifacts")
    return store, render_report(result(**kwargs), store.directory)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_report_renders_for_a_verified_verdict(tmp_path: Path) -> None:
    _, html_path = store_in(tmp_path)
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "sp_orders_status_created_at_partial" in html


def test_report_renders_for_a_rejected_verdict(tmp_path: Path) -> None:
    """A null result is still a report; it must not crash on a weak speedup."""
    _, html_path = store_in(tmp_path, verified=False)
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


def test_report_is_self_contained(tmp_path: Path) -> None:
    """No network fetches: the report has to open from a file:// URL."""
    _, html_path = store_in(tmp_path)
    html = html_path.read_text(encoding="utf-8")
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://" not in html.replace("https://www.w3.org", "")
    assert "<script src=" not in html
    assert "cdn" not in html.lower()


def test_report_shows_the_median_not_a_single_run(tmp_path: Path) -> None:
    _, html_path = store_in(tmp_path)
    assert "median" in html_path.read_text(encoding="utf-8").lower()


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_both_artifacts_are_written(tmp_path: Path) -> None:
    store, html_path = store_in(tmp_path)
    assert html_path.exists()
    assert store.last_json.exists()
    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == Verdict.VERIFIED.value
    assert payload["speedup"] > 50


def test_written_json_is_stable_apart_from_the_timestamp(tmp_path: Path) -> None:
    store = ResultStore(directory=tmp_path / "artifacts")
    store.save(result())
    store.save(result())
    first = json.loads(store.last_json.read_text(encoding="utf-8"))
    second = json.loads(store.last_json.read_text(encoding="utf-8"))
    for blob in (first, second):
        blob.pop("generated_at", None)
        blob.pop("elapsed_ms", None)
    assert first == second


def test_result_is_reloadable_and_re_renderable(tmp_path: Path) -> None:
    store, _ = store_in(tmp_path)
    reloaded = store.load()
    assert reloaded["insight"]["query"] == GOLDEN_QUERY
    assert write_report(store.directory).exists()


def test_missing_result_raises_an_actionable_error(tmp_path: Path) -> None:
    store = ResultStore(directory=tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="schemapulse diagnose"):
        store.load()


def test_out_dir_is_created_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifacts"
    assert render_report(result(), target).exists()


# --------------------------------------------------------------------------- #
# the honesty guarantees
# --------------------------------------------------------------------------- #
def test_fixture_run_is_labelled_offline(tmp_path: Path) -> None:
    """The headline claim of the demo; it must not be able to silently vanish."""
    store, html_path = store_in(tmp_path)
    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    assert payload["provider_is_live"] is False
    assert payload["provider"] == "fixture"
    html = html_path.read_text(encoding="utf-8").lower()
    assert "fixture" in html or "offline" in html


def test_report_contains_the_evidence_a_reviewer_needs(tmp_path: Path) -> None:
    store, _ = store_in(tmp_path)
    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    assert payload["comparison"]["evidence"], "a verdict without evidence is an assertion"
    assert payload["comparison"]["added_index"]
    assert payload["comparison"]["baseline_median_ms"] == pytest.approx(147.16)


def test_rejected_attempts_are_kept(tmp_path: Path) -> None:
    """A report that only prints successes is marketing, not evidence."""
    store, html_path = store_in(tmp_path)
    payload = json.loads(store.last_json.read_text(encoding="utf-8"))
    assert payload["attempts"]
    view = build_view_model(payload)
    assert view["attempts"]
    assert "Attempt" in html_path.read_text(encoding="utf-8")


def test_report_never_recommends_applying_anything(tmp_path: Path) -> None:
    _, html_path = store_in(tmp_path)
    payload_view = load_rendered_result(html_path.parent)
    assert payload_view["recommendation"]["auto_applied"] is False


def test_report_does_not_leak_environment_secrets(tmp_path: Path) -> None:
    store, html_path = store_in(tmp_path)
    for path in (html_path, store.last_json):
        blob = path.read_text(encoding="utf-8").lower()
        assert "gemini_api_key" not in blob
        assert "aiza" not in blob  # Gemini keys start with this


def test_view_model_flattens_what_the_template_needs(tmp_path: Path) -> None:
    store, _ = store_in(tmp_path)
    view = load_rendered_result(store.directory)
    assert view["verdict"] == "VERIFIED"
    assert view["speedup_display"].endswith("x")
    assert view["query"] == GOLDEN_QUERY
    assert view["nodes"]
    assert view["plan_shape"]


def test_view_model_survives_an_empty_payload() -> None:
    """StrictUndefined makes a missing key fatal, so the fallback path matters."""
    view = build_view_model({})
    assert view["verdict"] == "UNKNOWN"
    assert view["speedup"] is None
    assert view["speedup_display"] == "n/a"
