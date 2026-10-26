"""Extract a :class:`CandidateSet` from whatever a model actually returned.

Local models in particular are inconsistent about JSON: fenced blocks,
preamble sentences, trailing commas, single quotes.  Rather than pretend that
never happens, this module makes extraction tolerant and *reports* what it had
to repair, so the engine can decide whether to trust the result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from schemapulse.agent.schemas import CandidateSet

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_blob(text: str) -> str:
    """Return the most plausible JSON object inside ``text``."""
    if not text or not text.strip():
        raise ValueError("model returned an empty response")

    # 1. fenced block, if any
    for match in _FENCE_RE.finditer(text):
        blob = match.group(1).strip()
        if blob.startswith("{"):
            return blob

    # 2. outermost brace span
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("no JSON object found in the model response")


def parse_candidate_set(text: str) -> tuple[CandidateSet, str]:
    """Parse and validate a candidate set.

    Returns
    -------
    (candidate_set, repair_note)
        ``repair_note`` is empty when the model produced clean JSON. Anything
        else is recorded in the trace so the demo never overstates model quality.
    """
    notes: list[str] = []

    blob = extract_json_blob(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        notes.append("initial JSON was malformed; repaired")
        data = _repair_json(blob)
        if data is None:
            raise

    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object at the top level, got {type(data).__name__}")

    if "candidates" not in data:
        # tolerate a bare list of candidates
        if isinstance(data.get("result"), list):
            data = {"candidates": data["result"]}
            notes.append("model returned a bare candidate list; wrapped")
        elif isinstance(data.get("candidates"), str):
            data["candidates"] = [{"candidate_sql": data["candidates"]}]
            notes.append("candidates was a bare SQL string; wrapped")

    data = _drop_unknown_keys(data)
    return CandidateSet.model_validate(data), "; ".join(notes)


def _drop_unknown_keys(data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(CandidateSet.model_fields)
    trimmed = {k: v for k, v in data.items() if k in allowed}
    if isinstance(trimmed.get("candidates"), list):
        candidate_allowed = set(_candidate_fields())
        cleaned = []
        for item in trimmed["candidates"]:
            if isinstance(item, dict):
                cleaned.append({k: v for k, v in item.items() if k in candidate_allowed})
            else:
                cleaned.append(item)
        trimmed["candidates"] = cleaned
    return trimmed


def _candidate_fields() -> tuple[str, ...]:
    from schemapulse.agent.schemas import OptimizationCandidate

    return tuple(OptimizationCandidate.model_fields)


def _repair_json(blob: str) -> dict[str, Any] | None:
    """Last-resort repairs for common model JSON defects."""
    candidate = blob
    # trailing commas before } or ]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # single-quoted keys/strings -> double-quoted (only if no double quotes present)
    if '"' not in candidate and "'" in candidate:
        candidate = re.sub(r"'([^'\\]*)'", r'"\1"', candidate)
    # Python literals leaking into JSON
    candidate = candidate.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
