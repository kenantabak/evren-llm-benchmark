"""Consistent latency percentiles, finish_reason taxonomy, and cache metrics."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


def compute_p50_p95(latencies_s: Sequence[float]) -> Tuple[float, float]:
    """Compute consistent ``p50`` (``statistics.median(all)``) and ``p95``.

    All calls — including wall-cap timeouts — are included in the distribution so
    timeout-heavy runs cannot artificially look faster than completed runs.
    """
    vals = sorted(float(x) for x in latencies_s)
    if not vals:
        return 0.0, 0.0
    p50 = round(float(statistics.median(vals)), 2)
    idx95 = min(len(vals) - 1, max(0, math.ceil(0.95 * len(vals)) - 1))
    p95 = round(vals[idx95], 2)
    return p50, p95


def classify_finish_reason(
    finish_reason: Optional[str],
    passed: bool,
    status_code: Optional[int] = 200,
    error: Optional[str] = None,
) -> str:
    """Map raw API termination telemetry to a canonical benchmark outcome bucket."""
    if passed:
        return "passed"

    fr = (finish_reason or "").strip().lower()
    err = (error or "").strip().lower()

    if fr in {"wall_cap", "timeout"} or "wall_cap" in err or "timeout" in err:
        return "wall_cap_timeout"
    if fr == "length":
        return "length_cap"
    if status_code == 503 or fr == "http_503" or "503" in err:
        return "server_503"
    if (status_code is not None and status_code != 200) or fr.startswith("http_") or fr == "error":
        return "http_error"
    if fr in {"stop", "tool_calls", "end_turn"}:
        return "wrong_answer"
    return "http_error" if err else "wrong_answer"


def summarize_outcomes(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate canonical outcome counts across a sequence of result rows."""
    counts: Counter[str] = Counter()
    for row in records:
        bucket = classify_finish_reason(
            finish_reason=row.get("finish_reason"),
            passed=bool(row.get("passed")),
            status_code=row.get("status_code", 200),
            error=row.get("error"),
        )
        counts[bucket] += 1
    return {
        "passed": counts["passed"],
        "wrong_answer": counts["wrong_answer"],
        "wall_cap_timeout": counts["wall_cap_timeout"],
        "length_cap": counts["length_cap"],
        "server_503": counts["server_503"],
        "http_error": counts["http_error"],
    }


def accuracy_when_finished(passed: int, wrong_answer: int) -> float:
    """Compute conditional accuracy percentage among normally finished responses."""
    finished = passed + wrong_answer
    if finished <= 0:
        return 0.0
    return round((passed / finished) * 100.0, 1)


def get_cached_tokens(usage: Optional[Dict[str, Any]]) -> int:
    """Extract prefix-cache hit token count from OpenAI-compatible ``usage`` payload."""
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return int(details["cached_tokens"])
    return int(usage.get("cached_tokens") or 0)
