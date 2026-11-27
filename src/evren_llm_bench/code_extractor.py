"""Target-symbol code extraction and deterministic calibration helpers.

Fixes false negatives caused by naive first-block fenced extraction when a model
presents an initial sketch followed by a revised final implementation, or when
Turkish thousand separators (e.g., ``116.280``) appear in numeric answers.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple


_FENCE_CLOSED_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.S | re.I)
_FENCE_OPEN_RE = re.compile(r"```(?:python|py)?\s*\n(.*)$", flags=re.S | re.I)
_ANY_DEF_RE = re.compile(r"^\s*(?:def|class)\s+\w+", flags=re.M)


def extract_target_code(
    content: Optional[str],
    required_symbols: Sequence[str] = (),
) -> Tuple[str, Dict[str, Any]]:
    """Extract the most complete/recent code block defining required symbols.

    Ranking priority:
      1. Number of ``required_symbols`` defined via ``def`` or ``class``.
      2. Closed markdown code fence over unclosed trailing fence.
      3. Presence of any top-level/indented ``def`` or ``class``.
      4. Latest block index (preferring the model's final revised solution).
      5. Character length as tie-breaker.
    """
    text = content or ""
    candidates: List[Dict[str, Any]] = []

    last_close = 0
    for idx, match in enumerate(_FENCE_CLOSED_RE.finditer(text)):
        block = match.group(1).strip()
        last_close = match.end()
        if block:
            candidates.append(
                {"idx": idx, "code": block, "closed": True, "source": "fenced"}
            )

    tail = text[last_close:]
    open_match = _FENCE_OPEN_RE.search(tail)
    if open_match and open_match.group(1).strip():
        candidates.append(
            {
                "idx": len(candidates),
                "code": open_match.group(1).strip(),
                "closed": False,
                "source": "unclosed_fence",
            }
        )

    if not candidates and text.strip():
        candidates.append(
            {"idx": 0, "code": text.strip(), "closed": True, "source": "raw"}
        )

    if not candidates:
        return "", {
            "selected_idx": None,
            "total_blocks": 0,
            "closed": False,
            "source": "empty",
            "symbols_found": [],
            "selected_len": 0,
        }

    def _defined_symbols(code: str) -> List[str]:
        return [
            sym
            for sym in required_symbols
            if re.search(rf"^\s*(?:def|class)\s+{re.escape(sym)}\b", code, flags=re.M)
        ]

    def _score(cand: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        code = cand["code"]
        found = _defined_symbols(code)
        has_any_def = 1 if _ANY_DEF_RE.search(code) else 0
        return (
            len(found),
            1 if cand["closed"] else 0,
            has_any_def,
            cand["idx"],
            len(code),
        )

    best = max(candidates, key=_score)
    found_symbols = _defined_symbols(best["code"])
    return best["code"], {
        "selected_idx": best["idx"],
        "total_blocks": len(candidates),
        "closed": best["closed"],
        "source": best["source"],
        "symbols_found": found_symbols,
        "selected_len": len(best["code"]),
    }


def has_forbidden_dynamic_compile(code: str) -> bool:
    """Detect bare builtin ``compile(...)`` calls without flagging ``re.compile(...)``."""
    return bool(re.search(r"(?<![.\w])compile\s*\(", code or ""))


def normalize_turkish_number(text: str) -> str:
    """Normalize Turkish thousand-grouped integers (e.g. ``116.280`` -> ``116280``).

    Also strips whitespace and normalizes Unicode curly apostrophes.
    """
    cleaned = unicodedata.normalize("NFKC", (text or "").strip())
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", cleaned):
        return cleaned.replace(".", "")
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", cleaned):
        return cleaned.replace(",", "")
    if re.fullmatch(r"-?\d{1,3}(?: \d{3})+", cleaned):
        return cleaned.replace(" ", "")
    return cleaned


def run_pytest_clean(
    code: str,
    test_code: str,
    timeout_s: int = 20,
    prepend_import: bool = False,
) -> Tuple[bool, int, str]:
    """Run ``pytest`` against extracted solution code inside an isolated temp dir."""
    if not (code or "").strip():
        return False, -1, "empty_extracted_code"

    final_test = (
        "from solution import *\n" + test_code
        if prepend_import and "from solution import" not in test_code
        else test_code
    )
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code + "\n")
        with open(os.path.join(td, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(final_test + "\n")
        try:
            res = subprocess.run(
                ["pytest", "-q", "--tb=short", "test_solution.py"],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = (res.stdout + "\n" + res.stderr).strip()
            return res.returncode == 0, res.returncode, out[-1200:]
        except subprocess.TimeoutExpired:
            return False, -2, f"pytest_timeout_{timeout_s}s"
