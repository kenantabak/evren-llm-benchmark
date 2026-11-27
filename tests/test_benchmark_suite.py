"""Unit and integrity tests for the EVREN LLM Benchmark suite."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import pytest

from evren_llm_bench.code_extractor import (
    extract_target_code,
    has_forbidden_dynamic_compile,
    normalize_turkish_number,
    run_pytest_clean,
)
from evren_llm_bench.metrics import (
    accuracy_when_finished,
    classify_finish_reason,
    compute_p50_p95,
    get_cached_tokens,
    summarize_outcomes,
)
from evren_llm_bench.runner import EvrenLLMRunner, build_ocr_multipart
from tasks.final_suite import build_turkish_haystack
from tasks.full_80 import build_full_tasks
from tasks.hard_30 import build_hard_v2_tasks
from tasks.patch23_effort_lc import TARGET_SYMBOLS
from tasks.ultra_hard_20 import build_ultra_hard_v3_tasks


REPO_ROOT = Path(__file__).resolve().parent.parent


def _pct_half_up(num: int, den: int) -> float:
    val = (Decimal(num) * Decimal(100)) / Decimal(den)
    return float(val.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def test_extract_target_code_prefers_final_block_with_required_symbols():
    content = (
        "İlk taslak:\n"
        "```python\n"
        "def helper():\n"
        "    return 1\n"
        "```\n"
        "Gözden geçirilmiş ilk sürüm:\n"
        "```python\n"
        "class MVCCStore:\n"
        "    def get(self): return 0\n"
        "```\n"
        "Nihai çalışan sürüm:\n"
        "```python\n"
        "class MVCCStore:\n"
        "    def get(self): return 42\n"
        "```\n"
    )
    code, meta = extract_target_code(content, ["MVCCStore"])
    assert "return 42" in code
    assert meta["selected_idx"] == 2
    assert meta["total_blocks"] == 3
    assert meta["symbols_found"] == ["MVCCStore"]
    assert meta["closed"] is True


def test_extract_target_code_handles_unclosed_trailing_fence():
    content = "Çözüm:\n```python\ndef solve_linear_system(a, b):\n    return [1]\n"
    code, meta = extract_target_code(content, ["solve_linear_system"])
    assert "def solve_linear_system" in code
    assert meta["closed"] is False
    assert meta["source"] == "unclosed_fence"
    assert meta["symbols_found"] == ["solve_linear_system"]


def test_extract_target_code_empty_input():
    code, meta = extract_target_code("", ["MVCCStore"])
    assert code == ""
    assert meta["source"] == "empty"
    assert meta["total_blocks"] == 0


def test_forbidden_compile_ignores_re_compile():
    assert has_forbidden_dynamic_compile("import re\np = re.compile(r'\\d+')") is False
    assert has_forbidden_dynamic_compile("obj = compile('1+1', '<str>', 'eval')") is True


def test_normalize_turkish_number():
    assert normalize_turkish_number("116.280") == "116280"
    assert normalize_turkish_number("1.234.567") == "1234567"
    assert normalize_turkish_number("123,456") == "123456"
    assert normalize_turkish_number("3.14") == "3.14"


def test_run_pytest_clean_executes_valid_solution():
    code = "def add(a, b):\n    return a + b\n"
    tests = "from solution import add\ndef test_add():\n    assert add(2, 3) == 5\n"
    ok, rc, out = run_pytest_clean(code, tests)
    assert ok is True
    assert rc == 0
    assert "1 passed" in out


def test_metrics_p50_p95_and_finish_reason_classification():
    p50, p95 = compute_p50_p95([10.0, 20.0, 30.0, 40.0, 420.0])
    assert p50 == 30.0
    assert p95 == 420.0

    rows = [
        {"finish_reason": "stop", "passed": True, "status_code": 200},
        {"finish_reason": "stop", "passed": False, "status_code": 200},
        {"finish_reason": "wall_cap", "passed": False, "status_code": 200, "error": "wall_cap_420s"},
        {"finish_reason": "length", "passed": False, "status_code": 200},
        {"finish_reason": "http_503", "passed": False, "status_code": 503},
    ]
    summary = summarize_outcomes(rows)
    assert summary == {
        "passed": 1,
        "wrong_answer": 1,
        "wall_cap_timeout": 1,
        "length_cap": 1,
        "server_503": 1,
        "http_error": 0,
    }
    assert accuracy_when_finished(41, 0) == 100.0
    assert accuracy_when_finished(45, 5) == 90.0
    assert classify_finish_reason("tool_calls", passed=True) == "passed"


def test_get_cached_tokens():
    assert get_cached_tokens({"prompt_tokens_details": {"cached_tokens": 244736}}) == 244736
    assert get_cached_tokens({"cached_tokens": 512}) == 512
    assert get_cached_tokens({}) == 0
    assert get_cached_tokens(None) == 0


def test_runner_env_key_resolution_and_ocr_multipart(monkeypatch):
    monkeypatch.delenv("EVREN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("EVREN_LLM_API_KEY_REASONING_OFF", raising=False)
    monkeypatch.delenv("EVREN_LLM_API_KEY_REASONING_ON", raising=False)
    runner = EvrenLLMRunner()
    with pytest.raises(ValueError, match="Missing EVREN API key"):
        runner.resolve_key("reasoning_off")

    monkeypatch.setenv("EVREN_LLM_API_KEY", "test-placeholder-value")
    runner2 = EvrenLLMRunner()
    assert runner2.resolve_key("reasoning_on") == "test-placeholder-value"

    sample_img = REPO_ROOT / "tasks" / "vision_assets" / "V05_tr_invoice.png"
    assert sample_img.exists()
    body, content_type = build_ocr_multipart(str(sample_img), model="deepseek-ocr-2", boundary="testbound")
    assert content_type == "multipart/form-data; boundary=testbound"
    assert b'name="model"\r\n\r\ndeepseek-ocr-2' in body
    assert b'filename="V05_tr_invoice.png"' in body


def test_summary_metrics_json_integrity():
    metrics_path = REPO_ROOT / "data" / "summary_metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert data["totals"]["logical_tasks"] == 1727
    assert data["totals"]["http_retries"] == 543
    assert data["totals"]["total_live_http_calls"] == 2270
    assert data["totals"]["logical_tasks"] + data["totals"]["http_retries"] == data["totals"]["total_live_http_calls"]

    lb = data["leaderboard_9_configs"]
    assert len(lb) == 9
    for row in lb:
        d80 = row["daily_80"]
        h50 = row["hard_50"]
        assert d80["total"] == 80
        assert h50["total"] == 50
        assert _pct_half_up(d80["passed"], d80["total"]) == d80["pct"]
        assert _pct_half_up(h50["passed"], h50["total"]) == h50["pct"]
        if "outcomes" in h50:
            assert sum(h50["outcomes"].values()) == 50
            assert h50["outcomes"]["passed"] == h50["passed"]

    charts = sorted((REPO_ROOT / "assets").glob("chart_0*.png"))
    assert len(charts) == 5
    vis_assets = sorted((REPO_ROOT / "tasks" / "vision_assets").glob("V*.png"))
    assert len(vis_assets) == 15


def test_task_builders_schema_and_counts():
    full_tasks = build_full_tasks()
    assert len(full_tasks) == 80
    assert len({t["task_id"] for t in full_tasks}) == 80

    hard_tasks = build_hard_v2_tasks()
    assert len(hard_tasks) == 30
    assert len({t["task_id"] for t in hard_tasks}) == 30

    ultra_tasks = build_ultra_hard_v3_tasks()
    assert len(ultra_tasks) == 20
    assert len({t["task_id"] for t in ultra_tasks}) == 20

    ultra_ids = {t["task_id"] for t in ultra_tasks}
    assert set(TARGET_SYMBOLS.keys()).issubset(ultra_ids)

    haystack = build_turkish_haystack(4000)
    assert "KOD-7491" in haystack
    assert "2029-04-17" in haystack
    assert "Bursa-Nilüfer-Depo-4" in haystack
