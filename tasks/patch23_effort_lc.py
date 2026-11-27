#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EVREN LLM v5 Clean Patch Suite (23 calls):
  - F3 Long-Context 3 gap calls (LC_350K deepseek off, LC_200K glm on, LC_200K deepseek on)
  - F1 Effort SweetSpot clean re-run (20 calls: glm-5.3 & qwen3.8-flash-next x low/medium x UC01..UC05, wall_cap=420s, max_tokens=16384)
Saves FULL raw content, reasoning_content, usage (incl. cached_tokens), finish_reason, status_code, retries, latency_s,
plus fixed target-symbol code extraction & pytest output.
"""
import os, sys, json, time, re, datetime, tempfile, subprocess, http.client
from urllib.parse import urlparse

from tasks import ultra_hard_20 as uh3

BASE_URL = os.environ.get("EVREN_LLM_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1").rstrip("/")
KEY_OFF = os.environ.get("EVREN_LLM_API_KEY_REASONING_OFF") or os.environ.get("EVREN_LLM_API_KEY", "")
KEY_ON = os.environ.get("EVREN_LLM_API_KEY_REASONING_ON") or os.environ.get("EVREN_LLM_API_KEY", "")

OUT_DIR = os.environ.get("EVREN_BENCH_OUT_DIR", "results/patch23_effort_lc")
OUT_PATH = os.path.join(OUT_DIR, "results.jsonl")

TARGET_SYMBOLS = {
    "UC01_mvcc_snapshot_kv": ["MVCCStore"],
    "UC02_sql_mini_engine": ["execute_query"],
    "UC03_interval_multiset": ["IntervalMultiset"],
    "UC04_exact_frac_linear": ["solve_linear_system"],
    "UC05_dfa_equiv_minimize": ["are_dfas_equivalent", "minimal_state_count"],
}

def tsi_now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S TSİ")

def load_done():
    done = set()
    if os.path.exists(OUT_PATH):
        for line in open(OUT_PATH, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["run_key"])
                except Exception:
                    pass
    return done

def append_row(row):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

def call_api_stream(model, mode, messages, max_tokens=16384, wall_cap=420, extra_body=None):
    api_key = KEY_ON if mode == "reasoning_on" else KEY_OFF
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        payload.update(extra_body)

    u = urlparse(f"{BASE_URL}/chat/completions")
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    retries = 0
    last_status = 0
    last_err = None
    for attempt in range(4):
        if attempt > 0:
            retries += 1
        t0 = time.perf_counter()
        content_parts, reas_parts = [], []
        finish_reason, routed_model, usage, err = None, None, {}, None
        conn = None
        try:
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=120)
            conn.request("POST", u.path, body=body, headers=headers)
            resp = conn.getresponse()
            last_status = resp.status
            if resp.status in (429, 502, 503, 504):
                err_txt = resp.read().decode("utf-8", "ignore")[:240]
                conn.close()
                last_err = f"HTTP {resp.status}: {err_txt}"
                time.sleep(8 * (attempt + 1))
                continue
            if resp.status != 200:
                err_txt = resp.read().decode("utf-8", "ignore")[:300]
                conn.close()
                return {
                    "ok_http": False, "status_code": resp.status, "retries": retries,
                    "content": "", "reasoning_content": "",
                    "finish_reason": f"http_{resp.status}", "routed_model": None,
                    "latency_s": round(time.perf_counter() - t0, 2), "usage": {}, "error": err_txt,
                }
            while True:
                if time.perf_counter() - t0 > wall_cap:
                    finish_reason = "wall_cap"
                    err = f"wall_cap_{wall_cap}s"
                    break
                chunk = resp.readline()
                if not chunk:
                    break
                line = chunk.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    ev = json.loads(data_str)
                except Exception:
                    continue
                if ev.get("model") and not routed_model:
                    routed_model = ev.get("model")
                if ev.get("usage"):
                    usage = ev["usage"]
                choices = ev.get("choices") or []
                if choices:
                    c0 = choices[0]
                    if c0.get("finish_reason"):
                        finish_reason = c0["finish_reason"]
                    delta = c0.get("delta") or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        reas_parts.append(rc)
            conn.close()
            return {
                "ok_http": True, "status_code": 200, "retries": retries,
                "content": "".join(content_parts), "reasoning_content": "".join(reas_parts),
                "finish_reason": finish_reason, "routed_model": routed_model,
                "latency_s": round(time.perf_counter() - t0, 2), "usage": usage, "error": err,
            }
        except Exception as e:
            if conn:
                try: conn.close()
                except Exception: pass
            last_err = str(e)[:240]
            time.sleep(6 * (attempt + 1))

    return {
        "ok_http": False, "status_code": last_status, "retries": retries,
        "content": "", "reasoning_content": "",
        "finish_reason": "error", "routed_model": None,
        "latency_s": wall_cap, "usage": {}, "error": last_err,
    }

def extract_target_code(content, required_symbols):
    """Extract the most complete/recent code block defining the required target def/class symbols."""
    text = content or ""
    candidates = []
    # 1. All closed fenced blocks
    for idx, m in enumerate(re.finditer(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)):
        blk = m.group(1).strip()
        if blk:
            candidates.append({"idx": idx, "code": blk, "closed": True, "source": "fenced"})
    # 2. Check for an unclosed trailing fence after the last closed fence
    last_close = 0
    for m in re.finditer(r"```(?:python|py)?\s*\n.*?```", text, flags=re.S | re.I):
        last_close = m.end()
    tail = text[last_close:]
    m_open = re.search(r"```(?:python|py)?\s*\n(.*)$", tail, flags=re.S | re.I)
    if m_open and m_open.group(1).strip():
        candidates.append({"idx": len(candidates), "code": m_open.group(1).strip(), "closed": False, "source": "unclosed_fence"})

    if not candidates and text.strip():
        candidates.append({"idx": 0, "code": text.strip(), "closed": True, "source": "raw"})

    if not candidates:
        return "", {"selected_idx": None, "total_blocks": 0, "symbols_found": [], "source": "empty"}

    def score_cand(c):
        code = c["code"]
        found = [s for s in required_symbols if re.search(rf"^\s*(?:def|class)\s+{re.escape(s)}\b", code, re.M)]
        has_any_def = 1 if re.search(r"^\s*(?:def|class)\s+\w+", code, re.M) else 0
        # Prioritize: (1) count of target symbols defined, (2) closed fence, (3) has any def/class, (4) latest block index (revised final code), (5) length
        return (len(found), 1 if c["closed"] else 0, has_any_def, c["idx"], len(code))

    best = max(candidates, key=score_cand)
    found = [s for s in required_symbols if re.search(rf"^\s*(?:def|class)\s+{re.escape(s)}\b", best["code"], re.M)]
    return best["code"], {
        "selected_idx": best["idx"],
        "total_blocks": len(candidates),
        "closed": best["closed"],
        "source": best["source"],
        "symbols_found": found,
        "selected_len": len(best["code"]),
    }

def run_pytest_clean(code, test_code):
    if not code.strip():
        return False, -1, "empty_extracted_code"
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(td, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write(test_code)
        try:
            res = subprocess.run(
                ["pytest", "-q", "--tb=short", "test_solution.py"],
                cwd=td, capture_output=True, text=True, timeout=20
            )
            out = (res.stdout + "\n" + res.stderr).strip()
            return res.returncode == 0, res.returncode, out[-1200:]
        except subprocess.TimeoutExpired:
            return False, -2, "pytest_timeout_20s"

def build_turkish_haystack(target_tokens):
    para = (
        "EVREN dağıtık yapay zeka altyapısı yüksek hızlı NVLink bağlantılı sunucu düğümleri üzerinde "
        "bilgisayarlı görü, doğal dil işleme ve çok modlu çıkarım iş yüklerini paralel kuyruk mimarisiyle yürütür. "
        "Veri hattı parçalı aktarım doğrulaması, SHA-256 bütünlük kontrolü ve otomatik şema denetimi içerir. "
        "İzleme katmanı her saniye GPU bellek kullanımı, kuyruk bekleme süresi ve ağ gecikmesini kaydeder.\n"
    )
    n_paras = max(30, int(target_tokens / 78))
    p1 = int(n_paras * 0.10)
    p2 = int(n_paras * 0.50)
    p3 = int(n_paras * 0.90)
    lines = []
    for i in range(n_paras):
        lines.append(f"[Bölüm {i+1:04d}] {para}")
        if i == p1:
            lines.append("[KRİTİK KAYIT A] Ana yedekleme sunucusunun doğrulama kodu KOD-7491 olarak atanmıştır.\n")
        elif i == p2:
            lines.append("[KRİTİK KAYIT B] Sistem büyük bakım penceresi kesin tarih olarak 2029-04-17 belirlenmiştir.\n")
        elif i == p3:
            lines.append("[KRİTİK KAYIT C] Fiziksel soğuk arşiv kasa konumu Bursa-Nilüfer-Depo-4 tesisindedir.\n")
    return "".join(lines)

def get_cached_tokens(usage):
    if not isinstance(usage, dict):
        return 0
    ptd = usage.get("prompt_tokens_details") or {}
    return int(ptd.get("cached_tokens") or usage.get("cached_tokens") or 0)

def main():
    done = load_done()
    UH3_MAP = {t["task_id"]: t for t in uh3.build_ultra_hard_v3_tasks()}
    print(f"=== V5 CLEAN PATCH23 START ({tsi_now()}) | total=23 | already_done={len(done)} ===", flush=True)

    # ---------- PART B FIRST (3 fast F3 Long-Context gap calls) ----------
    needle_q = (
        "\n\nYukarıdaki uzun teknik raporda 3 adet [KRİTİK KAYIT] gizlenmiştir. "
        "YALNIZCA şu JSON formatında yanıt ver: "
        "{\"kod\": \"...\", \"tarih\": \"...\", \"depo\": \"...\"}"
    )
    f3_jobs = [
        ("350K", 350000, "deepseek-v4.1-flash", "reasoning_off"),
        ("200K", 200000, "glm-5.3", "reasoning_on"),
        ("200K", 200000, "deepseek-v4.1-flash", "reasoning_on"),
    ]
    hay_cache = {}
    idx = 0
    for sz, tok_target, m, mode in f3_jobs:
        idx += 1
        rk = f"V5_F3_needle_{sz}__{m}__{mode}"
        if rk in done:
            print(f"[{idx:2d}/23] SKIP {rk}", flush=True)
            continue
        if sz not in hay_cache:
            hay_cache[sz] = build_turkish_haystack(tok_target)
        prompt = hay_cache[sz] + needle_q
        res = call_api_stream(m, mode, [{"role": "user", "content": prompt}], max_tokens=2048, wall_cap=300)
        txt = res["content"] or ""
        n1 = "KOD-7491" in txt
        n2 = "2029-04-17" in txt
        n3 = ("Bursa-Nilüfer-Depo-4" in txt) or ("Bursa-Nilufer-Depo-4" in txt)
        found = int(n1) + int(n2) + int(n3)
        clean_ok = res["ok_http"] and res["finish_reason"] not in ("wall_cap", "error") and (found == 3)
        reason = f"needles={found}/3 (A={n1},B={n2},C={n3})" if res["ok_http"] else f"HTTP {res['status_code']}: {str(res['error'])[:80]}"
        u = res["usage"] or {}
        row = {
            "run_key": rk,
            "module": "F3_LongContext_Gap",
            "task_id": f"LC_{sz}",
            "model": m,
            "mode": mode,
            "effort": None,
            "routed_model": res["routed_model"],
            "status_code": res["status_code"],
            "retries": res["retries"],
            "finish_reason": res["finish_reason"],
            "latency_s": res["latency_s"],
            "usage": u,
            "cached_tokens": get_cached_tokens(u),
            "error": res["error"],
            "content": res["content"],
            "reasoning_content": res["reasoning_content"],
            "evaluation": {
                "passed": bool(clean_ok),
                "needles_found": found,
                "reason": reason,
            },
            "created_at_tsi": tsi_now(),
        }
        append_row(row)
        print(f"[{idx:2d}/23] {'PASS' if clean_ok else 'FAIL'} | {rk:52s} | {res['latency_s']:6.2f}s | fr={res['finish_reason']} | pt={u.get('prompt_tokens',0)} cached={get_cached_tokens(u)} ct={u.get('completion_tokens',0)} | {reason}", flush=True)

    # ---------- PART A (20 clean F1_Effort_SweetSpot calls, wall_cap=420s) ----------
    coding_ids = [
        "UC01_mvcc_snapshot_kv",
        "UC02_sql_mini_engine",
        "UC03_interval_multiset",
        "UC04_exact_frac_linear",
        "UC05_dfa_equiv_minimize",
    ]
    for m in ["glm-5.3", "qwen3.8-flash-next"]:
        for eff in ["low", "medium"]:
            for tid in coding_ids:
                idx += 1
                rk = f"V5_F1_effort__{tid}__{m}__eff_{eff}"
                if rk in done:
                    print(f"[{idx:2d}/23] SKIP {rk}", flush=True)
                    continue
                t = UH3_MAP[tid]
                res = call_api_stream(
                    m, "reasoning_on",
                    [{"role": "user", "content": t["prompt"]}],
                    max_tokens=16384, wall_cap=420,
                    extra_body={"reasoning_effort": eff},
                )
                code, ext_meta = extract_target_code(res["content"], TARGET_SYMBOLS[tid])
                pt_ok, pt_rc, pt_out = run_pytest_clean(code, t["test_code"])
                is_wall_cap = (res["finish_reason"] == "wall_cap")
                is_http_err = (not res["ok_http"])
                if is_http_err:
                    passed = False
                    reason = f"HTTP {res['status_code']}: {str(res['error'])[:80]}"
                elif is_wall_cap:
                    passed = False
                    reason = f"wall_cap_420s (partial_pytest={'PASS' if pt_ok else 'FAIL'})"
                elif res["finish_reason"] == "length":
                    passed = False
                    reason = f"length_16k (partial_pytest={'PASS' if pt_ok else 'FAIL'})"
                else:
                    passed = bool(pt_ok)
                    last_line = pt_out.splitlines()[-1] if pt_out else ""
                    reason = "pytest PASS" if pt_ok else f"pytest FAIL: {last_line[:120]}"

                u = res["usage"] or {}
                row = {
                    "run_key": rk,
                    "module": "F1_Effort_SweetSpot_Clean",
                    "task_id": tid,
                    "model": m,
                    "mode": "reasoning_on",
                    "effort": eff,
                    "routed_model": res["routed_model"],
                    "status_code": res["status_code"],
                    "retries": res["retries"],
                    "finish_reason": res["finish_reason"],
                    "latency_s": res["latency_s"],
                    "usage": u,
                    "cached_tokens": get_cached_tokens(u),
                    "error": res["error"],
                    "content": res["content"],
                    "reasoning_content": res["reasoning_content"],
                    "extraction": ext_meta,
                    "pytest": {
                        "passed_raw": bool(pt_ok),
                        "returncode": pt_rc,
                        "output": pt_out,
                    },
                    "evaluation": {
                        "passed": bool(passed),
                        "reason": reason,
                    },
                    "created_at_tsi": tsi_now(),
                }
                append_row(row)
                print(f"[{idx:2d}/23] {'PASS' if passed else 'FAIL'} | {rk:52s} | {res['latency_s']:6.2f}s | fr={res['finish_reason']} | blk={ext_meta.get('selected_idx')}/{ext_meta.get('total_blocks')} | {reason[:60]}", flush=True)

    print(f"=== V5 CLEAN PATCH23 ALL DONE ({tsi_now()}) ===", flush=True)

if __name__ == "__main__":
    main()

