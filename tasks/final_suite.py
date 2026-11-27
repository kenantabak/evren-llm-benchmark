#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EVREN LLM Final Benchmark Suite v4 (137 calls):
  F1: reasoning_effort (low/medium) on 5 ultra-hard coding tasks + deepseek 64K native cap check (33 calls)
  F2: Tool / Function Calling (single, parallel, restraint, typed params) & strict JSON schema (45 calls)
  F3: Long-Context Needle-in-a-Haystack (32K, 128K, 200K, 350K Turkish technical corpus) (23 calls)
  F4: Hallucination Resistance & Multi-Turn Constraint Tracking (36 calls)
Drive-checkpointed, sequential (max_workers=1), zero self-429.
"""
import os, sys, json, time, re, datetime, urllib.request, urllib.error, http.client
from urllib.parse import urlparse

from tasks import ultra_hard_20 as uh3

BASE_URL = os.environ.get("EVREN_LLM_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1").rstrip("/")
KEY_OFF = os.environ.get("EVREN_LLM_API_KEY_REASONING_OFF") or os.environ.get("EVREN_LLM_API_KEY", "")
KEY_ON = os.environ.get("EVREN_LLM_API_KEY_REASONING_ON") or os.environ.get("EVREN_LLM_API_KEY", "")

OUT_DIR = os.environ.get("EVREN_BENCH_OUT_DIR", "results/final_suite")
OUT_PATH = os.path.join(OUT_DIR, "results.jsonl")

MAIN_MODELS = ["deepseek-v4.1-flash", "glm-5.3", "qwen3.8-flash-next", "gemma-4-31b"]
ALL_9 = [(m, "reasoning_off") for m in MAIN_MODELS] + [(m, "reasoning_on") for m in MAIN_MODELS] + [("auto", "reasoning_off")]

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

def call_api_stream(model, mode, messages, max_tokens=16384, wall_cap=240, extra_body=None):
    """SSE streaming call with hard wall-clock cap and retry on 429/503."""
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

    for attempt in range(4):
        t0 = time.perf_counter()
        content_parts, reas_parts = [], []
        finish_reason, routed_model, usage, err = None, None, {}, None
        conn = None
        try:
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=90)
            conn.request("POST", u.path, body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status in (429, 502, 503, 504):
                err_txt = resp.read().decode("utf-8", "ignore")[:240]
                conn.close()
                time.sleep(8 * (attempt + 1))
                err = f"HTTP {resp.status}: {err_txt}"
                continue
            if resp.status != 200:
                err_txt = resp.read().decode("utf-8", "ignore")[:300]
                conn.close()
                return {
                    "ok_http": False, "status": resp.status, "content": "", "reasoning": "",
                    "finish_reason": f"http_{resp.status}", "routed_model": None,
                    "latency_s": round(time.perf_counter() - t0, 2), "usage": {}, "error": err_txt,
                }
            # Read SSE line-by-line
            buf = b""
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
                "ok_http": True, "status": 200,
                "content": "".join(content_parts), "reasoning": "".join(reas_parts),
                "finish_reason": finish_reason, "routed_model": routed_model,
                "latency_s": round(time.perf_counter() - t0, 2), "usage": usage, "error": err,
            }
        except Exception as e:
            if conn:
                try: conn.close()
                except Exception: pass
            err = str(e)[:240]
            time.sleep(6 * (attempt + 1))

    return {
        "ok_http": False, "status": 0, "content": "", "reasoning": "",
        "finish_reason": "error", "routed_model": None,
        "latency_s": wall_cap, "usage": {}, "error": err,
    }

def call_api_sync(model, mode, messages, max_tokens=2048, timeout=110, extra_body=None):
    """Non-streaming call (ideal for tool_calls and strict response_format)."""
    api_key = KEY_ON if mode == "reasoning_on" else KEY_OFF
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
                ch = (d.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                return {
                    "ok_http": True, "status": r.status,
                    "content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "tool_calls": msg.get("tool_calls") or [],
                    "finish_reason": ch.get("finish_reason"),
                    "routed_model": d.get("model"),
                    "latency_s": round(time.perf_counter() - t0, 2),
                    "usage": d.get("usage") or {},
                    "error": None,
                }
        except urllib.error.HTTPError as e:
            err_txt = e.read().decode("utf-8", "ignore")[:260]
            if e.code in (429, 502, 503, 504):
                time.sleep(8 * (attempt + 1))
                continue
            return {
                "ok_http": False, "status": e.code, "content": "", "reasoning": "", "tool_calls": [],
                "finish_reason": f"http_{e.code}", "routed_model": None,
                "latency_s": round(time.perf_counter() - t0, 2), "usage": {}, "error": err_txt,
            }
        except Exception as e:
            if attempt < 3:
                time.sleep(6 * (attempt + 1))
                continue
            return {
                "ok_http": False, "status": 0, "content": "", "reasoning": "", "tool_calls": [],
                "finish_reason": "timeout_or_err", "routed_model": None,
                "latency_s": round(time.perf_counter() - t0, 2), "usage": {}, "error": str(e)[:240],
            }

def extract_json_obj(text):
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ===================== BUILD TASK LISTS =====================
UH3_MAP = {t["task_id"]: t for t in uh3.build_ultra_hard_v3_tasks()}

TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Belirtilen şehir için güncel hava durumunu getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir adı"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": "İki para birimi arasındaki anlık döviz kurunu döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Kaynak para birimi kodu (ör. USD)"},
                    "target": {"type": "string", "description": "Hedef para birimi kodu (ör. TRY)"},
                },
                "required": ["base", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Müşteri siparişlerini filtreleyerek listeler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "shipped", "delivered", "cancelled"]},
                    "min_total": {"type": "integer"},
                },
                "required": ["customer_id", "status", "min_total"],
            },
        },
    },
]

def build_turkish_haystack(target_tokens):
    """Build deterministic ~target_tokens Turkish technical text with 3 embedded needles at 10%, 50%, 90%."""
    para = (
        "EVREN dağıtık yapay zeka altyapısı yüksek hızlı NVLink bağlantılı sunucu düğümleri üzerinde "
        "bilgisayarlı görü, doğal dil işleme ve çok modlu çıkarım iş yüklerini paralel kuyruk mimarisiyle yürütür. "
        "Veri hattı parçalı aktarım doğrulaması, SHA-256 bütünlük kontrolü ve otomatik şema denetimi içerir. "
        "İzleme katmanı her saniye GPU bellek kullanımı, kuyruk bekleme süresi ve ağ gecikmesini kaydeder.\n"
    )
    # ~75 tokens per para (~400 chars in Turkish)
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

def run_all():
    done = load_done()
    total_planned = 33 + 45 + 23 + 36
    idx = 0
    print(f"=== FINAL SUITE V4 START ({tsi_now()}) | total={total_planned} | already_done={len(done)} ===")

    # ---------- F1A: reasoning_effort (low / medium) on 5 ultra-hard coding tasks (30 calls) ----------
    coding_ids = ["UC01_mvcc_snapshot_kv", "UC02_sql_mini_engine", "UC03_interval_multiset", "UC04_exact_frac_linear", "UC05_dfa_equiv_minimize"]
    for m in ["deepseek-v4.1-flash", "glm-5.3", "qwen3.8-flash-next"]:
        for eff in ["low", "medium"]:
            for tid in coding_ids:
                idx += 1
                rk = f"F1_effort__{tid}__{m}__eff_{eff}"
                if rk in done:
                    continue
                t = UH3_MAP[tid]
                res = call_api_stream(m, "reasoning_on", [{"role": "user", "content": t["prompt"]}], max_tokens=16384, wall_cap=240, extra_body={"reasoning_effort": eff})
                passed, reason = (lambda ev: (ev["passed"], ev["reason"]) if isinstance(ev, dict) else (ev[0], ev[-1]))(uh3.eval_ultra(t, res["content"])) if res["content"] else (False, res["error"] or "empty_output")
                row = {"run_key": rk, "module": "F1_Effort_SweetSpot", "task_id": tid, "model": m, "mode": f"reasoning_on_{eff}", "effort": eff, "routed_model": res["routed_model"], "telemetry": {"latency_s": res["latency_s"], "usage": res["usage"], "finish_reason": res["finish_reason"], "error": res["error"]}, "evaluation": {"passed": bool(passed), "reason": str(reason)[:160]}, "created_at_tsi": tsi_now()}
                append_row(row)
                print(f"[{idx:3d}/{total_planned}] {'PASS' if passed else 'FAIL'} | {rk:56s} | {res['latency_s']:5.1f}s | {str(reason)[:60]}")

    # ---------- F1B: deepseek-v4.1-flash 64K native cap check (3 calls) ----------
    for tid in ["UM02_matrix_recurrence_1e9", "UC02_sql_mini_engine", "UC05_dfa_equiv_minimize"]:
        idx += 1
        rk = f"F1_64k__{tid}__deepseek-v4.1-flash__reasoning_on"
        if rk in done:
            continue
        t = UH3_MAP[tid]
        res = call_api_stream("deepseek-v4.1-flash", "reasoning_on", [{"role": "user", "content": t["prompt"]}], max_tokens=65536, wall_cap=480)
        passed, reason = (lambda ev: (ev["passed"], ev["reason"]) if isinstance(ev, dict) else (ev[0], ev[-1]))(uh3.eval_ultra(t, res["content"])) if res["content"] else (False, res["error"] or "empty_output")
        row = {"run_key": rk, "module": "F1_DeepSeek_64K", "task_id": tid, "model": "deepseek-v4.1-flash", "mode": "reasoning_on_64k", "routed_model": res["routed_model"], "telemetry": {"latency_s": res["latency_s"], "usage": res["usage"], "finish_reason": res["finish_reason"], "error": res["error"]}, "evaluation": {"passed": bool(passed), "reason": str(reason)[:160]}, "created_at_tsi": tsi_now()}
        append_row(row)
        print(f"[{idx:3d}/{total_planned}] {'PASS' if passed else 'FAIL'} | {rk:56s} | {res['latency_s']:5.1f}s | {str(reason)[:60]}")

    # ---------- F2: Tool Calling & Strict JSON Schema (45 calls) ----------
    f2_tasks = [
        ("FT01_single_tool_exact", [{"role": "user", "content": "Ankara için hava durumunu santigrat (celsius) cinsinden öğren."}], {"tools": TOOLS_DEF, "tool_choice": "auto"}),
        ("FT02_parallel_2tools", [{"role": "user", "content": "Aynı anda hem İzmir hava durumunu (celsius) hem de USD -> TRY döviz kurunu ilgili araçlarla sorgula."}], {"tools": TOOLS_DEF, "tool_choice": "auto"}),
        ("FT03_no_tool_restraint", [{"role": "user", "content": "Türkiye Cumhuriyeti hangi yılda ilan edilmiştir? Araç çağırmadan yalnızca 4 haneli yılı yaz."}], {"tools": TOOLS_DEF, "tool_choice": "auto"}),
        ("FT04_typed_order_tool", [{"role": "user", "content": "CUST-4821 numaralı müşterinin kargoya verilmiş (shipped) ve tutarı en az 1500 TL olan siparişlerini ara."}], {"tools": TOOLS_DEF, "tool_choice": "auto"}),
        ("FT05_strict_json_invoice", [{"role": "user", "content": "Aşağıdaki faturayı YALNIZCA geçerli JSON nesnesi olarak çıkar. Şema: {\"invoice_no\": str, \"total_try\": float, \"items\": [{\"name\": str, \"qty\": int, \"unit_price\": float}]}.\nFatura No: INV-2026-88. Kalemler: 2 adet SSD Disk (birim 1450.50 TL), 1 adet RAM 32GB (birim 2800.00 TL), 3 adet Kablo (birim 120.00 TL). Toplam tutarı da doğru hesapla."}], {"response_format": {"type": "json_object"}}),
    ]

    def eval_f2(tid, res):
        if not res["ok_http"]:
            return False, f"HTTP {res['status']}: {str(res['error'])[:80]}"
        tcs = res.get("tool_calls") or []
        if tid == "FT01_single_tool_exact":
            for tc in tcs:
                fn = tc.get("function", {})
                if fn.get("name") == "get_weather":
                    try: args = json.loads(fn.get("arguments", "{}"))
                    except Exception: args = {}
                    if "ankara" in str(args.get("city", "")).lower() and args.get("unit") == "celsius":
                        return True, "OK"
            return False, f"tool_calls={str(tcs)[:120]}"
        if tid == "FT02_parallel_2tools":
            names = {tc.get("function", {}).get("name") for tc in tcs}
            if {"get_weather", "get_fx_rate"}.issubset(names):
                return True, "OK (parallel 2/2)"
            return False, f"got_tools={sorted(names)}"
        if tid == "FT03_no_tool_restraint":
            if len(tcs) == 0 and "1923" in res["content"]:
                return True, "OK (no spurious tool)"
            return False, f"tcs={len(tcs)} content={res['content'][:60]}"
        if tid == "FT04_typed_order_tool":
            for tc in tcs:
                fn = tc.get("function", {})
                if fn.get("name") == "search_orders":
                    try: args = json.loads(fn.get("arguments", "{}"))
                    except Exception: args = {}
                    if args.get("customer_id") == "CUST-4821" and args.get("status") == "shipped" and int(args.get("min_total", -1)) == 1500:
                        return True, "OK"
            return False, f"tool_calls={str(tcs)[:120]}"
        if tid == "FT05_strict_json_invoice":
            obj = extract_json_obj(res["content"])
            if not isinstance(obj, dict):
                return False, f"invalid JSON: {res['content'][:80]}"
            expected_total = 2 * 1450.50 + 2800.0 + 3 * 120.0  # 6061.0
            items = obj.get("items")
            ok_no = obj.get("invoice_no") == "INV-2026-88"
            ok_tot = abs(float(obj.get("total_try", -1)) - expected_total) < 0.05
            ok_it = isinstance(items, list) and len(items) == 3 and all(isinstance(x.get("qty"), int) for x in items)
            if ok_no and ok_tot and ok_it:
                return True, "OK"
            return False, f"no={ok_no} tot={obj.get('total_try')} items={len(items) if isinstance(items, list) else 'bad'}"
        return False, "unknown"

    for m, mode in ALL_9:
        for tid, msgs, extra in f2_tasks:
            idx += 1
            rk = f"F2__{tid}__{m}__{mode}"
            if rk in done:
                continue
            res = call_api_sync(m, mode, msgs, max_tokens=2048, timeout=110, extra_body=extra)
            passed, reason = eval_f2(tid, res)
            row = {"run_key": rk, "module": "F2_ToolCalling_JSON", "task_id": tid, "model": m, "mode": mode, "routed_model": res["routed_model"], "telemetry": {"latency_s": res["latency_s"], "usage": res["usage"], "finish_reason": res["finish_reason"], "error": res["error"]}, "evaluation": {"passed": bool(passed), "reason": str(reason)[:160]}, "created_at_tsi": tsi_now()}
            append_row(row)
            print(f"[{idx:3d}/{total_planned}] {'PASS' if passed else 'FAIL'} | {rk:56s} | {res['latency_s']:5.1f}s | {str(reason)[:60]}")

    # ---------- F3: Long-Context Needle-in-a-Haystack (23 calls) ----------
    haystacks = {
        "32K": build_turkish_haystack(32000),
        "128K": build_turkish_haystack(128000),
        "200K": build_turkish_haystack(200000),
        "350K": build_turkish_haystack(350000),
    }
    needle_q = (
        "\n\nYukarıdaki uzun teknik raporda 3 adet [KRİTİK KAYIT] gizlenmiştir. "
        "YALNIZCA şu JSON formatında yanıt ver: "
        "{\"kod\": \"...\", \"tarih\": \"...\", \"depo\": \"...\"}"
    )

    f3_plan = []
    for sz in ["32K", "128K", "200K"]:
        for m in MAIN_MODELS + ["auto"]:
            f3_plan.append((sz, m, "reasoning_off"))
    for m in MAIN_MODELS:
        f3_plan.append(("128K", m, "reasoning_on"))
    for m in ["deepseek-v4.1-flash", "glm-5.3"]:
        f3_plan.append(("350K", m, "reasoning_off"))
        f3_plan.append(("350K", m, "reasoning_on"))

    for sz, m, mode in f3_plan:
        idx += 1
        rk = f"F3_needle_{sz}__{m}__{mode}"
        if rk in done:
            continue
        prompt = haystacks[sz] + needle_q
        res = call_api_stream(m, mode, [{"role": "user", "content": prompt}], max_tokens=2048, wall_cap=240)
        txt = res["content"] or ""
        n1 = "KOD-7491" in txt
        n2 = "2029-04-17" in txt
        n3 = "Bursa-Nilüfer-Depo-4" in txt or "Bursa-Nilufer-Depo-4" in txt
        found = int(n1) + int(n2) + int(n3)
        passed = (found == 3)
        reason = f"needles={found}/3 (A={n1},B={n2},C={n3})" if res["ok_http"] else f"HTTP {res['status']}: {str(res['error'])[:80]}"
        row = {"run_key": rk, "module": "F3_LongContext_Needle", "task_id": f"LC_{sz}", "model": m, "mode": mode, "routed_model": res["routed_model"], "telemetry": {"latency_s": res["latency_s"], "usage": res["usage"], "finish_reason": res["finish_reason"], "error": res["error"]}, "evaluation": {"passed": bool(passed), "needles_found": found, "reason": reason}, "created_at_tsi": tsi_now()}
        append_row(row)
        print(f"[{idx:3d}/{total_planned}] {'PASS' if passed else 'FAIL'} | {rk:56s} | {res['latency_s']:5.1f}s | {reason[:60]}")

    # ---------- F4: Hallucination Resistance & Multi-Turn Tracking (36 calls) ----------
    f4_tasks = [
        ("FH01_unanswerable_trap_tr", [
            {"role": "user", "content": "Rapor Özeti: Anadolu Teknoloji A.Ş. 2024 yılında 42 milyon TL, 2025 yılında 68 milyon TL ciro elde etti. Toplam çalışan sayısı 114'tür.\n\nSoru: Rapora göre şirketin 2027 yılı AR-GE bütçesi kaç milyon TL'dir? Eğer bilgi raporda yoksa uydurma, YALNIZCA 'METINDE_YOK' yaz."}
        ]),
        ("FH02_false_premise_stdlib", [
            {"role": "user", "content": "Python 3.12 standart kütüphanesinde yer alan `quantum_zipfile_v9` modülünün `entangle_archive()` fonksiyonu standart kütüphanede gerçekten var mı? YALNIZCA JSON döndür: {\"exists\": true/false, \"note\": \"kısa açıklama\"}."}
        ]),
        ("FH03_multiturn_constraint_override", [
            {"role": "user", "content": "Şu 4 şehri madde işaretli listele: Ankara, İzmir, Bursa, Antalya."},
            {"role": "assistant", "content": "- Ankara\n- İzmir\n- Bursa\n- Antalya"},
            {"role": "user", "content": "Bu listeden Bursa'yı çıkar, yerine Trabzon ekle."},
            {"role": "assistant", "content": "- Ankara\n- İzmir\n- Antalya\n- Trabzon"},
            {"role": "user", "content": "Şimdi güncel listeyi alfabetik sıraya koy ve madde işareti veya numara kullanmadan TEK SATIRDA virgülle ayırarak yaz (yalnızca şehir adları)."},
        ]),
        ("FH04_multiturn_selective_update", [
            {"role": "user", "content": "Proje kartını aklında tut: PROJE_KODU=EVR-904, BUTCE_TL=420000, TESLIM=14 Ekim."},
            {"role": "assistant", "content": "Not aldım: EVR-904, 420000 TL bütçe, teslim 14 Ekim."},
            {"role": "user", "content": "Kısa bir ara soru: Linux'ta bir dosyanın son 10 satırını hangi komut gösterir?"},
            {"role": "assistant", "content": "`tail -n 10 dosya` komutu gösterir."},
            {"role": "user", "content": "Proje kartında yalnızca bütçeyi 515000 TL olarak güncelle, diğer alanlar aynı kalsın. YALNIZCA şu JSON ile dön: {\"kod\": \"...\", \"butce_tl\": int, \"teslim\": \"...\"}."},
        ]),
    ]

    def eval_f4(tid, res):
        if not res["ok_http"]:
            return False, f"HTTP {res['status']}: {str(res['error'])[:80]}"
        txt = (res["content"] or "").strip()
        if tid == "FH01_unanswerable_trap_tr":
            if "METINDE_YOK" in txt.upper() and not re.search(r"\b\d+\s*milyon", txt.lower()):
                return True, "OK (refused to hallucinate)"
            return False, f"got: {txt[:80]}"
        if tid == "FH02_false_premise_stdlib":
            obj = extract_json_obj(txt)
            if isinstance(obj, dict) and obj.get("exists") is False:
                return True, "OK (caught false premise)"
            return False, f"got: {txt[:80]}"
        if tid == "FH03_multiturn_constraint_override":
            clean = txt.strip().strip(".")
            lines = [l.strip() for l in clean.splitlines() if l.strip()]
            expected = ["Ankara", "Antalya", "İzmir", "Trabzon"]
            if len(lines) == 1 and not lines[0].startswith(("-", "*", "1")):
                parts = [p.strip() for p in lines[0].split(",")]
                if parts == expected or [p.replace("İ", "I") for p in parts] == [p.replace("İ", "I") for p in expected]:
                    return True, "OK"
            return False, f"got: {txt[:80]}"
        if tid == "FH04_multiturn_selective_update":
            obj = extract_json_obj(txt)
            if isinstance(obj, dict) and obj.get("kod") == "EVR-904" and int(obj.get("butce_tl", -1)) == 515000 and "14 ekim" in str(obj.get("teslim", "")).lower():
                return True, "OK"
            return False, f"got: {txt[:80]}"
        return False, "unknown"

    for m, mode in ALL_9:
        for tid, msgs in f4_tasks:
            idx += 1
            rk = f"F4__{tid}__{m}__{mode}"
            if rk in done:
                continue
            res = call_api_sync(m, mode, msgs, max_tokens=2048, timeout=110)
            passed, reason = eval_f4(tid, res)
            row = {"run_key": rk, "module": "F4_Hallucination_MultiTurn", "task_id": tid, "model": m, "mode": mode, "routed_model": res["routed_model"], "telemetry": {"latency_s": res["latency_s"], "usage": res["usage"], "finish_reason": res["finish_reason"], "error": res["error"]}, "evaluation": {"passed": bool(passed), "reason": str(reason)[:160]}, "created_at_tsi": tsi_now()}
            append_row(row)
            print(f"[{idx:3d}/{total_planned}] {'PASS' if passed else 'FAIL'} | {rk:56s} | {res['latency_s']:5.1f}s | {str(reason)[:60]}")

    print(f"=== FINAL SUITE V4 COMPLETED ({tsi_now()}) ===")

if __name__ == "__main__":
    run_all()
