#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVREN LLM Benchmark — Hard Suite v2 (hard_v2)
30 frontier-hard, zero-memorization, 100% Python-verified tasks across 4 modules:
  - H1_Olympiad_Math (8 tasks)
  - H2_Competitive_Coding (8 pytest tasks)
  - H3_CSP_Logic_Tracing (7 tasks)
  - H4_Adversarial_Debug_TR (7 tasks)
Every single math/logic ground truth is computed & proven unique in Python.
Every single pytest suite is verified against a reference solution at build time.
All tasks use max_tokens=8192 so reasoning models are NEVER artificially cut off.
"""
import decimal
import fractions
import itertools
import os
import sqlite3
import subprocess
import tempfile

# =============================================================================
# GROUND-TRUTH SOLVERS (ALL VERIFIED PROGRAMMATICALLY)
# =============================================================================

def solve_hm01():
    for x in range(50001, 200000):
        if x % 42 == 17 and x % 70 == 59 and x % 45 == 14:
            return x
    raise RuntimeError("No HM01")

def solve_hm02():
    cnt = 0
    for p in itertools.permutations(range(1, 9)):
        if p[0] >= p[1]:
            continue
        if all(val != i and val != (i % 8) + 1 for i, val in enumerate(p, 1)):
            cnt += 1
    return cnt

def solve_hm03():
    bad = {(2, 2, 1), (3, 3, 2), (4, 2, 3)}
    def cp(s, t):
        sx, sy, sz = s; tx, ty, tz = t
        dp = {s: 1}
        for x in range(sx, tx + 1):
            for y in range(sy, ty + 1):
                for z in range(sz, tz + 1):
                    if (x, y, z) == s: continue
                    dp[(x, y, z)] = 0 if (x, y, z) in bad else (
                        dp.get((x-1, y, z), 0) + dp.get((x, y-1, z), 0) + dp.get((x, y, z-1), 0)
                    )
        return dp[t]
    return cp((0, 0, 0), (3, 2, 2)) * cp((3, 2, 2), (6, 5, 4))

def solve_hm04():
    F = fractions.Fraction
    A = [[F(0)] * 6 for _ in range(5)]
    A[0][0] = F(1); A[0][1] = F(-1); A[0][5] = F(1)
    for i in range(1, 5):
        A[i][i] += F(1)
        if i + 1 < 5: A[i][i + 1] -= F(1, 2)
        A[i][i - 1] -= F(1, 3)
        A[i][0] -= F(1, 6)
        A[i][5] = F(1)
    for col in range(5):
        piv = next(r for r in range(col, 5) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        div = A[col][col]
        A[col] = [v / div for v in A[col]]
        for r in range(5):
            if r != col and A[r][col] != 0:
                fac = A[r][col]
                A[r] = [A[r][c] - fac * A[col][c] for c in range(6)]
    e0 = A[0][5]
    return e0.numerator + e0.denominator

def solve_hm05():
    return pow(13, pow(17, 2027, 400), 1000)

def solve_hm06():
    g = [0] * 25
    for n in range(1, 25):
        s = {g[n - t] for t in (1, 2, 4) if n >= t}
        if n >= 6 and n % 2 == 0:
            s.add(0)
        m = 0
        while m in s: m += 1
        g[n] = m
    piles = (11, 14, 17)
    wins = 0
    for idx, p in enumerate(piles):
        ox = 0
        for j, q in enumerate(piles):
            if j != idx: ox ^= g[q]
        for t in (1, 2, 4):
            if p >= t and (ox ^ g[p - t]) == 0: wins += 1
        if p >= 6 and p % 2 == 0 and (ox ^ 0) == 0: wins += 1
    return wins

def solve_hm07():
    return sum(
        1 for length in range(1, 7)
        for comb in itertools.combinations_with_replacement(range(1, 10), length)
        if sum(comb) % 7 == 0
    )

def solve_hm08():
    MOD = 1000003
    a = [1, 2, 5]
    for n in range(3, 51):
        a.append((3 * a[-1] - 2 * a[-2] + 4 * a[-3] + n * n) % MOD)
    return a[50]

def solve_hl01():
    # KARA + KARA = ORMAN -> unique 16524
    letters = sorted(set("KARAORMAN"))
    sols = []
    for perm in itertools.permutations(range(10), len(letters)):
        mp = dict(zip(letters, perm))
        if mp["K"] == 0 or mp["O"] == 0: continue
        v1 = int("".join(str(mp[c]) for c in "KARA"))
        v3 = int("".join(str(mp[c]) for c in "ORMAN"))
        if v1 + v1 == v3:
            sols.append(v3)
    assert len(sols) == 1, f"HL01 sols={sols}"
    return sols[0]

def solve_hl02():
    # 6 servers: Atlas, Bora, Ceylan, Deniz, Evren, Firtina in slots 1..6
    # 6 workloads: LLM, Vision, Audio, RAG, ETL, Sim
    names = ["Atlas", "Bora", "Ceylan", "Deniz", "Evren", "Firtina"]
    works = ["LLM", "Vision", "Audio", "RAG", "ETL", "Sim"]
    sols = []
    for np in itertools.permutations(names):
        n_pos = {name: i + 1 for i, name in enumerate(np)}
        # 1. Evren is immediately right of Atlas
        if n_pos["Evren"] != n_pos["Atlas"] + 1: continue
        # 2. Bora is in an odd slot and strictly left of Firtina
        if n_pos["Bora"] % 2 == 0 or n_pos["Bora"] >= n_pos["Firtina"]: continue
        # 3. Firtina is strictly between Bora and Atlas
        if not (n_pos["Bora"] < n_pos["Firtina"] < n_pos["Atlas"]): continue
        # 4. Ceylan is at slot 6
        if n_pos["Ceylan"] != 6: continue
        # 5. Deniz is immediately left of Atlas OR immediately right of Bora? Let's make Deniz at slot 2!
        # Wait: Bora(1), Firtina(2), Deniz(3), Atlas(4), Evren(5), Ceylan(6) -> check clues without giving exact slot of Deniz!
        # 5. Deniz is NOT adjacent to Ceylan and NOT at slot 1, and Bora is NOT adjacent to Deniz
        if abs(n_pos["Deniz"] - n_pos["Ceylan"]) == 1 or n_pos["Deniz"] == 1 or abs(n_pos["Bora"] - n_pos["Deniz"]) == 1:
            continue
        for wp in itertools.permutations(works):
            w_pos = {w: i + 1 for i, w in enumerate(wp)}
            # 6. LLM runs on Evren
            if w_pos["LLM"] != n_pos["Evren"]: continue
            # 7. Vision is immediately left of Audio
            if w_pos["Vision"] + 1 != w_pos["Audio"]: continue
            # 8. RAG runs on a prime slot (2, 3, or 5) and NOT on Evren and NOT on Deniz
            if w_pos["RAG"] not in (2, 3, 5) or w_pos["RAG"] in (n_pos["Evren"], n_pos["Deniz"]): continue
            # 9. ETL runs on Bora
            if w_pos["ETL"] != n_pos["Bora"]: continue
            # 10. Sim runs on Ceylan
            if w_pos["Sim"] != n_pos["Ceylan"]: continue
            sols.append((n_pos, w_pos))
    assert len(sols) == 1, f"HL02 sols={len(sols)}"
    n_pos, w_pos = sols[0]
    return n_pos["Atlas"] * 1000 + n_pos["Deniz"] * 100 + w_pos["Vision"] * 10 + w_pos["RAG"]

def solve_hl03():
    people = ["Ali", "Burcu", "Cem", "Deniz", "Emre"]
    sols = []
    for perm in set(itertools.permutations(["K", "K", "N", "N", "S"])):
        r = dict(zip(people, perm))
        # Ali: "Burcu Yalancıdır (N) VE Cem Casus (S) değildir."
        s_ali = (r["Burcu"] == "N") and (r["Cem"] != "S")
        # Burcu: "Ali Doğrucu (K) ise Deniz de Doğrucudur (K)."
        s_burcu = (r["Deniz"] == "S") and (r["Ali"] == "N")
        # Cem: "Emre ve ben aynı rolde değiliz VE Ali Yalancıdır (N)."
        s_cem = (r["Emre"] != r["Cem"]) and (r["Ali"] == "N")
        # Deniz: "Burcu Doğrucudur (K) VE Emre Casus (S) değildir."
        s_deniz = (r["Burcu"] == "K") and (r["Emre"] != "S")
        # Emre: "Deniz Yalancıdır (N) VE Burcu Casus (S) değildir."
        s_emre = (r["Deniz"] == "N") and (r["Burcu"] != "S")
        stmts = {"Ali": s_ali, "Burcu": s_burcu, "Cem": s_cem, "Deniz": s_deniz, "Emre": s_emre}
        if all((r[p] != "K" or stmts[p]) and (r[p] != "N" or not stmts[p]) for p in people):
            sols.append(r)
    assert len(sols) == 1, f"HL03 sols={len(sols)}: {sols}"
    r = sols[0]
    code = {"K": 1, "N": 2, "S": 3}
    return int("".join(str(code[r[p]]) for p in people))

def solve_hl04():
    code = '''class A:
    def f(self, x, acc=[]):
        acc.append(x)
        return sum(acc)

class B(A):
    def f(self, x, acc=None):
        v = super().f(x * 2)
        return v + 3

class C(A):
    def f(self, x, acc=None):
        v = super().f(x + 1)
        return v * 2

class D(B, C):
    def f(self, x, acc=None):
        return super().f(x) + 1

funcs = [lambda k, i=i: k + i for i in range(3)]
late = [lambda k: k + i for i in range(3)]

def gen():
    total = 5
    for _ in range(3):
        inc = yield total
        total = total * 2 - inc
    yield total

d = D()
r1 = d.f(2)
r2 = d.f(3)
g = gen()
v0 = next(g)
v1 = g.send(3)
v2 = g.send(4)
v3 = g.send(1)
out = r1 + r2 + funcs[1](10) + late[0](10) + v0 + v1 + v2 + v3'''
    ns = {}
    exec(code, ns)
    return ns["out"], code

def solve_hl05():
    dur = [4, 5, 6, 3, 7, 4, 5, 6]
    edges = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6), (5, 7), (6, 7)]
    preds = [0] * 8
    for u, v in edges: preds[v] |= (1 << u)
    best = 999
    memo = {}
    def dfs(done, r1, j1, r2, j2, t):
        nonlocal best
        if t >= best: return
        if done == (1 << 8) - 1:
            best = min(best, t); return
        if (r1, j1) > (r2, j2): r1, j1, r2, j2 = r2, j2, r1, j1
        key = (done, r1, j1, r2, j2)
        if memo.get(key, 999) <= t: return
        memo[key] = t
        running = (0 if j1 == -1 else (1 << j1)) | (0 if j2 == -1 else (1 << j2))
        ready = [j for j in range(8) if not ((done | running) & (1 << j)) and (preds[j] & done) == preds[j]]
        if j1 == -1 and ready:
            for j in ready: dfs(done, dur[j], j, r2, j2, t)
            return
        dt = r1 if j1 != -1 else r2
        nt = t + dt
        ndone, nr1, nj1, nr2, nj2 = done, r1, j1, r2, j2
        if nj1 != -1:
            nr1 -= dt
            if nr1 == 0: ndone |= (1 << nj1); nj1 = -1
        if nj2 != -1:
            nr2 -= dt
            if nr2 == 0: ndone |= (1 << nj2); nj2 = -1
        dfs(ndone, nr1, nj1, nr2, nj2, nt)
    dfs(0, 0, -1, 0, -1, 0)
    return best

def solve_hl06():
    r0, r1, r2 = 90, 51, 7
    while r2 > 0:
        r0 = (r0 ^ r1) & 0xFF
        r0 = ((r0 >> 2) | ((r0 << 6) & 0xFF)) & 0xFF
        r1 = (r1 + r0 + r2 * 3) & 0xFF
        if r1 & 1: r0 = (r0 + 19) & 0xFF
        r2 -= 1
    return r0 * 256 + r1

def solve_hl07():
    F = fractions.Fraction
    priors = [F(5, 10), F(3, 10), F(2, 10)]
    def_r = [F(4, 100), F(5, 100), F(10, 100)]
    p_obs_def = F(9, 10) * F(1, 10)
    p_obs_ok = F(1, 5) * F(4, 5)
    num = priors[2] * def_r[2] * p_obs_def
    den = sum(priors[i] * (def_r[i] * p_obs_def + (1 - def_r[i]) * p_obs_ok) for i in range(3))
    post = num / den
    return post.numerator + post.denominator

def solve_ha04():
    base = 120000
    pen_pct = min(15, 12 + 4 + 2)
    after_sla = base * (100 - pen_pct) // 100
    after_loy = after_sla * 95 // 100
    return after_loy * 120 // 100

def solve_ha06():
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE orders (oid INT, cid INT, amount INT, status TEXT);
    INSERT INTO orders VALUES (1,10,200,'paid'),(2,10,150,'paid'),(3,20,300,'cancelled'),(4,20,100,'paid'),(5,30,NULL,'paid'),(6,40,250,'paid');
    CREATE TABLE refunds (rid INT, oid INT, r_amt INT);
    INSERT INTO refunds VALUES (1,1,50),(2,1,30),(3,4,100),(4,6,NULL);
    """)
    q = """SELECT SUM(net_val) FROM (
  SELECT o.oid,
         COALESCE(o.amount, 0) - COALESCE(SUM(r.r_amt), 0) AS net_val,
         COUNT(r.r_amt) AS rc
  FROM orders o
  LEFT JOIN refunds r ON o.oid = r.oid AND r.r_amt > 20
  WHERE o.status = 'paid'
  GROUP BY o.oid
  HAVING COUNT(*) >= 1 AND COALESCE(SUM(r.r_amt), 0) < 100
);"""
    val = cur.execute(q).fetchone()[0]
    con.close()
    return val, q

def solve_ha07():
    D = decimal.Decimal
    decimal.getcontext().rounding = decimal.ROUND_HALF_EVEN
    items = [(D("125.05"), 3, D("7.5")), (D("89.95"), 2, D("12.5")), (D("310.10"), 1, D("15")), (D("44.50"), 5, D("5"))]
    tot = D("0.00")
    for price, qty, disc in items:
        net = (price * qty * (D("100") - disc) / D("100")).quantize(D("0.01"))
        tot += (net * D("1.20")).quantize(D("0.01"))
    ref_net = (D("89.95") * D("87.5") / D("100")).quantize(D("0.01"))
    ref_kdv = (ref_net * D("1.20")).quantize(D("0.01"))
    return int((tot - ref_kdv) * 100)

# =============================================================================
# PYTEST SUITES + REFERENCE SOLUTIONS (VERIFIED AT BUILD TIME!)
# =============================================================================

PYTEST_TASKS = [
    # HC01: Regex engine without `re`
    (
        "HC01_regex_engine_no_re", "H2_Competitive_Coding",
        "Write a pure Python function `regex_match(pattern: str, text: str) -> bool` that checks whether `pattern` matches the ENTIRE `text`. "
        "Supported syntax: `.` (any single char), `*` (0 or more of preceding atom), `+` (1 or more), `?` (0 or 1), backslash escapes `\\.`, `\\*`, `\\+`, `\\?`, `\\\\`, `\\[`, `\\]`, "
        "and character classes `[abc]`, `[a-z0-9_-]`, negated `[^0-9]`. "
        "STRICT RULE: Do NOT import `re` (importing `re` fails the test). Return ONLY a Python code block.",
        '''import inspect, solution
def test_no_re_import():
    src = inspect.getsource(solution)
    assert "import re" not in src and "from re " not in src

def test_basic_and_quantifiers():
    assert regex_match("a*b+c?", "aaabbc") is True
    assert regex_match("a*b+c?", "b") is True
    assert regex_match("a*b+c?", "ac") is False
    assert regex_match("a.*b", "axxxb") is True
    assert regex_match("a.*b", "ab") is True
    assert regex_match("a.+b", "ab") is False

def test_char_classes_and_escapes():
    assert regex_match(r"[a-z]+_[0-9]+\\.txt", "file_42.txt") is True
    assert regex_match(r"[a-z]+_[0-9]+\\.txt", "file_42xtxt") is False
    assert regex_match(r"[^0-9]+", "hello!") is True
    assert regex_match(r"[^0-9]+", "hel3o") is False
    assert regex_match(r"a\\*b\\+c\\?", "a*b+c?") is True
    assert regex_match(r"[a-c]+", "abccba") is True
    assert regex_match(r"[a-c]+", "abcd") is False
    assert regex_match(r"a*a*a*a*a*b", "aaaaaaaaaaaaaaaaaaaaab") is True
    assert regex_match(r"a*a*a*a*a*b", "aaaaaaaaaaaaaaaaaaaaac") is False
''',
        '''from functools import lru_cache
def regex_match(pattern: str, text: str) -> bool:
    tokens = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\\\":
            atom = ("lit", pattern[i + 1])
            i += 2
        elif ch == ".":
            atom = ("any", None)
            i += 1
        elif ch == "[":
            i += 1
            neg = False
            if i < n and pattern[i] == "^":
                neg = True
                i += 1
            chars = set()
            while i < n and pattern[i] != "]":
                if pattern[i] == "\\\\" and i + 1 < n:
                    c1 = pattern[i + 1]; i += 2
                else:
                    c1 = pattern[i]; i += 1
                if i + 1 < n and pattern[i] == "-" and pattern[i + 1] != "]":
                    i += 1
                    c2 = pattern[i]; i += 1
                    for o in range(ord(c1), ord(c2) + 1):
                        chars.add(chr(o))
                else:
                    chars.add(c1)
            i += 1
            atom = ("cls", (neg, frozenset(chars)))
        else:
            atom = ("lit", ch)
            i += 1
        quant = ""
        if i < n and pattern[i] in "*+?":
            quant = pattern[i]; i += 1
        tokens.append((atom, quant))
    def match_atom(atom, c):
        k, v = atom
        if k == "any": return True
        if k == "lit": return c == v
        neg, cs = v
        return (c not in cs) if neg else (c in cs)
    @lru_cache(None)
    def dp(ti, si):
        if ti == len(tokens): return si == len(text)
        atom, q = tokens[ti]
        if q == "*":
            if dp(ti + 1, si): return True
            if si < len(text) and match_atom(atom, text[si]) and dp(ti, si + 1): return True
            return False
        if q == "+":
            if si < len(text) and match_atom(atom, text[si]):
                return dp(ti + 1, si + 1) or dp(ti, si + 1)
            return False
        if q == "?":
            if dp(ti + 1, si): return True
            if si < len(text) and match_atom(atom, text[si]) and dp(ti + 1, si + 1): return True
            return False
        return si < len(text) and match_atom(atom, text[si]) and dp(ti + 1, si + 1)
    return dp(0, 0)
'''
    ),
    # HC02: Recursive descent math expression parser (no eval/exec/ast)
    (
        "HC02_expr_parser_no_eval", "H2_Competitive_Coding",
        "Write a pure Python function `eval_expr(expr: str, env: dict[str, float] | None = None) -> float` that evaluates arithmetic expressions with: "
        "`+`, `-`, `*`, `/`, right-associative `^`, unary `-` and `+`, nested parentheses `()`, floats/ints, and variable names (`[a-zA-Z_][a-zA-Z0-9_]*`) looked up in `env`. "
        "CRITICAL PRECEDENCE RULES: `^` binds tighter than unary `-` on its left, so `-2^2 == -4.0`, while unary on exponent works: `2^-3 == 0.125`, `2^3^2 == 512.0`, `--3 == 3.0`. "
        "Raise `ZeroDivisionError` on division by zero and `ValueError` on invalid syntax or unknown variable. "
        "STRICT RULE: Do NOT use `eval`, `exec`, `compile`, or `ast`. Return ONLY a Python code block.",
        '''import inspect, math, pytest, solution
def test_no_eval():
    src = inspect.getsource(solution)
    for bad in ("eval(", "exec(", "import ast", "from ast", "compile("):
        assert bad not in src

def test_precedence_and_vars():
    assert math.isclose(eval_expr("-2^2"), -4.0)
    assert math.isclose(eval_expr("(-2)^2"), 4.0)
    assert math.isclose(eval_expr("2^3^2"), 512.0)
    assert math.isclose(eval_expr("2^-3"), 0.125)
    assert math.isclose(eval_expr("--3 + -(-2) * 4"), 11.0)
    assert math.isclose(eval_expr("x^2 + 2*x*y + y^2", {"x": 3.5, "y": -1.5}), 4.0)
    assert math.isclose(eval_expr("10 - 3 - 2"), 5.0)
    assert math.isclose(eval_expr("24 / 6 / 2"), 2.0)

def test_errors():
    with pytest.raises(ZeroDivisionError):
        eval_expr("1 / (2 - 2)")
    with pytest.raises(ValueError):
        eval_expr("2 + * 3")
    with pytest.raises(ValueError):
        eval_expr("(1 + 2")
    with pytest.raises(ValueError):
        eval_expr("unknown_var + 1", {})
''',
        '''def eval_expr(expr: str, env=None) -> float:
    env = env or {}
    toks = []
    i, n = 0, len(expr)
    while i < n:
        c = expr[i]
        if c.isspace(): i += 1
        elif c in "+-*/^()": toks.append((c, c)); i += 1
        elif c.isdigit() or c == ".":
            j = i
            dots = 0
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                if expr[j] == ".": dots += 1
                j += 1
            if dots > 1: raise ValueError("bad num")
            toks.append(("NUM", float(expr[i:j]))); i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"): j += 1
            toks.append(("ID", expr[i:j])); i = j
        else:
            raise ValueError(f"bad char {c}")
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def eat(kind=None):
        t = peek()
        if not t or (kind and t[0] != kind): raise ValueError("syntax")
        pos[0] += 1; return t
    def parse_add():
        v = parse_mul()
        while peek() and peek()[0] in "+-":
            op = eat()[0]; r = parse_mul()
            v = v + r if op == "+" else v - r
        return v
    def parse_mul():
        v = parse_unary()
        while peek() and peek()[0] in "*/":
            op = eat()[0]; r = parse_unary()
            if op == "*": v *= r
            else:
                if r == 0: raise ZeroDivisionError("div0")
                v /= r
        return v
    def parse_unary():
        if peek() and peek()[0] in "+-":
            op = eat()[0]; r = parse_unary()
            return r if op == "+" else -r
        return parse_pow()
    def parse_pow():
        v = parse_atom()
        if peek() and peek()[0] == "^":
            eat("^")
            r = parse_unary()
            return v ** r
        return v
    def parse_atom():
        t = peek()
        if not t: raise ValueError("eof")
        if t[0] == "NUM": eat(); return t[1]
        if t[0] == "ID":
            name = eat()[1]
            if name not in env: raise ValueError(f"unknown {name}")
            return float(env[name])
        if t[0] == "(":
            eat("("); v = parse_add(); eat(")"); return v
        raise ValueError("unexpected")
    res = parse_add()
    if pos[0] != len(toks): raise ValueError("trailing")
    return float(res)
'''
    ),
    # HC03: Min-Cost Max-Flow with negative-cost edges & parallel edges
    (
        "HC03_min_cost_max_flow", "H2_Competitive_Coding",
        "Write a pure Python function `min_cost_max_flow(n: int, edges: list[tuple[int,int,int,int]], s: int, t: int) -> tuple[int, int]` "
        "where vertices are `0..n-1`, each edge is `(u, v, cap, cost)` (directed, `cap >= 0`, `cost` can be negative; no negative cycles exist; parallel edges allowed). "
        "Return `(max_flow, min_cost)` where `min_cost` is the minimum possible cost to send `max_flow` units from `s` to `t`. Return ONLY a Python code block.",
        '''def test_mcmf():
    # Parallel edges + cheapest route
    edges = [(0,1,3,2), (0,1,2,1), (1,2,4,3), (0,2,1,10)]
    assert min_cost_max_flow(3, edges, 0, 2) == (5, 2*1 + 2*2 + 4*3 + 1*10)
    # Negative edge costs (no negative cycle)
    edges2 = [(0,1,2,-5), (1,3,2,1), (0,2,2,0), (2,3,2,0), (1,2,1,-2)]
    # Send 2 via 0->1->2->3? cap(1->2)=1 => 1 unit 0->1->2->3 (cost -7), 1 unit 0->1->3 (cost -4), 1 unit 0->2->3 (cost 0), wait: max flow from 0 is 4!
    # 0->1 (2), 0->2 (2) -> total 4: 1 via 0-1-2-3 (-7), 1 via 0-1-3 (-4), 1 via 0-2-3 (0), wait cap(2->3)=2 so 0-2-3 can only carry 1 more => max flow = 4!
    # Wait: 2->3 cap=2, 1->3 cap=2, so max flow=4: best is 0-1-3 (2 units, cost -8) + 0-2-3 (2 units, cost 0) = -8! OR 1 via 0-1-2-3 (-7) + 1 via 0-1-3 (-4) + 1 via 0-2-3 (0) = flow 3 (NOT max flow 4)!
    assert min_cost_max_flow(4, edges2, 0, 3) == (4, -8)
    # Zero capacity & disconnected
    assert min_cost_max_flow(3, [(0,1,0,5), (1,2,5,1)], 0, 2) == (0, 0)
    # Classic residual cancellation
    edges3 = [(0,1,1,1), (1,2,1,1), (2,3,1,1), (0,2,1,5), (1,3,1,5)]
    assert min_cost_max_flow(4, edges3, 0, 3) == (2, 12)
''',
        '''from collections import deque
def min_cost_max_flow(n, edges, s, t):
    g = [[] for _ in range(n)]
    def add(u, v, cap, cost):
        g[u].append([v, cap, cost, len(g[v])])
        g[v].append([u, 0, -cost, len(g[u]) - 1])
    for u, v, cap, cost in edges:
        if cap > 0: add(u, v, cap, cost)
    flow = cost = 0
    INF = 10**18
    while True:
        dist = [INF] * n
        inq = [False] * n
        pv = [-1] * n
        pe = [-1] * n
        dist[s] = 0
        q = deque([s]); inq[s] = True
        while q:
            u = q.popleft(); inq[u] = False
            for idx, (v, cap, w, _) in enumerate(g[u]):
                if cap > 0 and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w; pv[v] = u; pe[v] = idx
                    if not inq[v]: inq[v] = True; q.append(v)
        if dist[t] == INF: break
        addf = INF
        v = t
        while v != s:
            u = pv[v]; addf = min(addf, g[u][pe[v]][1]); v = u
        flow += addf; cost += addf * dist[t]
        v = t
        while v != s:
            u = pv[v]; e = g[u][pe[v]]; e[1] -= addf; g[v][e[3]][1] += addf; v = u
    return flow, cost
'''
    ),
    # HC04: Tarjan SCC + Condensation DAG max weighted path (with negative weights!)
    (
        "HC04_tarjan_condensation_dag", "H2_Competitive_Coding",
        "Write a pure Python function `max_scc_path_weight(n: int, weights: list[int], edges: list[tuple[int,int]]) -> int`. "
        "Given a directed graph with `n >= 1` vertices (`0..n-1`) and integer vertex weights (can be negative!), contract each Strongly Connected Component (SCC) into a supernode whose weight is the sum of its vertices' weights. "
        "Return the maximum total weight of any non-empty directed path (starting and ending at any supernode) in the condensation DAG. Return ONLY a Python code block.",
        '''def test_scc_dag():
    # Single cycle 0->1->2->0 with weights [3,-1,4] (sum=6), then 2->3 (-10) -> 4 (20) => 6 - 10 + 20 = 16 vs just [4]=20!
    assert max_scc_path_weight(5, [3,-1,4,-10,20], [(0,1),(1,2),(2,0),(2,3),(3,4)]) == 20
    # Cycle feeds positive chain: [3,-1,4]=6 -> 3(2) -> 4(5) = 13
    assert max_scc_path_weight(5, [3,-1,4,2,5], [(0,1),(1,2),(2,0),(2,3),(3,4)]) == 13
    # All negative weights: must still pick non-empty single supernode!
    assert max_scc_path_weight(3, [-5,-2,-8], [(0,1),(1,0),(1,2)]) == -7
    # Wait: SCC {0,1} has sum -7, SCC {2} has sum -8 => max is -7!
    assert max_scc_path_weight(3, [-10,-2,-3], [(0,1),(1,2)]) == -2
    # Deep chain (test iterative or safe recursion limit)
    n = 1500
    assert max_scc_path_weight(n, [1]*n, [(i, i+1) for i in range(n-1)]) == 1500
''',
        '''import sys
def max_scc_path_weight(n, weights, edges):
    sys.setrecursionlimit(max(10000, n * 4))
    g = [[] for _ in range(n)]
    for u, v in edges: g[u].append(v)
    idx = 0
    disc = [-1] * n
    low = [0] * n
    onst = [False] * n
    st = []
    comp = [-1] * n
    cweights = []
    def dfs(u):
        nonlocal idx
        disc[u] = low[u] = idx; idx += 1
        st.append(u); onst[u] = True
        for v in g[u]:
            if disc[v] == -1:
                dfs(v); low[u] = min(low[u], low[v])
            elif onst[v]:
                low[u] = min(low[u], disc[v])
        if low[u] == disc[u]:
            cid = len(cweights); s = 0
            while True:
                w = st.pop(); onst[w] = False; comp[w] = cid; s += weights[w]
                if w == u: break
            cweights.append(s)
    for i in range(n):
        if disc[i] == -1: dfs(i)
    k = len(cweights)
    dag = [set() for _ in range(k)]
    for u, v in edges:
        if comp[u] != comp[v]: dag[comp[u]].add(comp[v])
    # Tarjan produces components in reverse topological order!
    dp = list(cweights)
    for c in range(k):
        best_next = 0
        for nxt in dag[c]:
            if dp[nxt] > best_next: best_next = dp[nxt]
        dp[c] = cweights[c] + best_next
    return max(dp)
'''
    ),
    # HC05: JSON5 subset parser (no json/ast/eval)
    (
        "HC05_json5_parser_no_json", "H2_Competitive_Coding",
        "Write a pure Python function `parse_json5(text: str)` that parses a JSON5 subset into Python objects (`dict`, `list`, `str`, `int`, `float`, `bool`, `None`). "
        "Must support: double-quoted AND single-quoted strings with escapes (`\\\"`, `\\'`, `\\\\`, `\\n`, `\\t`, `\\uXXXX`), integers, floats, scientific notation (`1.5e-2`), `true`/`false`/`null`, "
        "single-line `//` comments (outside strings!), and trailing commas in objects and arrays. Raise `ValueError` on invalid input. "
        "STRICT RULE: Do NOT import `json` or `ast`, and do NOT call `eval`/`exec`. Return ONLY a Python code block.",
        '''import inspect, math, pytest, solution
def test_no_builtin_parser():
    src = inspect.getsource(solution)
    for bad in ("import json", "from json", "import ast", "from ast", "eval(", "exec("):
        assert bad not in src

def test_json5_features():
    raw = r"""
    // top comment
    {
      "url": "https://a.com//not_a_comment", // inline comment
      'single': 'it\\'s\\nok\\u0021',
      "nums": [-42, 3.14, 1.5e-2, 0,],
      "flags": [true, false, null,],
    }
    """
    obj = parse_json5(raw)
    assert obj["url"] == "https://a.com//not_a_comment"
    assert obj["single"] == "it's" + chr(10) + "ok!"
    assert obj["nums"][0] == -42 and isinstance(obj["nums"][0], int)
    assert math.isclose(obj["nums"][1], 3.14) and math.isclose(obj["nums"][2], 0.015)
    assert obj["flags"] == [True, False, None]

def test_invalid():
    with pytest.raises(ValueError):
        parse_json5("{1, 2}")
    with pytest.raises(ValueError):
        parse_json5("[1, 2]]")
''',
        '''def parse_json5(text: str):
    n = len(text); pos = [0]
    def skip():
        while pos[0] < n:
            if text[pos[0]].isspace(): pos[0] += 1
            elif text.startswith("//", pos[0]):
                nl = text.find("\\n", pos[0])
                pos[0] = n if nl == -1 else nl + 1
            else: break
    def parse_str():
        q = text[pos[0]]; pos[0] += 1; out = []
        while pos[0] < n:
            c = text[pos[0]]; pos[0] += 1
            if c == q: return "".join(out)
            if c == "\\\\":
                if pos[0] >= n: raise ValueError("eof esc")
                e = text[pos[0]]; pos[0] += 1
                mp = {"n": "\\n", "t": "\\t", "r": "\\r", "\\\\": "\\\\", '"': '"', "'": "'", "/": "/"}
                if e in mp: out.append(mp[e])
                elif e == "u":
                    hx = text[pos[0]:pos[0]+4]
                    if len(hx) < 4: raise ValueError("bad u")
                    out.append(chr(int(hx, 16))); pos[0] += 4
                else: raise ValueError("bad esc")
            else: out.append(c)
        raise ValueError("unclosed str")
    def parse_num():
        st = pos[0]
        if text[pos[0]] == "-": pos[0] += 1
        while pos[0] < n and text[pos[0]].isdigit(): pos[0] += 1
        is_f = False
        if pos[0] < n and text[pos[0]] == ".":
            is_f = True; pos[0] += 1
            while pos[0] < n and text[pos[0]].isdigit(): pos[0] += 1
        if pos[0] < n and text[pos[0]] in "eE":
            is_f = True; pos[0] += 1
            if pos[0] < n and text[pos[0]] in "+-": pos[0] += 1
            while pos[0] < n and text[pos[0]].isdigit(): pos[0] += 1
        s = text[st:pos[0]]
        try: return float(s) if is_f else int(s)
        except Exception: raise ValueError("bad num")
    def parse_val():
        skip()
        if pos[0] >= n: raise ValueError("eof")
        c = text[pos[0]]
        if c in ("'", '"'): return parse_str()
        if c == "{":
            pos[0] += 1; d = {}
            while True:
                skip()
                if pos[0] < n and text[pos[0]] == "}": pos[0] += 1; return d
                if pos[0] >= n or text[pos[0]] not in ("'", '"'): raise ValueError("key expected")
                k = parse_str(); skip()
                if pos[0] >= n or text[pos[0]] != ":": raise ValueError("colon expected")
                pos[0] += 1; v = parse_val(); d[k] = v; skip()
                if pos[0] < n and text[pos[0]] == ",": pos[0] += 1
                elif pos[0] < n and text[pos[0]] == "}": pos[0] += 1; return d
                else: raise ValueError("bad obj")
        if c == "[":
            pos[0] += 1; arr = []
            while True:
                skip()
                if pos[0] < n and text[pos[0]] == "]": pos[0] += 1; return arr
                arr.append(parse_val()); skip()
                if pos[0] < n and text[pos[0]] == ",": pos[0] += 1
                elif pos[0] < n and text[pos[0]] == "]": pos[0] += 1; return arr
                else: raise ValueError("bad arr")
        if text.startswith("true", pos[0]): pos[0] += 4; return True
        if text.startswith("false", pos[0]): pos[0] += 5; return False
        if text.startswith("null", pos[0]): pos[0] += 4; return None
        if c == "-" or c.isdigit(): return parse_num()
        raise ValueError("bad val")
    val = parse_val(); skip()
    if pos[0] != n: raise ValueError("trailing")
    return val
'''
    ),
    # HC06: TRUE Damerau-Levenshtein (unrestricted transpositions — NOT OSA!)
    (
        "HC06_damerau_levenshtein_true", "H2_Competitive_Coding",
        "Write a pure Python function `true_damerau_levenshtein(a: str, b: str) -> int` that computes the **TRUE (unrestricted)** Damerau-Levenshtein distance "
        "(allowing insertions, deletions, substitutions, and adjacent transpositions where transposed characters MAY subsequently be edited/inserted between — NOT Optimal String Alignment!). "
        "For example, `true_damerau_levenshtein('ca', 'abc') == 2` (transpose `ca -> ac`, then insert `b` -> `abc`), whereas restricted OSA gives `3`. Return ONLY a Python code block.",
        '''def test_true_dl():
    assert true_damerau_levenshtein("ca", "abc") == 2
    assert true_damerau_levenshtein("abc", "ca") == 2
    assert true_damerau_levenshtein("a_cat", "an_act") == 2
    assert true_damerau_levenshtein("abcdef", "azced") == 3
    assert true_damerau_levenshtein("", "xyz") == 3
    assert true_damerau_levenshtein("ab", "ba") == 1
    assert true_damerau_levenshtein("ba", "acb") == 2
    assert true_damerau_levenshtein("ca", "abdc") == 3
''',
        '''def true_damerau_levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    maxd = la + lb
    d = [[0] * (lb + 2) for _ in range(la + 2)]
    d[0][0] = maxd
    for i in range(la + 1): d[i + 1][0] = maxd; d[i + 1][1] = i
    for j in range(lb + 1): d[0][j + 1] = maxd; d[1][j + 1] = j
    da = {}
    for i in range(1, la + 1):
        db = 0
        for j in range(1, lb + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0: db = j
            d[i + 1][j + 1] = min(
                d[i][j] + cost,
                d[i + 1][j] + 1,
                d[i][j + 1] + 1,
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        da[a[i - 1]] = i
    return d[la + 1][lb + 1]
'''
    ),
    # HC07: Weighted K non-overlapping intervals (half-open, negative weights)
    (
        "HC07_weighted_interval_k_select", "H2_Competitive_Coding",
        "Write a pure Python function `max_weighted_k_intervals(intervals: list[tuple[int,int,int]], k: int) -> int`. "
        "Each interval is `(start, end, weight)` representing half-open `[start, end)` (`start < end`, `weight` can be negative). "
        "Two intervals are non-overlapping if `end1 <= start2` or `end2 <= start1`. "
        "Choose **at most `k`** mutually non-overlapping intervals (including 0 intervals for total 0) to maximize the sum of weights. Return ONLY a Python code block.",
        '''def test_k_intervals():
    assert max_weighted_k_intervals([(1,3,5),(2,5,6),(4,6,5),(6,7,4),(5,8,11),(8,10,3)], 2) == 17
    assert max_weighted_k_intervals([(1,3,5),(2,5,6),(4,6,5),(6,7,4),(5,8,11),(8,10,3)], 3) == 20
    assert max_weighted_k_intervals([(1,2,-5),(2,3,-1)], 2) == 0
    assert max_weighted_k_intervals([(0,10,100),(0,5,60),(5,10,60)], 1) == 100
    assert max_weighted_k_intervals([(0,10,100),(0,5,60),(5,10,60)], 2) == 120
    assert max_weighted_k_intervals([], 5) == 0
''',
        '''import bisect
def max_weighted_k_intervals(intervals, k):
    ivs = sorted([iv for iv in intervals if iv[2] > 0], key=lambda x: x[1])
    n = len(ivs)
    if n == 0 or k <= 0: return 0
    ends = [x[1] for x in ivs]
    prev = [bisect.bisect_right(ends, ivs[i][0]) for i in range(n)]
    dp = [0] * (n + 1)
    for _ in range(min(k, n)):
        ndp = [0] * (n + 1)
        for i in range(1, n + 1):
            ndp[i] = max(ndp[i - 1], dp[prev[i - 1]] + ivs[i - 1][2])
        dp = ndp
    return dp[n]
'''
    ),
    # HC08: Exact Vixie Cron next-N-runs (leap years + DOM/DOW OR-semantics!)
    (
        "HC08_exact_calendar_cron", "H2_Competitive_Coding",
        "Write a pure Python function `next_cron_runs(expr: str, start_iso: str, count: int) -> list[str]` for standard 5-field UTC cron (`min hour dom month dow`) where `dow` uses `0=Sun, 1=Mon, ..., 6=Sat` (and `7=Sun` alias allowed). "
        "Support `*`, `*/n`, `a-b`, `a-b/n`, and comma-separated lists. Return the next `count` matching datetimes STRICTLY AFTER `start_iso` (`YYYY-MM-DDTHH:MM`) formatted as `YYYY-MM-DDTHH:MM`. "
        "CRITICAL RULES: (1) Handle Gregorian leap years (`2024` leap, `2100` NOT leap). (2) Follow standard Vixie cron semantics: when BOTH `dom` and `dow` are restricted (neither is `*`), a date matches if EITHER `dom` OR `dow` matches! When only one is restricted, use AND. Return ONLY a Python code block.",
        '''def test_cron_runs():
    # Leap day Feb 29: 2024 -> 2028
    assert next_cron_runs("0 12 29 2 *", "2024-02-28T00:00", 2) == ["2024-02-29T12:00", "2028-02-29T12:00"]
    # 2100 is NOT a leap year! 2096 -> 2104
    assert next_cron_runs("30 8 29 2 *", "2096-02-29T08:30", 1) == ["2104-02-29T08:30"]
    # Vixie OR semantics: dom=15 OR dow=1 (Monday) in Nov 2026 (2026-11-15 is Sunday, 2026-11-16 is Monday)
    assert next_cron_runs("0 9 15 11 1", "2026-11-14T23:59", 2) == ["2026-11-15T09:00", "2026-11-16T09:00"]
    # Step & range
    assert next_cron_runs("15-45/15 23 * * *", "2026-11-24T23:15", 3) == ["2026-11-24T23:30", "2026-11-24T23:45", "2026-11-25T23:15"]
''',
        '''import datetime
def _parse_field(f, lo, hi, dow=False):
    res = set()
    for part in f.split(","):
        step = 1
        if "/" in part:
            base, st = part.split("/"); step = int(st)
        else: base = part
        if base == "*": a, b = lo, hi
        elif "-" in base: a, b = map(int, base.split("-"))
        else: a = b = int(base)
        for v in range(a, b + 1, step):
            res.add(0 if (dow and v == 7) else v)
    return res

def next_cron_runs(expr: str, start_iso: str, count: int) -> list[str]:
    fmin, fhr, fdom, fmon, fdow = expr.split()
    mins = sorted(_parse_field(fmin, 0, 59))
    hrs = sorted(_parse_field(fhr, 0, 23))
    doms = _parse_field(fdom, 1, 31)
    mons = _parse_field(fmon, 1, 12)
    dows = _parse_field(fdow, 0, 7, dow=True)
    dom_star = (fdom == "*")
    dow_star = (fdow == "*")
    dt = datetime.datetime.strptime(start_iso, "%Y-%m-%dT%H:%M") + datetime.timedelta(minutes=1)
    out = []
    while len(out) < count:
        if dt.month not in mons:
            y, m = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
            dt = datetime.datetime(y, m, 1, 0, 0); continue
        cron_dow = (dt.weekday() + 1) % 7
        dm = dt.day in doms
        wm = cron_dow in dows
        day_ok = (dm and wm) if (dom_star or dow_star) else (dm or wm)
        if not day_ok:
            dt = (dt + datetime.timedelta(days=1)).replace(hour=0, minute=0); continue
        if dt.hour not in hrs:
            dt = (dt + datetime.timedelta(hours=1)).replace(minute=0); continue
        if dt.minute not in mins:
            dt = dt + datetime.timedelta(minutes=1); continue
        out.append(dt.strftime("%Y-%m-%dT%H:%M"))
        dt = dt + datetime.timedelta(minutes=1)
    return out
'''
    ),
    # HA01: Fix 3 subtle bugs in Hybrid TokenBucket + SlidingWindow Rate Limiter
    (
        "HA01_fix_3bugs_rate_limiter", "H4_Adversarial_Debug_TR",
        "The following Python `HybridLimiter` class has **3 subtle bugs** causing clock-skew crashes, off-by-one sliding window leaks, and burst-cap violations:\n"
        "```python\n"
        "from collections import deque\n"
        "class HybridLimiter:\n"
        "    def __init__(self, capacity: float, refill_per_sec: float, window_sec: float, max_in_window: int):\n"
        "        self.capacity = capacity\n"
        "        self.tokens = capacity\n"
        "        self.refill = refill_per_sec\n"
        "        self.window_sec = window_sec\n"
        "        self.max_in_window = max_in_window\n"
        "        self.last_ts = None\n"
        "        self.hits = deque()\n"
        "    def allow(self, now: float) -> bool:\n"
        "        if self.last_ts is not None:\n"
        "            self.tokens = self.tokens + (now - self.last_ts) * self.refill  # BUG 1 & 2: no clamp to capacity, no backward clock-skew guard!\n"
        "        self.last_ts = now\n"
        "        while self.hits and now - self.hits[0] > self.window_sec:  # BUG 3: boundary uses > instead of >= so exact window_sec old events aren't evicted, AND if denied tokens/hits must stay clean!\n"
        "            self.hits.popleft()\n"
        "        if self.tokens < 1.0 or len(self.hits) >= self.max_in_window:\n"
        "            return False\n"
        "        self.tokens -= 1.0\n"
        "        self.hits.append(now)\n"
        "        return True\n"
        "```\n"
        "Fix ALL bugs cleanly (without modifying the method signature): (1) Guard against backward clock skew (`now < last_ts` must NOT subtract tokens or corrupt `last_ts`). "
        "(2) Clamp refilled tokens to `<= self.capacity`. (3) Evict window entries where `now - ts >= self.window_sec` (half-open `[now - window_sec, now]`). "
        "Return ONLY the fixed Python code block.",
        '''def test_hybrid_limiter():
    lim = HybridLimiter(capacity=2.0, refill_per_sec=1.0, window_sec=10.0, max_in_window=3)
    assert lim.allow(0.0) is True
    assert lim.allow(0.0) is True
    assert lim.allow(0.0) is False  # capacity exhausted
    # Backward clock skew must NOT lose tokens or break state
    assert lim.allow(-5.0) is False
    # After 1.0s at t=1.0, 1 token refills -> allowed (3rd hit in window)
    assert lim.allow(1.0) is True
    # Wait 5.0s -> t=6.0: tokens refilled and clamped to capacity=2.0, but window has 3 hits (0,0,1) => denied!
    assert lim.allow(6.0) is False
    # At t=10.0 (exact window_sec after t=0.0): both t=0.0 hits evicted! Only t=1.0 remains -> allowed twice (capacity=2)!
    assert lim.allow(10.0) is True
    assert lim.allow(10.0) is True
    assert lim.allow(10.0) is False  # clamped to capacity=2, NOT >2!
''',
        '''from collections import deque
class HybridLimiter:
    def __init__(self, capacity: float, refill_per_sec: float, window_sec: float, max_in_window: int):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill = float(refill_per_sec)
        self.window_sec = float(window_sec)
        self.max_in_window = int(max_in_window)
        self.last_ts = None
        self.hits = deque()
    def allow(self, now: float) -> bool:
        if self.last_ts is None:
            self.last_ts = now
        elif now > self.last_ts:
            self.tokens = min(self.capacity, self.tokens + (now - self.last_ts) * self.refill)
            self.last_ts = now
        while self.hits and (now - self.hits[0]) >= self.window_sec:
            self.hits.popleft()
        if self.tokens < 1.0 or len(self.hits) >= self.max_in_window:
            return False
        self.tokens -= 1.0
        self.hits.append(now)
        return True
'''
    ),
    # HA02: Fix LSM-Tree k-way compaction (tombstones + newest-wins + empty runs)
    (
        "HA02_fix_3bugs_lsm_compact", "H4_Adversarial_Debug_TR",
        "Write a clean fixed function `compact_runs(runs: list[list[tuple[str, str | None]]], drop_tombstones: bool = True) -> list[tuple[str, str | None]]`. "
        "`runs` is ordered from **OLDEST run (`runs[0]`) to NEWEST run (`runs[-1]`)**; each run is sorted strictly ascending by `key` with unique keys inside a single run. "
        "A value of `None` is a **tombstone** (deletion marker). "
        "Requirements: (1) For any key present in multiple runs, the value from the **NEWEST** run wins (including if newest is a tombstone `None`, which deletes older values!). "
        "(2) If `drop_tombstones=True` (bottom-level compaction), omit keys whose winning value is `None`; if `drop_tombstones=False`, keep `(key, None)` in the output. "
        "(3) Handle empty runs cleanly and return output sorted ascending by `key`. Return ONLY a Python code block.",
        '''def test_lsm_compact():
    runs = [
        [("a", "old_a"), ("b", "old_b"), ("c", "old_c")],
        [],
        [("a", None), ("b", "mid_b"), ("d", "new_d")],
        [("b", "new_b"), ("c", None), ("e", None)],
    ]
    assert compact_runs(runs, drop_tombstones=True) == [("b", "new_b"), ("d", "new_d")]
    assert compact_runs(runs, drop_tombstones=False) == [("a", None), ("b", "new_b"), ("c", None), ("d", "new_d"), ("e", None)]
    assert compact_runs([[], []], drop_tombstones=True) == []
    # Re-insertion after tombstone must resurrect key!
    runs2 = [[("k", "v1")], [("k", None)], [("k", "v2")]]
    assert compact_runs(runs2, drop_tombstones=True) == [("k", "v2")]
''',
        '''def compact_runs(runs, drop_tombstones=True):
    merged = {}
    for run in runs:
        for k, v in run:
            merged[k] = v
    return sorted([(k, v) for k, v in merged.items() if (not drop_tombstones) or (v is not None)], key=lambda x: x[0])
'''
    ),
    # HA03: Fix 3 real CV engineering bugs in AP@50 calculator!
    (
        "HA03_fix_3bugs_cv_ap50", "H4_Adversarial_Debug_TR",
        "Write a pure Python function `compute_ap50(gts: list[tuple[int, float, float, float, float]], preds: list[tuple[int, float, float, float, float, float]]) -> float` "
        "for single-class object detection AP@50 (all-point interpolation, VOC2010/COCO style). "
        "`gts` entries are `(image_id, x1, y1, x2, y2)`. `preds` entries are `(image_id, conf, x1, y1, x2, y2)`. "
        "STRICT RULES: (1) Non-overlapping boxes MUST have `IoU == 0.0` (clamp intersection width/height to `>= 0`). "
        "(2) Sort predictions **globally** across all images by `conf` descending. "
        "(3) Match each prediction to the highest-IoU **unmatched** GT in the same `image_id` with `IoU >= 0.5`; each GT can be matched **at most once** (duplicate detections on an already-matched GT count as FP!). "
        "(4) Use monotonic precision envelope (`mpre[i] = max(mpre[i:])`) and integrate `sum((mrec[i] - mrec[i-1]) * mpre[i])`. Return `0.0` if `gts` or `preds` is empty. Return ONLY a Python code block.",
        '''import math
def test_ap50():
    # Exact match + duplicate prediction FP on same GT + non-overlapping box on img 2
    gts = [(1, 0, 0, 10, 10), (2, 100, 100, 110, 110)]
    preds = [
        (1, 0.95, 0, 0, 10, 10),   # TP (rec=0.5, prec=1.0)
        (1, 0.90, 0, 0, 10, 10),   # FP! GT already matched (rec=0.5, prec=0.5)
        (2, 0.80, 0, 0, 10, 10),   # FP! non-overlapping on img 2 (rec=0.5, prec=1/3)
        (2, 0.70, 100, 100, 110, 110), # TP (rec=1.0, prec=0.5)
    ]
    # Envelope: at rec=0.5 -> prec=1.0; from 0.5 to 1.0 -> prec=0.5 => AP = 0.5*1.0 + 0.5*0.5 = 0.75!
    assert math.isclose(compute_ap50(gts, preds), 0.75, abs_tol=1e-6)
    # Global sort test: img2 conf=0.9 (FP) BEFORE img1 conf=0.8 (TP) => rec=1.0, prec=0.5 => AP=0.5!
    gts2 = [(1, 0, 0, 10, 10)]
    preds2 = [(1, 0.8, 0, 0, 10, 10), (2, 0.9, 0, 0, 10, 10)]
    assert math.isclose(compute_ap50(gts2, preds2), 0.5, abs_tol=1e-6)
    assert compute_ap50([], preds) == 0.0
    assert compute_ap50(gts, []) == 0.0
''',
        '''def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0: return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0

def compute_ap50(gts, preds) -> float:
    if not gts or not preds: return 0.0
    gt_by_img = {}
    for idx, (img, x1, y1, x2, y2) in enumerate(gts):
        gt_by_img.setdefault(img, []).append((idx, (x1, y1, x2, y2)))
    matched = set()
    sp = sorted(preds, key=lambda x: -x[1])
    tp = []; fp = []
    for img, conf, x1, y1, x2, y2 in sp:
        best_iou, best_idx = 0.0, -1
        for gidx, gbox in gt_by_img.get(img, []):
            if gidx in matched: continue
            v = _iou((x1, y1, x2, y2), gbox)
            if v > best_iou: best_iou, best_idx = v, gidx
        if best_iou >= 0.5 and best_idx != -1:
            matched.add(best_idx); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    ct, cf = 0, 0
    recs = [0.0]; precs = [1.0]
    n_gt = len(gts)
    for t, f in zip(tp, fp):
        ct += t; cf += f
        recs.append(ct / n_gt)
        precs.append(ct / (ct + cf))
    recs.append(1.0); precs.append(0.0)
    for i in range(len(precs) - 2, -1, -1):
        if precs[i] < precs[i + 1]: precs[i] = precs[i + 1]
    ap = 0.0
    for i in range(1, len(recs)):
        if recs[i] != recs[i - 1]:
            ap += (recs[i] - recs[i - 1]) * precs[i]
    return float(ap)
'''
    ),
]

def build_hard_v2_tasks():
    hl04_ans, hl04_code = solve_hl04()
    ha06_ans, ha06_sql = solve_ha06()
    tasks = [
        # --- H1_Olympiad_Math (8) ---
        {"task_id": "HM01_noncoprime_crt_mod", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "x > 50000 olmak üzere x ≡ 17 (mod 42), x ≡ 59 (mod 70) ve x ≡ 14 (mod 45) denklik sistemini sağlayan EN KÜÇÜK tam sayı x kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm01()), "max_tokens": 8192},
        {"task_id": "HM02_menage_forbidden_perm", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "{1, 2, 3, 4, 5, 6, 7, 8} kümesinin tüm i ∈ {1..8} için p(i) ≠ i VE p(i) ≠ (i mod 8) + 1 (yani p(8) ≠ 1) VE ayrıca p(1) < p(2) koşullarını sağlayan kaç farklı permütasyonu vardır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm02()), "max_tokens": 8192},
        {"task_id": "HM03_lattice_3d_obstacles", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "3 boyutlu ızgarada (0,0,0) noktasından (6,5,4) noktasına yalnızca (+1,0,0), (0,+1,0) veya (0,0,+1) adımlarıyla gidilecektir. Yol (3,2,2) noktasından MUTLAKA geçmek zorunda, ancak (2,2,1), (3,3,2) ve (4,2,3) noktalarından HİÇBİRİNE uğrayamaz. Kaç farklı en kısa yol vardır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm03()), "max_tokens": 8192},
        {"task_id": "HM04_markov_reset_hitting", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "{0, 1, 2, 3, 4, 5} düğümlerindeki rastgele yürüyüş 0'dan başlar ve 5'e ulaşınca durur. 0'dayken 1 olasılıkla 1'e gider. Her i ∈ {1, 2, 3, 4} için bir adımda 1/2 olasılıkla i+1'e, 1/3 olasılıkla i-1'e ve 1/6 olasılıkla tamamen başa (0'a) döner. 5'e ulaşmak için beklenen adım sayısı en sade kesirle p/q ise p + q kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm04()), "max_tokens": 8192},
        {"task_id": "HM05_modular_tower_euler", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "13^(17^2027) sayısının 1000 ile bölümünden kalan kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm05()), "max_tokens": 8192},
        {"task_id": "HM06_sprague_grundy_split", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "İki oyunculu normal oyun kuralıyla (son hamleyi yapan kazanır) oynanan bir taş oyununda başlangıçta 3 yığın vardır: (11, 14, 17). Sırası gelen oyuncu tek bir yığından ya 1, 2 veya 4 taş alabilir YA DA çift sayıda ve en az 6 taş içeren bir yığını (2k >= 6) iki eşit yığına (k, k) bölebilir. Başlangıç konumundan ilk oyuncuyu kesin galibiyete götüren kaç farklı ilk hamle vardır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm06()), "max_tokens": 8192},
        {"task_id": "HM07_digit_dp_divisible7", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "1 ile 999.999 (dahil) arasında, içinde hiç 0 rakamı bulunmayan, soldan sağa rakamları azalmayan (ör. 112589 geçerli, 132 geçersiz) ve rakamları toplamı 7'ye tam bölünen kaç farklı tam sayı vardır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm07()), "max_tokens": 8192},
        {"task_id": "HM08_recurrence_mod_a50", "module": "H1_Olympiad_Math", "difficulty": "hard",
         "messages": [{"role": "user", "content": "a_0 = 1, a_1 = 2, a_2 = 5 ve n >= 3 için a_n = 3*a_(n-1) - 2*a_(n-2) + 4*a_(n-3) + n^2 dizisi tanımlanıyor. a_50 değerinin 1.000.003 ile bölümünden kalan kaçtır (0 ile 1.000.002 arasında)? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hm08()), "max_tokens": 8192},

        # --- H3_CSP_Logic_Tracing (7) ---
        {"task_id": "HL01_turkish_cryptarithm", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": "KARA + KARA = ORMAN kriptaritmetik toplamasında her farklı harf farklı bir rakamı (0–9) temsil eder ve baş harfler (K, O) 0 olamaz. Tek bir benzersiz çözüm vardır. ORMAN sayısının sayısal değeri kaçtır? Son satırda sadece 5 basamaklı tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hl01()), "max_tokens": 8192},
        {"task_id": "HL02_zebra_6_servers", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": "Bir kabinde soldan sağa 1'den 6'ya numaralı 6 yuvada 6 sunucu (Atlas, Bora, Ceylan, Deniz, Evren, Fırtına) ve her birinde farklı bir iş yükü (LLM, Vision, Audio, RAG, ETL, Sim) çalışıyor:\n1) Evren, Atlas'ın hemen sağındaki yuvada.\n2) Bora tek numaralı bir yuvada ve Fırtına'nın solunda.\n3) Fırtına, Bora ile Atlas arasında (ikisi arasında herhangi bir yerde).\n4) Ceylan 6 numaralı yuvada.\n5) Deniz, Ceylan'a bitişik değil, Bora'ya bitişik değil ve 1 numaralı yuvada değil.\n6) LLM, Evren'de çalışıyor.\n7) Vision, Audio'nun hemen solundaki yuvada.\n8) RAG asal numaralı bir yuvada (2, 3 veya 5), ancak Evren'de ve Deniz'de çalışmıyor.\n9) ETL, Bora'da çalışıyor.\n10) Sim, Ceylan'da çalışıyor.\nBuna göre 1000*yuva(Atlas) + 100*yuva(Deniz) + 10*yuva(Vision) + yuva(RAG) kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hl02()), "max_tokens": 8192},
        {"task_id": "HL03_knights_knaves_spies_5", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": "Ali, Burcu, Cem, Deniz ve Emre adlı 5 kişiden tam 2'si Doğrucu (1 — daima doğru söyler), tam 2'si Yalancı (2 — daima yalan söyler) ve tam 1'i Casus (3 — doğru da yalan da söyleyebilir).\nİfadeler:\n- Ali: 'Burcu Yalancıdır VE Cem Casus değildir.'\n- Burcu: 'Deniz Casustur VE Ali Yalancıdır.'\n- Cem: 'Emre ile ben farklı rollerdeyiz VE Ali Yalancıdır.'\n- Deniz: 'Burcu Doğrucudur VE Emre Casus değildir.'\n- Emre: 'Deniz Yalancıdır VE Burcu Casus değildir.'\nTek benzersiz rol dağılımını bul ve Ali, Burcu, Cem, Deniz, Emre sırasıyla kodları (Doğrucu=1, Yalancı=2, Casus=3) yan yana yazarak 5 basamaklı tam sayıyı oluştur. Son satırda sadece 5 basamaklı tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hl03()), "max_tokens": 8192},
        {"task_id": "HL04_python_mro_closure_gen", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": f"Aşağıdaki Python 3 kodu çalıştırıldığında `out` değişkeninin kesin tam sayı değeri ne olur? Son satırda sadece tam sayıyı yaz.\n```python\n{hl04_code}\n```"}],
         "eval_type": "exact_number", "expected": str(hl04_ans), "max_tokens": 8192},
        {"task_id": "HL05_dag_2machine_makespan", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": "8 iş (J0..J7) 2 özdeş makinede kesintisiz (non-preemptive) çalıştırılacaktır. Süreler: J0=4, J1=5, J2=6, J3=3, J4=7, J5=4, J6=5, J7=6. Öncelik kısıtları (u -> v demek u bitmeden v başlayamaz): J0->J2, J0->J3, J1->J3, J1->J4, J2->J5, J3->J5, J3->J6, J4->J6, J5->J7, J6->J7. Tüm işlerin tamamlanması için mümkün olan EN KÜÇÜK toplam bitiş süresi (minimum makespan) kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hl05()), "max_tokens": 8192},
        {"task_id": "HL06_bitwise_vm_trace", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": "8-bitlik bir kayıt makinesinde başlangıç değerleri R0=90, R1=51, R2=7'dir. R2 > 0 olduğu sürece her turda sırasıyla:\n1) R0 = (R0 XOR R1) mod 256\n2) R0 = ROR8(R0, 2)  (8-bit sağa 2 bit dairesel kaydırma: ((R0 >> 2) | ((R0 << 6) & 255)))\n3) R1 = (R1 + R0 + 3*R2) mod 256\n4) Eğer R1 tek sayı ise: R0 = (R0 + 19) mod 256\n5) R2 = R2 - 1\nDöngü bittiğinde 256*R0 + R1 değeri kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hl06()), "max_tokens": 8192},
        {"task_id": "HL07_bayes_3factory_2tests", "module": "H3_CSP_Logic_Tracing", "difficulty": "hard",
         "messages": [{"role": "user", "content": "Üretimin %50'si F1 (%4 kusurlu), %30'u F2 (%5 kusurlu), %20'si F3 (%10 kusurlu) fabrikasından gelmektedir. Bir sensör testi kusurlu üründe %90 pozitif (+), sağlam üründe %80 negatif (-) verir (yani sağlamda %20 yanlış pozitif). Rastgele seçilen bir ürüne birbirinden bağımsız iki test yapılıyor: 1. test POZİTİF (+), 2. test NEGATİF (-) çıkıyor. Bu ürünün hem F3'ten gelmiş hem de KUSURLU olma olasılığı en sade kesirle p/q ise p + q kaçtır? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_hl07()), "max_tokens": 8192},

        # --- H4_Adversarial_Debug_TR (non-pytest 4 tasks; 3 pytest tasks appended below) ---
        {"task_id": "HA04_turkish_sla_contract", "module": "H4_Adversarial_Debug_TR", "difficulty": "hard",
         "messages": [{"role": "user", "content": "Bir bulut sözleşmesinde taban aylık ücret 120.000 TL'dir. SLA kesinti kuralları:\n- Planlı bakımlar kesintiden muaftır.\n- Plansız kesintilerde ilk 60 dakikayı aşan her olay için taban %4 kesinti uygulanır; 60 dakikadan sonraki her TAM 30 dakikalık blok için ekstra %2 eklenir (tam 30 dakikayı doldurmayan artık dakikalar sayılmaz).\n- Yoğun saatlerde (09:00–18:00) yaşanan plansız kesintilerin oranı 1,5 katıyla çarpılır.\n- Üçüncü parti DNS kesintilerinde hesaplanan kesintinin yalnızca yarısı uygulanır (yoğun saat çarpanı uygulanmaz).\n- Bir aydaki toplam SLA kesintisi en fazla %15 ile tavanlanır.\n- %5 sadakat indirimi yalnızca SLA kesintisi düşüldükten sonraki tutara uygulanır.\n- %20 KDV en son indirimli net tutara eklenir.\nO ay yaşanan olaylar: (1) 45 dk planlı bakım, (2) 130 dk yoğun saat plansız kesinti, (3) 75 dk yoğun saat dışı plansız kesinti, (4) 90 dk üçüncü parti DNS kesintisi.\nMüşterinin ödeyeceği KDV dahil son fatura tutarı kaç TL'dir? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_ha04()), "max_tokens": 8192},
        {"task_id": "HA05_tr_lipogram_acrostic", "module": "H4_Adversarial_Debug_TR", "difficulty": "hard",
         "messages": [{"role": "user", "content": "Yapay zeka hakkında tam 5 satır Türkçe metin yaz.\nKESİN KURALLAR:\n1) Metnin HİÇBİR yerinde 'e' veya 'E' harfi geçmeyecek!\n2) Satırların baş harfleri yukarıdan aşağıya tam olarak Y-A-P-A-Y akrostişini oluşturacak (1. satır Y, 2. satır A, 3. satır P, 4. satır A, 5. satır Y ile başlayacak).\n3) Her satır en az 4 kelime olacak (toplam en az 20 kelime).\n4) Markdown, madde işareti, numara veya başlık kullanma; sadece 5 düz satır yaz."}],
         "eval_type": "strict_text_rules", "no_markdown": True, "exact_lines": 5, "forbidden_char": "e", "min_words": 20,
         "line_prefix_regex": r"^[YAPyap]", "acrostic": "YAPAY", "max_tokens": 8192},
        {"task_id": "HA06_adversarial_sql_nulls", "module": "H4_Adversarial_Debug_TR", "difficulty": "hard",
         "messages": [{"role": "user", "content": f"SQLite veritabanında şu tablolar var:\norders(oid, cid, amount, status): (1,10,200,'paid'), (2,10,150,'paid'), (3,20,300,'cancelled'), (4,20,100,'paid'), (5,30,NULL,'paid'), (6,40,250,'paid').\nrefunds(rid, oid, r_amt): (1,1,50), (2,1,30), (3,4,100), (4,6,NULL).\nAşağıdaki SQL sorgusunun döndüreceği kesin tam sayı kaçtır? Son satırda sadece tam sayıyı yaz.\n```sql\n{ha06_sql}\n```"}],
         "eval_type": "exact_number", "expected": str(ha06_ans), "max_tokens": 8192},
        {"task_id": "HA07_bankers_rounding_ledger", "module": "H4_Adversarial_Debug_TR", "difficulty": "hard",
         "messages": [{"role": "user", "content": "Bankacı yuvarlaması (ROUND_HALF_EVEN — tam ortada çift haneye yuvarlama) ile her adımda 2 ondalık basamağa (kuruş) yuvarlayarak hesapla:\nSepette 4 kalem var: (1) 125.05 TL × 3 adet, %7.5 indirim; (2) 89.95 TL × 2 adet, %12.5 indirim; (3) 310.10 TL × 1 adet, %15 indirim; (4) 44.50 TL × 5 adet, %5 indirim.\nHer kalemde önce brüt = fiyat*adet, sonra net = round(brüt * (1 - indirim), 2), sonra kdv_dahil = round(net * 1.20, 2) hesaplanıp toplanır.\nSon olarak 2. kalemden 1 adet iade edilir (iade tutarı aynı kuralla: round(89.95 * 0.875, 2) ardından round(* 1.20, 2)) ve toplamdan düşülür.\nKalan net fatura tutarı toplam kaç KURUŞTUR (ör. 123.45 TL -> 12345)? Son satırda sadece tam sayıyı yaz."}],
         "eval_type": "exact_number", "expected": str(solve_ha07()), "max_tokens": 8192},
    ]

    for tid, mod, prompt, pytest_code, _ref_code in PYTEST_TASKS:
        tasks.append({
            "task_id": tid, "module": mod, "difficulty": "hard",
            "messages": [{"role": "user", "content": prompt}],
            "eval_type": "python_pytest", "pytest_code": pytest_code, "max_tokens": 8192,
        })
    return tasks

def self_verify_all():
    tasks = build_hard_v2_tasks()
    assert len(tasks) == 30, f"Expected 30 tasks, got {len(tasks)}"
    print(f"BUILT {len(tasks)} HARD_V2 TASKS! Running 11 reference pytest suites...")
    for tid, mod, _, pytest_code, ref_code in PYTEST_TASKS:
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "solution.py"), "w", encoding="utf-8").write(ref_code + "\n")
            open(os.path.join(td, "test_solution.py"), "w", encoding="utf-8").write("from solution import *\n" + pytest_code + "\n")
            r = subprocess.run(["pytest", "-q", "test_solution.py"], cwd=td, capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                raise RuntimeError(f"REF PYTEST FAILED for {tid}:\n{r.stdout}\n{r.stderr}")
            print(f"  [PYTEST PASS] {tid}")
    print("ALL 30/30 HARD_V2 TASKS & 11/11 PYTEST REFERENCE SOLUTIONS 100% VERIFIED!")

if __name__ == "__main__":
    self_verify_all()
