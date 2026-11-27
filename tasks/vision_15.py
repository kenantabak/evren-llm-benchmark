#!/usr/bin/env python3
"""
EVREN LLM Benchmark — Ek Test 3: Görsel Anlama (vision_v1)
15 programmatically generated, 100% ground-truth visual tasks across 4 categories:
  - V1 Grafik & Veri Okuma (4)
  - V2 Türkçe Belge, Fatura & Tablo OCR (4)
  - V3 Görsel Sayma, Geometri & Uzamsal Akıl Yürütme (4)
  - V4 Arayüz, Kod & Diyagram Okuma (3)
Tested across ALL 4 EVREN vision-capable chat models:
  - gemma-4-31b (reasoning_off, reasoning_on)
  - deepseek-v4.1-flash (reasoning_off, reasoning_on)
  - qwen3.8-flash-next (reasoning_off, reasoning_on)
  - qwen3-vl-30b (reasoning_off — thinking: False)
Total: 15 * 7 = 105 calls.
"""
import base64, concurrent.futures, datetime, io, json, math, os, re, threading, time, requests
from collections import defaultdict, deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

TSI = datetime.timezone(datetime.timedelta(hours=3))
OUT_DIR = os.environ.get("EVREN_BENCH_OUT_DIR", "results/vision_v1")
IMG_DIR = os.environ.get("EVREN_VISION_ASSETS_DIR", os.path.join(os.path.dirname(__file__), "vision_assets"))
RES_JSONL = os.path.join(OUT_DIR, "results.jsonl")

plt.rcParams["font.family"] = "DejaVu Sans"

def get_font(size=20, bold=False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def save_fig(fig, name):
    p = os.path.join(IMG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p

def save_pil(im, name):
    p = os.path.join(IMG_DIR, name)
    im.save(p, "PNG")
    return p

def build_vision_tasks():
    tasks = []

    # V01: Bar chart exact sum (İzmir 418 + Bursa 275 = 693)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cities = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Konya"]
    vals = [612, 485, 418, 275, 340, 229]
    bars = ax.bar(cities, vals, color=["#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 12, str(v), ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 720); ax.set_ylabel("Tüketim (MWh)"); ax.set_title("Aylık Bölgesel Enerji Tüketimi (MWh)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    p = save_fig(fig, "V01_bar_chart.png")
    tasks.append({"task_id": "V01_bar_chart_sum", "module": "V1_Chart_Data", "img_path": p,
                  "prompt": "Görseldeki grafikte İzmir ve Bursa şehirlerinin toplam enerji tüketimi kaç MWh'dir? Yalnızca tek bir tam sayı (JSON: {\"answer\": <sayı>}) döndür.",
                  "check": "json_int", "expected": 693})

    # V02: Multi-line crossing quarter + Q6 gap
    fig, ax = plt.subplots(figsize=(8, 4.8))
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    a = [52, 58, 63, 66, 68, 71]
    b = [38, 49, 60, 72, 81, 89]
    ax.plot(qs, a, marker="o", lw=2.5, label="Model Alfa", color="#2563EB")
    ax.plot(qs, b, marker="s", lw=2.5, label="Model Beta", color="#EF4444")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i, va + 2.2, str(va), ha="center", color="#1D4ED8", fontweight="bold")
        ax.text(i, vb - 4.5, str(vb), ha="center", color="#B91C1C", fontweight="bold")
    ax.set_ylim(25, 102); ax.set_ylabel("Doğruluk (%)"); ax.set_title("Çeyrek Bazlı Model Doğruluk Karşılaştırması", fontweight="bold")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    p = save_fig(fig, "V02_multiline.png")
    tasks.append({"task_id": "V02_multiline_cross", "module": "V1_Chart_Data", "img_path": p,
                  "prompt": "Grafikte Model Beta ilk kez hangi çeyrekte Model Alfa'yı geçmiştir ve Q6 çeyreğindeki puan farkı (Beta - Alfa) kaçtır? JSON döndür: {\"quarter\": \"Q?\", \"q6_diff\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"quarter": "Q4", "q6_diff": 18}})

    # V03: Stacked bar highest Ar-Ge
    fig, ax = plt.subplots(figsize=(8, 4.8))
    depts = ["Birim-A", "Birim-B", "Birim-C", "Birim-D"]
    arge = np.array([120, 215, 165, 95])
    oper = np.array([200, 140, 180, 240])
    pzr  = np.array([80,  95,  110, 105])
    ax.bar(depts, arge, label="Ar-Ge", color="#10B981")
    ax.bar(depts, oper, bottom=arge, label="Operasyon", color="#3B82F6")
    ax.bar(depts, pzr, bottom=arge+oper, label="Pazarlama", color="#F59E0B")
    for i in range(4):
        ax.text(i, arge[i]/2, str(arge[i]), ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, arge[i]+oper[i]/2, str(oper[i]), ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, arge[i]+oper[i]+pzr[i]/2, str(pzr[i]), ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylim(0, 520); ax.set_ylabel("Bütçe (Bin TL)"); ax.set_title("Birimlere Göre Yıllık Bütçe Dağılımı", fontweight="bold")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
    p = save_fig(fig, "V03_stacked.png")
    tasks.append({"task_id": "V03_stacked_bar_arge", "module": "V1_Chart_Data", "img_path": p,
                  "prompt": "Grafikte en yüksek Ar-Ge bütçesine sahip birim hangisidir ve bu birimin toplam (Ar-Ge + Operasyon + Pazarlama) bütçesi kaç bin TL'dir? JSON döndür: {\"dept\": \"Birim-?\", \"total\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"dept": "Birim-B", "total": 450}})

    # V04: Scatter outlier identification
    fig, ax = plt.subplots(figsize=(8, 5.0))
    pts = [("P1", 45, 72), ("P2", 52, 75), ("P3", 61, 79), ("P4", 74, 82), ("P5", 83, 84),
           ("P6", 38, 68), ("P7", 18, 93), ("P8", 68, 77), ("P9", 91, 86), ("P10", 55, 71)]
    for pid, x, y in pts:
        col = "#EF4444" if pid == "P7" else "#2563EB"
        ax.scatter(x, y, s=110, c=col, edgecolors="black", zorder=5)
        ax.text(x + 1.8, y + 0.8, f"{pid} ({x},{y})", fontsize=9.5, fontweight="bold")
    ax.set_xlim(5, 108); ax.set_ylim(60, 100); ax.set_xlabel("Gecikme (ms)"); ax.set_ylabel("Doğruluk (%)")
    ax.set_title("Gecikme vs Doğruluk Dağılımı", fontweight="bold"); ax.grid(True, linestyle="--", alpha=0.4)
    p = save_fig(fig, "V04_scatter.png")
    tasks.append({"task_id": "V04_scatter_outlier", "module": "V1_Chart_Data", "img_path": p,
                  "prompt": "Sol üst bölgede yer alan (gecikme < 25 ms ve doğruluk > %90) aykırı noktanın kodu ve doğruluk değeri nedir? JSON döndür: {\"id\": \"P?\", \"accuracy\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"id": "P7", "accuracy": 93}})

    # V05: Turkish Invoice OCR + KDV %20 arithmetic (2*450 + 3*180 + 1*620 = 2060 -> +20% KDV = 2472)
    im = Image.new("RGB", (820, 540), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f_h, f_b, f_r = get_font(24, True), get_font(18, True), get_font(17, False)
    d.rectangle([20, 20, 800, 520], outline=(30, 41, 59), width=2)
    d.text((40, 38), "ANADOLU TEKNOLOJİ A.Ş. — E-FATURA", fill=(15, 23, 42), font=f_h)
    d.text((40, 80), "Fatura No: FTR-2026-8419    |    Tarih: 24.11.2026    |    KDV Oranı: %20", fill=(51, 65, 85), font=f_r)
    d.line([40, 118, 780, 118], fill=(148, 163, 184), width=2)
    rows_inv = [
        ("Ürün / Hizmet", "Adet", "Birim Fiyat", "Tutar"),
        ("Endüstriyel Kamera Modülü", "2", "450 TL", "900 TL"),
        ("Lens Filtre Seti (IR-Cut)", "3", "180 TL", "540 TL"),
        ("Kalibrasyon Hizmeti", "1", "620 TL", "620 TL"),
    ]
    y0 = 135
    for idx, (u, a, bf, t) in enumerate(rows_inv):
        fn = f_b if idx == 0 else f_r
        d.text((45, y0), u, fill=(15, 23, 42), font=fn)
        d.text((420, y0), a, fill=(15, 23, 42), font=fn)
        d.text((510, y0), bf, fill=(15, 23, 42), font=fn)
        d.text((660, y0), t, fill=(15, 23, 42), font=fn)
        y0 += 48
    d.line([40, 345, 780, 345], fill=(148, 163, 184), width=2)
    d.text((450, 368), "Ara Toplam:", fill=(15, 23, 42), font=f_b); d.text((650, 368), "2060 TL", fill=(15, 23, 42), font=f_b)
    d.text((450, 408), "KDV (%20):", fill=(15, 23, 42), font=f_b); d.text((650, 408), "412 TL", fill=(15, 23, 42), font=f_b)
    d.text((450, 452), "GENEL TOPLAM:", fill=(185, 28, 28), font=f_b); d.text((650, 452), "2472 TL", fill=(185, 28, 28), font=f_b)
    p = save_pil(im, "V05_tr_invoice.png")
    tasks.append({"task_id": "V05_tr_invoice_ocr", "module": "V2_Turkish_OCR", "img_path": p,
                  "prompt": "Görseldeki faturanın Fatura Numarası ve KDV dâhil Genel Toplam (TL) tutarı nedir? JSON döndür: {\"invoice_no\": \"...\", \"grand_total\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"invoice_no": "FTR-2026-8419", "grand_total": 2472}})

    # V06: Turkish Lab Report Table (Ferritin LOW 9.4 [15-150], Glukoz HIGH 128 [70-100])
    im = Image.new("RGB", (820, 460), (248, 250, 252))
    d = ImageDraw.Draw(im)
    d.text((35, 25), "LABORATUVAR SONUÇ RAPORU — BİYOKİMYA", fill=(15, 23, 42), font=f_h)
    tbl = [
        ("Parametre", "Sonuç", "Birim", "Referans Aralığı"),
        ("Hemoglobin", "14.2", "g/dL", "12.0 - 16.0"),
        ("Ferritin", "9.4", "ng/mL", "15.0 - 150.0"),
        ("TSH", "2.15", "mIU/L", "0.40 - 4.00"),
        ("Vitamin B12", "410", "pg/mL", "200 - 900"),
        ("Açlık Glukozu", "128", "mg/dL", "70 - 100"),
    ]
    y0 = 85
    for idx, (k, v, u, ref) in enumerate(tbl):
        bg = (226, 232, 240) if idx == 0 else ((255, 255, 255) if idx % 2 else (241, 245, 249))
        d.rectangle([30, y0, 790, y0+52], fill=bg, outline=(203, 213, 225))
        fn = f_b if idx == 0 else f_r
        d.text((48, y0+14), k, fill=(15, 23, 42), font=fn)
        d.text((300, y0+14), v, fill=(15, 23, 42), font=fn)
        d.text((430, y0+14), u, fill=(71, 85, 105), font=fn)
        d.text((560, y0+14), ref, fill=(15, 23, 42), font=fn)
        y0 += 54
    p = save_pil(im, "V06_tr_lab_table.png")
    tasks.append({"task_id": "V06_tr_lab_abnormal", "module": "V2_Turkish_OCR", "img_path": p,
                  "prompt": "Tabloda referans aralığının DIŞINDA kalan (düşük veya yüksek) tam olarak kaç parametre vardır ve adları nelerdir? JSON döndür: {\"count\": <sayı>, \"abnormal\": [\"...\", \"...\"]}.",
                  "check": "v06_lab", "expected": {"count": 2, "must": ["ferritin", "glukoz"]}})

    # V07: Rotated (-6 deg) noisy receipt
    im = Image.new("RGB", (600, 520), (245, 242, 232))
    d = ImageDraw.Draw(im)
    f_m = get_font(20, True); f_s = get_font(18, False)
    d.text((130, 35), "KAFE BOĞAZİÇİ — FİŞ", fill=(20, 20, 20), font=f_m)
    d.text((90, 75), "Fiş Kodu: KFB-7392", fill=(30, 30, 30), font=f_m)
    d.line([70, 110, 530, 110], fill=(80, 80, 80), width=2)
    items_r = [("Türk Kahvesi x2", "190 TL"), ("Filtre Kahve x1", "115 TL"), ("San Sebastian x1", "245 TL"), ("Su x2", "50 TL")]
    y0 = 135
    for k, v in items_r:
        d.text((85, y0), k, fill=(25, 25, 25), font=f_s)
        d.text((415, y0), v, fill=(25, 25, 25), font=f_s)
        y0 += 48
    d.line([70, 345, 530, 345], fill=(80, 80, 80), width=2)
    d.text((85, 368), "TOPLAM ÖDEME:", fill=(15, 23, 42), font=f_m)
    d.text((405, 368), "600 TL", fill=(15, 23, 42), font=f_m)
    im = im.rotate(-6, expand=True, fillcolor=(230, 228, 220))
    arr = np.array(im, dtype=np.int16)
    rng = np.random.default_rng(42)
    arr = np.clip(arr + rng.integers(-8, 9, size=arr.shape), 0, 255).astype(np.uint8)
    p = save_pil(Image.fromarray(arr), "V07_rotated_receipt.png")
    tasks.append({"task_id": "V07_rotated_receipt", "module": "V2_Turkish_OCR", "img_path": p,
                  "prompt": "Eğik ve hafif gürültülü fiş görselindeki Fiş Kodunu ve Toplam Ödeme tutarını (TL) okuyun. JSON döndür: {\"code\": \"...\", \"total\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"code": "KFB-7392", "total": 600}})

    # V08: Financial Matrix (Gelir - Gider per quarter)
    im = Image.new("RGB", (820, 420), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((35, 25), "2026 ÇEYREKLİK FİNANSAL ÖZET (BİN TL)", fill=(15, 23, 42), font=f_h)
    ftbl = [
        ("Kalem", "Q1", "Q2", "Q3", "Q4"),
        ("Gelir", "1420", "1680", "1950", "1810"),
        ("Operasyon Gideri", "890", "940", "1020", "1090"),
        ("Ar-Ge Gideri", "210", "240", "265", "280"),
    ]
    # Net kar: Q1=320, Q2=500, Q3=665, Q4=440
    y0 = 90
    xs_c = [40, 320, 440, 560, 680]
    for idx, row in enumerate(ftbl):
        bg = (30, 41, 59) if idx == 0 else ((248, 250, 252) if idx % 2 else (255, 255, 255))
        fg = (255, 255, 255) if idx == 0 else (15, 23, 42)
        d.rectangle([30, y0, 790, y0+58], fill=bg, outline=(203, 213, 225))
        fn = f_b if idx == 0 else f_r
        for x, cell in zip(xs_c, row):
            d.text((x, y0+18), cell, fill=fg, font=fn)
        y0 += 60
    p = save_pil(im, "V08_financial_table.png")
    tasks.append({"task_id": "V08_financial_net_profit", "module": "V2_Turkish_OCR", "img_path": p,
                  "prompt": "Tabloda Net Kâr = Gelir - (Operasyon Gideri + Ar-Ge Gideri) olarak tanımlanmaktadır. Q3 Net Kârı kaç bin TL'dir ve en yüksek Net Kâr hangi çeyrektedir? JSON döndür: {\"q3_net\": <tam_sayı>, \"best_quarter\": \"Q?\"}.",
                  "check": "json_kv", "expected": {"q3_net": 665, "best_quarter": "Q3"}})

    # V09: Colored Shape Counting (7 red circles, 5 blue squares, 4 green triangles — strictly non-overlapping grid!)
    im = Image.new("RGB", (760, 560), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 750, 550], outline=(203, 213, 225), width=2)
    # 4x4 grid of 16 well-separated cells: 7 red circles, 5 blue squares, 4 green triangles
    shapes = ["RC"]*7 + ["BS"]*5 + ["GT"]*4
    rng = np.random.default_rng(2026)
    rng.shuffle(shapes)
    for idx, sh in enumerate(shapes):
        r, c = divmod(idx, 4)
        cx, cy = 115 + c * 175, 95 + r * 125
        if sh == "RC":
            d.ellipse([cx-36, cy-36, cx+36, cy+36], fill=(220, 38, 38), outline=(127, 29, 29), width=3)
        elif sh == "BS":
            d.rectangle([cx-34, cy-34, cx+34, cy+34], fill=(37, 99, 235), outline=(30, 58, 138), width=3)
        else:
            d.polygon([(cx, cy-38), (cx-38, cy+34), (cx+38, cy+34)], fill=(22, 163, 74), outline=(20, 83, 45), width=3)
    p = save_pil(im, "V09_shapes_count.png")
    tasks.append({"task_id": "V09_color_shape_count", "module": "V3_Visual_Spatial", "img_path": p,
                  "prompt": "Görselde kaç adet KIRMIZI DAİRE ve kaç adet MAVİ KARE vardır? JSON döndür: {\"red_circles\": <tam_sayı>, \"blue_squares\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"red_circles": 7, "blue_squares": 5}})

    # V10: 6x6 Grid Maze Shortest Path (BFS-verified!)
    grid = [
        [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0],
    ]
    start, goal = (0, 0), (5, 5)
    # BFS verification
    q = deque([(start[0], start[1], 0)]); vis = {start}; bfs_dist = -1
    while q:
        r, c, dist = q.popleft()
        if (r, c) == goal: bfs_dist = dist; break
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 6 and 0 <= nc < 6 and grid[nr][nc] == 0 and (nr, nc) not in vis:
                vis.add((nr, nc)); q.append((nr, nc, dist+1))
    assert bfs_dist == 12, f"Unexpected BFS dist: {bfs_dist}"
    im = Image.new("RGB", (540, 580), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((40, 15), "6x6 Izgara — S'den G'ye En Kısa Yol (Yalnız 4 Yön)", fill=(15, 23, 42), font=get_font(18, True))
    cs = 76; ox, oy = 42, 62
    for r in range(6):
        for c in range(6):
            x0, y0 = ox + c*cs, oy + r*cs
            fill = (15, 23, 42) if grid[r][c] == 1 else (248, 250, 252)
            if (r, c) == start: fill = (34, 197, 94)
            if (r, c) == goal: fill = (239, 68, 68)
            d.rectangle([x0, y0, x0+cs, y0+cs], fill=fill, outline=(100, 116, 139), width=2)
            if (r, c) == start: d.text((x0+28, y0+22), "S", fill=(255,255,255), font=get_font(28, True))
            if (r, c) == goal:  d.text((x0+26, y0+22), "G", fill=(255,255,255), font=get_font(28, True))
    p = save_pil(im, "V10_grid_maze.png")
    tasks.append({"task_id": "V10_grid_shortest_path", "module": "V3_Visual_Spatial", "img_path": p,
                  "prompt": "6x6 ızgarada yeşil S hücresinden kırmızı G hücresine siyah duvarlardan geçmeden (sadece yukarı/aşağı/sola/sağa hareket ederek) en az kaç adımda ulaşılır? JSON döndür: {\"steps\": <tam_sayı>}.",
                  "check": "json_int_key", "key": "steps", "expected": bfs_dist})

    # V11: Analog Clock (03:40 — super clear hour & minute hands!)
    im = Image.new("RGB", (520, 520), (255, 255, 255))
    d = ImageDraw.Draw(im)
    cx, cy, R = 260, 260, 210
    d.ellipse([cx-R, cy-R, cx+R, cy+R], fill=(248, 250, 252), outline=(15, 23, 42), width=6)
    f_ck = get_font(24, True)
    for i in range(60):
        ang = math.radians(i * 6 - 90)
        r1 = R - (18 if i % 5 == 0 else 8)
        d.line([cx + r1*math.cos(ang), cy + r1*math.sin(ang), cx + (R-3)*math.cos(ang), cy + (R-3)*math.sin(ang)],
               fill=(15, 23, 42), width=4 if i % 5 == 0 else 1)
    for h_num in range(1, 13):
        ang = math.radians(h_num * 30 - 90)
        tx, ty = cx + (R - 42)*math.cos(ang), cy + (R - 42)*math.sin(ang)
        d.text((tx-12, ty-14), str(h_num), fill=(15, 23, 42), font=f_ck)
    # Time = 03:40 -> minute hand at 8 (240 deg - 90 = 150 deg), hour hand at 3 + 40/60 = 3.667 (110 deg - 90 = 20 deg)
    ang_m = math.radians(40 * 6 - 90)
    ang_h = math.radians((3 + 40/60.0) * 30 - 90)
    d.line([cx, cy, cx + 155*math.cos(ang_m), cy + 155*math.sin(ang_m)], fill=(37, 99, 235), width=7)
    d.line([cx, cy, cx + 105*math.cos(ang_h), cy + 105*math.sin(ang_h)], fill=(15, 23, 42), width=11)
    d.ellipse([cx-10, cy-10, cx+10, cy+10], fill=(220, 38, 38))
    p = save_pil(im, "V11_analog_clock.png")
    tasks.append({"task_id": "V11_analog_clock", "module": "V3_Visual_Spatial", "img_path": p,
                  "prompt": "Görseldeki analog saat tam olarak kaçı göstermektedir (12 saatlik HH:MM biçiminde, örn. 07:15)? Kısa siyah ibre akrep (saat), uzun mavi ibre yelkovan (dakika). JSON döndür: {\"time\": \"HH:MM\"}.",
                  "check": "v11_clock", "expected": "03:40"})

    # V12: Coordinate Localization (Mavi Drone at x=60..80, y=20..40 -> center (70, 30))
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 10)); ax.set_yticks(range(0, 101, 10))
    ax.grid(True, linestyle="--", alpha=0.55)
    for (x0, y0, w, h, col, name) in [
        (10, 60, 20, 20, "#EF4444", "Kırmızı Kamyon"),
        (60, 20, 20, 20, "#2563EB", "Mavi Drone"),
        (20, 10, 16, 16, "#10B981", "Yeşil Baret"),
    ]:
        rect = plt.Rectangle((x0, y0), w, h, facecolor=col, edgecolor="black", lw=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x0 + w/2, y0 + h + 2.5, name, ha="center", fontweight="bold", fontsize=10, color=col)
    ax.set_xlabel("X Koordinatı (0-100)"); ax.set_ylabel("Y Koordinatı (0-100)")
    ax.set_title("Nesne Konumlandırma Haritası", fontweight="bold")
    p = save_fig(fig, "V12_bbox_coord.png")
    tasks.append({"task_id": "V12_bbox_center", "module": "V3_Visual_Spatial", "img_path": p,
                  "prompt": "Koordinat haritasındaki 'Mavi Drone' dikdörtgeninin tam merkez noktası (x, y) nedir? JSON döndür: {\"x\": <tam_sayı>, \"y\": <tam_sayı>}.",
                  "check": "v12_coord", "expected": (70, 30)})

    # V13: Dark-mode Code Bug Screenshot
    im = Image.new("RGB", (820, 440), (15, 23, 42))
    d = ImageDraw.Draw(im)
    f_c = get_font(17, False); f_cb = get_font(18, True)
    d.text((25, 18), "metrics.py — VS Code", fill=(148, 163, 184), font=f_cb)
    code_lines = [
        " 1 | def compute_f1_scores(tp_list, fp_list, fn_list):",
        " 2 |     results = {}",
        " 3 |     for idx in range(len(tp_list)):",
        " 4 |         tp = tp_list[idx]",
        " 5 |         fp = fp_list[idx]",
        " 6 |         fn = fn_list[idx]",
        " 7 |         precision = tp / (tp + fp)",
        " 8 |         recall = tp / (tp + fn)",
        " 9 |         results[idx] = 2 * precision * recall / (precision + recall)",
        "10 |     return results",
        "11 |",
        "12 | # Terminal Output:",
        "13 | # compute_f1_scores([5, 0], [1, 0], [2, 3])",
        "14 | # Traceback (most recent call last):",
        "15 | #   File 'metrics.py', line 7, in compute_f1_scores",
        "16 | # ZeroDivisionError: division by zero",
    ]
    y0 = 58
    for i, ln in enumerate(code_lines):
        col = (248, 113, 113) if i >= 13 else ((148, 163, 184) if i >= 11 else (226, 232, 240))
        d.text((25, y0), ln, fill=col, font=f_c)
        y0 += 23
    p = save_pil(im, "V13_code_screenshot.png")
    tasks.append({"task_id": "V13_code_bug_screen", "module": "V4_UI_Code_Diag", "img_path": p,
                  "prompt": "Kod ekran görüntüsündeki hatanın oluştuğu satır numarası (line) ve istisna türü (exception) nedir? JSON döndür: {\"line\": <tam_sayı>, \"exception\": \"...\"}.",
                  "check": "json_kv", "expected": {"line": 7, "exception": "ZeroDivisionError"}})

    # V14: GPU Cluster Dashboard KPI Screenshot
    im = Image.new("RGB", (860, 420), (15, 23, 42))
    d = ImageDraw.Draw(im)
    d.text((30, 22), "EVREN GPU KÜME İZLEME PANELİ", fill=(248, 250, 252), font=get_font(22, True))
    nodes = [
        ("Düğüm-1", "74%", "64°C", "Aktif"),
        ("Düğüm-2", "82%", "71°C", "Aktif"),
        ("Düğüm-3", "95%", "83°C", "Yüksek Yük"),
        ("Düğüm-4", "61%", "58°C", "Aktif"),
    ]
    for i, (nm, vram, temp, st) in enumerate(nodes):
        x0 = 30 + i * 205
        border = (239, 68, 68) if nm == "Düğüm-3" else (51, 65, 85)
        d.rectangle([x0, 85, x0+185, 360], fill=(30, 41, 59), outline=border, width=3)
        d.text((x0+20, 105), nm, fill=(248, 250, 252), font=get_font(20, True))
        d.text((x0+20, 155), f"VRAM: {vram}", fill=(56, 189, 248), font=get_font(19, True))
        d.text((x0+20, 205), f"Sıcaklık: {temp}", fill=(251, 191, 36), font=get_font(19, True))
        d.text((x0+20, 265), f"Durum: {st}", fill=(248, 113, 113) if nm=="Düğüm-3" else (74, 222, 128), font=get_font(16, True))
    p = save_pil(im, "V14_gpu_dashboard.png")
    tasks.append({"task_id": "V14_dashboard_anomaly", "module": "V4_UI_Code_Diag", "img_path": p,
                  "prompt": "İzleme panelinde VRAM kullanımı %90'ı aşan düğüm hangisidir ve sıcaklığı kaç derecedir? JSON döndür: {\"node\": \"Düğüm-?\", \"temp_c\": <tam_sayı>}.",
                  "check": "json_kv", "expected": {"node": "Düğüm-3", "temp_c": 83}})

    # V15: Flowchart Execution Trace (x=14 -> x%2==0 True -> x=x//2=7 -> x>10 False -> x=x*3+5=26 -> Output 26)
    im = Image.new("RGB", (760, 560), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fb = get_font(18, True); fr = get_font(16, False)
    def box(xc, yc, w, h, txt, fill=(239, 246, 255), border=(37, 99, 235)):
        d.rectangle([xc-w//2, yc-h//2, xc+w//2, yc+h//2], fill=fill, outline=border, width=2)
        d.text((xc - len(txt)*4.5, yc - 10), txt, fill=(15, 23, 42), font=fb)
    box(380, 50, 220, 48, "BAŞLA: x = 14", (220, 252, 231), (22, 163, 74))
    d.line([380, 74, 380, 120], fill=(15, 23, 42), width=3)
    box(380, 145, 240, 50, "x çift sayı mı?", (254, 243, 199), (217, 119, 6))
    # EVET -> left (x = x / 2)
    d.line([260, 145, 170, 145, 170, 210], fill=(22, 163, 74), width=3)
    d.text((190, 120), "EVET", fill=(22, 163, 74), font=fb)
    box(170, 235, 200, 48, "x = x / 2")
    # HAYIR -> right (x = x + 3)
    d.line([500, 145, 590, 145, 590, 210], fill=(220, 38, 38), width=3)
    d.text((515, 120), "HAYIR", fill=(220, 38, 38), font=fb)
    box(590, 235, 200, 48, "x = x + 3")
    # Merge to x > 10?
    d.line([170, 259, 170, 325, 260, 325], fill=(15, 23, 42), width=3)
    d.line([590, 259, 590, 325, 500, 325], fill=(15, 23, 42), width=3)
    box(380, 325, 240, 50, "x > 10 mu?", (254, 243, 199), (217, 119, 6))
    # EVET -> left (x = x - 4)
    d.line([260, 325, 170, 325, 170, 400], fill=(22, 163, 74), width=3)
    d.text((185, 345), "EVET", fill=(22, 163, 74), font=fb)
    box(170, 425, 200, 48, "x = x - 4")
    # HAYIR -> right (x = x * 3 + 5)
    d.line([500, 325, 590, 325, 590, 400], fill=(220, 38, 38), width=3)
    d.text((515, 345), "HAYIR", fill=(220, 38, 38), font=fb)
    box(590, 425, 210, 48, "x = x * 3 + 5")
    # Merge to SONUÇ
    d.line([170, 449, 170, 515, 270, 515], fill=(15, 23, 42), width=3)
    d.line([590, 449, 590, 515, 490, 515], fill=(15, 23, 42), width=3)
    box(380, 515, 220, 48, "YAZDIR: x", (224, 231, 255), (79, 70, 229))
    p = save_pil(im, "V15_flowchart.png")
    tasks.append({"task_id": "V15_flowchart_trace", "module": "V4_UI_Code_Diag", "img_path": p,
                  "prompt": "Görseldeki akış şemasını x = 14 başlangıç değeriyle adım adım izleyin. En altta yazdırılan son x tam sayısı kaçtır? JSON döndür: {\"output\": <tam_sayı>}.",
                  "check": "json_int_key", "key": "output", "expected": 26})

    return tasks

def parse_json_loose(text):
    if not text: return None
    # Try fenced json first, then last {...}
    m = re.findall(r"\{[^{}]*\}", text, re.S)
    for cand in reversed(m):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None

def eval_vision(task, content):
    j = parse_json_loose(content or "")
    chk = task["check"]
    if not j:
        return False, 0.0, f"No valid JSON found in: {(content or '')[:60]}"
    if chk == "json_int":
        val = j.get("answer")
        try:
            ok = int(round(float(str(val).replace(".", "").replace(",", "")))) == task["expected"]
            return ok, 1.0 if ok else 0.0, "OK" if ok else f"Got {val} != {task['expected']}"
        except Exception:
            return False, 0.0, f"Bad int: {val}"
    if chk == "json_int_key":
        val = j.get(task["key"])
        try:
            ok = int(round(float(str(val)))) == task["expected"]
            return ok, 1.0 if ok else 0.0, "OK" if ok else f"Got {val} != {task['expected']}"
        except Exception:
            return False, 0.0, f"Bad int: {val}"
    if chk == "json_kv":
        for k, exp in task["expected"].items():
            got = j.get(k)
            if isinstance(exp, int):
                try:
                    if int(round(float(str(got).replace(".", "").replace(",", "")))) != exp:
                        return False, 0.0, f"Key {k}: got {got} != {exp}"
                except Exception:
                    return False, 0.0, f"Key {k}: bad int {got}"
            else:
                if str(got or "").strip().lower() != str(exp).strip().lower():
                    return False, 0.0, f"Key {k}: got {got} != {exp}"
        return True, 1.0, "OK"
    if chk == "v06_lab":
        cnt = j.get("count")
        abn = " ".join(str(x).lower() for x in (j.get("abnormal") or []))
        ok = (int(cnt or -1) == 2) and ("ferritin" in abn) and ("glukoz" in abn)
        return ok, 1.0 if ok else 0.0, "OK" if ok else f"Got count={cnt}, abn={abn}"
    if chk == "v11_clock":
        got = str(j.get("time") or "").strip()
        ok = got in ("03:40", "3:40", "15:40")
        return ok, 1.0 if ok else 0.0, "OK" if ok else f"Got time={got} != 03:40"
    if chk == "v12_coord":
        try:
            gx, gy = float(j.get("x")), float(j.get("y"))
            ex, ey = task["expected"]
            ok = abs(gx - ex) <= 4.0 and abs(gy - ey) <= 4.0
            return ok, 1.0 if ok else 0.0, "OK" if ok else f"Got ({gx},{gy}) != ({ex},{ey})"
        except Exception:
            return False, 0.0, f"Bad coord: {j}"
    return False, 0.0, "Unknown check"

def call_vision(base_url, api_key, model, b64_img, prompt, max_tokens=4096, timeout=300):
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ],
        }],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    delays = [15, 30, 50, 75, 100]
    retry_count = http_503 = 0
    for att in range(len(delays) + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code == 200:
                data = r.json()
                ch = (data.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                u = data.get("usage") or {}
                cd = u.get("completion_tokens_details") or {}
                ct = int(u.get("completion_tokens") or 0)
                return dict(ok=True, status=200, retry_count=retry_count, http_503=http_503,
                            total_ms=round(ms, 1), tok_per_sec=round(ct/(ms/1000.0), 2) if ms>0 and ct>0 else 0.0,
                            finish_reason=ch.get("finish_reason"), content=msg.get("content") or "",
                            prompt_tokens=int(u.get("prompt_tokens") or 0), completion_tokens=ct,
                            reasoning_tokens=int(cd.get("reasoning_tokens") or 0), total_tokens=int(u.get("total_tokens") or 0))
            if r.status_code in (429, 500, 502, 503, 504) and att < len(delays):
                if r.status_code == 503: http_503 += 1
                retry_count += 1; time.sleep(delays[att]); continue
            return dict(ok=False, status=r.status_code, retry_count=retry_count, http_503=http_503, total_ms=round(ms, 1), error=r.text[:200])
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            if att < 2:
                retry_count += 1; time.sleep(20); continue
            return dict(ok=False, status=0, retry_count=retry_count, http_503=http_503, total_ms=round(ms, 1), error=str(e)[:200])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    base_url = os.environ.get("EVREN_LLM_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1").rstrip("/")
    fallback = os.environ.get("EVREN_LLM_API_KEY", "")
    keys = {
        "reasoning_off": os.environ.get("EVREN_LLM_API_KEY_REASONING_OFF") or fallback,
        "reasoning_on": os.environ.get("EVREN_LLM_API_KEY_REASONING_ON") or fallback,
    }
    if not keys["reasoning_off"] and not keys["reasoning_on"]:
        raise SystemExit("Set EVREN_LLM_API_KEY before running live benchmark.")

    tasks = build_vision_tasks()
    b64_by_id = {t["task_id"]: base64.b64encode(open(t["img_path"], "rb").read()).decode() for t in tasks}

    model_modes = [
        ("gemma-4-31b", "reasoning_off", 4096),
        ("gemma-4-31b", "reasoning_on", 16384),
        ("deepseek-v4.1-flash", "reasoning_off", 4096),
        ("deepseek-v4.1-flash", "reasoning_on", 16384),
        ("qwen3.8-flash-next", "reasoning_off", 4096),
        ("qwen3.8-flash-next", "reasoning_on", 16384),
        ("qwen3-vl-30b", "reasoning_off", 4096),
    ]

    done_keys = set()
    if os.path.exists(RES_JSONL):
        for ln in open(RES_JSONL):
            if ln.strip():
                r = json.loads(ln)
                if (r.get("response") or {}).get("status_code") == 200:
                    done_keys.add(r["run_key"])

    by_model = defaultdict(list)
    total = 0
    for m, mode, mt in model_modes:
        for t in tasks:
            rk = f"{t['task_id']}|{m}|{mode}"
            if rk not in done_keys:
                by_model[m].append((t, mode, mt, rk))
                total += 1

    print(f"=== EK TEST 3: VISION_V1 (15 tasks x 7 combos = 105 | pending={total}) ===", flush=True)
    lock = threading.Lock()
    cnt = [0]

    def worker(model, items):
        for t, mode, mt, rk in items:
            res = call_vision(base_url, keys[mode], model, b64_by_id[t["task_id"]], t["prompt"], max_tokens=mt, timeout=360)
            ok, sc, rsn = eval_vision(t, res.get("content")) if res.get("status") == 200 else (False, 0.0, str(res.get("error")))
            rec = {
                "run_key": rk, "task_id": t["task_id"], "module": t["module"], "model": model, "mode": mode,
                "request": {"max_tokens": mt, "temperature": 0.0, "img_path": t["img_path"]},
                "response": {"status_code": res.get("status"), "finish_reason": res.get("finish_reason"), "content": res.get("content"), "error": res.get("error")},
                "telemetry": {"retry_count": res.get("retry_count", 0), "http_503_count": res.get("http_503", 0), "prompt_tokens": res.get("prompt_tokens", 0), "completion_tokens": res.get("completion_tokens", 0), "reasoning_tokens": res.get("reasoning_tokens", 0), "total_tokens": res.get("total_tokens", 0), "total_ms": res.get("total_ms", 0), "tok_per_sec": res.get("tok_per_sec", 0)},
                "evaluation": {"passed": ok, "score": sc, "reason": rsn},
                "created_at_tsi": datetime.datetime.now(TSI).strftime("%Y-%m-%d %H:%M:%S TSİ"),
            }
            with lock:
                with open(RES_JSONL, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jf.flush(); os.fsync(jf.fileno())
                cnt[0] += 1
                print(f"[{cnt[0]:3d}/{total}] {'PASS' if ok else 'FAIL':4s} | {t['task_id']:22s} | {model:19s} | {mode:13s} | {res.get('total_ms',0)/1000:5.1f}s | {rsn[:42]}", flush=True)
            time.sleep(1.5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        futs = [ex.submit(worker, m, items) for m, items in by_model.items()]
        for f in concurrent.futures.as_completed(futs):
            f.result()

    # Final summary
    rows = [json.loads(ln) for ln in open(RES_JSONL) if ln.strip()]
    latest = {r["run_key"]: r for r in rows}
    by_mm = defaultdict(list)
    for r in latest.values():
        by_mm[(r["model"], r["mode"])].append(r)
    print("\n================ VISION_V1 FINAL SUMMARY ================", flush=True)
    for (m, md), lst in sorted(by_mm.items(), key=lambda kv: (-sum(1 for x in kv[1] if x["evaluation"]["passed"]), kv[0])):
        n200 = sum(1 for x in lst if (x.get("response") or {}).get("status_code") == 200)
        p = sum(1 for x in lst if x["evaluation"]["passed"])
        print(f"  {m:20s} | {md:13s} | 200={n200:2d}/15 | Pass={p:2d}/15 ({p/15*100:5.1f}%)", flush=True)
    print("VISION_V1_ALL_DONE", flush=True)

if __name__ == "__main__":
    main()
