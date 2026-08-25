"""
calibrate_threshold.py — TEXT_THRESHOLD'u düşürmek işe yarar mı?

Mevcut benchmark_results_enhanced.json üzerinden çalışır, pipeline'ı
yeniden çalıştırmaz.

Mantık: eşiği DÜŞÜRMEK yalnızca şu an eşleşmemiş çiftler arasında yeni
eşleşme yaratabilir. Yani (missed_gt x false_positives) kümesindeki,
zamanda zaten örtüşen çiftler tam olarak "eşik düşerse kazanılacaklar"dır.
Bu küme JSON'da mevcut, dolayısıyla süpürmeyi kesin olarak hesaplayabiliriz.

Çıktı iki soruyu cevaplar:
  1. Eşiği T'ye düşürsem F1 ne olur?
  2. O F1 artışı GERÇEK eşleşmelerden mi geliyor, yoksa zamanda örtüşen
     ama alakasız çiftlerden mi? (örnekleri gözle kontrol edersiniz)

Kullanım:
    python calibrate_threshold.py
    python calibrate_threshold.py --samples 12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

try:
    from kpi_enhanced import TurkishTextProcessor, IOM_THRESHOLD, TEXT_THRESHOLD
except ImportError:
    print("❌ kpi_enhanced.py aynı klasörde olmalı.")
    sys.exit(1)

SWEEP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def _iom(a: List[float], b: List[float]) -> float:
    la, lb = a[1] - a[0], b[1] - b[0]
    denom = min(la, lb)
    if denom <= 0:
        return 0.0
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / denom


def collect(results: List[Dict[str, Any]]):
    """Her video için: mevcut TP sayısı, GT sayısı, pred sayısı ve
    zamanda örtüşen ama eşleşmemiş aday çiftler."""
    videos = []
    all_candidates: List[Tuple[str, str, str, float, float]] = []

    for row in results:
        ev = row["metrics"]["layer_analysis"]["event_analysis"]
        misses = ev.get("missed_gt", [])
        fps = ev.get("false_positives", [])

        cands: List[Tuple[int, int, float, float]] = []
        for gi, miss in enumerate(misses):
            g_iv = miss.get("interval")
            if not g_iv:
                continue
            for pi, fp in enumerate(fps):
                p_iv = fp.get("interval")
                if not p_iv:
                    continue
                iom = _iom(g_iv, p_iv)
                if iom < IOM_THRESHOLD:
                    continue
                sim = TurkishTextProcessor.similarity(
                    miss.get("text", ""), fp.get("text", ""))
                cands.append((gi, pi, iom, sim))
                all_candidates.append(
                    (row["video_id"], miss.get("text", ""), fp.get("text", ""), iom, sim))

        videos.append({
            "video_id": row["video_id"],
            "n_gt": ev.get("total_gt_events", 0),
            "n_pred": ev.get("detected_events", 0),
            "base_tp": ev.get("tp", 0),
            "candidates": cands,
        })
    return videos, all_candidates


def sweep_threshold(videos: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    """Verilen eşikte toplam TP/FP/FN ve makro-ortalama F1."""
    tot_tp = tot_fp = tot_fn = 0
    f1s: List[float] = []

    for v in videos:
        # eşiği geçen adaylar arasında bire-bir greedy atama
        elig = [c for c in v["candidates"] if c[3] >= threshold]
        elig.sort(key=lambda c: -(0.5 * c[2] + 0.5 * c[3]))
        used_g, used_p = set(), set()
        extra = 0
        for gi, pi, _, _ in elig:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            extra += 1

        tp = v["base_tp"] + extra
        fp = v["n_pred"] - tp
        fn = v["n_gt"] - tp
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)

    return {
        "threshold": threshold,
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmark_results_enhanced.json")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) and "results" in data else data

    videos, candidates = collect(results)

    print("\n" + "=" * 74)
    print("EŞİK SÜPÜRMESİ — TEXT_THRESHOLD düşürülürse ne olur?")
    print("=" * 74)
    print(f"Zamanda örtüşen ama metinde eşleşmemiş aday çift: {len(candidates)}")
    print(f"(IoM >= {IOM_THRESHOLD} şartını zaten sağlıyorlar)\n")

    print(f"{'eşik':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'makro F1':>11}   değişim")
    print("-" * 74)
    base = None
    rows = []
    for t in SWEEP:
        r = sweep_threshold(videos, t)
        rows.append(r)
        if abs(t - TEXT_THRESHOLD) < 1e-9:
            base = r
    for r in rows:
        marker = "  <= şu anki" if abs(r["threshold"] - TEXT_THRESHOLD) < 1e-9 else ""
        delta = ""
        if base and not marker:
            d = r["macro_f1"] - base["macro_f1"]
            delta = f"{d:+.3f}"
        print(f"{r['threshold']:>6.2f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
              f"{r['macro_f1']:>11.3f}   {delta}{marker}")

    # Benzerlik dağılımı — ayrışma var mı?
    print("\n" + "=" * 74)
    print("ADAY ÇİFTLERİN BENZERLİK DAĞILIMI")
    print("=" * 74)
    buckets = Counter()
    for _, _, _, _, sim in candidates:
        buckets[round(sim // 0.05 * 0.05, 2)] += 1
    for b in sorted(buckets):
        bar = "█" * int(buckets[b] * 40 / max(buckets.values()))
        flag = " <- eşik" if b <= TEXT_THRESHOLD < b + 0.05 else ""
        print(f"  {b:.2f}-{b + 0.05:.2f}  {buckets[b]:>4}  {bar}{flag}")

    print("\n  Yorum: dağılım eşiğin etrafında TEK TEPELİ ise, doğru ve yanlış")
    print("  eşleşmeler aynı bölgede toplanmış demektir — eşiği oynatmak ikisini")
    print("  birden içeri alır ya da dışarı atar. ÇİFT TEPELİ ise ayrım mümkündür.")

    print("\n" + "=" * 74)
    print(f"EŞİK 0.10'A DÜŞÜRÜLSE KABUL EDİLECEK ÇİFTLER (ilk {args.samples})")
    print("=" * 74)
    print("Bunları tek tek okuyun: AYNI olay mı, farklı olay mı?")
    print("Çoğu farklıysa eşiği düşürmek F1'i şişirir, ölçümü bozar.\n")
    admitted = sorted([c for c in candidates if 0.10 <= c[4] < TEXT_THRESHOLD],
                      key=lambda c: -c[4])
    if not admitted:
        print("  (0.10–0.15 aralığında aday yok)")
    for vid, gt_t, pred_t, iom, sim in admitted[:args.samples]:
        print(f"  {vid}  IoM={iom:.2f}  benzerlik={sim:.2f}")
        print(f"    GT   : {gt_t[:78]}")
        print(f"    Pred : {pred_t[:78]}")
        print()

    # Tekrarlanan tahmin tespiti
    print("=" * 74)
    print("TEKRARLANAN / GENEL TAHMİNLER")
    print("=" * 74)
    dup_total = 0
    for row in results:
        ev = row["metrics"]["layer_analysis"]["event_analysis"]
        texts = [f["text"] for f in ev.get("false_positives", [])]
        texts += [m["pred"] for m in ev.get("matches", [])]
        norm = [TurkishTextProcessor.normalize(t)[:60] for t in texts if t]
        counts = Counter(norm)
        dups = {t: c for t, c in counts.items() if c > 1}
        if dups:
            dup_total += sum(dups.values()) - len(dups)
            print(f"  {row['video_id']}: {len(dups)} tekrarlı metin")
            for t, c in list(dups.items())[:2]:
                print(f"      x{c}  {t[:62]}")
    if dup_total == 0:
        print("  Tekrarlanan tahmin metni yok.")
    else:
        print(f"\n  Toplam {dup_total} fazlalık. Model aynı cümleyi birden fazla olay")
        print("  olarak üretiyorsa bu doğrudan FP'ye dönüşür.")

    print("\n" + "=" * 74 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())