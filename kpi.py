"""
diagnose_kpi.py — "F1 neden düşük?" sorusunu veriye dayandırır.

Pipeline'ı YENİDEN ÇALIŞTIRMAZ. Mevcut benchmark_results_enhanced.json
içindeki missed_gt / false_positives kayıtlarını okuyup her kaçırılan GT
olayı için kabahatin nerede olduğunu sınıflandırır:

  ZAMAN     : hiçbir tahmin bu GT aralığıyla zamanda örtüşmüyor
              -> model olayı yanlış saniyeye koymuş VEYA GT süresi bozuk
  METİN     : zamanda örtüşen bir tahmin var ama metin benzerliği eşiğin altında
              -> model olayı gördü, farklı kelimelerle anlattı
  TAHMİN_YOK: model o video için hiç (eşleşmemiş) tahmin üretmemiş

Kullanım:
    python diagnose_kpi.py
    python diagnose_kpi.py --input benchmark_results_enhanced.json --top 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from kpi_enhanced import (TurkishTextProcessor, TEXT_THRESHOLD,
                              IOM_THRESHOLD, tolerance_for_duration)
except ImportError:
    print("❌ kpi_enhanced.py aynı klasörde olmalı.")
    sys.exit(1)


def _overlap(a: List[float], b: List[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _iom(a: List[float], b: List[float]) -> float:
    la, lb = a[1] - a[0], b[1] - b[0]
    denom = min(la, lb)
    return _overlap(a, b) / denom if denom > 0 else 0.0


def classify_miss(miss: Dict[str, Any],
                  fps: List[Dict[str, Any]]) -> Tuple[str, Optional[float], Optional[float]]:
    """Kaçırılan bir GT olayı için kabahati sınıflandır."""
    gt_iv = miss.get("interval")
    if not fps:
        return "TAHMIN_YOK", None, None
    if not gt_iv:
        return "ZAMAN", None, None

    best_iom, best_sim, best_sim_at_overlap = 0.0, 0.0, 0.0
    for fp in fps:
        fp_iv = fp.get("interval")
        if not fp_iv:
            continue
        iom = _iom(gt_iv, fp_iv)
        sim = TurkishTextProcessor.similarity(miss.get("text", ""), fp.get("text", ""))
        best_iom = max(best_iom, iom)
        best_sim = max(best_sim, sim)
        if iom >= IOM_THRESHOLD:
            best_sim_at_overlap = max(best_sim_at_overlap, sim)

    if best_iom < IOM_THRESHOLD:
        return "ZAMAN", best_iom, best_sim
    return "METIN", best_iom, best_sim_at_overlap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmark_results_enhanced.json")
    ap.add_argument("--top", type=int, default=8, help="Kaç kötü video listelensin")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"] if isinstance(data, dict) and "results" in data else data
    run_summary = data.get("run_summary", {}) if isinstance(data, dict) else {}

    reasons: Counter = Counter()
    per_video: List[Dict[str, Any]] = []
    total_gt = total_tp = total_fp = 0
    crit_missed = 0
    near_miss_text: List[Tuple[str, str, str, float, float]] = []

    for row in results:
        vid = row["video_id"]
        ev = row["metrics"]["layer_analysis"]["event_analysis"]
        misses = ev.get("missed_gt", [])
        fps = ev.get("false_positives", [])

        total_gt += ev.get("total_gt_events", 0)
        total_tp += ev.get("tp", 0)
        total_fp += ev.get("fp", 0)

        vid_reasons: Counter = Counter()
        for miss in misses:
            reason, iom, sim = classify_miss(miss, fps)
            reasons[reason] += 1
            vid_reasons[reason] += 1
            if miss.get("critical"):
                crit_missed += 1
            if reason == "METIN" and sim is not None:
                near_miss_text.append(
                    (vid, miss.get("text", "")[:48],
                     max((f.get("text", "") for f in fps), key=len, default="")[:48],
                     iom or 0.0, sim))

        per_video.append({
            "video": vid,
            "f1": row["metrics"]["accuracy"]["event_f1_score"],
            "gt": ev.get("total_gt_events", 0),
            "pred": ev.get("detected_events", 0),
            "tp": ev.get("tp", 0), "fp": ev.get("fp", 0), "fn": ev.get("fn", 0),
            "tol": ev.get("tolerance_sec"),
            "reasons": dict(vid_reasons),
            "unparsable": ev.get("unparsable_pred_times", 0),
        })

    total_missed = sum(reasons.values())

    print("\n" + "=" * 70)
    print("KAÇIRILAN OLAYLARIN KABAHAT DAĞILIMI")
    print("=" * 70)
    print(f"Toplam GT olayı      : {total_gt}")
    print(f"Eşleşen (TP)         : {total_tp}")
    print(f"Kaçırılan (FN)       : {total_missed}")
    print(f"Fazladan tahmin (FP) : {total_fp}")
    print(f"Kaçırılan kritik olay: {crit_missed}")
    print("-" * 70)
    for reason, count in reasons.most_common():
        pct = 100 * count / total_missed if total_missed else 0
        label = {
            "ZAMAN": "ZAMAN     — zamanda örtüşen tahmin yok",
            "METIN": "METİN     — zaman tuttu, kelimeler tutmadı",
            "TAHMIN_YOK": "TAHMİN_YOK— model o videoda tahmin üretmemiş",
        }.get(reason, reason)
        bar = "█" * int(pct / 2.5)
        print(f"  {label:<45} {count:>4}  %{pct:5.1f} {bar}")

    print("\n" + "=" * 70)
    print("YORUM")
    print("=" * 70)
    z = 100 * reasons.get("ZAMAN", 0) / total_missed if total_missed else 0
    t = 100 * reasons.get("METIN", 0) / total_missed if total_missed else 0
    if z > 55:
        print("→ Baskın sorun ZAMAN. Bu genelde modelin olayı gördüğü ama yanlış")
        print("  saniyeye koyduğu ya da GT'nin zaman ekseninin kaymış olduğu anlamına")
        print("  gelir. Önce duration uyarısı olan videoları ayıklayın; onlarda GT")
        print("  zaten farklı bir zaman eksenine göre etiketlenmiş olabilir.")
    elif t > 55:
        print("→ Baskın sorun METİN. Model olayları doğru anda tespit ediyor ama")
        print("  GT'den farklı kelimelerle anlatıyor. Bu bir tespit sorunu DEĞİL,")
        print("  bir ifade/eşik sorunu. Vision prompt'unda GT'ye yakın terminoloji")
        print("  istemek veya TEXT_THRESHOLD'u düşürmek gündeme gelir.")
    else:
        print("→ Zaman ve metin sorunları dengeli dağılmış; ikisi de tek başına")
        print("  baskın değil. Aşağıdaki video listesinden başlayın.")

    warnings = run_summary.get("duration_warnings", [])
    if warnings:
        bad = {w.split(":")[0] for w in warnings}
        affected = [v for v in per_video if v["video"] in bad]
        clean = [v for v in per_video if v["video"] not in bad]
        f1_bad = sum(v["f1"] for v in affected) / len(affected) if affected else 0
        f1_clean = sum(v["f1"] for v in clean) / len(clean) if clean else 0
        print("\n" + "=" * 70)
        print("SÜRE TUTARSIZLIĞININ ETKİSİ")
        print("=" * 70)
        print(f"  Süresi tutarsız {len(affected):>2} video -> ort. F1 = {f1_bad:.3f}")
        print(f"  Süresi tutarlı  {len(clean):>2} video -> ort. F1 = {f1_clean:.3f}")
        if f1_clean > f1_bad + 0.05:
            print(f"\n  ⚠️  Fark {f1_clean - f1_bad:.3f}. Düşük F1'in kayda değer bir kısmı")
            print("      ground_truth.json'daki yanlış duration_sec / kaymış zaman")
            print("      ekseninden geliyor. Model hatası değil, etiket hatası.")
        else:
            print("\n  Süre tutarsızlığı F1 üzerinde belirgin bir fark yaratmamış;")
            print("  sorun başka yerde.")

    print("\n" + "=" * 70)
    print(f"EN KÖTÜ {args.top} VİDEO")
    print("=" * 70)
    print(f"{'video':<24}{'F1':>6}{'GT':>4}{'Pr':>4}{'TP':>4}{'FP':>4}{'FN':>4}"
          f"{'tol':>6}  kabahat")
    for v in sorted(per_video, key=lambda x: x["f1"])[:args.top]:
        r = ", ".join(f"{k}×{n}" for k, n in v["reasons"].items()) or "-"
        tol = f"{v['tol']:.1f}" if v["tol"] else "-"
        print(f"{v['video']:<24}{v['f1']:>6.2f}{v['gt']:>4}{v['pred']:>4}"
              f"{v['tp']:>4}{v['fp']:>4}{v['fn']:>4}{tol:>6}  {r}")

    unparsable = sum(v["unparsable"] for v in per_video)
    if unparsable:
        print(f"\n⚠️  {unparsable} tahminin zaman alanı ayrıştırılamadı "
              f"(eşleşme şansı yok). Vision çıktısındaki time formatını kontrol edin.")

    if near_miss_text:
        print("\n" + "=" * 70)
        print("ZAMAN TUTTU, METİN TUTMADI — en yakın kaçanlar")
        print("=" * 70)
        print(f"(eşik: metin ≥ {TEXT_THRESHOLD})")
        for vid, gt_t, pred_t, iom, sim in sorted(
                near_miss_text, key=lambda x: -x[4])[:6]:
            print(f"\n  {vid}  IoM={iom:.2f}  benzerlik={sim:.2f}")
            print(f"    GT   : {gt_t}")
            print(f"    Pred : {pred_t}")

    print("\n" + "=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())